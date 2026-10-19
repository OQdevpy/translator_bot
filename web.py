"""Live seminar interpreter web app.

The browser cuts microphone (or tab) audio into phrases at pauses and streams them
over a WebSocket as WAV. seminar.LiveSession transcribes each phrase, works out who
is speaking and interprets every finished thought into Uzbek with the whole seminar
as context. Sessions are stored, so they can run all day, and a report can be made
for any time range. One port serves the page, the API and the WebSocket.
"""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import report
from seminar import KEEP_UZBEK, LiveSession, SeminarAI, SeminarStore, build_report, public_speaker
from translator import LANGS, PARTNER_LANGS, Interpreter, Speaker
from usage import UsageLog
from voiceprint import Voiceprints

STATIC_DIR = Path(__file__).parent / "static"
COOKIE = "stt_session"
MAX_SEGMENT_BYTES = 2_000_000  # ~60 s of 16 kHz 16-bit mono WAV
MAX_TTS_CHARS = 1000
MAX_QUEUED_SEGMENTS = 20  # per session; protects the API budget from a runaway client
UNITS_ON_PAGE = 400  # older parts stay in the database and the reports
SESSION_SECONDS = 30 * 24 * 3600
LOGIN_FREE_FAILS = 5  # wrong passwords per IP before it is locked out
LOGIN_LOCK_SECONDS = 60  # doubles with every further failure
LOGIN_LOCK_MAX = 3600
LOGIN_FAIL_DELAY = 1.0  # seconds; slows down guessing without holding a worker thread
UNLOAD_CHECK_SECONDS = 5.0
MAX_CLOCK_SKEW = 24 * 3600  # browser timestamps further off than this are ignored

log = logging.getLogger("stt_web")


class TtsRequest(BaseModel):
    text: str


class NewSession(BaseModel):
    title: str = ""


class SessionUpdate(BaseModel):
    title: str | None = None
    partner: str | None = None


class SpeakerUpdate(BaseModel):
    label: str


class SpeakerMerge(BaseModel):
    into: str


class ReportRequest(BaseModel):
    start: float | None = None
    end: float | None = None


def sign_session(secret: bytes, password: str, expires: int) -> str:
    return hmac.new(secret, f"{password}\n{expires}".encode(), hashlib.sha256).hexdigest()


def new_session(secret: bytes, password: str) -> str:
    expires = int(time.time()) + SESSION_SECONDS
    return f"{expires}.{sign_session(secret, password, expires)}"


def valid_session(secret: bytes, password: str, value: str) -> bool:
    """Signed with a server secret, so the cookie reveals nothing about the password; expires server-side."""
    expires, _, signature = value.partition(".")
    if not expires.isdigit() or int(expires) < time.time():
        return False
    return hmac.compare_digest(signature, sign_session(secret, password, int(expires)))


def same_origin(headers) -> bool:
    """Blocks cross-site WebSocket use of the session cookie."""
    origin = headers.get("origin")
    return not origin or urlsplit(origin).netloc in (headers.get("host"), headers.get("x-forwarded-host"))


def create_app(
    ai: SeminarAI,
    store: SeminarStore,
    voiceprints: Voiceprints | None = None,
    speaker: Speaker | None = None,
    password: str = "",
    secret: bytes | None = None,
    tz: ZoneInfo = ZoneInfo("Asia/Tashkent"),
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    secret = secret or secrets.token_bytes(32)
    login_failures: dict[str, tuple[int, float]] = {}  # ip -> (failures, locked until)
    live: dict[int, LiveSession] = {}  # sessions with connected clients or unfinished work

    def authorized(cookies) -> bool:
        return not password or valid_session(secret, password, cookies.get(COOKIE, ""))

    def require_auth(request: Request):
        if not authorized(request.cookies):
            raise HTTPException(401, "login required")

    def session_or_404(session_id: int) -> dict:
        row = store.session(session_id)
        if row is None:
            raise HTTPException(404, "session not found")
        return row

    def get_live(session_id: int) -> LiveSession:
        if session_id not in live:
            live[session_id] = LiveSession(session_id, store, ai, voiceprints, tz)
        return live[session_id]

    async def unload_when_idle(sess: LiveSession):
        """Keeps the session in memory until nobody is connected and its work is done."""
        while not sess.clients:
            if not sess.busy() and sess.idle_timer is None:
                sess.close()
                if live.get(sess.id) is sess:
                    live.pop(sess.id)
                return
            await asyncio.sleep(UNLOAD_CHECK_SECONDS)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index(request: Request):
        page = "index.html" if authorized(request.cookies) else "login.html"
        return FileResponse(STATIC_DIR / page, headers={"Cache-Control": "no-store"})

    @app.post("/login")
    async def login(request: Request, password_: str = Form("", alias="password")):
        ip = request.client.host if request.client else "?"
        now = time.monotonic()
        failures, locked_until = login_failures.get(ip, (0, 0.0))
        if locked_until > now:
            return RedirectResponse("/?error=locked", status_code=303)
        if not password or not hmac.compare_digest(password_.encode(), password.encode()):
            failures += 1
            lock = 0.0
            if failures >= LOGIN_FREE_FAILS:
                lock = min(LOGIN_LOCK_SECONDS * 2 ** (failures - LOGIN_FREE_FAILS), LOGIN_LOCK_MAX)
            if len(login_failures) > 10_000:
                login_failures.clear()
            login_failures[ip] = (failures, now + lock)
            log.warning("Wrong web password from %s (%d)", ip, failures)
            await asyncio.sleep(LOGIN_FAIL_DELAY)
            return RedirectResponse("/?error=1", status_code=303)
        login_failures.pop(ip, None)
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(
            COOKIE,
            new_session(secret, password),
            max_age=SESSION_SECONDS,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
        )
        return resp

    @app.post("/logout")
    def logout():
        resp = RedirectResponse("/", status_code=303)
        resp.delete_cookie(COOKIE)
        return resp

    @app.get("/api/config")
    def config(request: Request):
        require_auth(request)
        return {
            "langs": {code: {"flag": flag, "name": name} for code, (flag, name, _) in LANGS.items()},
            "partners": PARTNER_LANGS,
            "tts": speaker is not None,
            "auth": bool(password),
            "voiceprints": voiceprints is not None,
        }

    @app.post("/api/tts")
    async def tts(body: TtsRequest, request: Request):
        require_auth(request)
        if speaker is None:
            raise HTTPException(503, "TTS is not configured")
        text = body.text.strip()
        if not text or len(text) > MAX_TTS_CHARS:
            raise HTTPException(400, f"text must be 1..{MAX_TTS_CHARS} characters")
        try:
            audio = await asyncio.to_thread(speaker.synthesize, text)
        except Exception:
            log.exception("TTS failed")
            raise HTTPException(502, "TTS failed")
        return Response(audio, media_type="audio/mpeg")

    # ----- sessions

    @app.get("/api/sessions")
    def list_sessions(request: Request):
        require_auth(request)
        return store.sessions()

    @app.post("/api/sessions")
    def create_session(body: NewSession, request: Request):
        require_auth(request)
        title = body.title.strip()[:200] or f"Seminar {datetime.now(tz):%d.%m.%Y %H:%M}"
        return store.create_session(title)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: int, request: Request):
        require_auth(request)
        row = session_or_404(session_id)
        sess = live.get(session_id)
        pending = sess.pending if sess else store.pending_segments(session_id)
        speakers = sess.speakers.values() if sess else store.speakers(session_id)
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "partner": row["partner"],
            "notes": row["notes"],
            "speakers": [public_speaker(sp) for sp in speakers],
            "units": store.units(session_id, limit=UNITS_ON_PAGE),
            "units_on_page": UNITS_ON_PAGE,
            "pending": [{"speaker": p["speaker"], "lang": p["lang"], "text": p["text"]} for p in pending],
            "reports": store.reports(session_id),
        }

    @app.patch("/api/sessions/{session_id}")
    def update_session(session_id: int, body: SessionUpdate, request: Request):
        require_auth(request)
        session_or_404(session_id)
        if body.title is not None and body.title.strip():
            store.update_session(session_id, title=body.title.strip()[:200])
        if body.partner is not None:
            if body.partner not in [KEEP_UZBEK, *PARTNER_LANGS]:
                raise HTTPException(400, "unknown language")
            if session_id in live:
                live[session_id].set_partner(body.partner)
            else:
                store.update_session(session_id, partner=body.partner)
        return {"ok": True}

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: int, request: Request):
        require_auth(request)
        session_or_404(session_id)
        sess = live.pop(session_id, None)
        if sess:
            sess.close()
        store.delete_session(session_id)
        return {"ok": True}

    @app.patch("/api/sessions/{session_id}/speakers/{speaker_id}")
    async def rename_speaker(session_id: int, speaker_id: str, body: SpeakerUpdate, request: Request):
        require_auth(request)
        session_or_404(session_id)
        label = body.label.strip()[:80]
        sess = get_live(session_id)
        if speaker_id not in sess.speakers or not label:
            raise HTTPException(400, "unknown speaker or empty name")
        await sess.rename_speaker(speaker_id, label)
        asyncio.create_task(unload_when_idle(sess))
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/speakers/{speaker_id}/merge")
    async def merge_speaker(session_id: int, speaker_id: str, body: SpeakerMerge, request: Request):
        require_auth(request)
        session_or_404(session_id)
        sess = get_live(session_id)
        if speaker_id == body.into or speaker_id not in sess.speakers or body.into not in sess.speakers:
            raise HTTPException(400, "unknown speaker")
        await sess.merge_speakers(speaker_id, body.into)
        asyncio.create_task(unload_when_idle(sess))
        return {"ok": True}

    # ----- reports

    @app.post("/api/sessions/{session_id}/reports")
    async def make_report(session_id: int, body: ReportRequest, request: Request):
        require_auth(request)
        session_or_404(session_id)
        start = body.start if body.start is not None else 0.0
        end = body.end if body.end is not None else time.time() + 60
        if start >= end:
            raise HTTPException(400, "Boshlanish vaqti tugashdan oldin bo'lishi kerak")
        try:
            report_id = await build_report(store, ai, session_id, start, end, tz)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception:
            log.exception("Report failed for session %s", session_id)
            raise HTTPException(502, "Hisobot tayyorlab bo'lmadi, qayta urinib ko'ring")
        return {"id": report_id}

    def report_parts(report_id: int):
        row = store.report(report_id)
        if row is None:
            raise HTTPException(404, "report not found")
        session = store.session(row["session_id"])
        units = store.units(row["session_id"], row["start"], row["end"])
        speakers = {sp["id"]: sp["label"] for sp in store.speakers(row["session_id"])}
        return row, session, units, speakers

    @app.get("/api/reports/{report_id:int}.md")
    def report_markdown(report_id: int, request: Request):
        require_auth(request)
        md = report.to_markdown(*report_parts(report_id), tz)
        return PlainTextResponse(
            md,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="hisobot-{report_id}.md"'},
        )

    @app.get("/api/reports/{report_id:int}")
    def report_page(report_id: int, request: Request):
        if not authorized(request.cookies):
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(report.to_html(*report_parts(report_id), tz))

    @app.get("/api/sessions/{session_id}/transcript.txt")
    def transcript(session_id: int, request: Request, start: float | None = None, end: float | None = None):
        require_auth(request)
        row = session_or_404(session_id)
        units = store.units(session_id, start, end)
        speakers = {sp["id"]: sp["label"] for sp in store.speakers(session_id)}
        text = f"{row['title']}\n\n" + report.transcript_text(units, speakers, tz)
        return PlainTextResponse(
            text, headers={"Content-Disposition": f'attachment; filename="transkript-{session_id}.txt"'}
        )

    # ----- live audio

    @app.websocket("/ws")
    async def live_socket(ws: WebSocket):
        await ws.accept()  # accept first: a rejected handshake reaches the browser only as code 1006
        if not authorized(ws.cookies):
            await ws.close(code=4401)  # the page reloads into the login form
            return
        if not same_origin(ws.headers):
            log.warning("WebSocket origin %s does not match host %s", ws.headers.get("origin"), ws.headers.get("host"))
            await ws.close(code=4403)  # usually a proxy that rewrites Host; the page shows a warning
            return
        session_id = ws.query_params.get("session", "")
        if not session_id.isdigit() or store.session(int(session_id)) is None:
            await ws.close(code=4404)
            return
        sess = get_live(int(session_id))
        sess.clients.add(ws)
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    # frame: 4-byte big-endian segment id, 8-byte capture end time, WAV file
                    if len(data) < 56 or data[12:16] != b"RIFF" or data[20:24] != b"WAVE":
                        await ws.send_json({"type": "error", "id": None, "message": "Noto'g'ri audio"})
                        continue
                    seg_id, ended_at = struct.unpack(">Id", data[:12])
                    if len(data) > MAX_SEGMENT_BYTES:
                        await ws.send_json({"type": "error", "id": seg_id, "message": "Segment juda uzun"})
                        continue
                    if sess.listen_queue.qsize() >= MAX_QUEUED_SEGMENTS:
                        await ws.send_json({"type": "error", "id": seg_id, "message": "Navbat to'lib ketdi"})
                        continue
                    now = time.time()
                    # the browser's clock keeps the timeline right for phrases sent late after a reconnect
                    received = ended_at if now - MAX_CLOCK_SKEW < ended_at < now + 60 else now
                    await sess.add_audio(seg_id, data[12:], received)
                elif message.get("text") is not None:
                    try:
                        msg = json.loads(message["text"])
                    except ValueError:
                        continue
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") == "partner" and msg.get("partner") in [KEEP_UZBEK, *PARTNER_LANGS]:
                        sess.set_partner(msg["partner"])
                    elif msg.get("type") == "flush":
                        sess.flush()  # listening stopped: interpret what was said so far
        except WebSocketDisconnect:
            pass
        finally:
            sess.clients.discard(ws)
            if not sess.clients:
                asyncio.create_task(unload_when_idle(sess))

    return app


def ssl_files(data_dir: str) -> tuple[str, str] | tuple[None, None]:
    """Certificate for HTTPS: SSL_CERTFILE/SSL_KEYFILE, or a generated self-signed one."""
    cert, key = os.getenv("SSL_CERTFILE"), os.getenv("SSL_KEYFILE")
    if cert and key:
        return cert, key
    if os.getenv("SSL_SELF_SIGNED", "").lower() not in ("1", "true", "yes"):
        return None, None
    cert, key = os.path.join(data_dir, "selfsigned.crt"), os.path.join(data_dir, "selfsigned.key")
    if not (os.path.exists(cert) and os.path.exists(key)):
        generate_self_signed(cert, key)
    return cert, key


def generate_self_signed(cert_path: str, key_path: str):
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "stt-web")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    os.makedirs(os.path.dirname(cert_path) or ".", exist_ok=True)
    with open(os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    log.info("Self-signed certificate created: %s", cert_path)


def session_secret(data_dir: str) -> bytes:
    """Kept in the data dir so logins survive restarts."""
    path = os.path.join(data_dir, "session.key")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
            f.write(secrets.token_bytes(32))
    with open(path, "rb") as f:
        return f.read()


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_dir = os.getenv("DATA_DIR", "data")
    usage_log = UsageLog(os.path.join(data_dir, "usage.db"))
    password = os.getenv("WEB_PASSWORD", "")
    if not password:
        log.warning("WEB_PASSWORD is empty: anyone who opens the site uses your API keys")
    tz = ZoneInfo(os.getenv("TIMEZONE") or "Asia/Tashkent")
    voiceprints = None if os.getenv("VOICEPRINTS", "1").lower() in ("0", "false", "no") else Voiceprints.load(data_dir)
    app = create_app(
        SeminarAI(Interpreter.from_env(usage_log), os.getenv("GEMINI_REPORT_MODEL") or None),
        SeminarStore(os.path.join(data_dir, "seminars.db")),
        voiceprints,
        Speaker.from_env(usage_log),
        password,
        session_secret(data_dir),
        tz,
    )
    host, port = os.getenv("HOST", "0.0.0.0"), int(os.getenv("PORT", "8000"))
    cert, key = ssl_files(data_dir)
    log.info("Web %s://%s:%s", "https" if cert else "http", host, port)
    uvicorn.run(app, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key, proxy_headers=True)


if __name__ == "__main__":
    main()
