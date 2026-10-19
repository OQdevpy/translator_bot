"""Real-time interpreter web app.

The browser cuts microphone (or tab) audio into speech segments at pauses and
sends each one over a WebSocket as WAV; Gemini detects the language and
translates it (other languages -> Uzbek, Uzbek -> chosen partner language),
and the result comes back on the same socket. One port serves everything.
"""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import struct
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from google.genai import types
from pydantic import BaseModel

from translator import DEFAULT_PARTNER, HISTORY_SIZE, LANGS, PARTNER_LANGS, Interpreter, Speaker
from usage import UsageLog

STATIC_DIR = Path(__file__).parent / "static"
COOKIE = "stt_session"
MAX_SEGMENT_BYTES = 2_000_000  # ~60 s of 16 kHz 16-bit mono WAV
MAX_TTS_CHARS = 1000
MAX_QUEUED_SEGMENTS = 20  # per connection; protects the API budget from a runaway client

log = logging.getLogger("stt_web")


class TtsRequest(BaseModel):
    text: str


def session_token(password: str) -> str:
    return hmac.new(password.encode(), b"stt-web-session", hashlib.sha256).hexdigest()


def same_origin(headers) -> bool:
    """Blocks cross-site WebSocket use of the session cookie."""
    origin = headers.get("origin")
    return not origin or urlsplit(origin).netloc in (headers.get("host"), headers.get("x-forwarded-host"))


def create_app(interpreter: Interpreter, speaker: Speaker | None = None, password: str = "") -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    token = session_token(password) if password else ""

    def authorized(cookies) -> bool:
        return not password or hmac.compare_digest(cookies.get(COOKIE, ""), token)

    def require_auth(request: Request):
        if not authorized(request.cookies):
            raise HTTPException(401, "login required")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index(request: Request):
        page = "index.html" if authorized(request.cookies) else "login.html"
        return FileResponse(STATIC_DIR / page, headers={"Cache-Control": "no-store"})

    @app.post("/login")
    def login(request: Request, password_: str = Form("", alias="password")):
        if not password or not hmac.compare_digest(session_token(password_), token):
            return RedirectResponse("/?error=1", status_code=303)
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(
            COOKIE,
            token,
            max_age=30 * 24 * 3600,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
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
            "default_partner": DEFAULT_PARTNER,
            "tts": speaker is not None,
            "auth": bool(password),
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

    @app.websocket("/ws")
    async def live(ws: WebSocket):
        await ws.accept()  # accept first: a rejected handshake reaches the browser only as code 1006
        if not authorized(ws.cookies) or not same_origin(ws.headers):
            await ws.close(code=4401)
            return
        state = {"partner": DEFAULT_PARTNER}
        history = deque(maxlen=HISTORY_SIZE)
        segments: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()

        async def translate_segments():
            # One at a time, in order, so every segment sees the previous ones as context.
            while True:
                seg_id, wav = await segments.get()
                part = types.Part.from_bytes(data=wav, mime_type="audio/wav")
                try:
                    result = await asyncio.to_thread(
                        interpreter.interpret, [part], state["partner"], list(history), True
                    )
                except Exception:
                    log.exception("Segment %s failed", seg_id)
                    await ws.send_json({"type": "error", "id": seg_id, "message": "Tarjima qilib bo'lmadi"})
                    continue
                if not result.text or not result.translation:
                    await ws.send_json({"type": "empty", "id": seg_id})
                    continue
                history.append(result)
                await ws.send_json(
                    {
                        "type": "result",
                        "id": seg_id,
                        "source_lang": result.source_lang,
                        "text": result.text,
                        "translation": result.translation,
                        "target": state["partner"] if result.source_lang == "uz" else "uz",
                    }
                )

        worker = asyncio.create_task(translate_segments())
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    # frame: 4-byte big-endian segment id + WAV file
                    if len(data) < 48 or data[4:8] != b"RIFF" or data[12:16] != b"WAVE":
                        await ws.send_json({"type": "error", "id": None, "message": "Noto'g'ri audio"})
                        continue
                    (seg_id,) = struct.unpack(">I", data[:4])
                    if len(data) > MAX_SEGMENT_BYTES:
                        await ws.send_json({"type": "error", "id": seg_id, "message": "Segment juda uzun"})
                        continue
                    if segments.qsize() >= MAX_QUEUED_SEGMENTS:
                        await ws.send_json({"type": "error", "id": seg_id, "message": "Navbat to'lib ketdi"})
                        continue
                    await segments.put((seg_id, data[4:]))
                elif message.get("text") is not None:
                    try:
                        msg = json.loads(message["text"])
                    except ValueError:
                        continue
                    if msg.get("type") == "config" and msg.get("partner") in PARTNER_LANGS:
                        state["partner"] = msg["partner"]
                    elif msg.get("type") == "reset":
                        history.clear()
        except WebSocketDisconnect:
            pass
        finally:
            worker.cancel()

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
    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    os.chmod(key_path, 0o600)
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    log.info("Self-signed certificate created: %s", cert_path)


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_dir = os.getenv("DATA_DIR", "data")
    usage_log = UsageLog(os.path.join(data_dir, "usage.db"))
    password = os.getenv("WEB_PASSWORD", "")
    if not password:
        log.warning("WEB_PASSWORD is empty: anyone who opens the site uses your API keys")
    app = create_app(Interpreter.from_env(usage_log), Speaker.from_env(usage_log), password)
    host, port = os.getenv("HOST", "0.0.0.0"), int(os.getenv("PORT", "8000"))
    cert, key = ssl_files(data_dir)
    log.info("Web %s://%s:%s", "https" if cert else "http", host, port)
    uvicorn.run(app, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key, proxy_headers=True)


if __name__ == "__main__":
    main()
