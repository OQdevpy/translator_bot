"""Tests for the seminar web app (web.py): login and security, the sessions API, speakers,
reports and the live WebSocket.

Gemini (SeminarAI), ElevenLabs and the voiceprint model are fakes; nothing touches the network.
Every app runs inside one TestClient portal (one event loop), like under uvicorn, so the
LiveSession workers created by one request keep running for the next one.
"""
import asyncio
import json
import math
import re
import ssl
import struct
import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import seminar
import web
from conftest import make_wav
from seminar import Heard, Notes, Rendered, Report, SeminarStore

TZ = ZoneInfo("Asia/Tashkent")
WAV = make_wav(1.0)  # exactly 1.0 s for seminar.wav_seconds
BASE = datetime(2026, 3, 2, 10, 0, tzinfo=TZ).timestamp()  # a seminar morning, safely in the past


# ---------------------------------------------------------------- fakes


def heard(text="Привет", lang="ru", speaker="new", name="", voice="", complete=True) -> Heard:
    return Heard(speaker=speaker, speaker_name=name, voice=voice, source_lang=lang, text=text, complete=complete)


def make_report(**overrides) -> Report:
    data = {
        "title": "Hisobot",
        "overview": "Birinchi xatboshi.\nIkkinchi xatboshi.",
        "speakers": [{"speaker_id": "S1", "name": "", "role": "", "summary": "Salomlashdi.", "key_points": ["Salom"]}],
        "timeline": [{"time": "10:00", "speaker_id": "S1", "event": "Ochilish"}],
        "topics": [{"title": "Kirish", "summary": "Tanishuv"}],
        "qa": [{"time": "10:10", "asked_by": "S2", "question": "Savol?", "answered_by": "S1", "answer": "Javob."}],
        "decisions": ["Qaror"],
        "action_items": ["Vazifa"],
        "glossary": [{"term": "API", "translation": "API", "note": "interfeys"}],
    }
    data.update(overrides)
    return Report.model_validate(data)


class FakeAI:
    """Stands in for SeminarAI: scripted answers, recorded calls, gates that hold a stage."""

    def __init__(self):
        self.heards: list = []  # Heard or Exception per listen() call; default heard() when empty
        self.interpret_errors: list[Exception] = []
        self.report_result: Report | Exception = make_report()
        self.listen_calls, self.interpret_calls, self.notes_calls, self.report_calls = [], [], [], []
        self.listen_gate: threading.Event | None = None
        self.interpret_gate: threading.Event | None = None
        self.listening = threading.Event()  # set once listen() is entered
        self.interpreting = threading.Event()

    def listen(self, wav, references, similarities, recent, notes, identified=None):
        self.listening.set()
        if self.listen_gate:
            assert self.listen_gate.wait(5)
        self.listen_calls.append(SimpleNamespace(
            wav=wav, references=[r["id"] for r in references], similarities=dict(similarities),
            recent=list(recent), notes=dict(notes),
        ))
        result = self.heards.pop(0) if self.heards else heard()
        if isinstance(result, Exception):
            raise result
        return result

    def interpret(self, text, source_lang, speaker, target, recent, notes):
        self.interpreting.set()
        if self.interpret_gate:
            assert self.interpret_gate.wait(5)
        self.interpret_calls.append(SimpleNamespace(
            text=text, source_lang=source_lang, speaker=speaker, target=target, recent=list(recent), notes=dict(notes),
        ))
        if self.interpret_errors:
            raise self.interpret_errors.pop(0)
        return Rendered(translation=f"uz: {text}", note="")

    def update_notes(self, notes, new_lines, speakers):
        self.notes_calls.append(SimpleNamespace(notes=dict(notes), new_lines=list(new_lines), speakers=list(speakers)))
        return Notes(summary="Seminar haqida", glossary=[{"term": "AI", "translation": "sun'iy intellekt", "note": ""}])

    def report(self, transcript, speakers, period):
        self.report_calls.append(SimpleNamespace(transcript=transcript, speakers=speakers, period=period))
        if isinstance(self.report_result, Exception):
            raise self.report_result
        return self.report_result


class FakeSpeaker:
    def __init__(self, fail=False):
        self.texts = []
        self.fail = fail

    def synthesize(self, text):
        if self.fail:
            raise RuntimeError("ElevenLabs 500")
        self.texts.append(text)
        return b"ID3-mp3"


class FakeVoiceprints:
    """embed() returns the given vectors in order."""

    def __init__(self, *vectors):
        self.vectors = [np.asarray(v, dtype=np.float32) for v in vectors]

    def embed(self, wav):
        return self.vectors.pop(0)


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(web, "LOGIN_FAIL_DELAY", 0)
    monkeypatch.setattr(web, "UNLOAD_CHECK_SECONDS", 0.02)
    monkeypatch.setattr(web, "UNLOAD_GRACE_SECONDS", 0.0)  # tests that need the grace period set it
    monkeypatch.setattr(seminar, "RETRY_DELAY", 0.01)
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 30.0)  # thoughts are flushed explicitly unless a test shortens it


@pytest.fixture
def store(tmp_path):
    return SeminarStore(str(tmp_path / "seminars.db"))


@pytest.fixture
def ai():
    return FakeAI()


@pytest.fixture
def make(store, ai):
    """make(**options) -> a started TestClient for a new app; all are stopped after the test."""
    clients = []

    def make(password="", speaker=None, voiceprints=None, secret=None, **client_options):
        app = web.create_app(ai, store, voiceprints, speaker, password, secret, TZ)
        client = TestClient(app, **client_options)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for gate in (ai.listen_gate, ai.interpret_gate):
        if gate:
            gate.set()  # never leave a worker thread blocked at shutdown
    for client in reversed(clients):
        client.__exit__(None, None, None)


@pytest.fixture
def client(make):
    return make()


@pytest.fixture
def sid(store):
    return store.create_session("Seminar")["id"]


@pytest.fixture
def lives(monkeypatch):
    """Every LiveSession the app loads, with how often and from where it was closed."""
    loaded = []

    class SpyLiveSession(seminar.LiveSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            self.closed_off_loop = False
            loaded.append(self)

        def close(self):
            try:
                asyncio.get_running_loop()
            except RuntimeError:  # called from a worker thread, not from the event loop
                self.closed_off_loop = True
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(web, "LiveSession", SpyLiveSession)
    return loaded


def frame(seg_id: int, wav: bytes = WAV, age: float = 0.0) -> bytes:
    """Browser frame: 4-byte big-endian id, 8-byte big-endian float64 seconds since the phrase ended, WAV."""
    return struct.pack(">Id", seg_id, age) + wav


def recv(ws, timeout: float = 3.0) -> dict:
    """Next JSON message from the server; fails instead of hanging when nothing comes."""

    async def receive():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()

    message = ws.portal.call(receive)
    ws._raise_on_close(message)
    return json.loads(message["text"])


def recv_until(ws, kind: str, timeout: float = 3.0) -> tuple[dict, list[dict]]:
    """Reads messages up to the first one of this type; returns it and everything seen."""
    seen = []
    while True:
        seen.append(recv(ws, timeout))
        if seen[-1]["type"] == kind:
            return seen[-1], seen


def wait_for(condition, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the condition")
        time.sleep(0.01)


def login(client, password="secret", **kwargs):
    resp = client.post("/login", data={"password": password}, follow_redirects=False, **kwargs)
    assert resp.status_code == 303 and resp.headers["location"] == "/"
    return resp


def expect_close(client, code: int, url: str, **kwargs):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(url, **kwargs) as ws:
            # a wrongly accepted socket answers (or times out) here instead of closing
            ws.send_bytes(frame(1))
            recv(ws)
    assert exc.value.code == code


def add_speaker(store, sid, speaker_id, label, voice="", centroid=None, samples=0, ref_wav=None, named=False):
    store.save_speaker(sid, {
        "id": speaker_id, "label": label, "named": named, "voice": voice, "ref_wav": ref_wav,
        "ref_seconds": 1.0 if ref_wav else 0.0,
        "centroid": None if centroid is None else np.asarray(centroid, dtype=np.float32), "samples": samples,
    })


def speaker_row(store, sid, speaker_id) -> dict:
    return next(sp for sp in store.speakers(sid) if sp["id"] == speaker_id)


def add_unit(store, sid, start, end, speaker, source, translation, lang="ru", target="uz", note="") -> dict:
    return store.add_unit(sid, {"start": start, "end": end, "speaker": speaker, "lang": lang, "target": target,
                                "source": source, "translation": translation, "note": note}, [])


@pytest.fixture
def seeded(store, sid):
    """A morning of a seminar: two speakers, three units, the last one left untranslated."""
    add_speaker(store, sid, "S1", "Dr. Karimov")
    add_speaker(store, sid, "S2", "Spiker 2")
    return [
        add_unit(store, sid, BASE, BASE + 10, "S1", "Добрый день", "Xayrli kun"),
        add_unit(store, sid, BASE + 600, BASE + 610, "S2", "Any questions?", "Savollar bormi?", lang="en"),
        add_unit(store, sid, BASE + 3600, BASE + 3605, "S1", "Rahmat", "", lang="uz"),
    ]


# ---------------------------------------------------------------- pages, config, static files


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_open_access_without_password(client):
    page = client.get("/")
    assert page.status_code == 200 and "Jonli tarjimon" in page.text and "/static/app.js" in page.text
    assert page.headers["cache-control"] == "no-store"
    config = client.get("/api/config").json()
    assert config["partners"] == ["zh", "tr", "en", "ru"]
    assert set(config["langs"]) == {"uz", "zh", "tr", "en", "ru"}
    assert config["langs"]["uz"] == {"flag": "🇺🇿", "name": "O'zbekcha"}
    assert config["tts"] is False and config["auth"] is False and config["voiceprints"] is False


def test_config_reports_tts_and_voiceprints(make):
    config = make(speaker=FakeSpeaker(), voiceprints=FakeVoiceprints()).get("/api/config").json()
    assert config["tts"] is True and config["voiceprints"] is True


def test_static_files_are_served(client):
    for path in ("/static/app.js", "/static/audio.js", "/static/pcm-worklet.js", "/static/style.css"):
        assert client.get(path).status_code == 200, path
    assert client.get("/static/nope.js").status_code == 404


# ---------------------------------------------------------------- login and cookies


def test_password_login_flow(make):
    client = make(password="secret")
    assert 'name="password"' in client.get("/").text
    assert client.get("/api/config").status_code == 401

    wrong = client.post("/login", data={"password": "nope"}, follow_redirects=False)
    assert wrong.status_code == 303 and wrong.headers["location"] == "/?error=1"
    assert "stt_session" not in client.cookies

    cookie = login(client).headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "secure" not in cookie
    assert "/static/app.js" in client.get("/").text
    assert client.get("/api/config").json()["auth"] is True

    out = client.post("/logout", follow_redirects=False)
    assert out.status_code == 303 and "max-age=0" in out.headers["set-cookie"].lower()
    assert "stt_session" not in client.cookies
    assert client.get("/api/config").status_code == 401
    assert 'name="password"' in client.get("/").text


def test_cookie_is_secure_behind_an_https_proxy(make):
    client = make(password="secret")
    cookie = login(client, headers={"x-forwarded-proto": "https"}).headers["set-cookie"].lower()
    assert "secure" in cookie


def test_cookie_from_another_password_is_rejected(make):
    old = make(password="old")
    login(old, "old")
    new = make(password="new")
    new.cookies.update(old.cookies)
    assert new.get("/api/config").status_code == 401


def test_session_cookie_is_signed_and_expires(make, monkeypatch):
    client = make(password="secret")
    login(client)
    value = client.cookies["stt_session"]
    expires, _, signature = value.partition(".")
    assert expires.isdigit() and len(signature) == 64
    assert "secret" not in value
    assert int(expires) == pytest.approx(time.time() + web.SESSION_SECONDS, abs=5)

    client.cookies.set("stt_session", f"{int(expires) + 999}.{signature}")  # extended by hand
    assert client.get("/api/config").status_code == 401
    client.cookies.set("stt_session", f"{expires}.{'0' * 64}")
    assert client.get("/api/config").status_code == 401
    client.cookies.set("stt_session", "garbage")
    assert client.get("/api/config").status_code == 401

    client.cookies.set("stt_session", value)
    assert client.get("/api/config").status_code == 200
    later = SimpleNamespace(time=lambda: time.time() + web.SESSION_SECONDS + 10, monotonic=time.monotonic)
    monkeypatch.setattr(web, "time", later)
    assert client.get("/api/config").status_code == 401


def test_session_survives_restart_with_the_same_secret(make):
    secret = b"k" * 32
    first = make(password="secret", secret=secret)
    login(first)
    second = make(password="secret", secret=secret)
    second.cookies.update(first.cookies)
    assert second.get("/api/config").status_code == 200
    third = make(password="secret")  # a random secret: old cookies are void
    third.cookies.update(first.cookies)
    assert third.get("/api/config").status_code == 401


def test_session_secret_file_is_created_once(tmp_path):
    a = web.session_secret(str(tmp_path / "data"))
    assert len(a) == 32 and web.session_secret(str(tmp_path / "data")) == a
    assert oct((tmp_path / "data" / "session.key").stat().st_mode & 0o777) == "0o600"
    assert web.session_secret(str(tmp_path / "other")) != a


def test_login_is_locked_after_repeated_wrong_passwords(make):
    client = make(password="secret")
    for _ in range(web.LOGIN_FREE_FAILS):
        resp = client.post("/login", data={"password": "guess"}, follow_redirects=False)
        assert resp.headers["location"] == "/?error=1"
    locked = client.post("/login", data={"password": "secret"}, follow_redirects=False)
    assert locked.headers["location"] == "/?error=locked"
    assert "stt_session" not in client.cookies


def test_a_few_typos_do_not_lock_the_owner_out(make):
    client = make(password="secret")
    for _ in range(web.LOGIN_FREE_FAILS - 1):
        client.post("/login", data={"password": "typo"}, follow_redirects=False)
    login(client)
    assert client.get("/api/config").status_code == 200


def test_lockout_expires_doubles_and_resets_after_success(make, monkeypatch):
    clock = SimpleNamespace(offset=0.0)
    monkeypatch.setattr(web, "time", SimpleNamespace(time=time.time, monotonic=lambda: time.monotonic() + clock.offset))
    client = make(password="secret")

    def attempt(password):
        return client.post("/login", data={"password": password}, follow_redirects=False).headers["location"]

    for _ in range(web.LOGIN_FREE_FAILS):
        attempt("guess")
    assert attempt("secret") == "/?error=locked"
    clock.offset += web.LOGIN_LOCK_SECONDS + 1
    assert attempt("guess") == "/?error=1"  # one more failure locks for twice as long
    clock.offset += web.LOGIN_LOCK_SECONDS + 1
    assert attempt("secret") == "/?error=locked"
    clock.offset += web.LOGIN_LOCK_SECONDS
    assert attempt("secret") == "/"
    # a successful login forgets the failures
    for _ in range(web.LOGIN_FREE_FAILS - 1):
        assert attempt("guess") == "/?error=1"
    assert attempt("secret") == "/"


def test_lockout_is_per_client_address(store, ai):
    app = web.create_app(ai, store, None, None, "secret", None, TZ)
    attacker = TestClient(app, client=("203.0.113.9", 4000))
    owner = TestClient(app, client=("198.51.100.7", 4000))
    for _ in range(web.LOGIN_FREE_FAILS + 2):
        attacker.post("/login", data={"password": "guess"}, follow_redirects=False)
    assert attacker.post("/login", data={"password": "secret"}, follow_redirects=False).headers["location"] == "/?error=locked"
    login(owner)


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("GET", "/api/config", None),
        ("GET", "/api/sessions", None),
        ("POST", "/api/sessions", {"title": "x"}),
        ("GET", "/api/sessions/{sid}", None),
        ("PATCH", "/api/sessions/{sid}", {"title": "hacked", "partner": "tr"}),
        ("DELETE", "/api/sessions/{sid}", None),
        ("PATCH", "/api/sessions/{sid}/speakers/S1", {"label": "x"}),
        ("POST", "/api/sessions/{sid}/speakers/S2/merge", {"into": "S1"}),
        ("POST", "/api/sessions/{sid}/reports", {}),
        ("GET", "/api/reports/1.md", None),
        ("GET", "/api/sessions/{sid}/transcript.txt", None),
        ("POST", "/api/tts", {"text": "x"}),
    ],
)
def test_api_requires_login(make, store, sid, seeded, ai, method, path, body):
    rid = store.save_report(sid, BASE, BASE + 10, make_report().model_dump())
    assert rid == 1
    client = make(password="secret", speaker=FakeSpeaker())
    resp = client.request(method, path.format(sid=sid), json=body)
    assert resp.status_code == 401
    row = store.session(sid)
    assert row["title"] == "Seminar" and row["partner"] == "uz"
    assert [sp["label"] for sp in store.speakers(sid)] == ["Dr. Karimov", "Spiker 2"]
    assert len(store.sessions()) == 1 and len(store.reports(sid)) == 1
    assert ai.report_calls == []


# ---------------------------------------------------------------- WebSocket access


def test_websocket_requires_login(make, sid):
    client = make(password="secret")
    expect_close(client, 4401, f"/ws?session={sid}")
    expect_close(client, 4401, "/ws?session=999")  # nothing about sessions is revealed before login
    login(client)
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        msg, _ = recv_until(ws, "heard")
        assert msg["id"] == 1


def test_websocket_rejects_other_origins(client, sid):
    url = f"/ws?session={sid}"
    # its own code, so the page warns instead of reloading forever
    expect_close(client, 4403, url, headers={"origin": "https://evil.example"})
    expect_close(client, 4403, url, headers={"origin": "null"})
    expect_close(client, 4403, url, headers={"origin": "http://testserver.evil.example"})
    for headers in (
        {"origin": "http://testserver"},
        {"origin": "https://testserver"},
        {"origin": "https://seminar.example", "x-forwarded-host": "seminar.example"},
        {},  # non-browser clients send no Origin
    ):
        with client.websocket_connect(url, headers=headers) as ws:
            ws.send_text("{}")
            ws.send_bytes(b"x")
            assert recv(ws)["type"] == "error"


@pytest.mark.parametrize("query", ["", "?session=", "?session=abc", "?session=-1", "?session=1.0", "?session=999", "?id=1"])
def test_websocket_needs_an_existing_session(client, sid, lives, query):
    assert sid == 1
    expect_close(client, 4404, f"/ws{query}")
    assert lives == []  # nothing was loaded for a bad request


# ---------------------------------------------------------------- live audio: the full flow


def test_phrase_is_heard_and_interpreted(client, ai, store, sid):
    ai.heards = [heard("Привет, коллеги", voice="deep male"), heard("Начнём", speaker="S1")]
    ended = time.time() - 5
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(7, WAV, age=5.0))  # ended 5 s ago
        unit_msg, seen = recv_until(ws, "unit")
        kinds = [m["type"] for m in seen]
        assert kinds.index("speaker") < kinds.index("heard") < kinds.index("unit")
        assert seen[kinds.index("speaker")] == {
            "type": "speaker", "speaker": {"id": "S1", "label": "Spiker 1", "voice": "deep male"},
        }
        assert seen[kinds.index("heard")] == {
            "type": "heard", "id": 7, "speaker": "S1", "lang": "ru", "text": "Привет, коллеги", "complete": True,
        }
        unit = unit_msg["unit"]
        assert unit["end"] == pytest.approx(ended, abs=1) and unit["start"] == pytest.approx(unit["end"] - 1.0)
        assert unit == {
            "id": unit["id"], "start": unit["start"], "end": unit["end"], "speaker": "S1", "lang": "ru", "target": "uz",
            "source": "Привет, коллеги", "translation": "uz: Привет, коллеги", "note": "",
        }
        listen = ai.listen_calls[0]
        assert listen.wav == WAV and listen.references == [] and listen.similarities == {} and listen.recent == []
        assert vars(ai.interpret_calls[0]) == {
            "text": "Привет, коллеги", "source_lang": "ru", "speaker": "Spiker 1", "target": "uz", "recent": [], "notes": {},
        }

        # the next phrase gets the conversation so far as context
        ws.send_bytes(frame(8))
        second, _ = recv_until(ws, "unit")
        assert second["unit"]["speaker"] == "S1" and second["unit"]["source"] == "Начнём"
        assert ai.listen_calls[1].references == ["S1"]
        assert ai.listen_calls[1].recent == ["[S1 Spiker 1] Привет, коллеги"]
        assert ai.interpret_calls[1].recent == ["[Spiker 1] uz: Привет, коллеги"]

        data = client.get(f"/api/sessions/{sid}").json()
    assert data["units"] == [unit, second["unit"]]
    assert data["speakers"] == [{"id": "S1", "label": "Spiker 1", "voice": "deep male"}]
    assert data["pending"] == []
    assert store.units(sid) == [unit, second["unit"]]


def test_voiceprints_decide_the_speaker_when_they_are_sure(make, ai, sid):
    voiceprints = FakeVoiceprints([1, 0], [1, 0], [0, 1])
    client = make(voiceprints=voiceprints)
    # Gemini is wrong twice; the voiceprints overrule it (long clips: short ones are left to Gemini)
    ai.heards = [heard("Bir"), heard("Ikki", speaker="new"), heard("Uch", speaker="S1")]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        units = []
        for i in (1, 2, 3):
            ws.send_bytes(frame(i, make_wav(3.0)))
            units.append(recv_until(ws, "unit")[0]["unit"])
    assert [u["speaker"] for u in units] == ["S1", "S1", "S2"]
    assert ai.listen_calls[1].similarities == {"S1": pytest.approx(1.0)}
    assert ai.listen_calls[2].similarities == {"S1": pytest.approx(0.0)}


def test_ai_failures_are_retried_then_reported(client, ai, sid, monkeypatch):
    ai.heards = [RuntimeError("Gemini 503"), RuntimeError("429"), heard("Снова"), RuntimeError("503"), heard("")]
    ai.interpret_errors = [RuntimeError("503"), RuntimeError("503")]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))  # fails twice, works on the third try
        assert recv_until(ws, "heard")[0]["text"] == "Снова"
        unit, _ = recv_until(ws, "unit")
        # interpreting failed twice: the original words are kept untranslated
        assert unit["unit"]["source"] == "Снова" and unit["unit"]["translation"] == ""
        monkeypatch.setattr(seminar, "LISTEN_RETRY_SECONDS", 0)  # the retry time is used up
        ws.send_bytes(frame(2))
        assert recv_until(ws, "error")[0] == {"type": "error", "id": 2, "message": "Eshitib bo'lmadi"}
        ws.send_bytes(frame(3))  # silence
        assert recv_until(ws, "empty")[0] == {"type": "empty", "id": 3}


def test_notes_are_updated_and_broadcast(client, ai, store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "NOTES_FIRST_AFTER_UNITS", 1, raising=False)
    monkeypatch.setattr(seminar, "NOTES_EVERY_UNITS", 1)
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        notes, _ = recv_until(ws, "notes")
    expected = {"summary": "Seminar haqida", "glossary": [{"term": "AI", "translation": "sun'iy intellekt", "note": ""}]}
    assert notes == {"type": "notes", "notes": expected}
    assert store.session(sid)["notes"] == expected
    assert client.get(f"/api/sessions/{sid}").json()["notes"] == expected


# ---------------------------------------------------------------- live audio: frames


def unit_for_age(client, sid, age) -> dict:
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1, WAV, age))
        return recv_until(ws, "unit")[0]["unit"]


@pytest.mark.parametrize("age", [0.0, 5.0, 3600.0, web.MAX_FRAME_AGE - 300])
def test_phrase_age_places_it_in_the_timeline(client, sid, age):
    """The age, not the device's clock: a phrase resent late keeps its real time, whatever the device's clock says."""
    before = time.time()
    unit = unit_for_age(client, sid, age)
    after = time.time()
    assert before - age <= unit["end"] <= after - age
    assert unit["start"] == pytest.approx(unit["end"] - 1.0)


@pytest.mark.parametrize(
    "age",
    [web.MAX_FRAME_AGE + 60, -1.0, -0.001, time.time(), time.time() * 1000, math.nan, math.inf, -math.inf],
    ids=["over-a-day", "negative", "slightly-negative", "absolute-time", "milliseconds", "nan", "inf", "-inf"],
)
def test_implausible_age_is_replaced_by_arrival_time(client, sid, age):
    before = time.time()
    unit = unit_for_age(client, sid, age)
    after = time.time()
    assert before <= unit["end"] <= after
    assert unit["start"] == pytest.approx(unit["end"] - 1.0)


def test_malformed_frames_are_rejected(client, ai, sid):
    header = struct.pack(">Id", 9, 0.0)
    bad = [
        b"",
        b"\x00\x01",
        header + b"x" * 60,  # not a WAV file
        struct.pack(">I", 9) + WAV,  # old frame format without the age
        header + b"RIFF" + WAV[4:8] + b"AVI " + WAV[12:],
        header + WAV[:43],  # cut inside the WAV header
    ]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        for data in bad:
            ws.send_bytes(data)
            assert recv(ws) == {"type": "error", "id": None, "message": "Noto'g'ri audio"}
        assert ai.listen_calls == []
        ws.send_bytes(frame(10))  # the socket still works
        assert recv_until(ws, "heard")[0]["id"] == 10
    assert len(ai.listen_calls) == 1


def test_oversized_frame_is_rejected(client, ai, sid, monkeypatch):
    monkeypatch.setattr(web, "MAX_SEGMENT_BYTES", len(frame(1)))
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(5, make_wav(1.01)))
        assert recv(ws) == {"type": "error", "id": 5, "message": "Segment juda uzun"}
        ws.send_bytes(frame(6))  # exactly at the limit
        assert recv_until(ws, "heard")[0]["id"] == 6
    assert len(ai.listen_calls) == 1 and ai.listen_calls[0].wav == WAV


def test_queue_limit_protects_the_budget(client, ai, sid, monkeypatch):
    monkeypatch.setattr(seminar, "MAX_QUEUED_SEGMENTS", 3)
    ai.listen_gate = threading.Event()
    ai.heards = [heard("")] * 5
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        assert ai.listening.wait(3)  # phrase 1 is being transcribed, the queue is empty again
        for i in range(2, 7):
            ws.send_bytes(frame(i))
        assert recv(ws) == {"type": "error", "id": 5, "message": "Navbat to'lib ketdi"}
        assert recv(ws) == {"type": "error", "id": 6, "message": "Navbat to'lib ketdi"}
        ai.listen_gate.set()
        assert [recv(ws) for _ in range(4)] == [{"type": "empty", "id": i} for i in (1, 2, 3, 4)]
        ws.send_bytes(frame(7))  # room again
        assert recv(ws) == {"type": "empty", "id": 7}
    assert len(ai.listen_calls) == 5


# ---------------------------------------------------------------- live audio: text messages


def test_partner_message_sets_the_target_for_uzbek_speech(client, ai, store, sid):
    ai.heards = [heard("Salom", lang="uz"), heard("Rahmat", lang="uz"), heard("Привет", lang="ru")]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_text(json.dumps({"type": "partner", "partner": "xx"}))  # ignored
        ws.send_text(json.dumps({"type": "partner"}))  # ignored
        ws.send_text(json.dumps({"type": "partner", "partner": "tr"}))
        ws.send_bytes(frame(1))
        assert recv_until(ws, "unit")[0]["unit"]["target"] == "tr"
        assert store.session(sid)["partner"] == "tr"

        ws.send_text(json.dumps({"type": "partner", "partner": "uz"}))  # keep Uzbek in Uzbek
        ws.send_bytes(frame(2))
        assert recv_until(ws, "unit")[0]["unit"]["target"] == "uz"

        ws.send_text(json.dumps({"type": "partner", "partner": "en"}))
        ws.send_bytes(frame(3))  # other languages always go into Uzbek
        assert recv_until(ws, "unit")[0]["unit"]["target"] == "uz"
    assert [c.target for c in ai.interpret_calls] == ["tr", "uz", "uz"]
    assert store.session(sid)["partner"] == "en"


def test_flush_message_interprets_the_unfinished_thought(client, ai, sid):
    ai.heards = [heard("Итак, сегодня", complete=False)]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_text(json.dumps({"type": "flush"}))  # nothing pending: no-op
        ws.send_bytes(frame(1))
        heard_msg, _ = recv_until(ws, "heard")
        assert heard_msg["complete"] is False
        pending, _ = recv_until(ws, "pending")
        assert pending == {
            "type": "pending", "pending": [{"speaker": "S1", "lang": "ru", "text": "Итак, сегодня"}], "interpreting": 0,
        }
        live = client.get(f"/api/sessions/{sid}").json()
        assert live["pending"] == pending["pending"]
        assert live["speakers"] == [{"id": "S1", "label": "Spiker 1", "voice": ""}]
        assert ai.interpret_calls == []

        ws.send_text(json.dumps({"type": "flush"}))  # listening stopped
        unit, _ = recv_until(ws, "unit")
        assert unit["unit"]["source"] == "Итак, сегодня"
        assert client.get(f"/api/sessions/{sid}").json()["pending"] == []
    assert len(ai.interpret_calls) == 1


def test_idle_pause_interprets_the_unfinished_thought(client, ai, sid, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.1)
    ai.heards = [heard("Итак", complete=False)]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        unit, _ = recv_until(ws, "unit")
    assert unit["unit"]["source"] == "Итак"


@pytest.mark.parametrize(
    "text",
    ["null", "1", "[]", '"x"', "{}", "not json", "NaN", '{"type": 5}', '{"type": "partner", "partner": ["tr"]}',
     '{"type": "partner", "partner": {"tr": 1}}', '{"type": "reset"}'],
)
def test_odd_text_messages_do_not_kill_the_socket(client, ai, store, sid, text):
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_text(text)
        ws.send_bytes(frame(1))
        assert recv_until(ws, "unit")[0]["unit"]["target"] == "uz"
    assert store.session(sid)["partner"] == "uz"


# ---------------------------------------------------------------- live audio: acknowledgements, resends, speech, ping


def kinds(messages) -> list[str]:
    return [m["type"] for m in messages]


def test_answers_about_a_phrase_go_only_to_the_device_that_sent_it(client, ai, sid):
    ai.heards = [heard("Привет"), heard("")]
    with client.websocket_connect(f"/ws?session={sid}&client=a") as sender, \
            client.websocket_connect(f"/ws?session={sid}&client=b") as other:
        sender.send_bytes(frame(1))
        _, seen_other = recv_until(other, "unit")
        heard_msg, seen_sender = recv_until(sender, "unit")
        assert "heard" in kinds(seen_sender) and "heard" not in kinds(seen_other)
        assert {"speaker", "unit"} <= set(kinds(seen_other))  # everything else is for every device
        sender.send_bytes(frame(2))  # silence
        assert recv_until(sender, "empty")[0] == {"type": "empty", "id": 2}
        other.send_text(json.dumps({"type": "ping"}))
        _, seen_other = recv_until(other, "pong")
        assert not {"heard", "empty", "pong"} & set(kinds(seen_other[:-1]))  # the pong only for who asked
        sender.send_text(json.dumps({"type": "ping"}))
        _, seen_sender = recv_until(sender, "pong")
        assert "pong" not in kinds(seen_sender[:-1])


def test_resent_phrase_is_acknowledged_without_processing_it_again(client, ai, sid, lives, monkeypatch):
    monkeypatch.setattr(web, "UNLOAD_GRACE_SECONDS", 30.0)
    ai.heards = [heard("Bir"), heard("Ikki", speaker="S1"), heard("Uch", speaker="S1")]
    url = f"/ws?session={sid}&client=page-1_A"
    with client.websocket_connect(url) as ws:
        ws.send_bytes(frame(1))
        recv_until(ws, "heard")
    with client.websocket_connect(url) as ws:  # its acknowledgement was lost: resent after a reconnect
        ws.send_bytes(frame(1, age=4.0))
        assert recv(ws) == {"type": "dup", "id": 1}
        ws.send_bytes(frame(2))
        assert recv_until(ws, "heard")[0]["id"] == 2
    with client.websocket_connect(f"/ws?session={sid}&client=page-2") as ws:  # another page's phrase 1
        ws.send_bytes(frame(1))
        assert recv_until(ws, "heard")[0] == {"type": "heard", "id": 1, "speaker": "S1", "lang": "ru", "text": "Uch",
                                              "complete": True}
    assert len(ai.listen_calls) == 3 and len(lives) == 1


def test_phrase_resent_while_still_queued_is_a_dup(client, ai, sid):
    ai.listen_gate = threading.Event()
    url = f"/ws?session={sid}&client=p"
    with client.websocket_connect(url) as ws:
        ws.send_bytes(frame(1))
        assert ai.listening.wait(3)
    with client.websocket_connect(url) as ws:
        ws.send_bytes(frame(1))
        assert recv(ws) == {"type": "dup", "id": 1}
        ai.listen_gate.set()
        _, seen = recv_until(ws, "unit")
        assert "heard" not in kinds(seen)  # the answer went to the socket that sent it
    assert len(ai.listen_calls) == 1


@pytest.mark.parametrize("query", ["", "&client=", "&client=bad%20id", "&client=" + "x" * 41, "&client=a.b"])
def test_without_a_valid_client_id_phrases_are_not_deduplicated(client, ai, sid, query):
    ai.heards = [heard("Bir"), heard("Ikki")]
    with client.websocket_connect(f"/ws?session={sid}{query}") as ws:
        for _ in range(2):
            ws.send_bytes(frame(1))
            assert recv_until(ws, "heard")[0]["id"] == 1
    assert len(ai.listen_calls) == 2


def test_speech_messages_hold_the_idle_flush(client, ai, store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.1)
    ai.heards = [heard("Главная проблема в том,", complete=False)]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_text(json.dumps({"type": "speech", "active": True}))  # the speaker goes on
        ws.send_bytes(frame(1))
        recv_until(ws, "pending")
        time.sleep(0.4)
        assert store.units(sid) == []
        ws.send_text(json.dumps({"type": "speech", "active": False}))
        stopped = time.monotonic()
        unit, _ = recv_until(ws, "unit")
        assert time.monotonic() - stopped >= 0.09  # the pause is counted from when the speech stopped
    assert unit["unit"]["source"] == "Главная проблема в том,"


@pytest.mark.parametrize("active", ["yes", 1, None])
def test_speech_active_must_be_true(client, ai, store, sid, monkeypatch, active):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.1)
    ai.heards = [heard("Итак", complete=False)]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_text(json.dumps({"type": "speech", "active": active}))
        ws.send_bytes(frame(1))
        assert recv_until(ws, "unit")[0]["unit"]["source"] == "Итак"


def test_a_device_that_disconnects_mid_speech_counts_as_silent(client, ai, store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.1)
    ai.heards = [heard("Итак", complete=False)]
    with client.websocket_connect(f"/ws?session={sid}") as listener:
        with client.websocket_connect(f"/ws?session={sid}") as recorder:
            recorder.send_text(json.dumps({"type": "speech", "active": True}))
            recorder.send_bytes(frame(1))
            recv_until(listener, "pending")
            time.sleep(0.3)
            assert store.units(sid) == []
        assert recv_until(listener, "unit")[0]["unit"]["source"] == "Итак"


def test_flush_waits_for_the_phrase_still_being_transcribed(client, ai, store, sid):
    """Stop is pressed right after the last phrase: it is part of the thought, not a separate unit."""
    ai.heards = [heard("Итак,", complete=False), heard("начнём", speaker="S1", complete=False)]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        recv_until(ws, "pending")
        ai.listening.clear()
        ai.listen_gate = threading.Event()
        ws.send_bytes(frame(2))
        assert ai.listening.wait(3)
        ws.send_text(json.dumps({"type": "flush"}))
        time.sleep(0.2)
        assert store.units(sid) == [] and ai.interpret_calls == []
        ai.listen_gate.set()
        unit, _ = recv_until(ws, "unit")
    assert unit["unit"]["source"] == "Итак, начнём"


def test_session_payload_counts_thoughts_in_interpretation(client, ai, sid):
    ai.interpret_gate = threading.Event()
    url = f"/api/sessions/{sid}"
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        assert ai.interpreting.wait(3)
        assert client.get(url).json()["interpreting"] == 1  # a page reloading now shows "being translated"
        ai.interpret_gate.set()
        recv_until(ws, "unit")
        wait_for(lambda: client.get(url).json()["interpreting"] == 0)


# ---------------------------------------------------------------- sessions API


def test_sessions_crud(client, store):
    assert client.get("/api/sessions").json() == []
    before = time.time()
    first = client.post("/api/sessions", json={"title": "  Kun 1  "}).json()
    assert first == {"id": first["id"], "title": "Kun 1", "created_at": first["created_at"], "partner": "uz", "notes": {}}
    assert before <= first["created_at"] <= time.time()
    second = client.post("/api/sessions", json={"title": "x" * 300}).json()
    assert second["title"] == "x" * 200

    add_unit(store, first["id"], BASE, BASE + 5, "S1", "a", "b")
    listed = client.get("/api/sessions").json()
    assert [s["id"] for s in listed] == [second["id"], first["id"]]  # newest first
    assert listed[1] | {"created_at": 0} == {
        "id": first["id"], "title": "Kun 1", "created_at": 0, "partner": "uz", "units": 1, "first": BASE, "last": BASE + 5,
    }
    assert listed[0]["units"] == 0 and listed[0]["first"] is None

    url = f"/api/sessions/{first['id']}"
    assert client.patch(url, json={"title": "  Kun birinchi  "}).json() == {"ok": True}
    assert client.get(url).json()["title"] == "Kun birinchi"
    assert client.patch(url, json={"title": "   "}).status_code == 200  # blank titles are ignored
    assert client.patch(url, json={}).status_code == 200
    assert client.patch(url, json={"title": "y" * 300}).status_code == 200
    assert store.session(first["id"])["title"] == "y" * 200
    for partner in ("tr", "uz", "zh"):
        assert client.patch(url, json={"partner": partner}).status_code == 200
        assert client.get(url).json()["partner"] == partner

    assert client.delete(url).json() == {"ok": True}
    assert client.get(url).status_code == 404
    assert client.delete(url).status_code == 404
    assert client.patch(url, json={"title": "z"}).status_code == 404
    assert [s["id"] for s in client.get("/api/sessions").json()] == [second["id"]]


@pytest.mark.parametrize("body", [{}, {"title": ""}, {"title": "   "}])
def test_new_session_gets_a_dated_title(client, body):
    title = client.post("/api/sessions", json=body).json()["title"]
    match = re.fullmatch(r"Seminar (\d\d\.\d\d\.\d{4} \d\d:\d\d)", title)
    assert match, title
    stamped = datetime.strptime(match.group(1), "%d.%m.%Y %H:%M")
    now = datetime.now(TZ).replace(tzinfo=None)  # the configured time zone, not the server's
    assert timedelta(0) <= now - stamped < timedelta(minutes=2)


@pytest.mark.parametrize("partner", ["xx", "", "UZ", "uzbek", "zh-CN"])
def test_unknown_partner_language_is_rejected(client, store, sid, partner):
    resp = client.patch(f"/api/sessions/{sid}", json={"partner": partner})
    assert resp.status_code == 400 and resp.json()["detail"] == "unknown language"
    assert store.session(sid)["partner"] == "uz"


def test_unknown_sessions_are_404(client):
    assert client.get("/api/sessions/999").status_code == 404
    assert client.patch("/api/sessions/999", json={"partner": "tr"}).status_code == 404
    assert client.delete("/api/sessions/999").status_code == 404
    assert client.patch("/api/sessions/999/speakers/S1", json={"label": "x"}).status_code == 404
    assert client.post("/api/sessions/999/speakers/S1/merge", json={"into": "S2"}).status_code == 404
    assert client.post("/api/sessions/999/reports", json={}).status_code == 404
    assert client.get("/api/sessions/999/transcript.txt").status_code == 404
    assert client.get("/api/sessions/abc").status_code == 422


def test_session_payload_from_the_store(client, store, sid, lives, monkeypatch):
    monkeypatch.setattr(web, "UNITS_ON_PAGE", 3)
    add_speaker(store, sid, "S1", "Dr. Karimov", voice="deep", centroid=[1, 0], samples=2, ref_wav=WAV, named=True)
    add_speaker(store, sid, "S2", "Spiker 2")
    units = [add_unit(store, sid, BASE + 10 * i, BASE + 10 * i + 5, "S1", f"src {i}", f"tr {i}") for i in range(5)]
    store.add_segment(sid, {"start": BASE + 100, "end": BASE + 101, "speaker": "S2", "lang": "en", "text": "and so",
                            "complete": False, "embedding": np.ones(2)})
    rid = store.save_report(sid, BASE, BASE + 45, make_report().model_dump())
    row = store.session(sid)

    data = client.get(f"/api/sessions/{sid}").json()
    assert data == {
        "id": sid,
        "title": "Seminar",
        "created_at": row["created_at"],
        "partner": "uz",
        "notes": {},
        # only what the page needs: no reference audio or voiceprints
        "speakers": [{"id": "S1", "label": "Dr. Karimov", "voice": "deep"}, {"id": "S2", "label": "Spiker 2", "voice": ""}],
        "units": units[-3:],  # the newest ones, oldest first
        "units_on_page": 3,
        "pending": [{"speaker": "S2", "lang": "en", "text": "and so"}],
        "interpreting": 0,
        "reports": [{"id": rid, "created_at": store.reports(sid)[0]["created_at"], "start": BASE, "end": BASE + 45,
                     "status": "done", "error": ""}],
    }
    assert lives == []  # reading a session does not start it


def test_partner_change_reaches_the_live_session(client, ai, store, sid, lives):
    ai.heards = [heard("Salom", lang="uz")]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        assert client.patch(f"/api/sessions/{sid}", json={"partner": "tr"}).status_code == 200
        assert lives[0].partner == "tr"
        ws.send_bytes(frame(1))
        assert recv_until(ws, "unit")[0]["unit"]["target"] == "tr"
    assert store.session(sid)["partner"] == "tr"


def test_delete_session_removes_everything(client, store, sid, seeded, lives):
    store.save_report(sid, BASE, BASE + 10, make_report().model_dump())
    other = store.create_session("Other")["id"]
    add_unit(store, other, BASE, BASE + 1, "S1", "a", "b")
    with client.websocket_connect(f"/ws?session={sid}"):
        assert client.delete(f"/api/sessions/{sid}").json() == {"ok": True}
        assert lives[0].close_calls == 1
    assert store.session(sid) is None
    assert store.units(sid) == [] and store.speakers(sid) == [] and store.reports(sid) == []
    assert len(store.units(other)) == 1  # other sessions are untouched
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    expect_close(client, 4404, f"/ws?session={sid}")


# ---------------------------------------------------------------- speakers


def test_rename_speaker(client, store, sid, lives):
    add_speaker(store, sid, "S1", "Spiker 1")
    add_speaker(store, sid, "S2", "Spiker 2")
    url = f"/api/sessions/{sid}/speakers/S1"
    assert client.patch(url, json={"label": "  Dr. Karimov  "}).json() == {"ok": True}
    s1 = speaker_row(store, sid, "S1")
    assert s1["label"] == "Dr. Karimov" and s1["named"] == 1
    assert speaker_row(store, sid, "S2")["label"] == "Spiker 2"
    labels = {sp["id"]: sp["label"] for sp in client.get(f"/api/sessions/{sid}").json()["speakers"]}
    assert labels == {"S1": "Dr. Karimov", "S2": "Spiker 2"}
    wait_for(lambda: lives[0].close_calls == 1)  # nobody listens: the session is unloaded again

    assert client.patch(url, json={"label": "k" * 100}).status_code == 200
    assert speaker_row(store, sid, "S1")["label"] == "k" * 80
    assert len(lives) == 2 and lives[1].speakers["S1"]["label"] == "k" * 80


def test_rename_speaker_rejects_bad_input(client, store, sid):
    add_speaker(store, sid, "S1", "Spiker 1")
    url = f"/api/sessions/{sid}/speakers"
    assert client.patch(f"{url}/S9", json={"label": "Ali"}).status_code == 400
    assert client.patch(f"{url}/S1", json={"label": "   "}).status_code == 400
    assert client.patch(f"{url}/S1", json={}).status_code == 422
    assert speaker_row(store, sid, "S1")["label"] == "Spiker 1"


def test_rename_is_broadcast_to_listeners(client, store, sid, lives):
    add_speaker(store, sid, "S1", "Spiker 1", voice="soft")
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        assert client.patch(f"/api/sessions/{sid}/speakers/S1", json={"label": "Aziz"}).status_code == 200
        assert recv(ws) == {"type": "speaker", "speaker": {"id": "S1", "label": "Aziz", "voice": "soft"}}
        time.sleep(0.1)
        assert lives[0].close_calls == 0  # still in use


def test_merge_speakers(client, ai, store, sid, lives, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.3)
    add_speaker(store, sid, "S1", "Dr. Karimov", centroid=[1, 0], samples=1)
    add_speaker(store, sid, "S2", "Spiker 2", centroid=[0, 1], samples=3)
    add_unit(store, sid, BASE, BASE + 5, "S1", "a", "a1")
    add_unit(store, sid, BASE + 10, BASE + 15, "S2", "b", "b1")
    # S2's unfinished thought from before a restart
    store.add_segment(sid, {"start": BASE + 20, "end": BASE + 21, "speaker": "S2", "lang": "ru", "text": "и ещё",
                            "complete": False})

    resp = client.post(f"/api/sessions/{sid}/speakers/S2/merge", json={"into": "S1"})
    assert resp.json() == {"ok": True}
    speakers = store.speakers(sid)
    assert [sp["id"] for sp in speakers] == ["S1"]
    assert speakers[0]["samples"] == 4
    np.testing.assert_allclose(speakers[0]["centroid"], [0.25, 0.75])  # weighted by samples
    assert {u["speaker"] for u in store.units(sid)} == {"S1"}

    # the pending thought is finished by the idle flush, under the merged speaker
    wait_for(lambda: len(store.units(sid)) == 3)
    assert store.units(sid)[-1]["speaker"] == "S1" and ai.interpret_calls[0].speaker == "Dr. Karimov"
    wait_for(lambda: lives[0].close_calls == 1)


def test_merge_rejects_bad_ids(client, store, sid):
    add_speaker(store, sid, "S1", "Spiker 1")
    add_speaker(store, sid, "S2", "Spiker 2")
    url = f"/api/sessions/{sid}/speakers"
    assert client.post(f"{url}/S1/merge", json={"into": "S1"}).status_code == 400
    assert client.post(f"{url}/S9/merge", json={"into": "S1"}).status_code == 400
    assert client.post(f"{url}/S1/merge", json={"into": "S9"}).status_code == 400
    assert client.post(f"{url}/S1/merge", json={}).status_code == 422
    assert [sp["id"] for sp in store.speakers(sid)] == ["S1", "S2"]


def test_merge_is_broadcast_to_listeners(client, store, sid):
    add_speaker(store, sid, "S1", "Spiker 1")
    add_speaker(store, sid, "S2", "Spiker 2")
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        assert client.post(f"/api/sessions/{sid}/speakers/S2/merge", json={"into": "S1"}).status_code == 200
        assert recv(ws) == {"type": "merged", "source": "S2", "target": "S1"}


# ---------------------------------------------------------------- reports


def transcript_lines(call) -> list[str]:
    return call.transcript.split("\n")


def make_report_now(client, sid, body) -> int:
    """POST a report and wait for the background job, like the page's polling does."""
    resp = client.post(f"/api/sessions/{sid}/reports", json=body)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["status"] == "pending"
    rid = resp.json()["id"]
    finished = {}

    def done():
        finished.update(client.get(f"/api/reports/{rid}/status").json())
        return finished["status"] != "pending"

    wait_for(done)
    return rid


def test_report_covers_the_whole_session_by_default(client, ai, store, sid, seeded):
    rid = make_report_now(client, sid, {})
    row = store.report(rid)
    # the saved range is what was actually said, not 1970..now
    assert (row["session_id"], row["start"], row["end"]) == (sid, BASE, BASE + 3605)
    assert row["report"] == make_report().model_dump()

    call = ai.report_calls[0]
    assert call.period == "2026-03-02 10:00 - 2026-03-02 11:00"
    assert call.speakers == "S1: Dr. Karimov\nS2: Spiker 2"
    assert transcript_lines(call) == [
        "[10:00] S1 (Dr. Karimov): Xayrli kun  <<ru: Добрый день>>",
        "[10:10] S2 (Spiker 2): Savollar bormi?  <<en: Any questions?>>",
        "[11:00] S1 (Dr. Karimov): Rahmat",  # untranslated: the original words
    ]
    reports = client.get(f"/api/sessions/{sid}").json()["reports"]
    assert reports == [{"id": rid, "created_at": row["created_at"], "start": BASE, "end": BASE + 3605,
                        "status": "done", "error": ""}]

    again = make_report_now(client, sid, {"start": None, "end": None})
    assert (store.report(again)["start"], store.report(again)["end"]) == (BASE, BASE + 3605)


@pytest.mark.parametrize(
    "body, sources, saved",
    [
        ({"start": BASE + 300, "end": BASE + 1200}, ["Any questions?"], (BASE + 600, BASE + 610)),
        ({"start": BASE + 5, "end": BASE + 605}, ["Добрый день", "Any questions?"], (BASE + 5, BASE + 605)),
        ({"start": BASE - 3600, "end": BASE + 30}, ["Добрый день"], (BASE, BASE + 10)),
        ({"start": BASE + 601}, ["Any questions?", "Rahmat"], (BASE + 601, BASE + 3605)),
        ({"end": BASE + 5}, ["Добрый день"], (BASE, BASE + 5)),
    ],
    ids=["middle", "overlapping-edges", "before-first", "start-only", "end-only"],
)
def test_report_for_a_chosen_time_range(client, ai, store, sid, seeded, body, sources, saved):
    rid = make_report_now(client, sid, body)
    lines = transcript_lines(ai.report_calls[0])
    assert len(lines) == len(sources) and all(src in line for src, line in zip(sources, lines))
    row = store.report(rid)
    assert (row["start"], row["end"]) == saved
    # the report page and its downloads show exactly the units of that range
    page = client.get(f"/api/reports/{rid}").text
    md = client.get(f"/api/reports/{rid}.md").text
    for unit in seeded:
        shown = unit["source"] in sources
        assert (unit["source"] in page) == shown and (unit["source"] in md) == shown, unit["source"]
    link = re.search(r"href='(/api/sessions/\d+/transcript\.txt\?[^']+)'", page).group(1)
    txt = client.get(link).text
    for unit in seeded:
        assert (unit["source"] in txt) == (unit["source"] in sources), unit["source"]


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"start": BASE + 4000, "end": BASE + 5000}, "Bu oraliqda tarjima qilingan gap yo'q"),
        ({"start": BASE + 700, "end": BASE + 800}, "Bu oraliqda tarjima qilingan gap yo'q"),
        ({"start": BASE + 10, "end": BASE + 10}, "Boshlanish vaqti tugashdan oldin bo'lishi kerak"),
        ({"start": BASE + 600, "end": BASE}, "Boshlanish vaqti tugashdan oldin bo'lishi kerak"),
        ({"end": -5}, "Boshlanish vaqti tugashdan oldin bo'lishi kerak"),
        ({"start": 4102444800}, "Boshlanish vaqti tugashdan oldin bo'lishi kerak"),  # 2100: after the default end
    ],
    ids=["after-all", "gap", "empty-range", "reversed", "end-before-epoch", "start-in-future"],
)
def test_report_rejects_bad_ranges(client, ai, store, sid, seeded, body, detail):
    resp = client.post(f"/api/sessions/{sid}/reports", json=body)
    assert resp.status_code == 400 and resp.json()["detail"] == detail
    assert ai.report_calls == [] and store.reports(sid) == []


def test_report_of_a_session_without_units_is_400(client, ai, sid):
    resp = client.post(f"/api/sessions/{sid}/reports", json={})
    assert resp.status_code == 400 and resp.json()["detail"] == "Bu oraliqda tarjima qilingan gap yo'q"
    assert ai.report_calls == []


def test_report_ai_failure_is_reported(client, ai, store, sid, seeded):
    ai.report_result = RuntimeError("Gemini 500")
    rid = make_report_now(client, sid, {})
    status = client.get(f"/api/reports/{rid}/status").json()
    assert status == {"id": rid, "status": "failed", "error": "Hisobot tayyorlab bo'lmadi, qayta urinib ko'ring"}
    assert [r["status"] for r in client.get(f"/api/sessions/{sid}").json()["reports"]] == ["failed"]
    page = client.get(f"/api/reports/{rid}")
    assert page.status_code == 500 and "Hisobot tayyorlab bo" in page.text
    assert client.get(f"/api/reports/{rid}.md").status_code == 409


def test_report_is_pending_until_written(client, ai, store, sid, seeded):
    gate = threading.Event()
    real = ai.report

    def slow_report(*args):
        gate.wait(3)
        return real(*args)

    ai.report = slow_report
    resp = client.post(f"/api/sessions/{sid}/reports", json={})
    rid = resp.json()["id"]
    assert resp.json() == {"id": rid, "status": "pending"}
    assert client.get(f"/api/reports/{rid}/status").json()["status"] == "pending"
    page = client.get(f"/api/reports/{rid}")
    assert page.status_code == 200 and "http-equiv='refresh'" in page.text and "tayyorlanmoqda" in page.text
    assert client.get(f"/api/reports/{rid}.md").status_code == 409
    gate.set()
    wait_for(lambda: client.get(f"/api/reports/{rid}/status").json()["status"] == "done")
    assert "<h2>Umumiy mazmun</h2>" in client.get(f"/api/reports/{rid}").text


def test_unfinished_reports_fail_after_a_restart(store, sid, ai):
    rid = store.start_report(sid, BASE, BASE + 5)
    web.create_app(ai, store)  # a new server process
    row = store.report(rid)
    assert row["status"] == "failed" and "qayta" in row["error"]


def test_status_of_unknown_report_is_404(client):
    assert client.get("/api/reports/999/status").status_code == 404


def test_report_page_escapes_html(client, store, sid):
    store.update_session(sid, title="Kun <1>")
    add_speaker(store, sid, "S1", "<img src=x onerror=alert(1)>")
    add_unit(store, sid, BASE, BASE + 5, "S1", "<script>bad()</script>", "Salom <b>dunyo</b>")
    report = make_report(title="<script>alert(1)</script>", overview="a < b & c")
    rid = store.save_report(sid, BASE, BASE + 5, report.model_dump())
    resp = client.get(f"/api/reports/{rid}")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    page = resp.text
    for raw in ("<script>alert(1)</script>", "<script>bad()</script>", "<img src=x", "<b>dunyo</b>", "Kun <1>"):
        assert raw not in page, raw
    for escaped in ("&lt;script&gt;alert(1)&lt;/script&gt;", "&lt;img src=x onerror=alert(1)&gt;",
                    "Salom &lt;b&gt;dunyo&lt;/b&gt;", "a &lt; b &amp; c", "Kun &lt;1&gt;"):
        assert escaped in page, escaped


def test_report_names_merged_away_speakers_by_who_they_became(client, store, sid, seeded, lives):
    """A report written before a merge still says S2: it reads as the person S2 was merged into."""
    rid = store.save_report(sid, BASE, BASE + 3605, make_report().model_dump())  # Q&A asked by S2
    assert client.post(f"/api/sessions/{sid}/speakers/S2/merge", json={"into": "S1"}).status_code == 200
    wait_for(lambda: lives[0].close_calls == 1)  # even once the session is unloaded
    page = client.get(f"/api/reports/{rid}").text
    md = client.get(f"/api/reports/{rid}.md").text
    assert "<b>10:10 Dr. Karimov:</b>" in page and "**10:10 Dr. Karimov:**" in md
    assert "S2" not in page and "Spiker 2" not in page



def test_report_downloads(client, store, sid, seeded):
    rid = make_report_now(client, sid, {})

    md = client.get(f"/api/reports/{rid}.md")
    assert md.status_code == 200
    assert md.headers["content-type"] == "text/markdown; charset=utf-8"
    assert md.headers["content-disposition"] == f'attachment; filename="hisobot-{rid}.md"'
    assert md.text.startswith("# Hisobot\n") and "**Seminar:** Seminar" in md.text
    assert "## To'liq transkript" in md.text and "Xayrli kun" in md.text

    txt = client.get(f"/api/sessions/{sid}/transcript.txt")
    assert txt.status_code == 200 and txt.headers["content-type"] == "text/plain; charset=utf-8"
    assert txt.headers["content-disposition"] == f'attachment; filename="transkript-{sid}.txt"'
    assert txt.text == (
        "Seminar\n\n"
        "[10:00] Dr. Karimov:\n  Xayrli kun\n    (Ruscha: Добрый день)\n\n"
        "[10:10] Spiker 2:\n  Savollar bormi?\n    (Inglizcha: Any questions?)\n\n"
        "[11:00] Dr. Karimov:\n  Rahmat\n"
    )
    part = client.get(f"/api/sessions/{sid}/transcript.txt", params={"start": BASE + 300, "end": BASE + 1200}).text
    assert "Savollar bormi?" in part and "Xayrli kun" not in part and "Rahmat" not in part


def test_report_pages_need_login(make, store, sid, seeded):
    rid = store.save_report(sid, BASE, BASE + 3605, make_report().model_dump())
    client = make(password="secret")
    page = client.get(f"/api/reports/{rid}", follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/"  # opened in a new tab: go to the login form
    assert client.get(f"/api/reports/{rid}.md").status_code == 401
    assert client.get(f"/api/sessions/{sid}/transcript.txt").status_code == 401
    login(client)
    assert client.get(f"/api/reports/{rid}").status_code == 200


def test_unknown_reports_are_404(client):
    assert client.get("/api/reports/999").status_code == 404
    assert client.get("/api/reports/999.md").status_code == 404
    assert client.get("/api/reports/abc").status_code == 404


# ---------------------------------------------------------------- unloading


def test_session_is_unloaded_when_the_last_client_leaves(client, ai, sid, lives):
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        recv_until(ws, "unit")
        with client.websocket_connect(f"/ws?session={sid}"):
            pass  # a second tab closes: the first one still listens
        time.sleep(0.1)
        assert len(lives) == 1 and lives[0].close_calls == 0
    wait_for(lambda: lives[0].close_calls == 1)

    # loading it again restores the context from the database
    ai.heards = [heard("Ещё", speaker="S1")]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(2))
        recv_until(ws, "unit")
    assert len(lives) == 2
    assert ai.listen_calls[1].references == ["S1"]
    assert ai.interpret_calls[1].recent == ["[Spiker 1] uz: Привет"]


def test_unloading_waits_for_interpreting_to_finish(client, ai, store, sid, lives):
    ai.interpret_gate = threading.Event()
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        assert ai.interpreting.wait(3)
    time.sleep(0.15)
    assert lives[0].close_calls == 0 and store.units(sid) == []
    ai.interpret_gate.set()
    wait_for(lambda: lives[0].close_calls == 1)
    assert [u["source"] for u in store.units(sid)] == ["Привет"]  # finished after the client left


def test_unloading_waits_for_the_unfinished_thought(client, ai, store, sid, lives, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.5)
    ai.heards = [heard("Итак", complete=False)]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        recv_until(ws, "heard")
    time.sleep(0.1)
    assert lives[0].close_calls == 0 and store.units(sid) == []  # the thought is still open
    wait_for(lambda: lives[0].close_calls == 1)
    assert [u["source"] for u in store.units(sid)] == ["Итак"]


def test_reconnecting_client_gets_the_work_started_before(client, ai, sid, lives):
    ai.interpret_gate = threading.Event()
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        assert ai.interpreting.wait(3)
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        assert len(lives) == 1  # still the same session in memory
        ai.interpret_gate.set()
        unit, _ = recv_until(ws, "unit")
        assert unit["unit"]["source"] == "Привет"
    wait_for(lambda: lives[0].close_calls == 1)


def test_session_stays_loaded_for_a_grace_period(client, ai, sid, lives, monkeypatch):
    """A phone dropping off the Wi-Fi for a moment comes back to the same session in memory."""
    monkeypatch.setattr(web, "UNLOAD_GRACE_SECONDS", 0.5)
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        recv_until(ws, "unit")
    time.sleep(0.2)
    assert lives[0].close_calls == 0
    with client.websocket_connect(f"/ws?session={sid}"):
        assert len(lives) == 1
    left = time.monotonic()
    wait_for(lambda: lives[0].close_calls == 1)
    assert time.monotonic() - left >= 0.45  # counted from when the last device left
    assert len(lives) == 1


def test_deleted_session_leaves_no_task_behind(client, ai, sid, lives):
    ai.heards = [heard("Итак", complete=False)]
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        ws.send_bytes(frame(1))
        recv_until(ws, "pending")  # an unfinished thought: the session is busy
        assert client.delete(f"/api/sessions/{sid}").status_code == 200

    def unload_tasks():
        tasks = client.portal.call(asyncio.all_tasks)
        return [t for t in tasks if t.get_coro().__qualname__.endswith("unload_when_idle")]

    wait_for(lambda: not unload_tasks())
    assert lives[0].close_calls == 1 and lives[0].idle_timer is None



# ---------------------------------------------------------------- text to speech


def test_tts(make):
    speaker = FakeSpeaker()
    client = make(speaker=speaker)
    resp = client.post("/api/tts", json={"text": "  Salom  "})
    assert resp.status_code == 200 and resp.headers["content-type"] == "audio/mpeg" and resp.content == b"ID3-mp3"
    assert speaker.texts == ["Salom"]
    assert client.post("/api/tts", json={"text": "a" * web.MAX_TTS_CHARS}).status_code == 200
    assert client.post("/api/tts", json={"text": " "}).status_code == 400
    assert client.post("/api/tts", json={"text": "a" * (web.MAX_TTS_CHARS + 1)}).status_code == 400
    assert client.post("/api/tts", json={}).status_code == 422
    assert len(speaker.texts) == 2


def test_tts_errors_and_auth(make):
    assert make().post("/api/tts", json={"text": "x"}).status_code == 503
    assert make(speaker=FakeSpeaker(fail=True)).post("/api/tts", json={"text": "x"}).status_code == 502
    speaker = FakeSpeaker()
    locked = make(speaker=speaker, password="secret")
    assert locked.post("/api/tts", json={"text": "x"}).status_code == 401
    assert speaker.texts == []


def test_tts_is_synthesized_once_per_text(make, monkeypatch):
    """Every device reading out a unit asks for the same text: ElevenLabs bills it once."""
    monkeypatch.setattr(web, "TTS_CACHE_ITEMS", 2)
    speaker = FakeSpeaker()
    client = make(speaker=speaker)
    for text in ("Salom", "Salom", " Salom ", "Xayr", "Salom", "Rahmat", "Xayr"):
        resp = client.post("/api/tts", json={"text": text})
        assert resp.status_code == 200 and resp.content == b"ID3-mp3"
    # the oldest clip makes room for new ones
    assert speaker.texts == ["Salom", "Xayr", "Rahmat", "Xayr"]


def test_tts_concurrent_requests_share_one_synthesis(make):
    gate, entered = threading.Event(), threading.Event()

    class SlowSpeaker(FakeSpeaker):
        def synthesize(self, text):
            entered.set()
            assert gate.wait(5)
            return super().synthesize(text)

    speaker = SlowSpeaker()
    client = make(speaker=speaker)
    results = []

    def read_out():
        results.append(client.post("/api/tts", json={"text": "Salom"}).content)

    threads = [threading.Thread(target=read_out) for _ in range(3)]
    threads[0].start()
    assert entered.wait(3)
    for t in threads[1:]:
        t.start()
    time.sleep(0.2)  # the others are waiting for the same clip
    gate.set()
    for t in threads:
        t.join(5)
    assert results == [b"ID3-mp3"] * 3 and speaker.texts == ["Salom"]


def test_tts_failure_is_not_cached(make):
    speaker = FakeSpeaker(fail=True)
    client = make(speaker=speaker)
    assert client.post("/api/tts", json={"text": "Salom"}).status_code == 502
    speaker.fail = False
    assert client.post("/api/tts", json={"text": "Salom"}).status_code == 200
    assert speaker.texts == ["Salom"]



# ---------------------------------------------------------------- HTTPS


def test_ssl_off_by_default(monkeypatch, tmp_path):
    for var in ("SSL_CERTFILE", "SSL_KEYFILE", "SSL_SELF_SIGNED"):
        monkeypatch.delenv(var, raising=False)
    assert web.ssl_files(str(tmp_path)) == (None, None)
    monkeypatch.setenv("SSL_CERTFILE", "/etc/cert.pem")  # a certificate without its key is not enough
    assert web.ssl_files(str(tmp_path)) == (None, None)
    assert list(tmp_path.iterdir()) == []


def test_ssl_uses_given_files(monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERTFILE", "/etc/cert.pem")
    monkeypatch.setenv("SSL_KEYFILE", "/etc/key.pem")
    monkeypatch.setenv("SSL_SELF_SIGNED", "1")
    assert web.ssl_files(str(tmp_path)) == ("/etc/cert.pem", "/etc/key.pem")


def test_self_signed_certificate_is_generated_once(monkeypatch, tmp_path):
    monkeypatch.delenv("SSL_CERTFILE", raising=False)
    monkeypatch.delenv("SSL_KEYFILE", raising=False)
    monkeypatch.setenv("SSL_SELF_SIGNED", "true")
    cert, key = web.ssl_files(str(tmp_path / "data"))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)  # a valid pair
    mtime = (tmp_path / "data" / "selfsigned.crt").stat().st_mtime_ns
    assert web.ssl_files(str(tmp_path / "data")) == (cert, key)
    assert (tmp_path / "data" / "selfsigned.crt").stat().st_mtime_ns == mtime
    assert oct((tmp_path / "data" / "selfsigned.key").stat().st_mode & 0o777) == "0o600"


# ---------------------------------------------------------------- known bugs


def test_huge_ids_are_not_found(make):
    client = make(raise_server_exceptions=False)
    huge = "9" * 20
    assert client.get(f"/api/sessions/{huge}").status_code == 404
    assert client.get(f"/api/reports/{huge}").status_code == 404


def test_websocket_with_a_huge_session_id_closes_with_4404(client):
    expect_close(client, 4404, "/ws?session=" + "9" * 20)


def test_rejected_update_changes_nothing(client, store, sid):
    resp = client.patch(f"/api/sessions/{sid}", json={"title": "New title", "partner": "xx"})
    assert resp.status_code == 400
    assert store.session(sid)["title"] == "Seminar"


def test_rejected_speaker_edit_does_not_keep_the_session_loaded(client, store, sid, lives):
    add_speaker(store, sid, "S1", "Spiker 1")
    assert client.patch(f"/api/sessions/{sid}/speakers/S9", json={"label": "Ali"}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/speakers/S9/merge", json={"into": "S1"}).status_code == 400
    wait_for(lambda: all(s.close_calls for s in lives), timeout=0.5)


def test_delete_closes_the_live_session_on_the_event_loop(client, sid, lives):
    with client.websocket_connect(f"/ws?session={sid}"):
        assert client.delete(f"/api/sessions/{sid}").status_code == 200
        assert lives[0].close_calls == 1
        assert not lives[0].closed_off_loop


def test_deleting_a_session_disconnects_its_listeners(client, ai, sid):
    with client.websocket_connect(f"/ws?session={sid}") as ws:
        assert client.delete(f"/api/sessions/{sid}").status_code == 200
        ws.send_bytes(frame(1))
        with pytest.raises(WebSocketDisconnect) as exc:
            recv(ws, timeout=0.5)
        assert exc.value.code == 4404
    assert ai.listen_calls == []


def test_malformed_ai_report_fails_the_job_not_the_range_check(client, ai, store, sid, seeded):
    try:
        Report.model_validate_json('{"title": "cut off by the token limit", "overv')
    except ValueError as e:  # pydantic's ValidationError is a ValueError: it must not look like a bad range
        ai.report_result = e
    rid = make_report_now(client, sid, {})
    status = client.get(f"/api/reports/{rid}/status").json()
    assert status["status"] == "failed" and "validation" not in status["error"]


def test_speaker_order_survives_a_rename(client, store, sid, lives):
    add_speaker(store, sid, "S1", "Spiker 1")
    add_speaker(store, sid, "S2", "Spiker 2")
    assert client.patch(f"/api/sessions/{sid}/speakers/S1", json={"label": "Aziz"}).status_code == 200
    wait_for(lambda: lives[0].close_calls == 1)
    assert [sp["id"] for sp in client.get(f"/api/sessions/{sid}").json()["speakers"]] == ["S1", "S2"]
