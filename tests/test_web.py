import json
import ssl
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import web
from conftest import make_wav, segment_frame
from translator import Interpretation


class FakeInterpreter:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
        self.gate = None  # threading.Event to hold calls

    def interpret(self, contents, partner, history=(), live=False):
        if self.gate:
            self.gate.wait(5)
        self.calls.append(SimpleNamespace(contents=contents, partner=partner, history=list(history), live=live))
        result = self.results.pop(0) if self.results else Interpretation(
            source_lang="ru", text="Привет", translation="Salom"
        )
        if isinstance(result, Exception):
            raise result
        return result


class FakeSpeaker:
    def __init__(self, fail=False):
        self.texts = []
        self.fail = fail

    def synthesize(self, text):
        if self.fail:
            raise RuntimeError("ElevenLabs 500")
        self.texts.append(text)
        return b"ID3-mp3"


def client_for(interpreter=None, speaker=None, password=""):
    return TestClient(web.create_app(interpreter or FakeInterpreter(), speaker, password))


def login(client, password="secret"):
    resp = client.post("/login", data={"password": password}, follow_redirects=False)
    assert resp.status_code == 303
    return resp


def test_healthz():
    assert client_for().get("/healthz").json() == {"ok": True}


def test_open_access_without_password():
    client = client_for()
    page = client.get("/")
    assert page.status_code == 200 and "Jonli tarjimon" in page.text and "app.js" in page.text
    config = client.get("/api/config").json()
    assert config["partners"] == ["zh", "tr", "en", "ru"]
    assert config["langs"]["uz"] == {"flag": "🇺🇿", "name": "O'zbekcha"}
    assert config["tts"] is False and config["auth"] is False


def test_static_files_are_served():
    client = client_for()
    for path in ("/static/app.js", "/static/audio.js", "/static/pcm-worklet.js", "/static/style.css"):
        assert client.get(path).status_code == 200, path


def test_password_login_flow():
    client = client_for(password="secret")
    assert 'name="password"' in client.get("/").text
    assert client.get("/api/config").status_code == 401

    wrong = client.post("/login", data={"password": "nope"}, follow_redirects=False)
    assert wrong.status_code == 303 and wrong.headers["location"] == "/?error=1"
    assert "stt_session" not in client.cookies

    ok = login(client)
    cookie = ok.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert "app.js" in client.get("/").text
    assert client.get("/api/config").json()["auth"] is True

    client.post("/logout", follow_redirects=False)
    client.cookies.clear()
    assert client.get("/api/config").status_code == 401


def test_cookie_from_another_password_is_rejected():
    old = client_for(password="old")
    login(old, "old")
    new = client_for(password="new")
    new.cookies.update(old.cookies)
    assert new.get("/api/config").status_code == 401


def expect_close(client, code, **kwargs):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws", **kwargs) as ws:
            # if the socket were wrongly accepted, this returns a result and the test fails instead of hanging
            ws.send_bytes(segment_frame(1))
            ws.receive_json()
    assert exc.value.code == code


def test_websocket_requires_login():
    client = client_for(password="secret")
    expect_close(client, 4401)
    login(client)
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(segment_frame(1))
        assert ws.receive_json()["type"] == "result"


def test_websocket_rejects_other_origins():
    client = client_for()
    expect_close(client, 4401, headers={"origin": "https://evil.example"})
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_bytes(segment_frame(1))
        assert ws.receive_json()["type"] == "result"


def test_segment_is_translated_to_uzbek_with_live_prompt():
    interpreter = FakeInterpreter()
    wav = make_wav(1.0)
    with client_for(interpreter).websocket_connect("/ws") as ws:
        ws.send_bytes(segment_frame(7, wav))
        msg = ws.receive_json()
    assert msg == {
        "type": "result",
        "id": 7,
        "source_lang": "ru",
        "text": "Привет",
        "translation": "Salom",
        "target": "uz",
    }
    call = interpreter.calls[0]
    assert call.live is True and call.partner == "zh" and call.history == []
    part = call.contents[0]
    assert part.inline_data.mime_type == "audio/wav" and part.inline_data.data == wav


def test_uzbek_speech_goes_to_the_chosen_partner_language():
    interpreter = FakeInterpreter(Interpretation(source_lang="uz", text="Salom", translation="Merhaba"))
    with client_for(interpreter).websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "config", "partner": "tr"}))
        ws.send_text(json.dumps({"type": "config", "partner": "xx"}))  # ignored
        ws.send_text("not json")  # ignored
        ws.send_bytes(segment_frame(1))
        msg = ws.receive_json()
    assert msg["target"] == "tr"
    assert interpreter.calls[0].partner == "tr"


def test_segments_are_answered_in_order_with_growing_context():
    interpreter = FakeInterpreter(
        Interpretation(source_lang="zh", text="一", translation="bir"),
        Interpretation(source_lang="zh", text="二", translation="ikki"),
        Interpretation(source_lang="zh", text="三", translation="uch"),
    )
    with client_for(interpreter).websocket_connect("/ws") as ws:
        for i in (1, 2, 3):
            ws.send_bytes(segment_frame(i))
        msgs = [ws.receive_json() for _ in range(3)]
    assert [m["id"] for m in msgs] == [1, 2, 3]
    assert [m["translation"] for m in msgs] == ["bir", "ikki", "uch"]
    assert [[h.text for h in c.history] for c in interpreter.calls] == [[], ["一"], ["一", "二"]]


def test_context_is_capped_and_reset_clears_it():
    interpreter = FakeInterpreter()
    with client_for(interpreter).websocket_connect("/ws") as ws:
        for i in range(web.HISTORY_SIZE + 3):
            ws.send_bytes(segment_frame(i))
            ws.receive_json()
        assert len(interpreter.calls[-1].history) == web.HISTORY_SIZE
        ws.send_text(json.dumps({"type": "reset"}))
        ws.send_bytes(segment_frame(99))
        ws.receive_json()
    assert interpreter.calls[-1].history == []


def test_no_speech_and_failures_do_not_stop_the_stream():
    interpreter = FakeInterpreter(
        Interpretation(source_lang="uz", text="", translation=""),
        RuntimeError("Gemini 503"),
        Interpretation(source_lang="en", text="Hi", translation="Salom"),
    )
    with client_for(interpreter).websocket_connect("/ws") as ws:
        for i in (1, 2, 3):
            ws.send_bytes(segment_frame(i))
        msgs = [ws.receive_json() for _ in range(3)]
    assert msgs[0] == {"type": "empty", "id": 1}
    assert msgs[1]["type"] == "error" and msgs[1]["id"] == 2
    assert msgs[2]["type"] == "result" and msgs[2]["id"] == 3
    # failed and empty segments are not used as context
    assert [h.text for h in interpreter.calls[2].history] == []


def test_invalid_and_oversized_frames_are_rejected():
    interpreter = FakeInterpreter()
    with client_for(interpreter).websocket_connect("/ws") as ws:
        ws.send_bytes(b"\x00\x00\x00\x01" + b"not a wav file at all" * 3)
        assert ws.receive_json() == {"type": "error", "id": None, "message": "Noto'g'ri audio"}
        ws.send_bytes(b"\x00\x01")
        assert ws.receive_json()["id"] is None
        ws.send_bytes(segment_frame(5, make_wav(70)))
        assert ws.receive_json() == {"type": "error", "id": 5, "message": "Segment juda uzun"}
    assert interpreter.calls == []


def test_queue_limit_protects_the_budget():
    interpreter = FakeInterpreter()
    interpreter.gate = threading.Event()
    with client_for(interpreter).websocket_connect("/ws") as ws:
        total = web.MAX_QUEUED_SEGMENTS + 5
        for i in range(total):
            ws.send_bytes(segment_frame(i))
        time.sleep(0.2)
        interpreter.gate.set()
        msgs = [ws.receive_json() for _ in range(total)]
    rejected = [m for m in msgs if m["type"] == "error"]
    assert 1 <= len(rejected) <= 5
    assert all(m["message"] == "Navbat to'lib ketdi" for m in rejected)
    assert len(interpreter.calls) == total - len(rejected)


def test_tts():
    speaker = FakeSpeaker()
    client = client_for(speaker=speaker)
    assert client.get("/api/config").json()["tts"] is True
    resp = client.post("/api/tts", json={"text": "  Salom  "})
    assert resp.status_code == 200 and resp.headers["content-type"] == "audio/mpeg" and resp.content == b"ID3-mp3"
    assert speaker.texts == ["Salom"]
    assert client.post("/api/tts", json={"text": " "}).status_code == 400
    assert client.post("/api/tts", json={"text": "a" * (web.MAX_TTS_CHARS + 1)}).status_code == 400


def test_tts_errors_and_auth():
    assert client_for().post("/api/tts", json={"text": "x"}).status_code == 503
    assert client_for(speaker=FakeSpeaker(fail=True)).post("/api/tts", json={"text": "x"}).status_code == 502
    locked = client_for(speaker=FakeSpeaker(), password="secret")
    assert locked.post("/api/tts", json={"text": "x"}).status_code == 401


def test_ssl_off_by_default(monkeypatch, tmp_path):
    for var in ("SSL_CERTFILE", "SSL_KEYFILE", "SSL_SELF_SIGNED"):
        monkeypatch.delenv(var, raising=False)
    assert web.ssl_files(str(tmp_path)) == (None, None)


def test_ssl_uses_given_files(monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERTFILE", "/etc/cert.pem")
    monkeypatch.setenv("SSL_KEYFILE", "/etc/key.pem")
    assert web.ssl_files(str(tmp_path)) == ("/etc/cert.pem", "/etc/key.pem")


def test_self_signed_certificate_is_generated_once(monkeypatch, tmp_path):
    monkeypatch.delenv("SSL_CERTFILE", raising=False)
    monkeypatch.delenv("SSL_KEYFILE", raising=False)
    monkeypatch.setenv("SSL_SELF_SIGNED", "1")
    cert, key = web.ssl_files(str(tmp_path / "data"))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)  # valid pair
    mtime = (tmp_path / "data" / "selfsigned.crt").stat().st_mtime_ns
    assert web.ssl_files(str(tmp_path / "data")) == (cert, key)
    assert (tmp_path / "data" / "selfsigned.crt").stat().st_mtime_ns == mtime
    assert oct((tmp_path / "data" / "selfsigned.key").stat().st_mode & 0o777) == "0o600"
