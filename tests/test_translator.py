import json
from types import SimpleNamespace

import pytest
from google import genai
from google.genai import types

import translator
from translator import Interpretation, Interpreter, Speaker
from usage import UsageLog


@pytest.fixture
def usage_log(tmp_path):
    return UsageLog(str(tmp_path / "usage.db"))


def fake_gemini(captured, payload, usage=None):
    """A real genai.Client whose HTTP layer returns `payload` and records the request body."""
    client = genai.Client(api_key="fake")

    def request(method, path, body, http_options=None):
        captured.append(body)
        response = {"candidates": [{"content": {"role": "model", "parts": [{"text": json.dumps(payload)}]}}]}
        if usage:
            response["usageMetadata"] = usage
        return SimpleNamespace(headers={}, body=json.dumps(response))

    client.models._api_client.request = request
    return client


def system_text(body):
    return body["systemInstruction"]["parts"][0]["text"]


def test_interpret_parses_and_normalizes(usage_log):
    captured = []
    client = fake_gemini(captured, {"source_lang": " RU ", "text": " Привет ", "translation": " Salom "})
    result = Interpreter(client, "gemini-test", usage_log).interpret("Привет", "zh")
    assert result == Interpretation(source_lang="ru", text="Привет", translation="Salom")
    body = captured[0]
    assert "Simplified Chinese (Mandarin)" in system_text(body)
    assert translator.LIVE_PROMPT.strip() not in system_text(body)
    assert "Conversation so far" not in system_text(body)
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["required"] == ["source_lang", "text", "translation"]


def test_live_mode_and_history_go_into_the_system_prompt(usage_log):
    captured = []
    client = fake_gemini(captured, {"source_lang": "zh", "text": "多少钱", "translation": "Qancha?"})
    history = [Interpretation(source_lang="uz", text="Bu ko'ylak {x}", translation="这件衬衫")]
    Interpreter(client, "m", usage_log).interpret("多少钱", "tr", history, live=True)
    text = system_text(captured[0])
    assert "Turkish speaker" in text
    assert "fragment of continuous live speech" in text
    assert "[uz] Bu ko'ylak {x} => 这件衬衫" in text
    assert text.index("fragment of continuous") < text.index("Conversation so far")


def test_usage_and_cost_are_recorded(usage_log):
    usage = {
        "promptTokenCount": 1000,
        "candidatesTokenCount": 100,
        "thoughtsTokenCount": 50,
        "promptTokensDetails": [
            {"modality": "AUDIO", "tokenCount": 800},
            {"modality": "TEXT", "tokenCount": 200},
        ],
    }
    client = fake_gemini([], {"source_lang": "uz", "text": "a", "translation": "b"}, usage)
    interpreter = Interpreter(client, "m", usage_log, price_in=1.0, price_audio_in=2.0, price_out=10.0)
    interpreter.interpret("a", "zh")
    t = usage_log.totals("gemini", 0)
    assert (t.requests, t.input_tokens, t.audio_tokens, t.output_tokens) == (1, 1000, 800, 150)
    assert t.cost == pytest.approx((200 * 1.0 + 800 * 2.0 + 150 * 10.0) / 1e6)


def test_malformed_model_output_raises(usage_log):
    client = fake_gemini([], {"unexpected": True})
    with pytest.raises(Exception):
        Interpreter(client, "m", usage_log).interpret("a", "zh")


def test_from_env_reads_prices(monkeypatch, usage_log):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "my-model")
    monkeypatch.setenv("GEMINI_PRICE_IN", "0.75")
    monkeypatch.delenv("GEMINI_PRICE_AUDIO_IN", raising=False)
    monkeypatch.setenv("GEMINI_PRICE_OUT", "3.75")
    it = Interpreter.from_env(usage_log)
    assert (it.model, it.price_in, it.price_audio_in, it.price_out) == ("my-model", 0.75, 0.75, 3.75)


def test_speaker_counts_credits(monkeypatch, usage_log):
    responses = [
        SimpleNamespace(status_code=200, headers={"character-cost": "17"}, content=b"mp3"),
        SimpleNamespace(status_code=200, headers={}, content=b"mp3"),
        SimpleNamespace(status_code=401, headers={}, content=b"", text="unauthorized"),
    ]
    monkeypatch.setattr(translator.requests, "post", lambda *a, **k: responses.pop(0))
    speaker = Speaker("key", "eleven_v3", "voice", usage_log)
    assert speaker.synthesize("hello") == b"mp3"
    assert speaker.synthesize("hello") == b"mp3"
    with pytest.raises(RuntimeError, match="ElevenLabs 401"):
        speaker.synthesize("hello")
    t = usage_log.totals("elevenlabs", 0)
    assert (t.requests, t.credits) == (2, 17 + len("hello"))


def test_speaker_from_env_is_optional(monkeypatch, usage_log):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert Speaker.from_env(usage_log) is None
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    assert Speaker.from_env(usage_log).api_key == "k"


def test_subscription_failure_returns_none(monkeypatch, usage_log):
    def boom(*a, **k):
        raise translator.requests.ConnectionError("down")

    monkeypatch.setattr(translator.requests, "get", boom)
    assert Speaker("k", "m", "v", usage_log).subscription() is None


def test_media_modality_enum_matches_rest_values():
    meta = types.GenerateContentResponseUsageMetadata.model_validate(
        {"prompt_tokens_details": [{"modality": "AUDIO", "token_count": 3}]}
    )
    assert meta.prompt_tokens_details[0].modality == types.MediaModality.AUDIO


def test_from_env_sets_a_request_timeout_and_tolerates_blank_values(monkeypatch, usage_log):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    for var in ("GEMINI_MODEL", "GEMINI_PRICE_IN", "GEMINI_PRICE_AUDIO_IN", "GEMINI_PRICE_OUT"):
        monkeypatch.setenv(var, "")
    it = Interpreter.from_env(usage_log)
    assert (it.model, it.price_in, it.price_audio_in, it.price_out) == ("gemini-3.6-flash", 1.5, 1.5, 7.5)
    assert it.client._api_client._http_options.timeout == translator.GEMINI_TIMEOUT_MS

    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_MODEL", "")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "")
    speaker = Speaker.from_env(usage_log)
    assert speaker.model == "eleven_v3" and speaker.voice_id


def test_history_is_copied_before_use(usage_log):
    from collections import deque

    captured = []
    client = fake_gemini(captured, {"source_lang": "uz", "text": "a", "translation": "b"})
    history = deque([Interpretation(source_lang="uz", text="x", translation="y")])
    Interpreter(client, "m", usage_log).interpret("a", "zh", history)
    assert "[uz] x => y" in system_text(captured[0])
