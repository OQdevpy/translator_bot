import base64
import json
from types import SimpleNamespace

import pytest
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

import translator
from translator import Interpretation, Interpreter, Speaker
from usage import UsageLog


@pytest.fixture
def usage_log(tmp_path):
    return UsageLog(str(tmp_path / "usage.db"))


def fake_gemini(captured, payload, usage=None, paths=None, options=None):
    """A real genai.Client whose HTTP layer returns `payload` and records the request body (and path,
    and the per-request HTTP options).

    payload None means a response without candidates, as when Gemini blocks a prompt."""
    client = genai.Client(api_key="fake")

    def request(method, path, body, http_options=None):
        captured.append(body)
        if paths is not None:
            paths.append(path)
        if options is not None:
            options.append(http_options)
        if payload is None:
            response = {"promptFeedback": {"blockReason": "SAFETY"}}
        else:
            text = payload if isinstance(payload, str) else json.dumps(payload)
            response = {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}
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
    assert system_text(body) == translator.INTERPRET_PROMPT.format(partner="Simplified Chinese (Mandarin)")
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Привет"}]}]
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"]["required"] == ["source_lang", "text", "translation"]


def test_history_goes_into_the_system_prompt(usage_log):
    captured = []
    client = fake_gemini(captured, {"source_lang": "zh", "text": "多少钱", "translation": "Qancha?"})
    history = [
        Interpretation(source_lang="uz", text="Bu ko'ylak {x}", translation="这件衬衫"),
        Interpretation(source_lang="zh", text="很好", translation="Juda yaxshi"),
    ]
    Interpreter(client, "m", usage_log).interpret("多少钱", "tr", history)
    text = system_text(captured[0])
    assert text.startswith(translator.INTERPRET_PROMPT.format(partner="Turkish"))
    assert "Turkish speaker" in text
    # oldest first, braces in user text are not treated as format fields
    assert text.endswith("translate only the new input:\n[uz] Bu ko'ylak {x} => 这件衬衫\n[zh] 很好 => Juda yaxshi")


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


# ---------------------------------------------------------------- Interpreter.generate


class Item(BaseModel):
    name: str
    count: int


class Basket(BaseModel):
    items: list[Item]
    paid: bool
    comment: str


BASKET = {"items": [{"name": "olma", "count": 3}, {"name": "non", "count": 1}], "paid": True, "comment": " ok "}


def test_generate_parses_into_the_given_schema(usage_log):
    captured, paths = [], []
    client = fake_gemini(captured, BASKET, paths=paths)
    result = Interpreter(client, "base-model", usage_log).generate(["part one", "part two"], "Be exact.", Basket)
    assert isinstance(result, Basket)
    assert result == Basket(items=[Item(name="olma", count=3), Item(name="non", count=1)], paid=True, comment=" ok ")
    body = captured[0]
    assert paths == ["models/base-model:generateContent"]
    assert system_text(body) == "Be exact."
    assert body["contents"] == [{"role": "user", "parts": [{"text": "part one"}, {"text": "part two"}]}]
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    schema = config["responseSchema"]
    assert schema["required"] == ["items", "paid", "comment"]
    assert schema["properties"]["items"]["type"] == "ARRAY"
    assert schema["properties"]["items"]["items"]["required"] == ["name", "count"]
    assert schema["properties"]["items"]["items"]["properties"]["count"]["type"] == "INTEGER"
    # no tools are declared, so the SDK never calls functions on its own
    assert "tools" not in body


def thinking_level(body):
    thinking = body["generationConfig"]["thinkingConfig"]
    return thinking.get("thinkingLevel") or thinking.get("thinking_level")


def test_generate_defaults_low_thinking_and_the_models_own_temperature(usage_log):
    captured, options = [], []
    Interpreter(fake_gemini(captured, BASKET, options=options), "m", usage_log).generate("x", "s", Basket)
    assert thinking_level(captured[0]) == "LOW"
    # Gemini 3 is tuned for its default temperature (lower values can make it loop)
    assert "temperature" not in captured[0]["generationConfig"]
    assert "maxOutputTokens" not in captured[0]["generationConfig"]
    assert options == [None]  # the client's own timeout applies


def test_generate_per_call_timeout_and_output_cap(usage_log):
    captured, options = [], []
    interpreter = Interpreter(fake_gemini(captured, BASKET, options=options), "m", usage_log)
    interpreter.generate("x", "s", Basket, timeout=600, max_tokens=32768)
    interpreter.generate("x", "s", Basket, timeout=2.5, max_tokens=100)
    assert [o.timeout for o in options] == [600_000, 2500]  # milliseconds
    assert [b["generationConfig"]["maxOutputTokens"] for b in captured] == [32768, 100]


def test_generate_passes_thinking_temperature_and_model_override(usage_log):
    captured, paths = [], []
    interpreter = Interpreter(fake_gemini(captured, BASKET, paths=paths), "base-model", usage_log)
    interpreter.generate("x", "s", Basket, thinking="medium", temperature=0.7, model="gemini-pro-test")
    interpreter.generate("x", "s", Basket, thinking="high", model=None)
    assert paths == ["models/gemini-pro-test:generateContent", "models/base-model:generateContent"]
    assert [thinking_level(b) for b in captured] == ["MEDIUM", "HIGH"]
    assert captured[0]["generationConfig"]["temperature"] == pytest.approx(0.7)
    assert interpreter.model == "base-model"  # the override is per call


def test_generate_records_usage_with_audio_priced_separately(usage_log):
    usage = {
        "promptTokenCount": 2000,
        "candidatesTokenCount": 300,
        "thoughtsTokenCount": 700,
        "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 1500}, {"modality": "TEXT", "tokenCount": 500}],
    }
    interpreter = Interpreter(
        fake_gemini([], BASKET, usage), "m", usage_log, price_in=0.5, price_audio_in=1.0, price_out=3.0
    )
    interpreter.generate("x", "s", Basket)
    interpreter.generate("x", "s", Basket)
    t = usage_log.totals("gemini", 0)
    assert (t.requests, t.input_tokens, t.audio_tokens, t.output_tokens) == (2, 4000, 3000, 2000)
    assert t.cost == pytest.approx(2 * (500 * 0.5 + 1500 * 1.0 + 1000 * 3.0) / 1e6)


def test_generate_without_usage_metadata_records_nothing(usage_log):
    Interpreter(fake_gemini([], BASKET), "m", usage_log).generate("x", "s", Basket)
    assert usage_log.totals("gemini", 0).requests == 0


def test_generate_handles_partial_usage_metadata(usage_log):
    interpreter = Interpreter(fake_gemini([], BASKET, {"promptTokenCount": 10}), "m", usage_log, price_in=1.0)
    interpreter.generate("x", "s", Basket)
    t = usage_log.totals("gemini", 0)
    assert (t.requests, t.input_tokens, t.audio_tokens, t.output_tokens) == (1, 10, 0, 0)
    assert t.cost == pytest.approx(10 / 1e6)


@pytest.mark.parametrize(
    "payload",
    [{"items": [], "paid": "maybe", "comment": "x"}, {"items": []}, "not json at all", None],
    ids=["wrong-type", "missing-field", "not-json", "blocked"],
)
def test_generate_rejects_output_that_does_not_fit_the_schema_but_still_bills_it(usage_log, payload):
    client = fake_gemini([], payload, {"promptTokenCount": 7, "candidatesTokenCount": 3})
    with pytest.raises(ValidationError):
        Interpreter(client, "m", usage_log).generate("x", "s", Basket)
    t = usage_log.totals("gemini", 0)
    assert (t.requests, t.input_tokens, t.output_tokens) == (1, 7, 3)


def test_generate_propagates_api_errors_without_recording_usage(usage_log):
    client = genai.Client(api_key="fake")

    def request(method, path, body, http_options=None):
        raise genai.errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})

    client.models._api_client.request = request
    with pytest.raises(genai.errors.ServerError):
        Interpreter(client, "m", usage_log).generate("x", "s", Basket)
    assert usage_log.totals("gemini", 0).requests == 0


def test_generate_sends_audio_parts_inline(usage_log):
    captured = []
    wav = b"RIFF....WAVEfmt "
    contents = ["NEW SEGMENT:", types.Part.from_bytes(data=wav, mime_type="audio/wav")]
    Interpreter(fake_gemini(captured, BASKET), "m", usage_log).generate(contents, "s", Basket)
    parts = captured[0]["contents"][0]["parts"]
    assert parts[0] == {"text": "NEW SEGMENT:"}
    inline = parts[1]["inlineData"]
    assert (inline.get("mimeType") or inline.get("mime_type")) == "audio/wav"
    assert base64.b64decode(inline["data"]) == wav


def test_report_call_uses_the_report_model_and_medium_thinking(usage_log):
    """SeminarAI.report goes through Interpreter.generate with its own model, more thinking and a long timeout."""
    import seminar
    from seminar import Report, SeminarAI

    report = {
        "title": "T", "overview": "O", "speakers": [], "timeline": [], "topics": [], "qa": [],
        "decisions": [], "action_items": [], "glossary": [],
    }
    captured, paths, options = [], [], []
    interpreter = Interpreter(fake_gemini(captured, report, paths=paths, options=options), "flash", usage_log)
    result = SeminarAI(interpreter, report_model="pro").report("[09:00] S1 (Ali): Salom", "S1: Ali", "period")
    # the client-wide timeout is for live calls; a day-long report gets minutes
    assert options[0].timeout == seminar.REPORT_TIMEOUT * 1000 > translator.GEMINI_TIMEOUT_MS
    assert isinstance(result, Report) and result.title == "T"
    assert paths == ["models/pro:generateContent"]
    assert thinking_level(captured[0]) == "MEDIUM"
    assert "temperature" not in captured[0]["generationConfig"]
    assert captured[0]["generationConfig"]["maxOutputTokens"] == seminar.REPORT_MAX_TOKENS

    paths.clear()
    SeminarAI(interpreter).report("t", "s", "p")  # no report model configured: the main model
    assert paths == ["models/flash:generateContent"]
