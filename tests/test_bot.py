import importlib
import sys
from types import SimpleNamespace

import pytest
from google.genai import types

from translator import Interpretation

ENV = {
    "BOT_TOKEN": "123:fake",
    "GEMINI_API_KEY": "fake",
    "GEMINI_MODEL": "gemini-test",
    "GEMINI_PRICE_IN": "1.50",
    "GEMINI_PRICE_AUDIO_IN": "1.50",
    "GEMINI_PRICE_OUT": "7.50",
    "ELEVENLABS_API_KEY": "fake",
    "ELEVENLABS_MODEL": "eleven_v3",
    "ADMIN_IDS": "42, 7",
    "GEMINI_MONTHLY_BUDGET_USD": "5",
    "TIMEZONE": "Asia/Tashkent",
    "REPORT_TIME": "21:00",
}


@pytest.fixture
def b(monkeypatch, tmp_path):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    sys.modules.pop("bot", None)
    bot = importlib.import_module("bot")
    bot.sent = []
    bot.calls = []
    tg = bot.bot
    tg.reply_to = lambda m, text, **kw: bot.sent.append(("reply", text, kw.get("reply_markup")))
    tg.send_message = lambda chat, text, **kw: bot.sent.append(("send", chat, text))
    tg.send_voice = lambda chat, audio, **kw: bot.sent.append(("voice", audio, kw))
    tg.send_chat_action = lambda *a, **k: None
    tg.answer_callback_query = lambda *a, **k: bot.sent.append(("answer", a))
    tg.edit_message_text = lambda text, *a, **kw: bot.sent.append(("edit", text, kw.get("reply_markup")))
    tg.get_file = lambda file_id: SimpleNamespace(file_path="voice.ogg")
    tg.download_file = lambda path: b"OGG"

    def interpret(contents, partner, history=(), live=False):
        bot.calls.append(SimpleNamespace(contents=contents, partner=partner, history=list(history)))
        if isinstance(contents, str) and contents.startswith("Привет"):
            return Interpretation(source_lang="ru", text=contents, translation="Salom")
        return Interpretation(source_lang="uz", text="Salom", translation=f"<{partner}> hello & bye")

    monkeypatch.setattr(bot.interpreter, "interpret", interpret)
    monkeypatch.setattr(bot.speaker, "synthesize", lambda text: f"MP3({text})".encode())
    monkeypatch.setattr(bot.speaker, "subscription", lambda: None)
    yield bot
    sys.modules.pop("bot", None)


def msg(chat=1, text=None, voice=None, user=None):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat),
        from_user=SimpleNamespace(id=user or chat),
        message_id=10,
        text=text,
        voice=voice,
        audio=None,
    )


def test_uzbek_voice_goes_to_default_partner_with_text_and_voice(b):
    b.handle_audio(msg(voice=SimpleNamespace(file_id="f", mime_type="audio/ogg")))
    call = b.calls[-1]
    assert call.partner == "zh" and isinstance(call.contents[0], types.Part) and call.history == []
    kind, text, kb = b.sent[0]
    assert text == "🇺🇿 <code>Salom</code>\n\n🇨🇳 <code>&lt;zh&gt; hello &amp; bye</code>"
    assert [btn.text for btn in kb.keyboard[0]] == ["📋 Asl matn", "📋 Tarjima"]
    assert b.sent[1][:2] == ("voice", b"MP3(<zh> hello & bye)")


def test_language_choice_and_context(b):
    call = SimpleNamespace(
        id="c", data="lang:tr", message=SimpleNamespace(chat=SimpleNamespace(id=1), message_id=5)
    )
    b.handle_lang(call)
    assert b.partner_langs[1] == "tr"
    edit = next(s for s in b.sent if s[0] == "edit")
    assert "Turkcha" in edit[1] and "turkchaga" in edit[1]
    b.handle_text(msg(text="  Salom  "))
    b.handle_text(msg(text="Qalesan"))
    assert b.calls[-1].partner == "tr" and [h.text for h in b.calls[-1].history] == ["Salom"]
    b.handle_lang(SimpleNamespace(id="c", data="lang:xx", message=call.message))
    assert b.partner_langs[1] == "tr"


def test_foreign_text_is_translated_to_uzbek(b):
    b.handle_text(msg(text="Привет"))
    assert b.sent[0][1] == "🇷🇺 <code>Привет</code>\n\n🇺🇿 <code>Salom</code>"


def test_long_texts_are_split(b, monkeypatch):
    monkeypatch.setattr(
        b.interpreter,
        "interpret",
        lambda c, p, h=(): Interpretation(source_lang="en", text="a" * 5000, translation="b" * 4000),
    )
    b.handle_text(msg(text="long"))
    replies = [s for s in b.sent if s[0] == "reply"]
    assert len(replies) == 4 and all(len(r[1]) < 4096 for r in replies)


def test_failures(b, monkeypatch):
    def tts_down(text):
        raise RuntimeError("ElevenLabs 401")

    monkeypatch.setattr(b.speaker, "synthesize", tts_down)
    b.handle_text(msg(text="Salom"))
    assert b.sent[0][0] == "reply" and "🔇" in b.sent[1][1]

    b.sent.clear()
    monkeypatch.setattr(
        b.interpreter, "interpret", lambda c, p, h=(): Interpretation(source_lang="uz", text="", translation="")
    )
    b.handle_audio(msg(voice=SimpleNamespace(file_id="f", mime_type=None)))
    assert b.sent == [("reply", "Nutq topilmadi.", None)]

    b.sent.clear()

    def broken(c, p, h=()):
        raise ValueError("bad json")

    monkeypatch.setattr(b.interpreter, "interpret", broken)
    b.handle_text(msg(text="x"))
    assert "Xatolik" in b.sent[0][1]


def test_history_is_per_chat_and_capped(b):
    for i in range(b.HISTORY_SIZE + 5):
        b.handle_text(msg(chat=2, text=f"gap {i}"))
    assert len(b.histories[2]) == b.HISTORY_SIZE
    assert len(b.calls[-1].history) == b.HISTORY_SIZE
    assert 1 not in b.histories


def test_usage_command_is_admin_only(b):
    b.handle_text(msg(text="Salom"))  # one real-looking request in the log
    b.usage_log.add("gemini", input_tokens=1000, audio_tokens=800, output_tokens=150, cost=0.002)
    b.usage_log.add("elevenlabs", credits=30)

    b.show_usage(msg(chat=99))
    assert "faqat admin" in b.sent[-1][1] and "<code>99</code>" in b.sent[-1][1]

    b.show_usage(msg(chat=42))
    report, kb = b.sent[-1][1], b.sent[-1][2]
    assert report.startswith("📊 <b>Usage</b>")
    assert "1 so'rov · 1 150 token (kirish 1 000, shundan audio 800; chiqish 150)" in report
    assert "Oylik limitdan qoldi: <b>≈ $5.00</b>" in report
    assert "30 kredit" in report and "User → Read" in report
    assert kb.keyboard[0][0].callback_data == "usage"


def test_usage_refresh_and_subscription(b, monkeypatch):
    monkeypatch.setattr(
        b.speaker,
        "subscription",
        lambda: {"tier": "free", "character_count": 2300, "character_limit": 10000,
                 "next_character_count_reset_unix": 1793000000},
    )
    message = SimpleNamespace(chat=SimpleNamespace(id=42), message_id=5)
    b.refresh_usage(SimpleNamespace(id="c", data="usage", from_user=SimpleNamespace(id=42), message=message))
    assert "qoldi <b>7 700</b>" in b.sent[-1][1]
    b.refresh_usage(SimpleNamespace(id="c", data="usage", from_user=SimpleNamespace(id=5), message=message))
    assert b.sent[-1] == ("answer", ("c", "Faqat admin uchun"))


def test_daily_report_goes_to_every_admin(b, monkeypatch):
    sleeps = []

    class Stop(Exception):
        pass

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise Stop

    monkeypatch.setattr(b.time, "sleep", fake_sleep)
    with pytest.raises(Stop):
        b.daily_report_loop()
    assert 0 < sleeps[0] <= 86400
    sent = [s for s in b.sent if s[0] == "send"]
    assert sorted(s[1] for s in sent) == [7, 42]
    assert all("Kunlik hisobot" in s[2] for s in sent)


def test_bot_works_without_elevenlabs(b, monkeypatch):
    monkeypatch.setattr(b, "speaker", None)
    b.handle_text(msg(text="Salom"))
    assert [s[0] for s in b.sent] == ["reply"]
    assert "ElevenLabs" not in b.usage_report()
