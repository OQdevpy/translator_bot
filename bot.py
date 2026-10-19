import html
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import telebot
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from telebot.types import (
    BotCommand,
    BotCommandScopeChat,
    CopyTextButton,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from usage import Totals, UsageLog

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah (premade)

# Telegram user IDs allowed to see /usage and receive the daily report
ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").replace(",", " ").split()}
TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", "Asia/Tashkent"))
REPORT_TIME = os.getenv("REPORT_TIME", "21:00")  # daily report, local time
DATA_DIR = os.getenv("DATA_DIR", "data")
# USD per 1M tokens; output includes thinking tokens
GEMINI_PRICE_IN = float(os.getenv("GEMINI_PRICE_IN", "1.50"))
GEMINI_PRICE_AUDIO_IN = float(os.getenv("GEMINI_PRICE_AUDIO_IN", str(GEMINI_PRICE_IN)))
GEMINI_PRICE_OUT = float(os.getenv("GEMINI_PRICE_OUT", "7.50"))
GEMINI_MONTHLY_BUDGET = float(os.getenv("GEMINI_MONTHLY_BUDGET_USD") or 0)  # 0: not shown

# code -> (flag, Uzbek name for buttons, name for the prompt)
LANGS = {
    "uz": ("🇺🇿", "O'zbekcha", "Uzbek (Latin script)"),
    "zh": ("🇨🇳", "Xitoycha", "Simplified Chinese (Mandarin)"),
    "tr": ("🇹🇷", "Turkcha", "Turkish"),
    "en": ("🇬🇧", "Inglizcha", "English"),
    "ru": ("🇷🇺", "Ruscha", "Russian"),
}
PARTNER_LANGS = [code for code in LANGS if code != "uz"]
DEFAULT_PARTNER = "zh"

INTERPRET_PROMPT = (
    "You are an interpreter between an Uzbek speaker and a {partner} speaker. "
    "Detect the language of the input (speech or text). "
    "If it is Uzbek, translate it into {partner}; "
    "if it is any other language, translate it into Uzbek (Latin script). "
    "Translate the meaning naturally, as spoken language, not word by word. "
    "Keep names, numbers and currencies as they are (e.g. 块/元 is yuan, not so'm).\n"
    "Return JSON fields:\n"
    "- source_lang: ISO 639-1 code of the input language (uz, zh, tr, en, ru, ...)\n"
    "- text: the input in its original language; for audio the exact transcription, "
    "empty if there is no speech\n"
    "- translation: only the translation, no pinyin, quotes, notes or explanations"
)
HISTORY_PROMPT = (
    "\n\nConversation so far, oldest first. Use it only as context to resolve meaning, "
    "names, pronouns and tone; translate only the new input:\n{history}"
)
HISTORY_SIZE = 10  # last exchanges per chat given to Gemini as context
TEXT_LIMIT = 3500  # below Telegram's 4096 after HTML markup
COPY_LIMIT = 256  # Telegram's limit for copy_text buttons

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("stt_bot")

client = genai.Client(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")
usage_log = UsageLog(os.path.join(DATA_DIR, "usage.db"))


class Interpretation(BaseModel):
    source_lang: str
    text: str
    translation: str


# Both kept in memory, so they reset on restart.
# chat_id -> partner language code
partner_langs: dict[int, str] = {}
# chat_id -> recent exchanges, used as translation context
histories: dict[int, deque[Interpretation]] = {}


def interpret(contents, partner: str, history=()) -> Interpretation:
    system = INTERPRET_PROMPT.format(partner=LANGS[partner][2])
    if history:
        system += HISTORY_PROMPT.format(
            history="\n".join(f"[{h.source_lang}] {h.text} => {h.translation}" for h in history)
        )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type="application/json",
            response_schema=Interpretation,
        ),
    )
    record_gemini_usage(response.usage_metadata)
    result = Interpretation.model_validate_json(response.text or "")
    result.source_lang = result.source_lang.strip().lower()
    result.text = result.text.strip()
    result.translation = result.translation.strip()
    return result


def synthesize(text: str) -> bytes:
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        json={"text": text, "model_id": ELEVENLABS_MODEL},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs {resp.status_code}: {resp.text[:300]}")
    cost = resp.headers.get("character-cost", "")
    usage_log.add("elevenlabs", credits=int(cost) if cost.isdigit() else len(text))
    return resp.content


def record_gemini_usage(meta: types.GenerateContentResponseUsageMetadata | None):
    if meta is None:
        return
    prompt = meta.prompt_token_count or 0
    audio = sum(
        d.token_count or 0
        for d in meta.prompt_tokens_details or []
        if d.modality == types.MediaModality.AUDIO
    )
    output = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
    cost = ((prompt - audio) * GEMINI_PRICE_IN + audio * GEMINI_PRICE_AUDIO_IN + output * GEMINI_PRICE_OUT) / 1e6
    usage_log.add("gemini", input_tokens=prompt, audio_tokens=audio, output_tokens=output, cost=cost)


def elevenlabs_subscription() -> dict | None:
    """Needs the "User: Read" permission on the API key; None when unavailable."""
    try:
        resp = requests.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning("ElevenLabs subscription unavailable: %s", e)
        return None


def num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def gemini_line(t: Totals) -> str:
    return (
        f"{t.requests} so'rov · {num(t.input_tokens + t.output_tokens)} token "
        f"(kirish {num(t.input_tokens)}, shundan audio {num(t.audio_tokens)}; "
        f"chiqish {num(t.output_tokens)}) · <b>≈ ${t.cost:.4f}</b>"
    )


def usage_report() -> str:
    now = datetime.now(TIMEZONE)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = day_start.replace(day=1)
    g_day = usage_log.totals("gemini", day_start.timestamp())
    g_month = usage_log.totals("gemini", month_start.timestamp())
    e_day = usage_log.totals("elevenlabs", day_start.timestamp())
    e_month = usage_log.totals("elevenlabs", month_start.timestamp())

    lines = [
        f"📊 <b>Usage</b> · {now:%d.%m.%Y %H:%M}",
        "",
        f"🤖 <b>Gemini</b> ({html.escape(GEMINI_MODEL)})",
        f"Bugun: {gemini_line(g_day)}",
        f"Shu oy: {gemini_line(g_month)}",
    ]
    if GEMINI_MONTHLY_BUDGET:
        left = GEMINI_MONTHLY_BUDGET - g_month.cost
        lines.append(f"Oylik limitdan qoldi: <b>≈ ${left:.2f}</b> / ${GEMINI_MONTHLY_BUDGET:.2f}")
    lines += [
        "",
        f"🔊 <b>ElevenLabs</b> ({html.escape(ELEVENLABS_MODEL)})",
        f"Bugun: {e_day.requests} so'rov · {num(e_day.credits)} kredit",
        f"Shu oy: {e_month.requests} so'rov · {num(e_month.credits)} kredit",
    ]
    sub = elevenlabs_subscription()
    if sub:
        used, limit = sub["character_count"], sub["character_limit"]
        reset = datetime.fromtimestamp(sub["next_character_count_reset_unix"], TIMEZONE)
        lines.append(
            f"Obuna ({html.escape(str(sub.get('tier')))}): {num(used)} / {num(limit)} kredit, "
            f"qoldi <b>{num(limit - used)}</b> (yangilanadi {reset:%d.%m})"
        )
    else:
        lines.append("Qolgan kredit: API kalitga «User → Read» ruxsatini bering")
    lines += [
        "",
        "<i>Gemini summasi bot hisoblagan taxminiy narx: Google balansni API orqali bermaydi.</i>",
    ]
    return "\n".join(lines)


def usage_keyboard() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton("🔄 Yangilash", callback_data="usage"))
    return kb


def daily_report_loop():
    hour, minute = map(int, REPORT_TIME.split(":"))
    while True:
        now = datetime.now(TIMEZONE)
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        time.sleep((next_run - now).total_seconds())
        try:
            report = usage_report()
        except Exception:
            log.exception("Daily report failed")
            continue
        for admin_id in ADMIN_IDS:
            try:
                bot.send_message(admin_id, "🌙 <b>Kunlik hisobot</b>\n\n" + report, reply_markup=usage_keyboard())
            except Exception:
                log.exception("Daily report to %s failed", admin_id)


def flag(code: str) -> str:
    return LANGS[code][0] if code in LANGS else "🌐"


def lang_keyboard(current: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(
        *(
            InlineKeyboardButton(
                f"{'✅ ' if code == current else ''}{LANGS[code][0]} {LANGS[code][1]}",
                callback_data=f"lang:{code}",
            )
            for code in PARTNER_LANGS
        )
    )
    return kb


def copy_keyboard(result: Interpretation) -> InlineKeyboardMarkup | None:
    buttons = [
        InlineKeyboardButton(label, copy_text=CopyTextButton(text))
        for label, text in (("📋 Asl matn", result.text), ("📋 Tarjima", result.translation))
        if len(text) <= COPY_LIMIT
    ]
    if not buttons:
        return None
    kb = InlineKeyboardMarkup()
    kb.row(*buttons)
    return kb


def reply_with_translation(message, result: Interpretation, target: str):
    """Original and translation as tap-to-copy blocks; split if too long for one message."""
    src, dst = flag(result.source_lang), flag(target)
    combined = (
        f"{src} <code>{html.escape(result.text)}</code>\n\n"
        f"{dst} <code>{html.escape(result.translation)}</code>"
    )
    if len(combined) <= TEXT_LIMIT:
        bot.reply_to(message, combined, reply_markup=copy_keyboard(result))
        return
    for f, text in ((src, result.text), (dst, result.translation)):
        for i in range(0, len(text), TEXT_LIMIT):
            bot.reply_to(message, f"{f} <code>{html.escape(text[i : i + TEXT_LIMIT])}</code>")


def reply_with_voice(message, text: str):
    bot.send_chat_action(message.chat.id, "record_voice")
    try:
        audio = synthesize(text)
    except Exception:
        log.exception("TTS failed")
        bot.reply_to(message, "🔇 Ovoz yaratib bo'lmadi, tarjima yuqorida matn ko'rinishida.")
        return
    bot.send_voice(message.chat.id, audio, reply_to_message_id=message.message_id)


def translate_and_reply(message, contents):
    partner = partner_langs.get(message.chat.id, DEFAULT_PARTNER)
    history = histories.setdefault(message.chat.id, deque(maxlen=HISTORY_SIZE))
    bot.send_chat_action(message.chat.id, "typing")
    try:
        result = interpret(contents, partner, history)
    except Exception:
        log.exception("Interpretation failed")
        bot.reply_to(message, "Xatolik yuz berdi, birozdan so'ng qayta urinib ko'ring.")
        return
    if not result.text or not result.translation:
        bot.reply_to(message, "Nutq topilmadi.")
        return
    history.append(result)
    target = partner if result.source_lang == "uz" else "uz"
    reply_with_translation(message, result, target)
    reply_with_voice(message, result.translation)


@bot.message_handler(commands=["start"])
def start(message):
    partner = partner_langs.get(message.chat.id, DEFAULT_PARTNER)
    bot.reply_to(
        message,
        "Salom! Men tarjimonman: 🇺🇿 o'zbekcha ↔ 🇨🇳 🇹🇷 🇬🇧 🇷🇺\n\n"
        "Ovozli yoki matnli xabar yuboring, tilini o'zim aniqlayman:\n"
        "• o'zbekcha bo'lsa, suhbatdosh tiliga tarjima qilaman;\n"
        "• boshqa tilda bo'lsa, o'zbekchaga tarjima qilaman.\n"
        "Suhbatning oldingi gaplarini hisobga olib tarjima qilaman, "
        "natijani matn va ovoz ko'rinishida yuboraman.\n\n"
        "Suhbatdosh tilini tanlang (keyin /til bilan o'zgartirasiz):",
        reply_markup=lang_keyboard(partner),
    )


@bot.message_handler(commands=["til", "lang"])
def choose_lang(message):
    partner = partner_langs.get(message.chat.id, DEFAULT_PARTNER)
    bot.reply_to(message, "Suhbatdosh tilini tanlang:", reply_markup=lang_keyboard(partner))


@bot.callback_query_handler(func=lambda call: call.data.startswith("lang:"))
def handle_lang(call):
    code = call.data.removeprefix("lang:")
    if code not in PARTNER_LANGS:
        bot.answer_callback_query(call.id)
        return
    partner_langs[call.message.chat.id] = code
    flag_, name, _ = LANGS[code]
    bot.answer_callback_query(call.id, f"Suhbatdosh tili: {name}")
    bot.edit_message_text(
        f"✅ Suhbatdosh tili: {flag_} <b>{name}</b>\n"
        f"O'zbekcha gaplar {name.lower()}ga, {name.lower()} gaplar o'zbekchaga tarjima qilinadi.",
        call.message.chat.id,
        call.message.message_id,
        reply_markup=lang_keyboard(code),
    )


@bot.message_handler(commands=["usage"])
def show_usage(message):
    if message.from_user.id not in ADMIN_IDS:
        log.info("/usage denied for user %s", message.from_user.id)
        bot.reply_to(
            message,
            "⛔ Bu buyruq faqat admin uchun.\n"
            f"Sizning ID: <code>{message.from_user.id}</code> (uni .env dagi ADMIN_IDS ga yozing).",
        )
        return
    bot.send_chat_action(message.chat.id, "typing")
    bot.reply_to(message, usage_report(), reply_markup=usage_keyboard())


@bot.callback_query_handler(func=lambda call: call.data == "usage")
def refresh_usage(call):
    if call.from_user.id not in ADMIN_IDS:
        bot.answer_callback_query(call.id, "Faqat admin uchun")
        return
    bot.answer_callback_query(call.id, "Yangilandi")
    try:
        bot.edit_message_text(
            usage_report(), call.message.chat.id, call.message.message_id, reply_markup=usage_keyboard()
        )
    except telebot.apihelper.ApiTelegramException as e:
        if "message is not modified" not in str(e):
            raise


@bot.message_handler(content_types=["voice", "audio"])
def handle_audio(message):
    media = message.voice or message.audio
    try:
        audio = bot.download_file(bot.get_file(media.file_id).file_path)
    except Exception:
        log.exception("Download failed")
        bot.reply_to(message, "Xatolik yuz berdi, birozdan so'ng qayta urinib ko'ring.")
        return
    translate_and_reply(message, [types.Part.from_bytes(data=audio, mime_type=media.mime_type or "audio/ogg")])


@bot.message_handler(content_types=["text"])
def handle_text(message):
    text = message.text.strip()
    if text:
        translate_and_reply(message, text)


if __name__ == "__main__":
    commands = [BotCommand("start", "Boshlash"), BotCommand("til", "Suhbatdosh tilini tanlash")]
    bot.set_my_commands(commands)
    for admin_id in ADMIN_IDS:
        try:  # fails until the admin has opened a chat with the bot
            bot.set_my_commands(
                [*commands, BotCommand("usage", "Gemini va ElevenLabs sarfi")],
                scope=BotCommandScopeChat(admin_id),
            )
        except Exception as e:
            log.warning("Admin commands for %s not set: %s", admin_id, e)
    if ADMIN_IDS:
        threading.Thread(target=daily_report_loop, daemon=True).start()
    log.info("Bot ishga tushdi (gemini: %s, elevenlabs: %s)", GEMINI_MODEL, ELEVENLABS_MODEL)
    bot.infinity_polling(timeout=30, long_polling_timeout=25)
