import html
import logging
import os

import requests
import telebot
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from telebot.types import BotCommand, CopyTextButton, InlineKeyboardButton, InlineKeyboardMarkup

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah (premade)

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
    "Translate the meaning naturally, as spoken language, not word by word. Keep names as they are.\n"
    "Return JSON fields:\n"
    "- source_lang: ISO 639-1 code of the input language (uz, zh, tr, en, ru, ...)\n"
    "- text: the input in its original language; for audio the exact transcription, "
    "empty if there is no speech\n"
    "- translation: only the translation, no pinyin, quotes, notes or explanations"
)
TEXT_LIMIT = 3500  # below Telegram's 4096 after HTML markup
COPY_LIMIT = 256  # Telegram's limit for copy_text buttons

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("stt_bot")

client = genai.Client(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")

# chat_id -> partner language code; kept in memory, resets to DEFAULT_PARTNER on restart
partner_langs: dict[int, str] = {}


class Interpretation(BaseModel):
    source_lang: str
    text: str
    translation: str


def interpret(contents, partner: str) -> Interpretation:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=INTERPRET_PROMPT.format(partner=LANGS[partner][2]),
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type="application/json",
            response_schema=Interpretation,
        ),
    )
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
    return resp.content


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
    bot.send_chat_action(message.chat.id, "typing")
    try:
        result = interpret(contents, partner)
    except Exception:
        log.exception("Interpretation failed")
        bot.reply_to(message, "Xatolik yuz berdi, birozdan so'ng qayta urinib ko'ring.")
        return
    if not result.text or not result.translation:
        bot.reply_to(message, "Nutq topilmadi.")
        return
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
        "Tarjimani matn va ovoz ko'rinishida yuboraman.\n\n"
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
    bot.set_my_commands(
        [BotCommand("start", "Boshlash"), BotCommand("til", "Suhbatdosh tilini tanlash")]
    )
    log.info("Bot ishga tushdi (gemini: %s, elevenlabs: %s)", GEMINI_MODEL, ELEVENLABS_MODEL)
    bot.infinity_polling(timeout=30, long_polling_timeout=25)
