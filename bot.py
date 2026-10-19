import html
import logging
import os

import requests
import telebot
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telebot.types import CopyTextButton, InlineKeyboardButton, InlineKeyboardMarkup

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah (premade)

STT_PROMPT = (
    "Ushbu audiodagi o'zbekcha gaplarni faqat matn ko'rinishida yozib ber. "
    "Hech qanday qo'shimcha izoh qo'shma."
)
TRANSLATE_PROMPT = (
    "You translate Uzbek text into natural, spoken Simplified Chinese (Mandarin). "
    "Translate the meaning, not word by word. Keep names as they are. "
    "Return only the Chinese translation: no pinyin, quotes, notes or explanations."
)
TEXT_LIMIT = 3500  # below Telegram's 4096 after HTML markup
COPY_LIMIT = 256  # Telegram's limit for copy_text buttons
CAPTION_LIMIT = 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("stt_bot")

client = genai.Client(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")


def gemini(contents, system: str | None = None) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return (response.text or "").strip()


def transcribe(audio: bytes, mime_type: str) -> str:
    return gemini([STT_PROMPT, types.Part.from_bytes(data=audio, mime_type=mime_type)])


def translate_to_chinese(text: str) -> str:
    return gemini(text, system=TRANSLATE_PROMPT)


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


def text_keyboard(text: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    buttons = [InlineKeyboardButton("🔊 Audioga (xitoycha)", callback_data="tts")]
    if len(text) <= COPY_LIMIT:
        buttons.append(InlineKeyboardButton("📋 Copy", copy_text=CopyTextButton(text)))
    kb.row(*buttons)
    return kb


def reply_with_text(message, text: str):
    """Uzbek text as a tap-to-copy block, with the audio/copy buttons."""
    for i in range(0, len(text), TEXT_LIMIT):
        part = text[i : i + TEXT_LIMIT]
        bot.reply_to(message, f"<code>{html.escape(part)}</code>", reply_markup=text_keyboard(part))


@bot.message_handler(commands=["start"])
def start(message):
    bot.reply_to(
        message,
        "Salom! Menga o'zbekcha ovozli xabar yuboring, men uni matnga o'giraman.\n\n"
        "🔊 <b>Audioga</b> tugmasi matnni xitoychaga tarjima qilib, ovozli qilib beradi.\n"
        "📋 <b>Copy</b> bilan matnni nusxalab, xatosini tuzatib qayta yuborishingiz mumkin.",
    )


@bot.message_handler(content_types=["voice", "audio"])
def handle_audio(message):
    media = message.voice or message.audio
    bot.send_chat_action(message.chat.id, "typing")
    try:
        audio = bot.download_file(bot.get_file(media.file_id).file_path)
        text = transcribe(audio, media.mime_type or "audio/ogg")
    except Exception:
        log.exception("Transcription failed")
        bot.reply_to(message, "Xatolik yuz berdi, birozdan so'ng qayta urinib ko'ring.")
        return
    if not text:
        bot.reply_to(message, "Audioda nutq topilmadi.")
        return
    reply_with_text(message, text)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    reply_with_text(message, message.text.strip())


@bot.callback_query_handler(func=lambda call: call.data == "tts")
def handle_tts(call):
    uz_text = call.message.text
    bot.answer_callback_query(call.id, "Tarjima qilinmoqda...")
    bot.send_chat_action(call.message.chat.id, "record_voice")
    try:
        zh_text = translate_to_chinese(uz_text)
        audio = synthesize(zh_text)
    except Exception:
        log.exception("Translate/TTS failed")
        bot.reply_to(call.message, "Audio yaratishda xatolik yuz berdi, qayta urinib ko'ring.")
        return
    caption = html.escape(zh_text)
    bot.send_voice(
        call.message.chat.id,
        audio,
        caption=caption if len(caption) <= CAPTION_LIMIT else None,
        reply_to_message_id=call.message.message_id,
    )
    if len(caption) > CAPTION_LIMIT:
        bot.reply_to(call.message, caption)


if __name__ == "__main__":
    log.info("Bot ishga tushdi (gemini: %s, elevenlabs: %s)", GEMINI_MODEL, ELEVENLABS_MODEL)
    bot.infinity_polling(timeout=30, long_polling_timeout=25)
