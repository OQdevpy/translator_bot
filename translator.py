"""Gemini interpreter and ElevenLabs voice, shared by the Telegram bot and the web app."""
import logging
import os
from typing import TypeVar

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

from usage import UsageLog

log = logging.getLogger("stt_bot")
T = TypeVar("T", bound=BaseModel)

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
HISTORY_SIZE = 10  # last exchanges given to Gemini as context
GEMINI_TIMEOUT_MS = 60_000


class Interpretation(BaseModel):
    source_lang: str
    text: str
    translation: str


class Interpreter:
    """Detects the language of speech or text and translates it with Gemini."""

    def __init__(
        self,
        client: genai.Client,
        model: str,
        usage_log: UsageLog,
        price_in: float = 1.50,
        price_audio_in: float | None = None,
        price_out: float = 7.50,
    ):
        self.client = client
        self.model = model
        self.usage_log = usage_log
        # USD per 1M tokens; output includes thinking tokens
        self.price_in = price_in
        self.price_audio_in = price_in if price_audio_in is None else price_audio_in
        self.price_out = price_out

    @classmethod
    def from_env(cls, usage_log: UsageLog) -> "Interpreter":
        price_in = float(os.getenv("GEMINI_PRICE_IN") or 1.50)
        return cls(
            genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                # without a timeout a stalled connection would hang its thread forever
                http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
            ),
            os.getenv("GEMINI_MODEL") or "gemini-3.6-flash",
            usage_log,
            price_in=price_in,
            price_audio_in=float(os.getenv("GEMINI_PRICE_AUDIO_IN") or price_in),
            price_out=float(os.getenv("GEMINI_PRICE_OUT") or 7.50),
        )

    def generate(
        self,
        contents,
        system: str,
        schema: type[T],
        thinking: str = "low",
        temperature: float | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """One structured Gemini call; usage and cost are recorded.

        temperature None keeps the model's default: Gemini 3 is tuned for 1.0, and lower values can make
        it loop. timeout (seconds) replaces the client's for this call: a long report needs minutes, a live
        phrase is better retried than awaited. max_tokens (thinking included) stops a looping answer early."""
        response = self.client.models.generate_content(
            model=model or self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                http_options=None if timeout is None else types.HttpOptions(timeout=int(timeout * 1000)),
                thinking_config=types.ThinkingConfig(thinking_level=thinking),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        self.record_usage(response.usage_metadata)
        return schema.model_validate_json(response.text or "")

    def interpret(self, contents, partner: str, history=()) -> Interpretation:
        history = list(history)  # a snapshot: callers may append from other threads
        system = INTERPRET_PROMPT.format(partner=LANGS[partner][2])
        if history:
            system += HISTORY_PROMPT.format(
                history="\n".join(f"[{h.source_lang}] {h.text} => {h.translation}" for h in history)
            )
        result = self.generate(contents, system, Interpretation)
        result.source_lang = result.source_lang.strip().lower()
        result.text = result.text.strip()
        result.translation = result.translation.strip()
        return result

    def record_usage(self, meta: types.GenerateContentResponseUsageMetadata | None):
        if meta is None:
            return
        prompt = meta.prompt_token_count or 0
        audio = sum(
            d.token_count or 0
            for d in meta.prompt_tokens_details or []
            if d.modality == types.MediaModality.AUDIO
        )
        output = (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)
        cost = (
            (prompt - audio) * self.price_in + audio * self.price_audio_in + output * self.price_out
        ) / 1e6
        self.usage_log.add("gemini", input_tokens=prompt, audio_tokens=audio, output_tokens=output, cost=cost)


class Speaker:
    """Text to speech with ElevenLabs."""

    API = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str, model: str, voice_id: str, usage_log: UsageLog):
        self.api_key = api_key
        self.model = model
        self.voice_id = voice_id
        self.usage_log = usage_log

    @classmethod
    def from_env(cls, usage_log: UsageLog) -> "Speaker | None":
        """None when ELEVENLABS_API_KEY is not set."""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key,
            os.getenv("ELEVENLABS_MODEL") or "eleven_v3",
            os.getenv("ELEVENLABS_VOICE_ID") or "EXAVITQu4vr4xnSDxMaL",  # Sarah (premade)
            usage_log,
        )

    def synthesize(self, text: str) -> bytes:
        resp = requests.post(
            f"{self.API}/text-to-speech/{self.voice_id}",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": self.api_key},
            json={"text": text, "model_id": self.model},
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs {resp.status_code}: {resp.text[:300]}")
        cost = resp.headers.get("character-cost", "")
        self.usage_log.add("elevenlabs", credits=int(cost) if cost.isdigit() else len(text))
        return resp.content

    def subscription(self) -> dict | None:
        """Needs the "User: Read" permission on the API key; None when unavailable."""
        try:
            resp = requests.get(
                f"{self.API}/user/subscription", headers={"xi-api-key": self.api_key}, timeout=20
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log.warning("ElevenLabs subscription unavailable: %s", e)
            return None
