"""Seminar mode: live interpreting the way a human interpreter works.

Audio arrives in phrases cut at pauses. Each phrase is transcribed and its speaker
identified (voiceprints + Gemini comparing the voices and following the context);
phrases are collected until the speaker finishes a thought, and only then is the
whole thought interpreted into clear Uzbek with the conversation, a running summary
and a terminology list as context. Everything is stored in SQLite, so a seminar can
run all day, survive reloads and be turned into a report for any time range.
"""
import asyncio
import json
import logging
import os
import sqlite3
import time
from collections import Counter, deque
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from google.genai import types
from pydantic import BaseModel, Field

from translator import LANGS, Interpreter
from voiceprint import Voiceprints, cosine

log = logging.getLogger("stt_web")

KEEP_UZBEK = "uz"  # partner setting: leave Uzbek speech in Uzbek
SAME_VOICE = 0.62  # voiceprint similarity above which it is surely the same person
NEW_VOICE = 0.45  # below this for every known speaker it is surely someone new
MAX_REFERENCE_SPEAKERS = 6  # reference recordings sent to Gemini with each phrase
GOOD_REFERENCE_SECONDS = (4.0, 10.0)
UNIT_MAX_SECONDS = 25.0  # interpret even if the speaker hasn't finished the thought by then
IDLE_FLUSH_SECONDS = 4.0  # ...or after this long without new speech
NOTES_FIRST_AFTER_UNITS = 4  # early notes help with names and terms from the start
NOTES_EVERY_UNITS = 12
RECENT_UNITS = 8
RETRY_DELAY = 1.5  # seconds before the one retry of a failed Gemini call

# ---------------------------------------------------------------- prompts

LISTEN_PROMPT = """You are the listening half of a professional conference interpreter at a live seminar.
You get reference recordings of the speakers identified so far, then a NEW SEGMENT of live audio.

1. text: transcribe the NEW SEGMENT verbatim in its original language and script (Uzbek in Latin script).
   Empty if there is no intelligible speech (silence, noise, music, applause, coughing).
2. speaker: who speaks in the NEW SEGMENT. Compare the voice (timbre, pitch, gender, accent, manner)
   with the reference recordings and use the context: who was given the floor, who is answering whom.
   Return a known id, or "new" if it is a person not among the references.
   Language does not decide identity: people switch languages.
   Voiceprint similarity (same person is usually 0.62 or more, different people 0.45 or less):
{similarities}
3. speaker_name: the speaker's own name or title if it becomes clear who this is (they introduce
   themselves, the floor was just handed to someone by name, or someone addresses them by name);
   otherwise empty. Never the name of someone they merely mention.
4. voice: a few words describing the voice (e.g. "deep male voice, slow").
5. source_lang: ISO 639-1 code of the language spoken.
6. complete: true if the segment ends at the end of a sentence or complete thought; false if the
   speaker stopped mid-sentence and will clearly continue.

Seminar notes (use these spellings for names and terms):
{notes}

Recent transcript (oldest first):
{recent}"""

INTERPRET_PROMPT = """You are an experienced conference interpreter working into {target}.
Interpret the speaker's words for an audience that reads your text live.

- Convey the meaning, intent and tone faithfully, not word for word. Do not add facts.
- Make it clear and easy to follow: well-formed sentences, correct punctuation, natural {target}.
- Resolve pronouns and references from the context; keep names, numbers, dates, units and
  currencies exact; use the established translations from the terminology list consistently.
- Drop disfluencies (um, repetitions, false starts) unless they carry meaning.
- If the source is already in {target}, return it cleaned up (punctuation, obvious recognition
  errors) without changing the meaning.
- note: only if a term, abbreviation, idiom or cultural reference would be unclear to the audience,
  a very short explanation in {target}; otherwise empty.

Seminar notes:
{notes}

Recent interpretation (oldest first):
{recent}"""

NOTES_PROMPT = """You keep the working notes of an interpreter at a seminar. Update them with the new part.
- summary: in Uzbek, 3-8 sentences: what the seminar is about, who the speakers are (names, roles),
  what has been covered so far and what is going on now.
- glossary: the important names, organisations, technical terms and abbreviations so far (at most 40),
  each with its established Uzbek rendering and, if useful, a few words of explanation.
  Keep earlier entries unless they were wrong."""

REPORT_PROMPT = """You are the interpreter who worked at this seminar. Write the report for the client in
Uzbek (Latin script), using only what is in the transcript below. Attribute every statement to the right
speaker, keep numbers, names and dates exact, and prefer concrete content over generic phrases.
Put speaker ids (S1, S2...) only into the speaker_id, asked_by and answered_by fields; everywhere in the
text refer to people by their name or label from the speaker list. Times are HH:MM local time.

- title: a short title of the event
- overview: 1-2 paragraphs: what the event was about and how it went
- speakers: for every speaker who spoke: name or label, role if known, what they said in 2-4 sentences
  and their key points
- timeline: the course of the event in order: who did what when (opening, talks, questions, breaks...)
- topics: the main topics discussed with a short summary each
- qa: questions and answers (who asked, who answered)
- decisions: decisions, agreements, conclusions (empty if none)
- action_items: tasks, next steps, deadlines that were mentioned (empty if none)
- glossary: important terms and names with their Uzbek rendering and a short explanation"""


# ---------------------------------------------------------------- schemas


class Heard(BaseModel):
    speaker: str
    speaker_name: str
    voice: str
    source_lang: str
    text: str
    complete: bool


class Rendered(BaseModel):
    translation: str
    note: str


class Term(BaseModel):
    term: str = Field(description="the term or name as it was said, in the original language")
    translation: str = Field(description="its Uzbek rendering")
    note: str = Field(description="a few words of explanation in Uzbek, or empty")


class Notes(BaseModel):
    summary: str
    glossary: list[Term]


class ReportSpeaker(BaseModel):
    speaker_id: str
    name: str
    role: str
    summary: str
    key_points: list[str]


class ReportEvent(BaseModel):
    time: str
    speaker_id: str
    event: str


class ReportTopic(BaseModel):
    title: str
    summary: str


class ReportQA(BaseModel):
    time: str
    asked_by: str
    question: str
    answered_by: str
    answer: str


class Report(BaseModel):
    title: str
    overview: str
    speakers: list[ReportSpeaker]
    timeline: list[ReportEvent]
    topics: list[ReportTopic]
    qa: list[ReportQA]
    decisions: list[str]
    action_items: list[str]
    glossary: list[Term]


# ---------------------------------------------------------------- AI calls


def notes_text(notes: dict) -> str:
    parts = []
    if notes.get("summary"):
        parts.append(notes["summary"])
    terms = notes.get("glossary") or []
    if terms:
        parts.append(
            "Terminology: "
            + "; ".join(f"{t['term']} = {t['translation']}" + (f" ({t['note']})" if t.get("note") else "") for t in terms)
        )
    return "\n".join(parts) or "(none yet)"


class SeminarAI:
    def __init__(self, interpreter: Interpreter, report_model: str | None = None):
        self.interpreter = interpreter
        self.report_model = report_model

    def listen(self, wav: bytes, references: list[dict], similarities: dict, recent: list[str], notes: dict) -> Heard:
        contents = []
        for ref in references:
            contents.append(f"Reference recording of speaker {ref['id']} ({ref['label']}; {ref['voice'] or 'voice not described'}):")
            contents.append(types.Part.from_bytes(data=ref["ref_wav"], mime_type="audio/wav"))
        contents += ["NEW SEGMENT:", types.Part.from_bytes(data=wav, mime_type="audio/wav")]
        sims = "\n".join(f"   - {sid}: {score:.2f}" for sid, score in similarities.items()) or "   (not available)"
        system = LISTEN_PROMPT.format(
            similarities=sims, notes=notes_text(notes), recent="\n".join(recent) or "(nothing yet)"
        )
        heard = self.interpreter.generate(contents, system, Heard, temperature=0.1)
        heard.text = heard.text.strip()
        heard.source_lang = heard.source_lang.strip().lower()
        heard.speaker = heard.speaker.strip()
        heard.speaker_name = heard.speaker_name.strip()
        return heard

    def interpret(self, text: str, source_lang: str, speaker: str, target: str, recent: list[str], notes: dict) -> Rendered:
        target_name = LANGS.get(target, LANGS["uz"])[2]
        system = INTERPRET_PROMPT.format(
            target=target_name, notes=notes_text(notes), recent="\n".join(recent) or "(nothing yet)"
        )
        source = LANGS[source_lang][2] if source_lang in LANGS else source_lang
        result = self.interpreter.generate(f"{speaker} ({source}): {text}", system, Rendered)
        result.translation = result.translation.strip()
        result.note = result.note.strip()
        return result

    def update_notes(self, notes: dict, new_lines: list[str], speakers: list[str]) -> Notes:
        contents = (
            f"Current notes:\n{notes_text(notes)}\n\nSpeakers: {', '.join(speakers)}\n\n"
            "New part of the seminar:\n" + "\n".join(new_lines)
        )
        return self.interpreter.generate(contents, NOTES_PROMPT, Notes)

    def report(self, transcript: str, speakers: str, period: str) -> Report:
        contents = f"Period: {period}\nSpeakers:\n{speakers}\n\nTranscript:\n{transcript}"
        return self.interpreter.generate(
            contents, REPORT_PROMPT, Report, thinking="medium", temperature=0.3, model=self.report_model
        )


# ---------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL,
    partner TEXT NOT NULL DEFAULT 'uz', notes TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS speakers (
    session_id INTEGER NOT NULL, id TEXT NOT NULL, label TEXT NOT NULL, named INTEGER NOT NULL DEFAULT 0,
    voice TEXT NOT NULL DEFAULT '', ref_wav BLOB, ref_seconds REAL NOT NULL DEFAULT 0,
    centroid BLOB, samples INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (session_id, id));
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
    speaker TEXT NOT NULL, lang TEXT NOT NULL, text TEXT NOT NULL, complete INTEGER NOT NULL,
    embedding BLOB, unit_id INTEGER);
CREATE INDEX IF NOT EXISTS segments_session ON segments (session_id, unit_id);
CREATE TABLE IF NOT EXISTS units (
    id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
    speaker TEXT NOT NULL, lang TEXT NOT NULL, target TEXT NOT NULL, source TEXT NOT NULL,
    translation TEXT NOT NULL, note TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS units_session ON units (session_id, start);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, created_at REAL NOT NULL,
    start REAL NOT NULL, end REAL NOT NULL, report TEXT NOT NULL);
"""


def pack(vec: np.ndarray | None) -> bytes | None:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


class SeminarStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _all(self, sql, params=()) -> list[dict]:
        with closing(self._connect()) as db:
            return [dict(r) for r in db.execute(sql, params)]

    def _write(self, sql, params=()) -> int:
        with closing(self._connect()) as db, db:
            return db.execute(sql, params).lastrowid

    # sessions
    def create_session(self, title: str, partner: str = KEEP_UZBEK) -> dict:
        sid = self._write(
            "INSERT INTO sessions (title, created_at, partner) VALUES (?, ?, ?)", (title, time.time(), partner)
        )
        return self.session(sid)

    def session(self, session_id: int) -> dict | None:
        rows = self._all("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if not rows:
            return None
        row = rows[0]
        row["notes"] = json.loads(row["notes"])
        return row

    def sessions(self) -> list[dict]:
        return self._all(
            "SELECT s.id, s.title, s.created_at, s.partner, COUNT(u.id) AS units, "
            "MIN(u.start) AS first, MAX(u.end) AS last FROM sessions s "
            "LEFT JOIN units u ON u.session_id = s.id GROUP BY s.id ORDER BY s.created_at DESC"
        )

    def update_session(self, session_id: int, **fields):
        if "notes" in fields:
            fields["notes"] = json.dumps(fields["notes"], ensure_ascii=False)
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._write(f"UPDATE sessions SET {sets} WHERE id = ?", (*fields.values(), session_id))

    def delete_session(self, session_id: int):
        with closing(self._connect()) as db, db:
            for table in ("units", "segments", "speakers", "reports"):
                db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # speakers
    def speakers(self, session_id: int) -> list[dict]:
        rows = self._all("SELECT * FROM speakers WHERE session_id = ? ORDER BY rowid", (session_id,))
        for r in rows:
            r["centroid"] = unpack(r["centroid"])
        return rows

    def save_speaker(self, session_id: int, sp: dict):
        self._write(
            "INSERT OR REPLACE INTO speakers (session_id, id, label, named, voice, ref_wav, ref_seconds, centroid, samples) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, sp["id"], sp["label"], int(sp["named"]), sp["voice"], sp["ref_wav"], sp["ref_seconds"],
             pack(sp["centroid"]), sp["samples"]),
        )

    def merge_speakers(self, session_id: int, source: str, target: str):
        with closing(self._connect()) as db, db:
            for table in ("segments", "units"):
                db.execute(f"UPDATE {table} SET speaker = ? WHERE session_id = ? AND speaker = ?", (target, session_id, source))
            db.execute("DELETE FROM speakers WHERE session_id = ? AND id = ?", (session_id, source))

    # segments and units
    def add_segment(self, session_id: int, seg: dict) -> int:
        return self._write(
            "INSERT INTO segments (session_id, start, end, speaker, lang, text, complete, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, seg["start"], seg["end"], seg["speaker"], seg["lang"], seg["text"], int(seg["complete"]),
             pack(seg.get("embedding"))),
        )

    def pending_segments(self, session_id: int) -> list[dict]:
        return self._all(
            "SELECT id, start, end, speaker, lang, text, complete FROM segments "
            "WHERE session_id = ? AND unit_id IS NULL ORDER BY id",
            (session_id,),
        )

    def add_unit(self, session_id: int, unit: dict, segment_ids: list[int]) -> dict:
        with closing(self._connect()) as db, db:
            uid = db.execute(
                "INSERT INTO units (session_id, start, end, speaker, lang, target, source, translation, note) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, unit["start"], unit["end"], unit["speaker"], unit["lang"], unit["target"],
                 unit["source"], unit["translation"], unit["note"]),
            ).lastrowid
            db.executemany("UPDATE segments SET unit_id = ? WHERE id = ?", [(uid, s) for s in segment_ids])
        return {"id": uid, **unit}

    def units(self, session_id: int, start: float | None = None, end: float | None = None, limit: int | None = None) -> list[dict]:
        sql = "SELECT id, start, end, speaker, lang, target, source, translation, note FROM units WHERE session_id = ?"
        params: list = [session_id]
        if start is not None:
            sql += " AND end >= ?"
            params.append(start)
        if end is not None:
            sql += " AND start <= ?"
            params.append(end)
        if limit:
            rows = self._all(sql + " ORDER BY start DESC, id DESC LIMIT ?", (*params, limit))
            return rows[::-1]
        return self._all(sql + " ORDER BY start, id", params)

    # reports
    def save_report(self, session_id: int, start: float, end: float, report: dict) -> int:
        return self._write(
            "INSERT INTO reports (session_id, created_at, start, end, report) VALUES (?, ?, ?, ?, ?)",
            (session_id, time.time(), start, end, json.dumps(report, ensure_ascii=False)),
        )

    def reports(self, session_id: int) -> list[dict]:
        return self._all(
            "SELECT id, created_at, start, end FROM reports WHERE session_id = ? ORDER BY created_at DESC", (session_id,)
        )

    def report(self, report_id: int) -> dict | None:
        rows = self._all("SELECT * FROM reports WHERE id = ?", (report_id,))
        if not rows:
            return None
        rows[0]["report"] = json.loads(rows[0]["report"])
        return rows[0]


# ---------------------------------------------------------------- live session


def wav_seconds(wav: bytes) -> float:
    return max(0.0, (len(wav) - 44) / 32000)


def public_speaker(sp: dict) -> dict:
    return {"id": sp["id"], "label": sp["label"], "voice": sp["voice"]}


class LiveSession:
    """Working state of one seminar. Listening and interpreting run as two ordered queues,
    so the next phrase is transcribed while the previous thought is being interpreted."""

    def __init__(self, session_id: int, store: SeminarStore, ai: SeminarAI, voiceprints: Voiceprints | None, tz: ZoneInfo):
        self.id = session_id
        self.store = store
        self.ai = ai
        self.voiceprints = voiceprints
        self.tz = tz
        row = store.session(session_id)
        self.partner = row["partner"]
        self.notes = row["notes"]
        self.speakers = {sp["id"]: sp for sp in store.speakers(session_id)}
        self.pending = store.pending_segments(session_id)  # phrases of the thought in progress
        self.recent_lines: deque[str] = deque(maxlen=8)
        self.recent_units: deque[str] = deque(
            (f"[{self.label(u['speaker'])}] {u['translation']}" for u in store.units(session_id, limit=RECENT_UNITS)),
            maxlen=RECENT_UNITS,
        )
        self.units_since_notes = 0
        self.working = 0  # phrases or thoughts being processed right now
        self.interpreting = 0  # thoughts being interpreted right now
        self.clients: set = set()
        self.listen_queue: asyncio.Queue = asyncio.Queue()
        self.unit_queue: asyncio.Queue = asyncio.Queue()
        self.idle_timer: asyncio.TimerHandle | None = None
        self.notes_task: asyncio.Task | None = None
        self.workers = [
            asyncio.create_task(self._listen_worker()),
            asyncio.create_task(self._interpret_worker()),
        ]
        if self.pending:  # left over from before a restart
            self._arm_idle_flush()

    # --- helpers
    def label(self, speaker_id: str) -> str:
        sp = self.speakers.get(speaker_id)
        return sp["label"] if sp else speaker_id

    def busy(self) -> bool:
        return bool(self.pending) or self.working > 0 or not self.listen_queue.empty() or not self.unit_queue.empty()

    async def broadcast(self, message: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)

    def close(self):
        for task in self.workers:
            task.cancel()
        if self.idle_timer:
            self.idle_timer.cancel()

    # --- input
    async def add_audio(self, client_id: int, wav: bytes, received: float | None = None):
        await self.listen_queue.put((client_id, wav, received or time.time()))

    def set_partner(self, partner: str):
        self.partner = partner
        self.store.update_session(self.id, partner=partner)

    def flush(self):
        """The thought in progress is interpreted now (speaker change, long pause, stop)."""
        if self.idle_timer:
            self.idle_timer.cancel()
            self.idle_timer = None
        if self.pending:
            segments, self.pending = self.pending, []
            self.unit_queue.put_nowait(segments)
            asyncio.ensure_future(self.broadcast_pending())

    async def broadcast_pending(self):
        """What is being said right now (not interpreted yet) and how many thoughts are in translation."""
        await self.broadcast({
            "type": "pending",
            "pending": [{"speaker": p["speaker"], "lang": p["lang"], "text": p["text"]} for p in self.pending],
            "interpreting": self.unit_queue.qsize() + self.interpreting,
        })

    # --- stage 1: listen
    async def _listen_worker(self):
        while True:
            client_id, wav, received = await self.listen_queue.get()
            self.working += 1
            try:
                await self._listen_with_retry(client_id, wav, received)
            finally:
                self.working -= 1

    async def _listen_with_retry(self, client_id: int, wav: bytes, received: float):
        for attempt in (1, 2):
            try:
                await self._listen(client_id, wav, received)
                return
            except Exception:
                log.exception("Listening failed in session %s (attempt %d)", self.id, attempt)
                if attempt == 1:
                    await asyncio.sleep(RETRY_DELAY)
        await self.broadcast({"type": "error", "id": client_id, "message": "Eshitib bo'lmadi"})

    async def _listen(self, client_id: int, wav: bytes, received: float):
        seconds = wav_seconds(wav)
        embedding = None
        if self.voiceprints:
            try:
                embedding = await asyncio.to_thread(self.voiceprints.embed, wav)
            except Exception:
                log.exception("Voiceprint failed")
        sims = {}
        if embedding is not None:
            sims = {sid: cosine(embedding, sp["centroid"]) for sid, sp in self.speakers.items() if sp["centroid"] is not None}
        references = self._references(sims)
        heard = await asyncio.to_thread(
            self.ai.listen, wav, references, {sid: sims[sid] for sid in sims}, list(self.recent_lines), self.notes
        )
        if not heard.text:
            await self.broadcast({"type": "empty", "id": client_id})
            return
        speaker_id = await self._resolve_speaker(heard, sims, embedding, wav, seconds)
        seg = {
            "start": received - seconds,
            "end": received,
            "speaker": speaker_id,
            "lang": heard.source_lang,
            "text": heard.text,
            "complete": heard.complete,
            "embedding": embedding,
        }
        if self.pending and self.pending[0]["speaker"] != speaker_id:
            self.flush()  # someone else started talking: the previous thought is over
        seg["id"] = self.store.add_segment(self.id, seg)
        seg.pop("embedding")
        self.pending.append(seg)
        self.recent_lines.append(f"[{speaker_id} {self.label(speaker_id)}] {heard.text}")
        await self.broadcast({"type": "heard", "id": client_id, "speaker": speaker_id, "lang": heard.source_lang,
                              "text": heard.text, "complete": heard.complete})
        pending_seconds = sum(s["end"] - s["start"] for s in self.pending)
        if heard.complete or pending_seconds >= UNIT_MAX_SECONDS:
            self.flush()
        else:
            self._arm_idle_flush()
            await self.broadcast_pending()

    def _arm_idle_flush(self):
        if self.idle_timer:
            self.idle_timer.cancel()
        self.idle_timer = asyncio.get_running_loop().call_later(IDLE_FLUSH_SECONDS, self._idle_flush)

    def _idle_flush(self):
        self.idle_timer = None
        if self.listen_queue.empty():  # more speech is already waiting: let it complete the thought
            self.flush()
        else:
            self._arm_idle_flush()

    def _references(self, sims: dict) -> list[dict]:
        """The most likely speakers' recordings, so Gemini can compare voices."""
        ranked = sorted(self.speakers.values(), key=lambda sp: -sims.get(sp["id"], 0))
        return [sp for sp in ranked if sp["ref_wav"]][:MAX_REFERENCE_SPEAKERS]

    async def _resolve_speaker(self, heard: Heard, sims: dict, embedding, wav: bytes, seconds: float) -> str:
        best_id, best = max(sims.items(), key=lambda kv: kv[1]) if sims else (None, 0.0)
        gemini_id = heard.speaker if heard.speaker in self.speakers else None
        if best_id and best >= SAME_VOICE:
            speaker_id, confident = best_id, True
        elif sims and best < NEW_VOICE and not (gemini_id and self.speakers[gemini_id]["centroid"] is None):
            speaker_id, confident = None, True
        else:  # voices are ambiguous (or no voiceprints): Gemini's judgement with the context decides
            speaker_id = gemini_id
            confident = gemini_id is not None and gemini_id == best_id
        changed = False
        if speaker_id is None:
            n = 1 + max([int(s[1:]) for s in self.speakers if s[1:].isdigit()] or [0])
            sp = {"id": f"S{n}", "label": heard.speaker_name or f"Spiker {n}", "named": bool(heard.speaker_name),
                  "voice": heard.voice, "ref_wav": wav, "ref_seconds": seconds, "centroid": embedding,
                  "samples": 1 if embedding is not None else 0}
            self.speakers[sp["id"]] = sp
            changed = True
        else:
            sp = self.speakers[speaker_id]
            if heard.speaker_name and not sp["named"]:
                sp["label"], sp["named"] = heard.speaker_name, True
                changed = True
            if embedding is not None and (confident or sp["centroid"] is None):
                n = sp["samples"]
                sp["centroid"] = embedding if sp["centroid"] is None else (sp["centroid"] * n + embedding) / (n + 1)
                sp["samples"] = n + 1
            low, high = GOOD_REFERENCE_SECONDS
            if sp["ref_seconds"] < low and seconds > sp["ref_seconds"] and seconds <= high:
                sp["ref_wav"], sp["ref_seconds"] = wav, seconds
            if not sp["voice"] and heard.voice:
                sp["voice"] = heard.voice
        self.store.save_speaker(self.id, sp)
        if changed:
            await self.broadcast({"type": "speaker", "speaker": public_speaker(sp)})
        return sp["id"]

    # --- stage 2: interpret
    async def _interpret_worker(self):
        while True:
            segments = await self.unit_queue.get()
            self.working += 1
            self.interpreting += 1
            try:
                await self._interpret(segments)
            except Exception:
                log.exception("Saving a unit failed in session %s", self.id)
            finally:
                self.working -= 1
                self.interpreting -= 1
                await self.broadcast_pending()

    async def _interpret(self, segments: list[dict]):
        speaker_id = segments[0]["speaker"]
        lang = Counter(s["lang"] for s in segments).most_common(1)[0][0]
        target = self.partner if lang == "uz" and self.partner != KEEP_UZBEK else "uz"
        source = " ".join(s["text"] for s in segments)
        rendered = Rendered(translation="", note="")  # kept with its original text if Gemini keeps failing
        for attempt in (1, 2):
            try:
                rendered = await asyncio.to_thread(
                    self.ai.interpret, source, lang, self.label(speaker_id), target, list(self.recent_units), self.notes
                )
                break
            except Exception:
                log.exception("Interpreting failed in session %s (attempt %d)", self.id, attempt)
                if attempt == 1:
                    await asyncio.sleep(RETRY_DELAY)
        unit = self.store.add_unit(
            self.id,
            {"start": segments[0]["start"], "end": segments[-1]["end"], "speaker": speaker_id, "lang": lang,
             "target": target, "source": source, "translation": rendered.translation, "note": rendered.note},
            [s["id"] for s in segments],
        )
        if rendered.translation:
            self.recent_units.append(f"[{self.label(speaker_id)}] {rendered.translation}")
        await self.broadcast({"type": "unit", "unit": unit})
        self.units_since_notes += 1
        due = NOTES_EVERY_UNITS if self.notes.get("summary") else NOTES_FIRST_AFTER_UNITS
        if self.units_since_notes >= due and not (self.notes_task and not self.notes_task.done()):
            self.units_since_notes = 0
            self.notes_task = asyncio.create_task(self._update_notes())

    async def _update_notes(self):
        try:
            units = self.store.units(self.id, limit=NOTES_EVERY_UNITS * 2)
            lines = [f"[{self.label(u['speaker'])}] {u['source']} => {u['translation']}" for u in units]
            labels = [f"{sid} = {sp['label']}" for sid, sp in self.speakers.items()]
            notes = await asyncio.to_thread(self.ai.update_notes, self.notes, lines, labels)
            self.notes = notes.model_dump()
            self.store.update_session(self.id, notes=self.notes)
            await self.broadcast({"type": "notes", "notes": self.notes})
        except Exception:
            log.exception("Notes update failed in session %s", self.id)

    # --- speakers edited by the user
    async def rename_speaker(self, speaker_id: str, label: str):
        sp = self.speakers[speaker_id]
        sp["label"], sp["named"] = label, True
        self.store.save_speaker(self.id, sp)
        await self.broadcast({"type": "speaker", "speaker": public_speaker(sp)})

    async def merge_speakers(self, source: str, target: str):
        src, dst = self.speakers.pop(source), self.speakers[target]
        if src["centroid"] is not None:
            if dst["centroid"] is None:
                dst["centroid"], dst["samples"] = src["centroid"], src["samples"]
            else:
                total = dst["samples"] + src["samples"]
                dst["centroid"] = (dst["centroid"] * dst["samples"] + src["centroid"] * src["samples"]) / max(total, 1)
                dst["samples"] = total
        self.store.merge_speakers(self.id, source, target)
        self.store.save_speaker(self.id, dst)
        for seg in self.pending:
            if seg["speaker"] == source:
                seg["speaker"] = target
        await self.broadcast({"type": "merged", "source": source, "target": target})


# ---------------------------------------------------------------- reports


def clock(ts: float, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%H:%M")


def transcript_lines(units: list[dict], speakers: dict[str, str], tz: ZoneInfo, with_source: bool = True) -> list[str]:
    lines = []
    for u in units:
        text = u["translation"] or u["source"]  # an untranslated unit still has its original words
        line = f"[{clock(u['start'], tz)}] {u['speaker']} ({speakers.get(u['speaker'], u['speaker'])}): {text}"
        if with_source and u["translation"] and u["lang"] != u["target"]:
            line += f"  <<{u['lang']}: {u['source']}>>"
        lines.append(line)
    return lines


async def build_report(store: SeminarStore, ai: SeminarAI, session_id: int, start: float, end: float, tz: ZoneInfo) -> int:
    units = store.units(session_id, start, end)
    if not units:
        raise ValueError("Bu oraliqda tarjima qilingan gap yo'q")
    # the report covers what was actually said, not the empty edges of the chosen range
    start, end = max(start, units[0]["start"]), min(end, max(u["end"] for u in units))
    speakers = {sp["id"]: sp["label"] for sp in store.speakers(session_id)}
    period = (
        f"{datetime.fromtimestamp(start, tz):%Y-%m-%d %H:%M} - {datetime.fromtimestamp(end, tz):%Y-%m-%d %H:%M}"
    )
    report = await asyncio.to_thread(
        ai.report,
        "\n".join(transcript_lines(units, speakers, tz)),
        "\n".join(f"{sid}: {label}" for sid, label in speakers.items()),
        period,
    )
    return store.save_report(session_id, start, end, report.model_dump())
