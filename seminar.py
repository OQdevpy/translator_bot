"""Seminar mode: live interpreting the way a human interpreter works.

Audio arrives in phrases cut at pauses. Each phrase is transcribed and its speaker
identified (voiceprints + Gemini comparing the voices and following the context);
phrases are collected until the speaker finishes a thought, and only then is the
whole thought interpreted into clear Uzbek with the conversation, a running summary
and a terminology list as context. Everything is stored in SQLite, so a seminar can
run all day, survive reloads and be turned into a report for any time range.
"""
import asyncio
import io
import json
import logging
import os
import re
import sqlite3
import time
import wave
from collections import Counter, OrderedDict, deque
from contextlib import closing
from datetime import datetime
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

import numpy as np
from google.genai import types
from pydantic import BaseModel, Field

from translator import LANGS, Interpreter
from voiceprint import Voiceprints, cosine

log = logging.getLogger("stt_web")

KEEP_UZBEK = "uz"  # partner setting: leave Uzbek speech in Uzbek
SAME_VOICE = 0.62  # voiceprint similarity above which it is surely the same person
NEW_VOICE = 0.30  # different people usually score below this (short clips of one person can too)
SURELY_NEW_VOICE = 0.20  # below this for everyone, on a long clip, it is surely someone new
NEW_VOICE_MIN_SECONDS = 3.0  # shorter clips score too low too often to declare someone new by voice
CENTROID_MIN_SECONDS = 2.0  # shorter clips make noisy voiceprints
REFERENCE_MIN_SECONDS = 1.0  # a speaker known only from a shorter clip can't be compared by voice
UNKNOWN_VOICE_SILENT_SECONDS = 30 * 60  # a speaker without a voiceprint silent this long is not expected back
MAX_REFERENCE_SPEAKERS = 6  # reference recordings sent to Gemini with each phrase
GOOD_REFERENCE_SECONDS = (4.0, 8.0)
UNIT_MAX_SECONDS = 25.0  # interpret even if the speaker hasn't finished the thought by then
IDLE_FLUSH_SECONDS = 4.0  # ...or after this long without speech
NOTES_FIRST_AFTER_UNITS = 4  # early notes help with names and terms from the start
NOTES_EVERY_UNITS = 12
RECENT_UNITS = 8
RECENT_LINES = 8
RETRY_DELAY = 1.5  # seconds before retrying a failed Gemini call (doubles for listening)
LISTEN_RETRY_SECONDS = 60.0  # a phrase is retried this long (an outage, a rate limit) before it is given up
MAX_QUEUED_SEGMENTS = 20  # per session; protects the API budget from a runaway client
DEDUPE_SECONDS = 15 * 60  # a phrase resent after a reconnect is recognised for this long
ECHO_SECONDS = 90.0  # a phrase that matches a translation this recent is its read-out picked up by a microphone
ECHO_SIMILARITY = 0.8
DUPLICATE_SECONDS = 30.0  # two devices recording the same room hear the same phrase within this time
DUPLICATE_SIMILARITY = 0.85
# per-call limits: a stuck call must not hold up the live queue, a long report must be allowed to finish
LISTEN_TIMEOUT, LISTEN_MAX_TOKENS = 20.0, 2048
INTERPRET_TIMEOUT, INTERPRET_MAX_TOKENS = 60.0, 2048
NOTES_TIMEOUT, NOTES_MAX_TOKENS = 60.0, 8192
REPORT_TIMEOUT, REPORT_MAX_TOKENS = 600.0, 32768

# ---------------------------------------------------------------- prompts

LISTEN_PROMPT = """You are the listening half of a professional conference interpreter at a live seminar.
You get reference recordings of the speakers identified so far, then a NEW SEGMENT of live audio.

1. text: transcribe the NEW SEGMENT verbatim in its original language and script (Uzbek in Latin script).
   Empty if there is no intelligible speech (silence, noise, music, applause, coughing).
2. source_lang: ISO 639-1 code of the language spoken.
3. complete: true if the segment ends at the end of a sentence or complete thought; false if the
   speaker stopped mid-sentence and will clearly continue.
4. voice: a few words describing the voice (e.g. "deep male voice, slow").
5. speaker_name: the name or title of the person speaking in the NEW SEGMENT, only if it is clear: they
   introduce themselves, or right before this segment someone else handed the floor to them or addressed
   them by name. Never the name of the person they address or hand the floor to, nor of anyone they
   merely mention; otherwise empty.
6. speaker: who speaks in the NEW SEGMENT. Compare the voice (timbre, pitch, gender, accent, manner)
   with the reference recordings and use the context: who was given the floor, who is answering whom.
   Return a known id, or "new" if it is a person not among the references.
   Language does not decide identity: people switch languages.
{voices}

Terminology (use these spellings for names and terms):
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
  what has been covered so far and what is going on now. Refer to people by their name or their label
  from the speaker list; never use ids like S1.
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
    # Gemini writes the fields in this order: the transcript first, identity last, decided with the words in view
    text: str
    source_lang: str
    complete: bool
    voice: str
    speaker_name: str
    speaker: str


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

    def listen(self, wav: bytes, references: list[dict], similarities: dict, recent: list[str], notes: dict,
               identified: str | None = None) -> Heard:
        """identified: the speaker id (or "new") when the voiceprints already settled who speaks."""
        contents = []
        for ref in references:
            contents.append(f"Reference recording of speaker {ref['id']} ({ref['label']}; {ref['voice'] or 'voice not described'}):")
            contents.append(types.Part.from_bytes(data=ref["ref_wav"], mime_type="audio/wav"))
        contents += ["NEW SEGMENT:", types.Part.from_bytes(data=wav, mime_type="audio/wav")]
        if identified == "new":  # told, so that speaker_name is about the right person
            voices = '   The voiceprints show this is someone not heard before: answer "new".'
        elif identified:
            voices = f'   The voiceprints identified this speaker as {identified}: answer "{identified}".'
        else:
            voices = (
                f"   Voiceprint similarity (the same person usually scores {SAME_VOICE:.2f} or more, different people"
                f" {NEW_VOICE:.2f} or less; short segments score lower):\n"
                + ("\n".join(f"   - {sid}: {score:.2f}" for sid, score in similarities.items()) or "   (not available)")
            )
        system = LISTEN_PROMPT.format(
            voices=voices,
            notes=notes_text({"glossary": notes.get("glossary")}),
            recent="\n".join(recent) or "(nothing yet)",
        )
        heard = self.interpreter.generate(
            contents, system, Heard, timeout=LISTEN_TIMEOUT, max_tokens=LISTEN_MAX_TOKENS
        )
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
        result = self.interpreter.generate(
            f"{speaker} ({source}): {text}", system, Rendered, timeout=INTERPRET_TIMEOUT, max_tokens=INTERPRET_MAX_TOKENS
        )
        result.translation = result.translation.strip()
        result.note = result.note.strip()
        return result

    def update_notes(self, notes: dict, new_lines: list[str], speakers: list[str]) -> Notes:
        """speakers: their labels only; ids would end up in the summary, which every device shows."""
        contents = (
            f"Current notes:\n{notes_text(notes)}\n\nSpeakers: {', '.join(speakers)}\n\n"
            "New part of the seminar:\n" + "\n".join(new_lines)
        )
        return self.interpreter.generate(
            contents, NOTES_PROMPT, Notes, timeout=NOTES_TIMEOUT, max_tokens=NOTES_MAX_TOKENS
        )

    def report(self, transcript: str, speakers: str, period: str) -> Report:
        contents = f"Period: {period}\nSpeakers:\n{speakers}\n\nTranscript:\n{transcript}"
        return self.interpreter.generate(
            contents, REPORT_PROMPT, Report, thinking="medium", model=self.report_model,
            timeout=REPORT_TIMEOUT, max_tokens=REPORT_MAX_TOKENS,
        )


# ---------------------------------------------------------------- storage

# AUTOINCREMENT: ids are never reused, so late background work of a deleted session or report
# (or a device still holding its id) can't land in a new one. merged: merged-away speaker id -> the one it
# became, so those ids are never issued again; notes_unit: the last unit the notes cover.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, created_at REAL NOT NULL,
    partner TEXT NOT NULL DEFAULT 'uz', notes TEXT NOT NULL DEFAULT '{}',
    merged TEXT NOT NULL DEFAULT '{}', notes_unit INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS speakers (
    session_id INTEGER NOT NULL, id TEXT NOT NULL, label TEXT NOT NULL, named INTEGER NOT NULL DEFAULT 0,
    voice TEXT NOT NULL DEFAULT '', ref_wav BLOB, ref_seconds REAL NOT NULL DEFAULT 0,
    centroid BLOB, samples INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (session_id, id));
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
    speaker TEXT NOT NULL, lang TEXT NOT NULL, text TEXT NOT NULL, complete INTEGER NOT NULL,
    embedding BLOB, unit_id INTEGER);
CREATE INDEX IF NOT EXISTS segments_session ON segments (session_id, unit_id);
CREATE TABLE IF NOT EXISTS units (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
    speaker TEXT NOT NULL, lang TEXT NOT NULL, target TEXT NOT NULL, source TEXT NOT NULL,
    translation TEXT NOT NULL, note TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS units_session ON units (session_id, start);
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL, created_at REAL NOT NULL,
    start REAL NOT NULL, end REAL NOT NULL, report TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'done', error TEXT NOT NULL DEFAULT '');
"""
MAX_ID = 2**63 - 1  # SQLite INTEGER range; larger ids can't exist


class ReportRangeError(Exception):
    """The chosen time range can't be reported on (nothing said, or too long for one report)."""


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
        if not 0 < session_id <= MAX_ID:
            return None
        rows = self._all("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if not rows:
            return None
        row = rows[0]
        row["notes"] = json.loads(row["notes"])
        row["merged"] = json.loads(row["merged"])
        return row

    def sessions(self) -> list[dict]:
        return self._all(
            "SELECT s.id, s.title, s.created_at, s.partner, COUNT(u.id) AS units, "
            "MIN(u.start) AS first, MAX(u.end) AS last FROM sessions s "
            "LEFT JOIN units u ON u.session_id = s.id GROUP BY s.id ORDER BY s.created_at DESC"
        )

    def update_session(self, session_id: int, **fields):
        for key in ("notes", "merged"):
            if key in fields:
                fields[key] = json.dumps(fields[key], ensure_ascii=False)
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
            "INSERT INTO speakers (session_id, id, label, named, voice, ref_wav, ref_seconds, centroid, samples) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (session_id, id) DO UPDATE SET label = excluded.label, "
            "named = excluded.named, voice = excluded.voice, ref_wav = excluded.ref_wav, "
            "ref_seconds = excluded.ref_seconds, centroid = excluded.centroid, samples = excluded.samples",
            (session_id, sp["id"], sp["label"], int(sp["named"]), sp["voice"], sp["ref_wav"], sp["ref_seconds"],
             pack(sp["centroid"]), sp["samples"]),
        )

    def merge_speakers(self, session_id: int, source: str, target: str):
        with closing(self._connect()) as db, db:
            for table in ("segments", "units"):
                db.execute(f"UPDATE {table} SET speaker = ? WHERE session_id = ? AND speaker = ?", (target, session_id, source))
            db.execute("DELETE FROM speakers WHERE session_id = ? AND id = ?", (session_id, source))
            (merged,) = db.execute("SELECT merged FROM sessions WHERE id = ?", (session_id,)).fetchone()
            merged = {old: target if new == source else new for old, new in json.loads(merged).items()}
            merged[source] = target
            db.execute("UPDATE sessions SET merged = ? WHERE id = ?", (json.dumps(merged), session_id))

    def speaker_labels(self, session_id: int) -> dict[str, str]:
        """Speaker id -> label, merged-away ids included: reports written before a merge still name them."""
        labels = {r["id"]: r["label"] for r in self._all(
            "SELECT id, label FROM speakers WHERE session_id = ? ORDER BY rowid", (session_id,))}
        row = self.session(session_id)
        for old, new in (row["merged"] if row else {}).items():
            if new in labels:
                labels[old] = labels[new]
        return labels

    def last_spoken(self, session_id: int) -> dict[str, float]:
        return {r["speaker"]: r["last"] for r in self._all(
            "SELECT speaker, MAX(end) AS last FROM segments WHERE session_id = ? GROUP BY speaker", (session_id,))}

    # segments and units
    def add_segment(self, session_id: int, seg: dict) -> int:
        return self._write(
            "INSERT INTO segments (session_id, start, end, speaker, lang, text, complete, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, seg["start"], seg["end"], seg["speaker"], seg["lang"], seg["text"], int(seg["complete"]),
             pack(seg.get("embedding"))),
        )

    def recent_segments(self, session_id: int, limit: int) -> list[dict]:
        rows = self._all(
            "SELECT speaker, text FROM segments WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit)
        )
        return rows[::-1]

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

    def count_units(self, session_id: int, after_id: int = 0) -> int:
        return self._all(
            "SELECT COUNT(*) AS n FROM units WHERE session_id = ? AND id > ?", (session_id, after_id)
        )[0]["n"]

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

    def start_report(self, session_id: int, start: float, end: float) -> int:
        return self._write(
            "INSERT INTO reports (session_id, created_at, start, end, report, status) VALUES (?, ?, ?, ?, '{}', 'pending')",
            (session_id, time.time(), start, end),
        )

    def finish_report(self, report_id: int, report: dict):
        self._write(
            "UPDATE reports SET report = ?, status = 'done' WHERE id = ?",
            (json.dumps(report, ensure_ascii=False), report_id),
        )

    def fail_report(self, report_id: int, error: str):
        self._write("UPDATE reports SET status = 'failed', error = ? WHERE id = ?", (error, report_id))

    def fail_unfinished_reports(self, error: str):
        """Reports that were being written when the server stopped will never finish."""
        self._write("UPDATE reports SET status = 'failed', error = ? WHERE status = 'pending'", (error,))

    def reports(self, session_id: int) -> list[dict]:
        return self._all(
            "SELECT id, created_at, start, end, status, error FROM reports WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        )

    def report(self, report_id: int) -> dict | None:
        if not 0 < report_id <= MAX_ID:
            return None
        rows = self._all("SELECT * FROM reports WHERE id = ?", (report_id,))
        if not rows:
            return None
        rows[0]["report"] = json.loads(rows[0]["report"])
        return rows[0]


# ---------------------------------------------------------------- live session


def wav_seconds(wav: bytes) -> float:
    return max(0.0, (len(wav) - 44) / 32000)


def trim_wav(wav: bytes, seconds: float) -> bytes:
    """The first `seconds` of a WAV clip (unchanged if shorter or unreadable)."""
    try:
        with wave.open(io.BytesIO(wav)) as r:
            params, frames = r.getparams(), int(seconds * r.getframerate())
            if r.getnframes() <= frames:
                return wav
            data = r.readframes(frames)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setparams(params)
            w.writeframes(data)
        return buf.getvalue()
    except (wave.Error, EOFError):
        return wav


def normalized(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())


def public_speaker(sp: dict) -> dict:
    return {"id": sp["id"], "label": sp["label"], "voice": sp["voice"]}


class LiveSession:
    """Working state of one seminar. Listening and interpreting run as two ordered queues,
    so the next phrase is transcribed while the previous thought is being interpreted.

    Answers about a phrase (heard, empty, error) go only to the socket that sent it, as its
    acknowledgement; everything else is broadcast to every device on the seminar."""

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
        self.merged: dict[str, str] = row["merged"]  # merged-away speaker id -> the one it became
        self.last_spoken: dict[str, float] = store.last_spoken(session_id)
        self.pending: list[dict] = []  # phrases of the thought in progress
        restored = self._regroup(store.pending_segments(session_id))
        # (speaker id, text): rendered with the current ids and labels, so renames and merges show up at once
        self.recent_lines: deque[tuple[str, str]] = deque(
            ((s["speaker"], s["text"]) for s in store.recent_segments(session_id, RECENT_LINES)), maxlen=RECENT_LINES
        )
        self.recent_units: deque[tuple[str, str]] = deque(
            ((u["speaker"], u["translation"]) for u in store.units(session_id, limit=RECENT_UNITS) if u["translation"]),
            maxlen=RECENT_UNITS,
        )
        self.read_out: deque[tuple[float, list[str]]] = deque()  # (time, normalized translation and its sentences)
        self.recent_heard: deque[tuple[float, object, str]] = deque()  # (time, sending socket, normalized phrase)
        self.units_since_notes = store.count_units(session_id, row["notes_unit"])
        self.working = 0  # phrases or thoughts being processed right now
        self.listening = 0  # phrases being transcribed right now
        self.interpreting = 0  # thoughts being interpreted right now
        self.clients: set = set()
        self.speaking: set = set()  # clients whose microphone hears speech right now
        self.idle_since = time.monotonic()  # when the last client left
        self.received: OrderedDict[tuple[str, int], float] = OrderedDict()  # (client, phrase id) -> when
        self.flush_requested = False
        self.closed = False
        self.listen_queue: asyncio.Queue = asyncio.Queue()
        self.unit_queue: asyncio.Queue = asyncio.Queue()
        self.idle_timer: asyncio.TimerHandle | None = None
        self.notes_task: asyncio.Task | None = None
        self.workers = [
            asyncio.create_task(self._listen_worker()),
            asyncio.create_task(self._interpret_worker()),
        ]
        for thought in restored:  # left over from before a restart
            self.unit_queue.put_nowait(thought)
        if self.pending:
            self._arm_idle_flush()

    def _regroup(self, segments: list[dict]) -> list[list[dict]]:
        """Splits stored, not yet interpreted phrases back into thoughts; the unfinished tail becomes pending."""
        thoughts, current = [], []
        for seg in segments:
            if current and current[0]["speaker"] != seg["speaker"]:
                thoughts.append(current)
                current = []
            current.append(seg)
            if seg["complete"]:
                thoughts.append(current)
                current = []
        self.pending = current
        return thoughts

    # --- helpers
    def label(self, speaker_id: str) -> str:
        sp = self.speakers.get(self.resolve(speaker_id))
        return sp["label"] if sp else speaker_id

    def resolve(self, speaker_id: str) -> str:
        """Follows merges, for thoughts and answers that were already on their way when the speakers were merged."""
        while speaker_id in self.merged:
            speaker_id = self.merged[speaker_id]
        return speaker_id

    def recent_transcript(self) -> list[str]:
        return [f"[{self.resolve(sid)} {self.label(sid)}] {text}" for sid, text in self.recent_lines]

    def recent_interpretation(self) -> list[str]:
        return [f"[{self.label(sid)}] {text}" for sid, text in self.recent_units]

    def busy(self) -> bool:
        return bool(self.pending) or self.working > 0 or not self.listen_queue.empty() or not self.unit_queue.empty()

    def interpreting_count(self) -> int:
        """Thoughts waiting for or in interpretation."""
        return self.unit_queue.qsize() + self.interpreting

    async def broadcast(self, message: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)

    async def _reply(self, ws, message: dict):
        """An answer about one phrase, to the device that sent it (it may have gone meanwhile)."""
        if ws is None:
            return
        try:
            await ws.send_json(message)
        except Exception:
            self.clients.discard(ws)

    def close(self):
        self.closed = True
        for task in self.workers:
            task.cancel()
        if self.notes_task:
            self.notes_task.cancel()
        if self.idle_timer:
            self.idle_timer.cancel()
            self.idle_timer = None

    # --- clients
    def join(self, ws):
        self.clients.add(ws)

    def leave(self, ws):
        self.clients.discard(ws)
        if ws in self.speaking:
            self.set_speaking(ws, False)  # a device that is gone is not talking
        if not self.clients:
            self.idle_since = time.monotonic()

    def set_speaking(self, ws, active: bool):
        """A device's microphone started or stopped hearing speech: a thought isn't over while someone talks,
        and the pause before the idle flush counts from when they stopped."""
        if active:
            self.speaking.add(ws)
            return
        self.speaking.discard(ws)
        if self.pending:
            self._arm_idle_flush()

    # --- input
    def queue_full(self) -> bool:
        return self.listen_queue.qsize() >= MAX_QUEUED_SEGMENTS

    def received_before(self, client: str, seg_id: int) -> bool:
        """A phrase resent after a reconnect that already arrived (it may still be waiting in the queue)."""
        self._forget_old_phrases()
        return bool(client) and (client, seg_id) in self.received

    def _forget_old_phrases(self):
        cutoff = time.monotonic() - DEDUPE_SECONDS
        while self.received and next(iter(self.received.values())) < cutoff:
            self.received.popitem(last=False)

    async def add_audio(self, seg_id: int, wav: bytes, received: float | None = None, origin=None, client: str = ""):
        """origin: the socket that sent the phrase and gets the answer; client: the sending page, for deduplication."""
        if client:
            self._forget_old_phrases()
            self.received[(client, seg_id)] = time.monotonic()
        await self.listen_queue.put((seg_id, wav, received or time.time(), origin))

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

    def request_flush(self):
        """A device stopped listening: what is left is interpreted once the phrases already sent are transcribed."""
        self.flush_requested = True
        self._flush_if_drained()

    def _flush_if_drained(self):
        if self.flush_requested and self.listen_queue.empty() and self.listening == 0:
            self.flush_requested = False
            self.flush()

    async def broadcast_pending(self):
        """What is being said right now (not interpreted yet) and how many thoughts are in translation."""
        await self.broadcast({
            "type": "pending",
            "pending": [{"speaker": p["speaker"], "lang": p["lang"], "text": p["text"]} for p in self.pending],
            "interpreting": self.interpreting_count(),
        })

    # --- stage 1: listen
    async def _listen_worker(self):
        while True:
            seg_id, wav, received, origin = await self.listen_queue.get()
            self.working += 1
            self.listening += 1
            try:
                await self._listen_with_retry(seg_id, wav, received, origin)
            finally:
                self.working -= 1
                self.listening -= 1
                self._flush_if_drained()

    async def _listen_with_retry(self, seg_id: int, wav: bytes, received: float, origin):
        embedding = None
        if self.voiceprints:
            try:
                embedding = await asyncio.to_thread(self.voiceprints.embed, wav)
            except Exception:
                log.exception("Voiceprint failed")
        # an outage or a rate limit passes: keep the phrase a while, unless the queue behind it is full
        deadline, delay = time.monotonic() + LISTEN_RETRY_SECONDS, RETRY_DELAY
        while True:
            try:
                await self._listen(seg_id, wav, received, origin, embedding)
                return
            except Exception:
                log.exception("Listening failed in session %s", self.id)
            if self.closed or self.queue_full() or time.monotonic() + delay > deadline:
                break
            await asyncio.sleep(delay)
            delay *= 2
        await self._reply(origin, {"type": "error", "id": seg_id, "message": "Eshitib bo'lmadi"})

    def _similarities(self, embedding) -> dict[str, float]:
        if embedding is None:
            return {}
        return {sid: cosine(embedding, sp["centroid"]) for sid, sp in self.speakers.items() if sp["centroid"] is not None}

    def _unknown_voices(self) -> bool:
        """Is there a speaker Gemini could recognise by their reference clip but the voiceprints can't
        (no voiceprint yet), who still speaks in this part of the seminar?"""
        latest = max(self.last_spoken.values(), default=0.0)
        return any(
            sp["centroid"] is None and sp["ref_wav"] and sp["ref_seconds"] >= REFERENCE_MIN_SECONDS
            and latest - self.last_spoken.get(sp["id"], latest) < UNKNOWN_VOICE_SILENT_SECONDS
            for sp in self.speakers.values()
        )

    def _voice_verdict(self, sims: dict, seconds: float) -> tuple[bool, str | None]:
        """Whether the voiceprints alone settle who speaks, and who (None: someone new). The same rule
        decides whether Gemini gets reference recordings and whether its answer is overruled."""
        if not sims:
            return False, None
        best_id, best = max(sims.items(), key=lambda kv: kv[1])
        if best >= SAME_VOICE:
            return True, best_id
        # a short clip of a known person often scores low: only a long, clearly different voice is someone new
        if best < SURELY_NEW_VOICE and seconds >= NEW_VOICE_MIN_SECONDS and not self._unknown_voices():
            return True, None
        return False, None

    async def _listen(self, seg_id: int, wav: bytes, received: float, origin, embedding):
        seconds = wav_seconds(wav)
        sims = self._similarities(embedding)
        decided, voice_id = self._voice_verdict(sims, seconds)
        # Reference recordings are most of the cost of a phrase: send them only when Gemini's answer counts.
        references = [] if decided else self._references(sims)
        heard = await asyncio.to_thread(
            self.ai.listen, wav, references, sims, self.recent_transcript(), self.notes,
            identified=(voice_id or "new") if decided else None,
        )
        if not heard.text or self._is_echo(heard.text) or self._heard_by_another_device(heard.text, origin):
            await self._reply(origin, {"type": "empty", "id": seg_id})
            return
        # resolved again: the speakers may have been merged while the speaker message was being sent
        speaker_id = self.resolve(await self._resolve_speaker(heard, embedding, wav, seconds))
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
        self.recent_lines.append((speaker_id, heard.text))
        self.last_spoken[speaker_id] = max(received, self.last_spoken.get(speaker_id, 0.0))
        await self._reply(origin, {"type": "heard", "id": seg_id, "speaker": speaker_id, "lang": heard.source_lang,
                                   "text": heard.text, "complete": heard.complete})
        pending_seconds = sum(s["end"] - s["start"] for s in self.pending)
        if heard.complete or pending_seconds >= UNIT_MAX_SECONDS:
            self.flush()
        else:
            self._arm_idle_flush()
            await self.broadcast_pending()

    def _heard_by_another_device(self, text: str, origin) -> bool:
        """Two phones recording the same seminar send every phrase twice: keep the first one only."""
        now = time.monotonic()
        while self.recent_heard and self.recent_heard[0][0] < now - DUPLICATE_SECONDS:
            self.recent_heard.popleft()
        if origin is None:
            return False
        phrase = normalized(text)
        duplicate = bool(phrase) and any(
            device is not origin and SequenceMatcher(None, phrase, other).ratio() >= DUPLICATE_SIMILARITY
            for _, device, other in self.recent_heard
        )
        if not duplicate:
            self.recent_heard.append((now, origin, phrase))
        return duplicate

    def _is_echo(self, text: str) -> bool:
        """A translation read out aloud by a device in the room and picked up by a microphone again:
        interpreting it would feed it back to the read-out, again and again."""
        cutoff = time.monotonic() - ECHO_SECONDS
        while self.read_out and self.read_out[0][0] < cutoff:
            self.read_out.popleft()
        phrase = normalized(text)
        return bool(phrase) and any(
            SequenceMatcher(None, phrase, part).ratio() >= ECHO_SIMILARITY for _, parts in self.read_out for part in parts
        )

    def _arm_idle_flush(self):
        if self.closed:
            return
        if self.idle_timer:
            self.idle_timer.cancel()
        self.idle_timer = asyncio.get_running_loop().call_later(IDLE_FLUSH_SECONDS, self._idle_flush)

    def _idle_flush(self):
        self.idle_timer = None
        if self.speaking or not self.listen_queue.empty() or self.listening:
            self._arm_idle_flush()  # someone is still talking, or more speech is on its way: it may complete the thought
        else:
            self.flush()

    def _references(self, sims: dict) -> list[dict]:
        """The likeliest speakers' recordings, so Gemini can compare voices: by voiceprint similarity (unknown
        ranks mid-way), then by who spoke last; whoever is in the recent transcript is always among them."""
        mid = (SAME_VOICE + NEW_VOICE) / 2
        ranked = sorted(
            (sp for sp in self.speakers.values() if sp["ref_wav"]),
            key=lambda sp: (-sims.get(sp["id"], mid), -self.last_spoken.get(sp["id"], 0.0)),
        )
        recent = {self.resolve(sid) for sid, _ in self.recent_lines}
        chosen = [sp["id"] for sp in ranked if sp["id"] in recent][:MAX_REFERENCE_SPEAKERS]
        chosen += [sp["id"] for sp in ranked if sp["id"] not in recent][: MAX_REFERENCE_SPEAKERS - len(chosen)]
        return [sp for sp in ranked if sp["id"] in chosen]

    async def _resolve_speaker(self, heard: Heard, embedding, wav: bytes, seconds: float) -> str:
        # compared again: the speakers may have been merged while Gemini was listening
        sims = self._similarities(embedding)
        best_id, best = max(sims.items(), key=lambda kv: kv[1]) if sims else (None, 0.0)
        named = self.resolve(heard.speaker)
        gemini_id = named if named in self.speakers else None
        decided, voice_id = self._voice_verdict(sims, seconds)
        if decided:
            speaker_id, confident = voice_id, True
        else:  # the voices alone can't tell: Gemini's judgement with the context decides
            speaker_id = gemini_id
            # trusted enough to learn from: the voiceprint agrees, or without one there was nobody else to pick
            confident = gemini_id is not None and (
                gemini_id == best_id and best >= NEW_VOICE if sims else len(self.speakers) == 1
            )
        # Gemini's name for the speaker counts only if it meant the same person
        name = heard.speaker_name if gemini_id == speaker_id else ""
        good_voiceprint = embedding is not None and seconds >= CENTROID_MIN_SECONDS
        changed = False
        if speaker_id is None:
            used = [*self.speakers, *self.merged]  # never reissue a merged-away id
            n = 1 + max([int(s[1:]) for s in used if s[1:].isdigit()] or [0])
            sp = {"id": f"S{n}", "label": name or f"Spiker {n}", "named": bool(name), "voice": heard.voice,
                  "ref_wav": trim_wav(wav, GOOD_REFERENCE_SECONDS[1]),
                  "ref_seconds": min(seconds, GOOD_REFERENCE_SECONDS[1]),
                  "centroid": embedding if good_voiceprint else None, "samples": 1 if good_voiceprint else 0}
            self.speakers[sp["id"]] = sp
            changed = True
        else:
            sp = self.speakers[speaker_id]
            if name and not sp["named"]:
                sp["label"], sp["named"] = name, True
                changed = True
            if good_voiceprint and (confident or sp["centroid"] is None):
                n = sp["samples"]
                sp["centroid"] = embedding if sp["centroid"] is None else (sp["centroid"] * n + embedding) / (n + 1)
                sp["samples"] = n + 1
            if confident:  # a wrong attribution must not give someone another person's voice
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
        speaker_id = self.resolve(segments[0]["speaker"])
        lang = Counter(s["lang"] for s in segments).most_common(1)[0][0]
        target = self.partner if lang == "uz" and self.partner != KEEP_UZBEK else "uz"
        source = " ".join(s["text"] for s in segments)
        rendered = Rendered(translation="", note="")  # kept with its original text if Gemini keeps failing
        for attempt in (1, 2):
            try:
                rendered = await asyncio.to_thread(
                    self.ai.interpret, source, lang, self.label(speaker_id), target, self.recent_interpretation(), self.notes
                )
                break
            except Exception:
                log.exception("Interpreting failed in session %s (attempt %d)", self.id, attempt)
                if attempt == 1:
                    await asyncio.sleep(RETRY_DELAY)
        speaker_id = self.resolve(speaker_id)  # the speakers may have been merged meanwhile
        unit = self.store.add_unit(
            self.id,
            {"start": segments[0]["start"], "end": segments[-1]["end"], "speaker": speaker_id, "lang": lang,
             "target": target, "source": source, "translation": rendered.translation, "note": rendered.note},
            [s["id"] for s in segments],
        )
        if rendered.translation:
            self.recent_units.append((speaker_id, rendered.translation))
            # a read-out may reach a microphone cut into sentences
            sentences = re.split(r"(?<=[.!?…])\s+", rendered.translation)
            parts = {normalized(p) for p in (rendered.translation, *sentences)} - {""}
            self.read_out.append((time.monotonic(), sorted(parts)))
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
            labels = [sp["label"] for sp in self.speakers.values()]
            notes = await asyncio.to_thread(self.ai.update_notes, self.notes, lines, labels)
            self.notes = notes.model_dump()
            # where the notes stand survives a reload, so a device reconnecting often doesn't keep resetting the count
            self.store.update_session(self.id, notes=self.notes, notes_unit=max((u["id"] for u in units), default=0))
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
        self.merged[source] = target
        for old, new in self.merged.items():
            if new == source:
                self.merged[old] = target
        if source in self.last_spoken:
            self.last_spoken[target] = max(self.last_spoken.pop(source), self.last_spoken.get(target, 0.0))
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


REPORT_SOURCE_LIMIT = 600_000  # characters of transcript; above it the original-language text is left out
REPORT_MAX_CHARS = 2_400_000  # about 600k tokens: more than one report can hold


def report_input(store: SeminarStore, session_id: int, start: float, end: float, tz: ZoneInfo) -> tuple[str, str, str, float, float]:
    """Transcript, speaker list and period for a report; raises ReportRangeError if it can't be made."""
    units = store.units(session_id, start, end)
    if not units:
        raise ReportRangeError("Bu oraliqda tarjima qilingan gap yo'q")
    # the report covers what was actually said, not the empty edges of the chosen range
    start, end = max(start, units[0]["start"]), min(end, max(u["end"] for u in units))
    speakers = {sp["id"]: sp["label"] for sp in store.speakers(session_id)}
    transcript = "\n".join(transcript_lines(units, speakers, tz))
    if len(transcript) > REPORT_SOURCE_LIMIT:
        transcript = "\n".join(transcript_lines(units, speakers, tz, with_source=False))
    if len(transcript) > REPORT_MAX_CHARS:
        raise ReportRangeError("Oraliq bitta hisobot uchun juda katta, qisqaroq vaqtni tanlang")
    period = f"{datetime.fromtimestamp(start, tz):%Y-%m-%d %H:%M} - {datetime.fromtimestamp(end, tz):%Y-%m-%d %H:%M}"
    roster = "\n".join(f"{sid}: {label}" for sid, label in speakers.items())
    return transcript, roster, period, start, end


async def build_report(store: SeminarStore, ai: SeminarAI, session_id: int, start: float, end: float, tz: ZoneInfo) -> int:
    """Writes the report right away (used by tests and scripts); the web app uses start_report_job."""
    transcript, roster, period, start, end = report_input(store, session_id, start, end, tz)
    report = await asyncio.to_thread(ai.report, transcript, roster, period)
    return store.save_report(session_id, start, end, report.model_dump())


def start_report_job(store: SeminarStore, ai: SeminarAI, session_id: int, start: float, end: float, tz: ZoneInfo) -> tuple[int, asyncio.Task]:
    """Reports on a long seminar take a minute or two, longer than proxies wait for a response:
    the report is stored as pending at once and written in the background."""
    transcript, roster, period, start, end = report_input(store, session_id, start, end, tz)
    report_id = store.start_report(session_id, start, end)

    async def run():
        try:
            report = await asyncio.to_thread(ai.report, transcript, roster, period)
            store.finish_report(report_id, report.model_dump())
        except Exception:
            log.exception("Report %s failed", report_id)
            store.fail_report(report_id, "Hisobot tayyorlab bo'lmadi, qayta urinib ko'ring")

    return report_id, asyncio.create_task(run())
