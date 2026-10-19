"""Per-request usage log (Gemini tokens, ElevenLabs credits) kept in SQLite."""
import logging
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from time import time

log = logging.getLogger("stt_bot")


@dataclass
class Totals:
    requests: int = 0
    input_tokens: int = 0
    audio_tokens: int = 0
    output_tokens: int = 0
    credits: int = 0
    cost: float = 0.0


class UsageLog:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._execute(
            "CREATE TABLE IF NOT EXISTS usage ("
            "ts REAL NOT NULL, service TEXT NOT NULL, "
            "input_tokens INTEGER NOT NULL DEFAULT 0, audio_tokens INTEGER NOT NULL DEFAULT 0, "
            "output_tokens INTEGER NOT NULL DEFAULT 0, credits INTEGER NOT NULL DEFAULT 0, "
            "cost REAL NOT NULL DEFAULT 0)"
        )

    def _execute(self, sql: str, params=()) -> list:
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute(sql, params).fetchall()

    def add(self, service: str, input_tokens=0, audio_tokens=0, output_tokens=0, credits=0, cost=0.0):
        """Never raises: losing a usage row must not break a translation."""
        try:
            self._execute(
                "INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time(), service, input_tokens, audio_tokens, output_tokens, credits, cost),
            )
        except Exception:
            log.exception("Usage write failed")

    def totals(self, service: str, since: float) -> Totals:
        (row,) = self._execute(
            "SELECT COUNT(*), TOTAL(input_tokens), TOTAL(audio_tokens), TOTAL(output_tokens), "
            "TOTAL(credits), TOTAL(cost) FROM usage WHERE service = ? AND ts >= ?",
            (service, since),
        )
        return Totals(*(int(v) for v in row[:5]), cost=row[5])
