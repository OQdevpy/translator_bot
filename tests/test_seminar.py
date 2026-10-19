"""Tests for seminar.py: storage, the live interpreting pipeline and reports.

Gemini and the voiceprint model are replaced by fakes; nothing touches the network."""
import asyncio
import math
import threading
import time
from contextlib import closing
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import seminar
from conftest import make_wav
from seminar import (
    Heard,
    LiveSession,
    Notes,
    Rendered,
    Report,
    SeminarAI,
    SeminarStore,
    Term,
    build_report,
    notes_text,
)

TZ = ZoneInfo("Asia/Tashkent")
T0 = 1_700_000_000.0  # capture times used as `received`


# ---------------------------------------------------------------- fakes and helpers


def heard(text="Salom", speaker="new", name="", voice="", lang="uz", complete=True) -> Heard:
    return Heard(speaker=speaker, speaker_name=name, voice=voice, source_lang=lang, text=text, complete=complete)


class FakeAI:
    """Stands in for SeminarAI. listen() pops scripted results (Heard, Exception or callable)."""

    def __init__(self, listen=(), interpret=None):
        self.listen_results = list(listen)
        self.listen_calls = []
        self.interpret_results = list(interpret or [])
        self.interpret_calls = []
        self.notes_calls = []
        self.notes_result = Notes(summary="Seminar haqida", glossary=[Term(term="AI", translation="SI", note="")])
        self.notes_gate: threading.Event | None = None
        self.report_calls = []
        self.report_result = make_report()

    def listen(self, wav, references, similarities, recent, notes, identified=None):
        self.listen_calls.append(
            {"wav": wav, "references": [r["id"] for r in references], "similarities": dict(similarities),
             "recent": list(recent), "notes": dict(notes), "identified": identified}
        )
        result = self.listen_results.pop(0)
        if callable(result) and not isinstance(result, Heard):
            result = result()
        if isinstance(result, Exception):
            raise result
        return result

    def interpret(self, text, source_lang, speaker, target, recent, notes):
        self.interpret_calls.append(
            {"text": text, "lang": source_lang, "speaker": speaker, "target": target,
             "recent": list(recent), "notes": dict(notes)}
        )
        result = self.interpret_results.pop(0) if self.interpret_results else None
        if callable(result):
            result = result()
        if isinstance(result, Exception):
            raise result
        return result or Rendered(translation=f"T:{text}", note="")

    def update_notes(self, notes, new_lines, speakers):
        self.notes_calls.append({"notes": dict(notes), "lines": list(new_lines), "speakers": list(speakers)})
        if self.notes_gate is not None:
            assert self.notes_gate.wait(5)
        if isinstance(self.notes_result, Exception):
            raise self.notes_result
        return self.notes_result

    def report(self, transcript, speakers, period):
        self.report_calls.append({"transcript": transcript, "speakers": speakers, "period": period})
        if isinstance(self.report_result, Exception):
            raise self.report_result
        return self.report_result


class FakeVoiceprints:
    """embed() returns the scripted vectors in order (an Exception item is raised)."""

    def __init__(self, embeddings=()):
        self.embeddings = list(embeddings)
        self.calls = 0

    def embed(self, wav):
        self.calls += 1
        item = self.embeddings.pop(0)
        if isinstance(item, Exception):
            raise item
        return None if item is None else np.asarray(item, dtype=np.float32)


class FakeWS:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)

    def types(self):
        return [m["type"] for m in self.messages]

    def of(self, kind):
        return [m for m in self.messages if m["type"] == kind]


class BrokenWS:
    def __init__(self):
        self.calls = 0

    async def send_json(self, message):
        self.calls += 1
        raise RuntimeError("socket closed")


def gated(result, gate: threading.Event, entered: threading.Event | None = None):
    """A scripted result that waits for `gate` (runs in the to_thread worker thread)."""

    def run():
        if entered is not None:
            entered.set()
        assert gate.wait(5), "test gate was never released"
        return result

    return run


E1 = np.array([1.0, 0.0, 0.0, 0.0])
E2 = np.array([0.0, 1.0, 0.0, 0.0])
E3 = np.array([0.0, 0.0, 1.0, 0.0])
E4 = np.array([0.0, 0.0, 0.0, 1.0])


def toward(base, other, cos):
    """A unit vector whose cosine with `base` is `cos` (base and other orthonormal)."""
    return cos * base + math.sqrt(1 - cos * cos) * other


def make_report(**kw) -> Report:
    fields = dict(title="Hisobot", overview="Umumiy", speakers=[], timeline=[], topics=[], qa=[],
                  decisions=[], action_items=[], glossary=[])
    fields.update(kw)
    return Report(**fields)


def seed_speaker(store, session_id, sid, label=None, centroid=None, samples=None, ref_wav=b"REF", ref_seconds=5.0,
                 named=False, voice=""):
    store.save_speaker(session_id, {
        "id": sid, "label": label or f"Spiker {sid[1:]}", "named": named, "voice": voice, "ref_wav": ref_wav,
        "ref_seconds": ref_seconds, "centroid": None if centroid is None else np.asarray(centroid, dtype=np.float32),
        "samples": samples if samples is not None else (0 if centroid is None else 1),
    })


async def wait_for(cond, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.002)


async def drain(sess: LiveSession):
    """Waits until every queued phrase is transcribed and every flushed thought interpreted."""
    await wait_for(lambda: sess.listen_queue.empty() and sess.unit_queue.empty() and sess.working == 0)
    if sess.notes_task is not None:
        await sess.notes_task
    for _ in range(5):  # let fire-and-forget broadcasts run
        await asyncio.sleep(0)


def start(store, session_id, ai, voiceprints=None) -> tuple[LiveSession, FakeWS]:
    sess = LiveSession(session_id, store, ai, voiceprints, TZ)
    ws = FakeWS()
    sess.clients.add(ws)
    return sess, ws


@pytest.fixture(autouse=True)
def fast_timing(monkeypatch):
    monkeypatch.setattr(seminar, "RETRY_DELAY", 0.01)
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 60.0)  # tests that need the idle flush shorten it


@pytest.fixture
def store(tmp_path):
    return SeminarStore(str(tmp_path / "seminars.db"))


@pytest.fixture
def sid(store):
    return store.create_session("Seminar")["id"]


# ---------------------------------------------------------------- notes_text


def test_notes_text_empty():
    assert notes_text({}) == "(none yet)"
    assert notes_text({"summary": "", "glossary": []}) == "(none yet)"


def test_notes_text_summary_and_glossary():
    notes = {
        "summary": "Seminar sun'iy intellekt haqida.",
        "glossary": [
            {"term": "AI", "translation": "sun'iy intellekt", "note": "texnologiya"},
            {"term": "GDP", "translation": "YaIM", "note": ""},
        ],
    }
    assert notes_text(notes) == (
        "Seminar sun'iy intellekt haqida.\nTerminology: AI = sun'iy intellekt (texnologiya); GDP = YaIM"
    )
    assert notes_text({"glossary": [{"term": "X", "translation": "Y"}]}) == "Terminology: X = Y"


# ---------------------------------------------------------------- SeminarAI prompts (fake Interpreter)


class FakeInterpreter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate(self, contents, system, schema, **kw):
        self.calls.append({"contents": contents, "system": system, "schema": schema, "kw": kw})
        return self.result


def test_seminar_ai_listen_builds_prompt_and_normalizes():
    interp = FakeInterpreter(Heard(speaker=" S1 ", speaker_name=" Aziz ", voice="deep", source_lang=" EN ",
                                   text="  Hello there  ", complete=True))
    ai = SeminarAI(interp)
    refs = [{"id": "S1", "label": "Aziz", "voice": "", "ref_wav": b"RIFF-ref"}]
    notes = {"summary": "Xulosa", "glossary": [{"term": "KPI", "translation": "asosiy ko'rsatkich", "note": ""}]}
    result = ai.listen(b"RIFF-new", refs, {"S1": 0.7012}, ["[S1 Aziz] earlier"], notes)
    assert (result.speaker, result.speaker_name, result.source_lang, result.text) == ("S1", "Aziz", "en", "Hello there")
    call = interp.calls[0]
    # a short timeout and answer cap: a stuck or looping call must not hold up the live queue
    assert call["schema"] is Heard and call["kw"] == {"timeout": seminar.LISTEN_TIMEOUT, "max_tokens": seminar.LISTEN_MAX_TOKENS}
    contents = call["contents"]
    assert contents[0] == "Reference recording of speaker S1 (Aziz; voice not described):"
    assert contents[1].inline_data.data == b"RIFF-ref"
    assert contents[2] == "NEW SEGMENT:" and contents[3].inline_data.data == b"RIFF-new"
    assert "   - S1: 0.70" in call["system"] and "0.62 or more, different people 0.30 or less" in call["system"]
    # the moderator handing the floor to "Professor Karimov" is not Professor Karimov
    assert "Never the name of the person they address or hand the floor to" in call["system"]
    # listening needs only the spellings; the summary is for interpreting (and costs tokens on every phrase)
    assert "KPI = asosiy ko'rsatkich" in call["system"] and "Xulosa" not in call["system"]
    assert "[S1 Aziz] earlier" in call["system"]

    ai.listen(b"RIFF-new", [], {}, [], {})
    system = interp.calls[1]["system"]
    assert "(not available)" in system and "(nothing yet)" in system and "(none yet)" in system
    assert interp.calls[1]["contents"][0] == "NEW SEGMENT:"


def test_seminar_ai_interpret_target_and_source_names():
    interp = FakeInterpreter(Rendered(translation="  Привет  ", note=" izoh "))
    ai = SeminarAI(interp)
    result = ai.interpret("Salom", "uz", "Aziz", "ru", [], {})
    assert (result.translation, result.note) == ("Привет", "izoh")
    assert "working into Russian" in interp.calls[0]["system"]
    assert interp.calls[0]["contents"] == "Aziz (Uzbek (Latin script)): Salom"
    assert interp.calls[0]["kw"] == {"timeout": seminar.INTERPRET_TIMEOUT, "max_tokens": seminar.INTERPRET_MAX_TOKENS}
    ai.interpret("Hallo", "de", "S2", "xx", ["[A] b"], {})  # unknown codes: raw source code, Uzbek target
    assert interp.calls[1]["contents"] == "S2 (de): Hallo"
    assert "working into Uzbek (Latin script)" in interp.calls[1]["system"]
    assert "[A] b" in interp.calls[1]["system"]


def test_seminar_ai_notes_and_report_calls():
    interp = FakeInterpreter(Notes(summary="s", glossary=[]))
    ai = SeminarAI(interp, report_model="gemini-pro-test")
    ai.update_notes({"summary": "Eski"}, ["[A] x => y"], ["A", "Spiker 2"])
    contents = interp.calls[0]["contents"]
    assert "Current notes:\nEski" in contents and "Speakers: A, Spiker 2" in contents and "[A] x => y" in contents
    assert "never use ids like S1" in interp.calls[0]["system"]  # the summary is shown on every device
    assert interp.calls[0]["kw"] == {"timeout": seminar.NOTES_TIMEOUT, "max_tokens": seminar.NOTES_MAX_TOKENS}
    interp.result = make_report()
    ai.report("TRANSCRIPT", "S1: A", "PERIOD")
    call = interp.calls[1]
    assert call["schema"] is Report
    # a day-long report needs minutes, far beyond the live calls' timeout
    assert call["kw"] == {"thinking": "medium", "model": "gemini-pro-test", "timeout": seminar.REPORT_TIMEOUT,
                          "max_tokens": seminar.REPORT_MAX_TOKENS}
    assert seminar.REPORT_TIMEOUT >= 300
    assert "Period: PERIOD" in call["contents"] and "TRANSCRIPT" in call["contents"]


# ---------------------------------------------------------------- SeminarStore


def test_store_creates_directory_and_session_roundtrip(tmp_path):
    store = SeminarStore(str(tmp_path / "nested" / "dir" / "s.db"))
    row = store.create_session("Kun 1")
    assert row["title"] == "Kun 1" and row["partner"] == "uz" and row["notes"] == {}
    assert store.create_session("Kun 2", partner="zh")["partner"] == "zh"
    notes = {"summary": "Ўзбек – xulosa", "glossary": [{"term": "块", "translation": "yuan", "note": ""}]}
    store.update_session(row["id"], notes=notes, partner="ru", title="Yangi")
    again = store.session(row["id"])
    assert again["notes"] == notes and again["partner"] == "ru" and again["title"] == "Yangi"
    assert store.session(999) is None


def test_store_sessions_list_counts_units_newest_first(store):
    a = store.create_session("A")["id"]
    b = store.create_session("B")["id"]
    store.update_session(a, created_at=100.0)
    store.update_session(b, created_at=200.0)
    for s, e in ((10.0, 20.0), (30.0, 45.0)):
        store.add_unit(a, {"start": s, "end": e, "speaker": "S1", "lang": "uz", "target": "uz", "source": "x",
                           "translation": "y", "note": ""}, [])
    rows = store.sessions()
    assert [r["id"] for r in rows] == [b, a]
    assert (rows[0]["units"], rows[0]["first"], rows[0]["last"]) == (0, None, None)
    assert (rows[1]["units"], rows[1]["first"], rows[1]["last"]) == (2, 10.0, 45.0)


def test_store_speakers_roundtrip(store, sid):
    seed_speaker(store, sid, "S1", label="Aziz", centroid=[0.5, 0.25, 0.0, 1.0], samples=3, named=True, voice="deep")
    seed_speaker(store, sid, "S2", ref_wav=None, ref_seconds=0.0)
    s1, s2 = store.speakers(sid)
    assert s1["id"] == "S1" and s1["label"] == "Aziz" and s1["named"] == 1 and s1["voice"] == "deep"
    assert s1["centroid"].dtype == np.float32 and s1["centroid"].tolist() == [0.5, 0.25, 0.0, 1.0]
    assert s1["samples"] == 3 and s1["ref_wav"] == b"REF" and s1["ref_seconds"] == 5.0
    assert s2["centroid"] is None and s2["ref_wav"] is None and s2["named"] == 0
    # saving again replaces the row instead of duplicating it
    seed_speaker(store, sid, "S2", label="Bobur")
    assert [sp["label"] for sp in store.speakers(sid) if sp["id"] == "S2"] == ["Bobur"]
    assert len(store.speakers(sid)) == 2


def test_store_speakers_keep_creation_order_after_update(store, sid):
    seed_speaker(store, sid, "S1")
    seed_speaker(store, sid, "S2")
    seed_speaker(store, sid, "S1", label="Aziz")  # LiveSession re-saves a speaker on every phrase
    assert [sp["id"] for sp in store.speakers(sid)] == ["S1", "S2"]


def test_store_segments_pending_until_linked_to_unit(store, sid):
    ids = [
        store.add_segment(sid, {"start": 1.0, "end": 2.0, "speaker": "S1", "lang": "uz", "text": "a",
                                "complete": False, "embedding": np.ones(4)}),
        store.add_segment(sid, {"start": 2.0, "end": 3.0, "speaker": "S1", "lang": "uz", "text": "b",
                                "complete": True}),
        store.add_segment(sid, {"start": 3.0, "end": 4.0, "speaker": "S2", "lang": "en", "text": "c",
                                "complete": False}),
    ]
    pending = store.pending_segments(sid)
    assert [p["id"] for p in pending] == ids
    assert pending[0] == {"id": ids[0], "start": 1.0, "end": 2.0, "speaker": "S1", "lang": "uz", "text": "a",
                          "complete": 0}
    unit = store.add_unit(sid, {"start": 1.0, "end": 3.0, "speaker": "S1", "lang": "uz", "target": "uz",
                                "source": "a b", "translation": "A B", "note": ""}, ids[:2])
    assert unit["id"] and unit["source"] == "a b"
    assert [p["id"] for p in store.pending_segments(sid)] == [ids[2]]
    assert store.units(sid) == [{"id": unit["id"], "start": 1.0, "end": 3.0, "speaker": "S1", "lang": "uz",
                                 "target": "uz", "source": "a b", "translation": "A B", "note": ""}]


def _units(store, sid, spans):
    out = []
    for i, (s, e) in enumerate(spans):
        out.append(store.add_unit(sid, {"start": s, "end": e, "speaker": "S1", "lang": "uz", "target": "uz",
                                        "source": f"u{i}", "translation": f"t{i}", "note": ""}, [])["id"])
    return out


def test_store_units_time_range_overlap(store, sid):
    # inserted out of order on purpose: results are ordered by start time
    ids = _units(store, sid, [(20.0, 30.0), (0.0, 10.0), (10.0, 20.0), (30.0, 40.0)])
    by_start = [ids[1], ids[2], ids[0], ids[3]]
    assert [u["id"] for u in store.units(sid)] == by_start
    # every unit that overlaps the range, edges inclusive
    assert [u["id"] for u in store.units(sid, 15.0, 25.0)] == [ids[2], ids[0]]
    assert [u["id"] for u in store.units(sid, 20.0, 20.0)] == [ids[2], ids[0]]
    assert [u["id"] for u in store.units(sid, start=35.0)] == [ids[3]]
    assert [u["id"] for u in store.units(sid, end=5.0)] == [ids[1]]
    assert store.units(sid, 41.0, 50.0) == []
    other = store.create_session("other")["id"]
    assert store.units(other) == []


def test_store_units_limit_returns_latest_oldest_first(store, sid):
    ids = _units(store, sid, [(0.0, 1.0), (5.0, 6.0), (5.0, 7.0), (9.0, 10.0)])
    assert [u["id"] for u in store.units(sid, limit=2)] == [ids[2], ids[3]]
    assert [u["id"] for u in store.units(sid, limit=3)] == ids[1:]  # equal starts ordered by id
    assert [u["id"] for u in store.units(sid, limit=10)] == ids
    assert [u["id"] for u in store.units(sid, limit=0)] == ids  # 0 means no limit
    assert [u["id"] for u in store.units(sid, 0.0, 6.0, limit=1)] == [ids[2]]


def test_store_merge_speakers_only_in_that_session(store, sid):
    other = store.create_session("other")["id"]
    for s in (sid, other):
        seed_speaker(store, s, "S1")
        seed_speaker(store, s, "S2")
        store.add_segment(s, {"start": 0, "end": 1, "speaker": "S2", "lang": "uz", "text": "x", "complete": False})
        store.add_unit(s, {"start": 0, "end": 1, "speaker": "S2", "lang": "uz", "target": "uz", "source": "x",
                           "translation": "y", "note": ""}, [])
    store.merge_speakers(sid, "S2", "S1")
    assert [sp["id"] for sp in store.speakers(sid)] == ["S1"]
    assert [p["speaker"] for p in store.pending_segments(sid)] == ["S1"]
    assert [u["speaker"] for u in store.units(sid)] == ["S1"]
    assert [sp["id"] for sp in store.speakers(other)] == ["S1", "S2"]
    assert [u["speaker"] for u in store.units(other)] == ["S2"]


def test_store_delete_session_removes_everything(store, sid):
    other = store.create_session("other")["id"]
    for s in (sid, other):
        seed_speaker(store, s, "S1")
        store.add_segment(s, {"start": 0, "end": 1, "speaker": "S1", "lang": "uz", "text": "x", "complete": False})
        _units(store, s, [(0.0, 1.0)])
        store.save_report(s, 0.0, 1.0, {"title": "r"})
    store.delete_session(sid)
    assert store.session(sid) is None
    assert (store.speakers(sid), store.pending_segments(sid), store.units(sid), store.reports(sid)) == ([], [], [], [])
    assert len(store.units(other)) == 1 and len(store.reports(other)) == 1 and store.session(other)


def test_store_reports(store, sid):
    first = store.save_report(sid, 10.0, 20.0, {"title": "Birinchi", "overview": "ü"})
    second = store.save_report(sid, 30.0, 40.0, {"title": "Ikkinchi"})
    with closing(store._connect()) as db, db:  # make created_at deterministic
        db.execute("UPDATE reports SET created_at = 1 WHERE id = ?", (first,))
        db.execute("UPDATE reports SET created_at = 2 WHERE id = ?", (second,))
    listed = store.reports(sid)
    assert [r["id"] for r in listed] == [second, first]
    assert set(listed[0]) == {"id", "created_at", "start", "end", "status", "error"}
    assert listed[0]["status"] == "done"
    row = store.report(first)
    assert row["session_id"] == sid and (row["start"], row["end"]) == (10.0, 20.0)
    assert row["report"] == {"title": "Birinchi", "overview": "ü"}
    assert store.report(12345) is None
    assert store.report(2**70) is None and store.session(2**70) is None  # beyond SQLite's integers


def test_store_report_jobs(store, sid):
    rid = store.start_report(sid, 10.0, 20.0)
    assert (store.report(rid)["status"], store.report(rid)["report"]) == ("pending", {})
    store.finish_report(rid, {"title": "Tayyor"})
    assert (store.report(rid)["status"], store.report(rid)["report"]) == ("done", {"title": "Tayyor"})
    failed = store.start_report(sid, 10.0, 20.0)
    store.fail_report(failed, "xato")
    assert (store.report(failed)["status"], store.report(failed)["error"]) == ("failed", "xato")
    stuck = store.start_report(sid, 10.0, 20.0)
    store.fail_unfinished_reports("restart")
    assert [store.report(r)["status"] for r in (rid, failed, stuck)] == ["done", "failed", "failed"]
    assert store.report(stuck)["error"] == "restart"


# ---------------------------------------------------------------- LiveSession: speaker resolution


def test_strong_voice_match_overrides_gemini(store, sid):
    seed_speaker(store, sid, "S1", centroid=E1, samples=2)
    seed_speaker(store, sid, "S2", centroid=E2, samples=1)
    emb = toward(E1, E3, 0.8)
    ai = FakeAI([heard("Davom etamiz", speaker="S2")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([emb]))
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    sims = ai.listen_calls[0]["similarities"]
    assert sims["S1"] == pytest.approx(0.8, abs=1e-6) and sims["S2"] == pytest.approx(0.0, abs=1e-6)
    assert ws.of("heard")[0]["speaker"] == "S1"
    s1 = sess.speakers["S1"]
    assert s1["samples"] == 3
    np.testing.assert_allclose(s1["centroid"], (E1 * 2 + emb) / 3, rtol=1e-6)
    saved = {sp["id"]: sp for sp in store.speakers(sid)}
    np.testing.assert_allclose(saved["S1"]["centroid"], (E1 * 2 + emb) / 3, rtol=1e-6)
    assert saved["S2"]["samples"] == 1 and saved["S2"]["centroid"].tolist() == E2.tolist()
    assert ws.of("speaker") == []  # nothing about the speaker list changed


def test_clearly_new_voice_creates_speaker_despite_gemini(store, sid):
    seed_speaker(store, sid, "S1", label="Aziz", centroid=E1, samples=4, named=True)
    emb = toward(E1, E2, 0.1)  # a long clip, very unlike anyone known
    wav = make_wav(3.0)
    ai = FakeAI([heard("Yangi odam", speaker="S1", voice="ayol ovozi")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([emb]))
        await sess.add_audio(1, wav, T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert list(sess.speakers) == ["S1", "S2"]
    s2 = sess.speakers["S2"]
    assert (s2["label"], s2["named"], s2["voice"], s2["samples"]) == ("Spiker 2", False, "ayol ovozi", 1)
    assert s2["ref_wav"] == wav and s2["ref_seconds"] == pytest.approx(3.0)
    np.testing.assert_allclose(s2["centroid"], emb, rtol=1e-6)
    assert sess.speakers["S1"]["samples"] == 4  # the rejected candidate is untouched
    assert ws.of("speaker") == [{"type": "speaker", "speaker": {"id": "S2", "label": "Spiker 2", "voice": "ayol ovozi"}}]
    assert ws.types().index("speaker") < ws.types().index("heard")
    assert store.units(sid)[0]["speaker"] == "S2"
    saved = {sp["id"]: sp for sp in store.speakers(sid)}
    assert saved["S2"]["ref_wav"] == wav and saved["S2"]["samples"] == 1


def test_ambiguous_voice_defers_to_gemini_and_updates_centroid_only_when_confident(store, sid):
    seed_speaker(store, sid, "S1", centroid=E1, samples=1)
    seed_speaker(store, sid, "S2", centroid=E2, samples=1)
    disagree = toward(E1, E3, 0.5)  # S1 is the best match, but only weakly
    agree = toward(E1, E4, 0.55)
    ai = FakeAI([heard("birinchi", speaker="S2"), heard("ikkinchi", speaker="S1")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([disagree, agree]))
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        after_first = {k: (sp["centroid"].copy(), sp["samples"]) for k, sp in sess.speakers.items()}
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws, after_first

    sess, ws, after_first = asyncio.run(main())
    assert ai.listen_calls[0]["similarities"]["S1"] == pytest.approx(0.5, abs=1e-6)
    assert [h["speaker"] for h in ws.of("heard")] == ["S2", "S1"]
    # Gemini picked someone other than the best voice match: accepted, but nobody's voiceprint learns
    np.testing.assert_allclose(after_first["S1"][0], E1)
    np.testing.assert_allclose(after_first["S2"][0], E2)
    assert after_first["S1"][1] == 1 and after_first["S2"][1] == 1
    # Gemini agreed with the best voice match: confident, the S1 voiceprint learns
    np.testing.assert_allclose(sess.speakers["S1"]["centroid"], (E1 + agree) / 2, rtol=1e-6)
    assert sess.speakers["S1"]["samples"] == 2 and sess.speakers["S2"]["samples"] == 1
    assert [u["speaker"] for u in store.units(sid)] == ["S2", "S1"]


def test_ambiguous_voice_and_gemini_says_new_or_unknown_id_creates_speakers(store, sid):
    seed_speaker(store, sid, "S1", centroid=E1, samples=1)
    ai = FakeAI([heard("a", speaker="new"), heard("b", speaker="S9")])  # S9 does not exist
    embs = [toward(E1, E2, 0.5), toward(E1, E3, 0.5)]

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints(embs))
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert list(sess.speakers) == ["S1", "S2", "S3"]
    assert [h["speaker"] for h in ws.of("heard")] == ["S2", "S3"]
    assert sess.speakers["S1"]["samples"] == 1
    # the second phrase was compared against the new S2 as well
    assert set(ai.listen_calls[1]["similarities"]) == {"S1", "S2"}


def test_gemini_names_speaker_without_voiceprint_when_voice_is_new(store, sid):
    """S2 was created before voiceprints existed for it: a low similarity to everyone else
    must not make a new speaker when Gemini recognises S2."""
    seed_speaker(store, sid, "S1", centroid=E1, samples=3)
    seed_speaker(store, sid, "S2", centroid=None, samples=0)
    emb = toward(E1, E2, 0.1)
    ai = FakeAI([heard("men S2", speaker="S2")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([emb]))
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert list(sess.speakers) == ["S1", "S2"]
    assert ws.of("heard")[0]["speaker"] == "S2"
    assert ai.listen_calls[0]["similarities"].keys() == {"S1"}  # no centroid, no similarity
    s2 = sess.speakers["S2"]
    assert s2["samples"] == 1  # the first voiceprint is adopted even though not "confident"
    np.testing.assert_allclose(s2["centroid"], emb, rtol=1e-6)
    np.testing.assert_allclose(store.speakers(sid)[-1]["centroid"], emb, rtol=1e-6)


def test_without_voiceprints_gemini_decides(store, sid):
    ai = FakeAI([heard("a", speaker="new", voice="past ovoz"), heard("b", speaker="S1"),
                 heard("c", speaker="new"), heard("d", speaker="S2")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i in range(4):
            await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert [h["speaker"] for h in ws.of("heard")] == ["S1", "S1", "S2", "S2"]
    assert all(c["similarities"] == {} for c in ai.listen_calls)
    assert all(sp["centroid"] is None and sp["samples"] == 0 for sp in sess.speakers.values())
    assert sess.speakers["S1"]["voice"] == "past ovoz"
    assert [u["speaker"] for u in store.units(sid)] == ["S1", "S1", "S2", "S2"]


@pytest.mark.parametrize("failure", [RuntimeError("onnx crashed"), None], ids=["embed-raises", "clip-too-short"])
def test_voiceprint_failure_falls_back_to_gemini(store, sid, failure):
    seed_speaker(store, sid, "S1", centroid=E1, samples=2)
    ai = FakeAI([heard("salom", speaker="S1")])
    vp = FakeVoiceprints([failure])

    async def main():
        sess, ws = start(store, sid, ai, vp)
        await sess.add_audio(1, make_wav(0.5), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert vp.calls == 1 and len(ai.listen_calls) == 1  # no retry of the whole phrase
    assert ai.listen_calls[0]["similarities"] == {}
    assert ws.of("heard")[0]["speaker"] == "S1"
    assert sess.speakers["S1"]["samples"] == 2  # nothing to learn from
    assert len(store.units(sid)) == 1


def test_speaker_name_sets_label_once(store, sid):
    ai = FakeAI([
        heard("a", speaker="new", name="Dr. Karimov"),  # introduces himself in the first phrase
        heard("b", speaker="new"),
        heard("c", speaker="S2", name="Nodira"),  # name becomes clear later
        heard("d", speaker="S2", name="Nodira opa"),  # never renamed again by Gemini
        heard("e", speaker="S1", name="Boshqa"),
    ])

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i in range(5):  # one at a time: the label at interpretation time is then deterministic
            await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
            await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert (sess.speakers["S1"]["label"], sess.speakers["S1"]["named"]) == ("Dr. Karimov", True)
    assert (sess.speakers["S2"]["label"], sess.speakers["S2"]["named"]) == ("Nodira", True)
    assert [m["speaker"]["label"] for m in ws.of("speaker")] == ["Dr. Karimov", "Spiker 2", "Nodira"]
    assert {sp["id"]: sp["label"] for sp in store.speakers(sid)} == {"S1": "Dr. Karimov", "S2": "Nodira"}
    # the interpreter is told the speaker's label
    assert [c["speaker"] for c in ai.interpret_calls] == ["Dr. Karimov", "Spiker 2", "Nodira", "Nodira", "Dr. Karimov"]


def test_user_rename_is_never_overridden_by_gemini(store, sid):
    seed_speaker(store, sid, "S1")
    ai = FakeAI([heard("a", speaker="S1", name="Gemini nomi")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.rename_speaker("S1", "Rais")
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert ws.of("speaker") == [{"type": "speaker", "speaker": {"id": "S1", "label": "Rais", "voice": ""}}]
    assert sess.speakers["S1"]["label"] == "Rais"
    saved = store.speakers(sid)[0]
    assert (saved["label"], saved["named"]) == ("Rais", 1)


def test_reference_clip_replaced_only_by_better_clip(store, sid):
    clips = [make_wav(s) for s in (2.0, 12.0, 1.5, 6.0, 8.0)]
    ai = FakeAI([heard(f"p{i}", speaker="new" if i == 0 else "S1", complete=True) for i in range(5)])
    refs = []

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i, wav in enumerate(clips):
            await sess.add_audio(i, wav, T0 + 20 * i, origin=ws)
            await drain(sess)
            refs.append((sess.speakers["S1"]["ref_wav"], sess.speakers["S1"]["ref_seconds"]))
        sess.close()

    asyncio.run(main())
    assert refs[0] == (clips[0], pytest.approx(2.0))
    assert refs[1] == (clips[0], pytest.approx(2.0))  # 12 s is longer than a good reference
    assert refs[2] == (clips[0], pytest.approx(2.0))  # shorter than what we have
    assert refs[3] == (clips[3], pytest.approx(6.0))  # within 4..10 s: better
    assert refs[4] == (clips[3], pytest.approx(6.0))  # already good: kept
    saved = store.speakers(sid)[0]
    assert saved["ref_wav"] == clips[3] and saved["ref_seconds"] == pytest.approx(6.0)


def test_listen_gets_ranked_references_recent_lines_and_notes(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "MAX_REFERENCE_SPEAKERS", 2)
    store.update_session(sid, notes={"summary": "Iqtisod", "glossary": []})
    seed_speaker(store, sid, "S1", centroid=E1, ref_wav=b"R1")
    seed_speaker(store, sid, "S2", centroid=E2, ref_wav=b"R2")
    seed_speaker(store, sid, "S3", centroid=E3, ref_wav=b"R3")
    seed_speaker(store, sid, "S4", centroid=None, ref_wav=None)  # nothing to send
    emb = 0.5 * E1 + 0.55 * E2 + 0.52 * E3  # S2 > S3 > S1, all in the ambiguous zone
    ai = FakeAI([heard("birinchi", speaker="S2"), heard("ikkinchi", speaker="S2")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([emb, emb]))
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    first, second = ai.listen_calls
    assert first["references"] == ["S2", "S3"]
    assert first["recent"] == [] and first["notes"] == {"summary": "Iqtisod", "glossary": []}
    assert second["recent"] == ["[S2 Spiker 2] birinchi"]


@pytest.mark.parametrize(
    "emb, seconds, unknown_voice, sends",
    [
        (E1, 2.0, None, False),  # surely S1: no need for Gemini to compare voices
        (E1, 2.0, "recent", False),  # surely S1 even with a speaker of unknown voice around
        (E3, 3.0, None, False),  # a long clip unlike anyone: surely someone new
        (E3, 2.0, None, True),  # a short clip scores low even for a known person: Gemini compares
        (toward(E1, E3, 0.25), 3.0, None, True),  # unlike everyone, but not clearly enough
        (E3, 3.0, "recent", True),  # clearly not S1/S2, but maybe the speaker whose voice is unknown
        (E3, 3.0, "tiny-ref", False),  # ...whose only clip is too short to compare with
        (E3, 3.0, "silent", False),  # ...who hasn't spoken for over half an hour
        (0.6 * E1 + 0.8 * E3, 2.0, None, True),  # ambiguous
        (None, 2.0, None, True),  # no voiceprint for this phrase (too short)
    ],
    ids=["same", "same-with-unknown", "new", "short-clip", "unlike-but-not-clearly", "new-or-unknown",
         "unknown-with-tiny-ref", "unknown-long-silent", "ambiguous", "no-voiceprint"],
)
def test_reference_recordings_only_when_voiceprints_cannot_decide(store, sid, emb, seconds, unknown_voice, sends):
    seed_speaker(store, sid, "S1", centroid=E1, ref_wav=b"R1")
    seed_speaker(store, sid, "S2", centroid=E2, ref_wav=b"R2")
    store.add_segment(sid, {"start": T0 - 2, "end": T0, "speaker": "S1", "lang": "uz", "text": "x", "complete": True})
    if unknown_voice:
        seed_speaker(store, sid, "S3", centroid=None, ref_wav=b"R3", ref_seconds=0.6 if unknown_voice == "tiny-ref" else 5.0)
        spoke = T0 - 31 * 60 if unknown_voice == "silent" else T0 - 60
        store.add_segment(sid, {"start": spoke - 2, "end": spoke, "speaker": "S3", "lang": "uz", "text": "y",
                                "complete": True})
    ai = FakeAI([heard("gap", speaker="S1")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([emb]))
        await sess.add_audio(1, make_wav(seconds), T0 + 5, origin=ws)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert bool(ai.listen_calls[0]["references"]) == sends


# ---------------------------------------------------------------- LiveSession: thoughts and units


def test_complete_phrase_becomes_a_unit(store, sid):
    ai = FakeAI([heard("Assalomu alaykum.", speaker="new")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(5, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert ws.of("heard") == [{"type": "heard", "id": 5, "speaker": "S1", "lang": "uz",
                               "text": "Assalomu alaykum.", "complete": True}]
    units = store.units(sid)
    assert len(units) == 1
    u = units[0]
    assert (u["start"], u["end"]) == (pytest.approx(T0 - 2.0), T0)
    assert (u["speaker"], u["lang"], u["target"], u["source"]) == ("S1", "uz", "uz", "Assalomu alaykum.")
    assert u["translation"] == "T:Assalomu alaykum."
    assert ws.of("unit") == [{"type": "unit", "unit": u}]
    assert ws.types().index("heard") < ws.types().index("unit")
    assert ws.of("pending")[-1] == {"type": "pending", "pending": [], "interpreting": 0}
    assert sess.pending == [] and store.pending_segments(sid) == [] and not sess.busy()
    assert sess.recent_interpretation() == ["[Spiker 1] T:Assalomu alaykum."]


def test_incomplete_then_complete_phrase_join_into_one_unit(store, sid):
    ai = FakeAI([heard("Bugun biz", speaker="new", complete=False),
                 heard("iqtisodiyot haqida gaplashamiz.", speaker="S1", complete=True)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
        assert sess.idle_timer is not None and store.units(sid) == []
        assert ws.of("pending")[-1] == {"type": "pending", "interpreting": 0,
                                        "pending": [{"speaker": "S1", "lang": "uz", "text": "Bugun biz"}]}
        await sess.add_audio(2, make_wav(3.0), T0 + 4, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert len(ai.interpret_calls) == 1
    assert ai.interpret_calls[0]["text"] == "Bugun biz iqtisodiyot haqida gaplashamiz."
    (u,) = store.units(sid)
    assert u["source"] == "Bugun biz iqtisodiyot haqida gaplashamiz."
    assert (u["start"], u["end"]) == (pytest.approx(T0 - 2.0), pytest.approx(T0 + 4))
    assert store.pending_segments(sid) == [] and sess.idle_timer is None


def test_speaker_change_flushes_previous_thought(store, sid):
    ai = FakeAI([heard("Savolim bor", speaker="new", complete=False),
                 heard("Marhamat, so'rang", speaker="new", complete=False)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        units = store.units(sid)
        pending = [p["text"] for p in sess.pending]
        sess.close()
        return units, pending

    units, pending = asyncio.run(main())
    assert [(u["speaker"], u["source"]) for u in units] == [("S1", "Savolim bor")]
    assert pending == ["Marhamat, so'rang"]  # S2's thought is still in progress


def test_idle_pause_flushes_thought(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.05)
    ai = FakeAI([heard("Xo'sh", speaker="new", complete=False)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await wait_for(lambda: len(store.units(sid)) == 1)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    assert store.units(sid)[0]["source"] == "Xo'sh"
    assert sess.pending == [] and sess.idle_timer is None and not sess.busy()


def test_idle_flush_waits_while_more_speech_is_queued(store, sid):
    gate, entered = threading.Event(), threading.Event()
    ai = FakeAI([heard("Birinchidan", speaker="new", complete=False),
                 gated(heard("ikkinchidan", speaker="S1", complete=False), gate, entered),
                 heard("uchinchidan.", speaker="S1", complete=True)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        try:
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
            await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
            await asyncio.to_thread(entered.wait, 5)
            await sess.add_audio(3, make_wav(2.0), T0 + 6, origin=ws)  # waits in the queue
            sess._idle_flush()
            assert len(sess.pending) == 1 and sess.unit_queue.empty()
            assert sess.idle_timer is not None  # re-armed
        finally:
            gate.set()
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert [u["source"] for u in store.units(sid)] == ["Birinchidan ikkinchidan uchinchidan."]


def test_idle_flush_waits_for_phrase_being_transcribed(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.1)
    gate, entered = threading.Event(), threading.Event()
    ai = FakeAI([heard("Birinchidan", speaker="new", complete=False),
                 gated(heard("ikkinchidan.", speaker="S1", complete=True), gate, entered)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        try:
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
            # the continuation arrives well inside the idle window, transcription is slow
            await sess.add_audio(2, make_wav(2.0), T0 + 1, origin=ws)
            await asyncio.to_thread(entered.wait, 5)
            await asyncio.sleep(0.3)
        finally:
            gate.set()
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert [u["source"] for u in store.units(sid)] == ["Birinchidan ikkinchidan."]


def test_unit_max_seconds_flushes_unfinished_thought(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "UNIT_MAX_SECONDS", 5.0)
    ai = FakeAI([heard(w, speaker="new" if i == 0 else "S1", complete=False)
                 for i, w in enumerate(["bir", "ikki", "uch"])])

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i in range(3):
            await sess.add_audio(i, make_wav(3.0), T0 + 3 * i, origin=ws)
        await drain(sess)
        pending = [p["text"] for p in sess.pending]
        sess.close()
        return pending

    pending = asyncio.run(main())
    assert [u["source"] for u in store.units(sid)] == ["bir ikki"]  # 6 s >= 5 s
    assert pending == ["uch"]


def test_explicit_flush(store, sid):
    gate = threading.Event()
    ai = FakeAI([heard("to'xtatildi", speaker="new", complete=False)],
                interpret=[gated(Rendered(translation="To'xtatildi.", note=""), gate)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        try:
            sess.flush()  # nothing pending: no-op
            await asyncio.sleep(0)
            assert sess.unit_queue.empty() and ws.messages == []
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
            assert sess.idle_timer is not None
            sess.flush()
            assert sess.idle_timer is None and sess.pending == [] and sess.unit_queue.qsize() == 1
            await wait_for(lambda: ws.messages[-1] == {"type": "pending", "pending": [], "interpreting": 1})
        finally:
            gate.set()
        await drain(sess)
        sess.close()
        return ws

    ws = asyncio.run(main())
    assert [(u["source"], u["translation"]) for u in store.units(sid)] == [("to'xtatildi", "To'xtatildi.")]
    assert {"type": "pending", "pending": [], "interpreting": 1} in ws.messages


def test_empty_transcription_is_ignored(store, sid):
    ai = FakeAI([heard("Gap", speaker="new", complete=False), heard("", speaker="new"),
                 heard("davomi.", speaker="S1", complete=True)])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([None, None, None]))
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await sess.add_audio(2, make_wav(1.0), T0 + 2, origin=ws)  # applause
        await wait_for(lambda: len(ai.listen_calls) == 2 and sess.working == 0)
        assert len(sess.pending) == 1 and len(sess.speakers) == 1 and len(store.pending_segments(sid)) == 1
        await sess.add_audio(3, make_wav(2.0), T0 + 5, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert {"type": "empty", "id": 2} in ws.messages
    assert len(ws.of("heard")) == 2 and list(sess.speakers) == ["S1"]
    assert [u["source"] for u in store.units(sid)] == ["Gap davomi."]


# ---------------------------------------------------------------- target language


@pytest.mark.parametrize(
    "partner, langs, expected",
    [
        ("uz", ["uz"], "uz"),  # keep Uzbek speech in Uzbek (cleaned up)
        ("ru", ["uz"], "ru"),  # Uzbek speech for a Russian-speaking partner
        ("ru", ["en"], "uz"),  # foreign speech always into Uzbek
        ("zh", ["zh"], "uz"),
        ("ru", ["uz", "en", "en"], "uz"),  # the thought's main language decides
        ("ru", ["uz", "uz", "en"], "ru"),
    ],
)
def test_target_language(store, partner, langs, expected):
    sid = store.create_session("S", partner=partner)["id"]
    ai = FakeAI([heard(f"p{i}", speaker="new" if i == 0 else "S1", lang=lang, complete=i == len(langs) - 1)
                 for i, lang in enumerate(langs)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i in range(len(langs)):
            await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    (call,) = ai.interpret_calls
    (u,) = store.units(sid)
    majority = max(set(langs), key=langs.count)
    assert (call["target"], call["lang"]) == (expected, majority)
    assert (u["target"], u["lang"]) == (expected, majority)


def test_set_partner_applies_to_next_thought_and_persists(store, sid):
    ai = FakeAI([heard("bir", speaker="new"), heard("ikki", speaker="S1")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.set_partner("zh")
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert [u["target"] for u in store.units(sid)] == ["uz", "zh"]
    assert store.session(sid)["partner"] == "zh"


# ---------------------------------------------------------------- failures and retries


@pytest.fixture
def fake_clock(monkeypatch):
    """seminar's asyncio.sleep and time.monotonic on a fake clock: sleeping advances it at once.
    Returns the list of sleeps."""
    clock, sleeps = [1000.0], []
    real_sleep = asyncio.sleep

    async def sleep(delay, *args):
        sleeps.append(delay)
        clock[0] += delay
        await real_sleep(0)

    fake_asyncio = SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("__")})
    fake_asyncio.sleep = sleep
    monkeypatch.setattr(seminar, "asyncio", fake_asyncio)
    monkeypatch.setattr(seminar, "time", SimpleNamespace(time=time.time, monotonic=lambda: clock[0]))
    return sleeps


def test_listen_retries_with_growing_delays_until_it_works(store, sid, fake_clock, monkeypatch):
    monkeypatch.setattr(seminar, "RETRY_DELAY", 1.5)
    ai = FakeAI([RuntimeError("503"), RuntimeError("429"), RuntimeError("503"), heard("salom", speaker="new")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return ws

    ws = asyncio.run(main())
    assert len(ai.listen_calls) == 4 and fake_clock == [1.5, 3.0, 6.0] and ws.of("error") == []
    assert ws.of("heard")[0]["id"] == 1
    assert [u["source"] for u in store.units(sid)] == ["salom"]


def test_listen_gives_up_after_about_a_minute_and_keeps_going(store, sid, fake_clock, monkeypatch):
    monkeypatch.setattr(seminar, "RETRY_DELAY", 1.5)
    ai = FakeAI([RuntimeError("503")] * 6 + [heard("keyingi", speaker="new")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(7, make_wav(2.0), T0, origin=ws)
        await sess.add_audio(8, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()
        return ws

    ws = asyncio.run(main())
    # 1.5 + 3 + 6 + 12 + 24 = 46.5 s of waiting; another 48 s would pass the 60 s budget
    assert fake_clock == [1.5, 3.0, 6.0, 12.0, 24.0]
    assert ws.of("error") == [{"type": "error", "id": 7, "message": "Eshitib bo'lmadi"}]
    assert [h["id"] for h in ws.of("heard")] == [8]
    assert [u["source"] for u in store.units(sid)] == ["keyingi"]


def test_listen_stops_retrying_when_the_queue_behind_is_full(store, sid, fake_clock, monkeypatch):
    monkeypatch.setattr(seminar, "MAX_QUEUED_SEGMENTS", 1)
    ai = FakeAI([RuntimeError("503"), heard("keyingi", speaker="new")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()
        return ws

    ws = asyncio.run(main())
    assert fake_clock == [] and len(ai.listen_calls) == 2  # newer speech goes first
    assert ws.of("error") == [{"type": "error", "id": 1, "message": "Eshitib bo'lmadi"}]
    assert [h["id"] for h in ws.of("heard")] == [2]


def test_interpret_retries_once_then_succeeds(store, sid):
    ai = FakeAI([heard("salom", speaker="new")],
                interpret=[RuntimeError("429"), Rendered(translation="Salom!", note="izoh")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    assert len(ai.interpret_calls) == 2
    (u,) = store.units(sid)
    assert (u["translation"], u["note"]) == ("Salom!", "izoh")
    assert sess.recent_interpretation() == ["[Spiker 1] Salom!"]


def test_interpret_failing_twice_keeps_original_words(store, sid):
    ai = FakeAI([heard("muhim gap", speaker="new"), heard("keyingi gap", speaker="S1")],
                interpret=[RuntimeError("500"), RuntimeError("500")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        recent = list(sess.recent_units)
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws, recent

    sess, ws, recent = asyncio.run(main())
    first, second = store.units(sid)
    assert (first["source"], first["translation"], first["note"]) == ("muhim gap", "", "")
    assert second["translation"] == "T:keyingi gap"  # the pipeline continues normally
    assert recent == []  # an empty translation is not context for the next thought
    assert ws.of("unit")[0]["unit"]["source"] == "muhim gap"
    assert store.pending_segments(sid) == []
    assert len(ai.interpret_calls) == 3


def test_failure_to_save_unit_keeps_segments_for_recovery(store, sid, monkeypatch):
    ai = FakeAI([heard("birinchi", speaker="new"), heard("ikkinchi", speaker="S1")])
    real_add_unit = store.add_unit
    calls = []

    def flaky_add_unit(session_id, unit, segment_ids):
        calls.append(unit["source"])
        if len(calls) == 1:
            raise RuntimeError("disk full")
        return real_add_unit(session_id, unit, segment_ids)

    monkeypatch.setattr(store, "add_unit", flaky_add_unit)

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert calls == ["birinchi", "ikkinchi"]
    assert [u["source"] for u in store.units(sid)] == ["ikkinchi"]
    # the words of the failed thought are still in the database, unlinked: a restart resumes them
    assert [p["text"] for p in store.pending_segments(sid)] == ["birinchi"]


# ---------------------------------------------------------------- persistence across restarts


def test_restart_resumes_pending_thought_and_state(store, sid, monkeypatch):
    seed_speaker(store, sid, "S1", label="Aziz", centroid=E1, samples=5, named=True)
    _ = store.add_unit(sid, {"start": T0 - 60, "end": T0 - 55, "speaker": "S1", "lang": "uz", "target": "uz",
                             "source": "oldin", "translation": "Oldingi gap.", "note": ""}, [])
    ai = FakeAI([heard("Gapimni tugatmadim", speaker="S1", complete=False)])
    emb = toward(E1, E2, 0.9)

    async def first_run():
        sess, ws = start(store, sid, ai, FakeVoiceprints([emb]))
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
        sess.close()  # the server stops before the thought is complete

    asyncio.run(first_run())
    assert [p["text"] for p in store.pending_segments(sid)] == ["Gapimni tugatmadim"]
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.05)

    async def second_run():
        sess, ws = start(store, sid, ai, FakeVoiceprints([]))
        assert [p["text"] for p in sess.pending] == ["Gapimni tugatmadim"] and sess.busy()
        assert sess.idle_timer is not None
        assert sess.recent_interpretation() == ["[Aziz] Oldingi gap."]
        assert sess.recent_transcript() == ["[S1 Aziz] Gapimni tugatmadim"]  # rebuilt from the stored phrases
        s1 = sess.speakers["S1"]
        assert (s1["label"], s1["samples"]) == ("Aziz", 6)
        np.testing.assert_allclose(s1["centroid"], (E1 * 5 + emb) / 6, rtol=1e-5)
        await wait_for(lambda: len(store.units(sid)) == 2)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(second_run())
    last = store.units(sid)[-1]
    assert (last["speaker"], last["source"], last["translation"]) == ("S1", "Gapimni tugatmadim", "T:Gapimni tugatmadim")
    assert ai.interpret_calls[0]["recent"] == ["[Aziz] Oldingi gap."]
    assert store.pending_segments(sid) == [] and not sess.busy()


def test_restart_pending_thought_continues_with_new_phrase(store, sid):
    seed_speaker(store, sid, "S1")
    store.add_segment(sid, {"start": T0 - 2, "end": T0, "speaker": "S1", "lang": "uz", "text": "Shunday qilib",
                            "complete": False})
    ai = FakeAI([heard("xulosa qilamiz.", speaker="S1", complete=True)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    (u,) = store.units(sid)
    assert u["source"] == "Shunday qilib xulosa qilamiz." and u["start"] == T0 - 2
    assert store.pending_segments(sid) == []


def test_restart_with_thoughts_of_two_speakers_keeps_them_apart(store, sid, monkeypatch):
    gate = threading.Event()
    ai = FakeAI([heard("Birinchi spiker gapi.", speaker="new", complete=True),
                 heard("Ikkinchi spiker", speaker="new", complete=False)],
                interpret=[gated(Rendered(translation="never stored", note=""), gate)])

    async def first_run():
        sess, ws = start(store, sid, ai, None)
        try:
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
            await wait_for(lambda: sess.interpreting == 1 and len(sess.pending) == 1 and sess.working == 1)
            sess.close()  # the server restarts while S1's thought is being interpreted
            await asyncio.sleep(0)
        finally:
            gate.set()

    asyncio.run(first_run())
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.05)

    async def second_run():
        sess, ws = start(store, sid, ai, None)
        await wait_for(lambda: not sess.pending)
        await drain(sess)
        sess.close()

    asyncio.run(second_run())
    assert [(u["speaker"], u["source"]) for u in store.units(sid)] == [
        ("S1", "Birinchi spiker gapi."), ("S2", "Ikkinchi spiker")]


# ---------------------------------------------------------------- rename and merge


def test_rename_speaker(store, sid):
    seed_speaker(store, sid, "S1", voice="yo'g'on")

    async def main():
        sess, ws = start(store, sid, FakeAI(), None)
        await sess.rename_speaker("S1", "Prof. Aliyev")
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert ws.messages == [{"type": "speaker", "speaker": {"id": "S1", "label": "Prof. Aliyev", "voice": "yo'g'on"}}]
    assert sess.label("S1") == "Prof. Aliyev" and sess.label("S9") == "S9"
    saved = store.speakers(sid)[0]
    assert (saved["label"], saved["named"]) == ("Prof. Aliyev", 1)


def test_merge_speakers_combines_voiceprints_and_relabels(store, sid):
    seed_speaker(store, sid, "S1", centroid=E1, samples=3)
    seed_speaker(store, sid, "S2", centroid=E2, samples=1)
    seed_speaker(store, sid, "S3", centroid=None)
    _units(store, sid, [(T0 - 50, T0 - 40)])
    store.add_unit(sid, {"start": T0 - 30, "end": T0 - 20, "speaker": "S2", "lang": "uz", "target": "uz",
                         "source": "x", "translation": "y", "note": ""}, [])
    ai = FakeAI([heard("tugallanmagan", speaker="S2", complete=False)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
        await sess.merge_speakers("S2", "S1")
        await sess.merge_speakers("S1", "S3")  # target without a voiceprint takes the source's
        sess.flush()
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert ws.of("merged") == [{"type": "merged", "source": "S2", "target": "S1"},
                               {"type": "merged", "source": "S1", "target": "S3"}]
    assert list(sess.speakers) == ["S3"]
    s3 = sess.speakers["S3"]
    assert s3["samples"] == 4
    np.testing.assert_allclose(s3["centroid"], (E1 * 3 + E2) / 4, rtol=1e-6)
    saved = store.speakers(sid)
    assert [sp["id"] for sp in saved] == ["S3"] and saved[0]["samples"] == 4
    assert {u["speaker"] for u in store.units(sid)} == {"S3"}  # old units and the flushed pending thought


def test_merge_source_without_voiceprint_keeps_target_centroid(store, sid):
    seed_speaker(store, sid, "S1", centroid=E1, samples=2)
    seed_speaker(store, sid, "S2", centroid=None)

    async def main():
        sess, ws = start(store, sid, FakeAI(), None)
        await sess.merge_speakers("S2", "S1")
        sess.close()
        return sess

    sess = asyncio.run(main())
    assert sess.speakers["S1"]["samples"] == 2 and sess.speakers["S1"]["centroid"].tolist() == E1.tolist()


def test_new_speaker_id_after_merge_does_not_collide(store, sid):
    seed_speaker(store, sid, "S1")
    seed_speaker(store, sid, "S2")
    seed_speaker(store, sid, "S3")
    ai = FakeAI([heard("yangi", speaker="new")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.merge_speakers("S2", "S1")
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    assert list(sess.speakers) == ["S1", "S3", "S4"]


def test_merge_while_thought_is_being_interpreted(store, sid):
    seed_speaker(store, sid, "S1")
    seed_speaker(store, sid, "S2")
    gate = threading.Event()
    ai = FakeAI([heard("noto'g'ri ajratilgan", speaker="S2", complete=True)],
                interpret=[gated(Rendered(translation="tarjima", note=""), gate)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        try:
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            await wait_for(lambda: sess.interpreting == 1)
            await sess.merge_speakers("S2", "S1")
        finally:
            gate.set()
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert [u["speaker"] for u in store.units(sid)] == ["S1"]


# ---------------------------------------------------------------- notes


def test_notes_first_early_then_every_n_units(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "NOTES_FIRST_AFTER_UNITS", 2)
    monkeypatch.setattr(seminar, "NOTES_EVERY_UNITS", 3)
    texts = [chr(ord("a") + i) for i in range(6)]
    ai = FakeAI([heard(t, speaker="new" if i == 0 else "S1", name="Aziz" if i == 0 else "") for i, t in enumerate(texts)])
    calls_after = []

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i in range(6):
            await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
            await drain(sess)
            calls_after.append(len(ai.notes_calls))
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    # no summary yet: the first notes come after NOTES_FIRST_AFTER_UNITS, then every NOTES_EVERY_UNITS
    assert calls_after == [0, 1, 1, 1, 2, 2]
    expected = {"summary": "Seminar haqida", "glossary": [{"term": "AI", "translation": "SI", "note": ""}]}
    assert ai.notes_calls[0] == {"notes": {}, "lines": ["[Aziz] a => T:a", "[Aziz] b => T:b"],
                                 "speakers": ["Aziz"]}  # labels only: ids must not reach the summary
    assert ai.notes_calls[1]["notes"] == expected
    assert ai.notes_calls[1]["lines"] == [f"[Aziz] {t} => T:{t}" for t in texts[:5]]  # last 2 * NOTES_EVERY_UNITS
    assert sess.notes == expected and store.session(sid)["notes"] == expected
    assert ws.of("notes") == [{"type": "notes", "notes": expected}] * 2
    assert ai.listen_calls[2]["notes"] == expected and ai.interpret_calls[2]["notes"] == expected
    assert ai.listen_calls[1]["notes"] == {}


def test_notes_without_summary_keep_the_early_cadence(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "NOTES_FIRST_AFTER_UNITS", 1)
    monkeypatch.setattr(seminar, "NOTES_EVERY_UNITS", 50)
    ai = FakeAI([heard("a", speaker="new"), heard("b", speaker="S1")])
    ai.notes_result = Notes(summary="", glossary=[Term(term="X", translation="Y", note="")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i in range(2):
            await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
            await drain(sess)
        sess.close()

    asyncio.run(main())
    assert len(ai.notes_calls) == 2


def test_notes_update_not_started_twice(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "NOTES_FIRST_AFTER_UNITS", 1)
    monkeypatch.setattr(seminar, "NOTES_EVERY_UNITS", 1)
    ai = FakeAI([heard(f"p{i}", speaker="new" if i == 0 else "S1") for i in range(4)])
    ai.notes_gate = threading.Event()

    async def main():
        sess, ws = start(store, sid, ai, None)
        try:
            for i in range(3):
                await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
            await wait_for(lambda: len(store.units(sid)) == 3 and sess.working == 0 and ai.notes_calls)
            first_task = sess.notes_task
            await asyncio.sleep(0.05)  # room for a (wrong) second update to show up
            assert len(ai.notes_calls) == 1 and sess.notes_task is first_task and not first_task.done()
        finally:
            ai.notes_gate.set()
        await sess.notes_task
        await sess.add_audio(3, make_wav(2.0), T0 + 9, origin=ws)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert len(ai.notes_calls) == 2
    assert len(ai.notes_calls[1]["lines"]) == 2  # limit is 2 * NOTES_EVERY_UNITS


def test_notes_failure_is_contained(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "NOTES_EVERY_UNITS", 1)
    store.update_session(sid, notes={"summary": "eski", "glossary": []})
    ai = FakeAI([heard("a", speaker="new"), heard("b", speaker="S1")])
    ai.notes_result = RuntimeError("quota")

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert len(ai.notes_calls) == 2 and ws.of("notes") == []
    assert sess.notes == {"summary": "eski", "glossary": []} == store.session(sid)["notes"]
    assert len(store.units(sid)) == 2


# ---------------------------------------------------------------- busy() and broadcasting


def test_busy_tracks_queued_in_flight_and_pending_work(store, sid):
    listen_gate, listen_entered = threading.Event(), threading.Event()
    interpret_gate = threading.Event()
    ai = FakeAI([gated(heard("bir", speaker="new", complete=False), listen_gate, listen_entered)],
                interpret=[gated(Rendered(translation="Bir", note=""), interpret_gate)])
    states = {}

    async def main():
        sess, ws = start(store, sid, ai, None)
        try:
            states["idle"] = sess.busy()
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            states["queued"] = sess.busy()
            await asyncio.to_thread(listen_entered.wait, 5)
            states["listening"] = (sess.busy(), sess.working, sess.listen_queue.empty())
            listen_gate.set()
            await wait_for(lambda: sess.working == 0 and len(sess.pending) == 1)
            states["pending"] = sess.busy()
            sess.flush()
            await wait_for(lambda: sess.interpreting == 1)
            states["interpreting"] = (sess.busy(), sess.pending, sess.unit_queue.empty())
        finally:
            listen_gate.set()
            interpret_gate.set()
        await drain(sess)
        states["done"] = sess.busy()
        sess.close()
        return ws

    ws = asyncio.run(main())
    assert states["idle"] is False and states["queued"] is True
    assert states["listening"] == (True, 1, True)
    assert states["pending"] is True
    assert states["interpreting"] == (True, [], True)
    assert states["done"] is False
    assert {"type": "pending", "pending": [], "interpreting": 1} in ws.messages


def test_broadcast_drops_failing_client(store, sid):
    async def main():
        sess = LiveSession(sid, store, FakeAI(), None, TZ)
        good, bad, other = FakeWS(), BrokenWS(), FakeWS()
        sess.clients.update({good, bad, other})
        await sess.broadcast({"type": "x"})
        await sess.broadcast({"type": "y"})
        sess.close()
        return sess, good, bad, other

    sess, good, bad, other = asyncio.run(main())
    assert good.types() == ["x", "y"] and other.types() == ["x", "y"]
    assert bad.calls == 1 and sess.clients == {good, other}


def test_close_cancels_workers_and_idle_timer(store, sid):
    seed_speaker(store, sid, "S1")
    store.add_segment(sid, {"start": 0, "end": 1, "speaker": "S1", "lang": "uz", "text": "x", "complete": False})

    async def main():
        sess = LiveSession(sid, store, FakeAI(), None, TZ)
        timer = sess.idle_timer
        assert timer is not None
        sess.close()
        await asyncio.sleep(0)
        return sess, timer

    sess, timer = asyncio.run(main())
    assert timer.cancelled() and all(t.cancelled() for t in sess.workers)


# ---------------------------------------------------------------- store: ids, merges, notes position


def test_store_ids_are_never_reused(store, sid):
    """Late background work of a deleted session or report must not land in a new one."""
    seg = store.add_segment(sid, {"start": 0, "end": 1, "speaker": "S1", "lang": "uz", "text": "x", "complete": True})
    (unit,) = _units(store, sid, [(0.0, 1.0)])
    report = store.start_report(sid, 0.0, 1.0)
    store.delete_session(sid)
    again = store.create_session("Yangi")["id"]
    assert again > sid
    assert store.add_segment(again, {"start": 0, "end": 1, "speaker": "S1", "lang": "uz", "text": "y",
                                     "complete": True}) > seg
    assert _units(store, again, [(0.0, 1.0)])[0] > unit
    assert store.start_report(again, 0.0, 1.0) > report
    store.finish_report(report, {"title": "eski"})  # the deleted session's job finishing late
    assert [r["status"] for r in store.reports(again)] == ["pending"]


def test_store_merge_map_and_labels_of_merged_ids(store, sid):
    seed_speaker(store, sid, "S1", label="Aziz")
    seed_speaker(store, sid, "S2")
    seed_speaker(store, sid, "S3")
    store.merge_speakers(sid, "S2", "S3")
    store.merge_speakers(sid, "S3", "S1")  # S2 follows S3 into S1
    assert store.session(sid)["merged"] == {"S2": "S1", "S3": "S1"}
    # a report written before the merges still names S2 and S3: they read as the person they became
    assert store.speaker_labels(sid) == {"S1": "Aziz", "S2": "Aziz", "S3": "Aziz"}
    assert store.speaker_labels(12345) == {}


def test_merged_ids_are_never_reissued_after_a_reload(store, sid):
    seed_speaker(store, sid, "S1")
    seed_speaker(store, sid, "S2")
    ai = FakeAI([heard("yangi odam", speaker="new")])

    async def first():
        sess, ws = start(store, sid, FakeAI(), None)
        await sess.merge_speakers("S2", "S1")
        sess.close()

    async def second():
        sess, ws = start(store, sid, ai, None)
        assert sess.merged == {"S2": "S1"}
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    asyncio.run(first())
    sess = asyncio.run(second())
    assert list(sess.speakers) == ["S1", "S3"]  # not S2 again: older reports mean someone else by it


def test_notes_keep_their_cadence_across_reloads(store, sid, monkeypatch):
    """A device reconnecting every few minutes reloads the session; the notes must still be updated."""
    monkeypatch.setattr(seminar, "NOTES_FIRST_AFTER_UNITS", 3)
    seed_speaker(store, sid, "S1")
    ai = FakeAI([heard(f"p{i}", speaker="S1") for i in range(4)])

    async def run(i):
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
        await drain(sess)
        sess.close()

    for i in range(3):
        asyncio.run(run(i))
    assert len(ai.notes_calls) == 1  # the third unit, although each session saw only one
    assert store.session(sid)["notes_unit"] == max(u["id"] for u in store.units(sid))
    asyncio.run(run(3))
    assert len(ai.notes_calls) == 1  # counting starts again from what the notes cover


# ---------------------------------------------------------------- replies, dedupe, echo


def test_answers_about_a_phrase_go_only_to_its_sender(store, sid, monkeypatch):
    ai = FakeAI([heard("Salom", speaker="new"), heard(""), RuntimeError("503")])

    async def main():
        sess, sender = start(store, sid, ai, None)
        other = FakeWS()
        sess.join(other)
        await sess.add_audio(1, make_wav(2.0), T0, origin=sender)
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=sender)
        await drain(sess)
        monkeypatch.setattr(seminar, "LISTEN_RETRY_SECONDS", 0)  # give up at once
        await sess.add_audio(3, make_wav(2.0), T0 + 6, origin=sender)
        await drain(sess)
        sess.close()
        return sender, other

    sender, other = asyncio.run(main())
    assert [(m["type"], m["id"]) for m in sender.messages if "id" in m] == [("heard", 1), ("empty", 2), ("error", 3)]
    assert not [m for m in other.messages if m["type"] in ("heard", "empty", "error")]
    assert {"speaker", "unit", "pending"} <= set(other.types())  # the rest is for every device


def test_resent_phrases_are_recognised_for_a_while(store, sid, fake_clock):
    async def main():
        sess, ws = start(store, sid, FakeAI([heard("")] * 3), None)
        assert not sess.received_before("abc", 1)
        await sess.add_audio(1, make_wav(1.0), T0, origin=ws, client="abc")
        await sess.add_audio(1, make_wav(1.0), T0, origin=ws, client="")  # no client id: no dedupe
        result = [sess.received_before("abc", 1), sess.received_before("abc", 2), sess.received_before("xyz", 1),
                  sess.received_before("", 1)]
        await seminar.asyncio.sleep(seminar.DEDUPE_SECONDS + 1)
        result.append(sess.received_before("abc", 1))
        await drain(sess)
        sess.close()
        return result, sess

    result, sess = asyncio.run(main())
    assert result == [True, False, False, False, False]
    assert not sess.received  # forgotten, not kept all day


def test_read_out_picked_up_by_a_microphone_is_dropped(store, sid, monkeypatch):
    """A device reading the translation aloud near another device's microphone must not start a loop."""
    ai = FakeAI(
        [heard("Здравствуйте, коллеги. Начнём семинар.", speaker="new", lang="ru"),
         heard("Assalomu alaykum hamkasblar, seminarni boshlaymiz", speaker="new"),  # the whole read-out
         heard("Seminarni boshlaymiz!", speaker="new"),  # one sentence of it
         heard("Rahmat, savolim bor.", speaker="new")],  # real speech
        interpret=[Rendered(translation="Assalomu alaykum, hamkasblar. Seminarni boshlaymiz.", note="")],
    )

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i in range(4):
            await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
            await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert [m["id"] for m in ws.of("empty")] == [1, 2]
    assert [u["source"] for u in store.units(sid)] == ["Здравствуйте, коллеги. Начнём семинар.", "Rahmat, savolim bor."]
    assert list(sess.speakers) == ["S1", "S2"]


def test_echo_window_expires(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "ECHO_SECONDS", 0.0)
    ai = FakeAI([heard("Hello", speaker="new", lang="en"), heard("T:Hello", speaker="new")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        for i in range(2):
            await sess.add_audio(i, make_wav(2.0), T0 + 3 * i, origin=ws)
            await drain(sess)
        sess.close()

    asyncio.run(main())
    assert len(store.units(sid)) == 2


def test_the_same_phrase_from_a_second_device_is_dropped(store, sid):
    """Two phones recording one seminar: every phrase arrives twice and must be kept once."""
    ai = FakeAI(
        [heard("Bugun logistika haqida gaplashamiz.", speaker="new"),
         heard("Bugun logistika haqida gaplashamiz", speaker="S1"),  # the other phone, slightly different
         heard("Bugun logistika haqida gaplashamiz.", speaker="S1")],  # the first phone again: a real repeat
        # translations unlike the source, so the read-out echo filter stays out of this test
        interpret=[Rendered(translation="Today we talk about logistics.", note="")] * 2,
    )

    async def main():
        sess, phone = start(store, sid, ai, None)
        laptop = FakeWS()
        sess.join(laptop)
        await sess.add_audio(1, make_wav(2.0), T0, origin=phone)
        await drain(sess)
        await sess.add_audio(1, make_wav(2.0), T0 + 0.5, origin=laptop)
        await drain(sess)
        await sess.add_audio(2, make_wav(2.0), T0 + 5, origin=phone)
        await drain(sess)
        sess.close()
        return phone, laptop

    phone, laptop = asyncio.run(main())
    assert [m["id"] for m in laptop.of("empty")] == [1]
    assert len(store.units(sid)) == 2


def test_duplicates_are_only_looked_for_within_a_short_window(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "DUPLICATE_SECONDS", 0.0)
    ai = FakeAI([heard("Rahmat.", speaker="new"), heard("Rahmat.", speaker="S1")],
                interpret=[Rendered(translation="Thank you.", note="")] * 2)

    async def main():
        sess, phone = start(store, sid, ai, None)
        laptop = FakeWS()
        sess.join(laptop)
        await sess.add_audio(1, make_wav(2.0), T0, origin=phone)
        await drain(sess)
        await sess.add_audio(1, make_wav(2.0), T0 + 60, origin=laptop)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert len(store.units(sid)) == 2


# ---------------------------------------------------------------- the idle flush and explicit flush


def test_idle_flush_waits_while_a_device_hears_speech(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.05)
    ai = FakeAI([heard("Asosiy muammo shundaki,", speaker="new", complete=False)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        sess.set_speaking(ws, True)  # the speaker goes on without a pause
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
        await asyncio.sleep(0.2)
        held = (len(sess.pending), store.units(sid), sess.idle_timer is not None)
        sess.set_speaking(ws, False)  # the pause starts now
        stopped = time.monotonic()
        await wait_for(lambda: len(store.units(sid)) == 1)
        waited = time.monotonic() - stopped
        sess.close()
        return held, waited

    held, waited = asyncio.run(main())
    assert held == (1, [], True)
    assert waited >= 0.04


def test_a_device_that_leaves_counts_as_silent(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "IDLE_FLUSH_SECONDS", 0.05)
    ai = FakeAI([heard("Gap", speaker="new", complete=False)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        other = FakeWS()
        sess.join(other)
        sess.set_speaking(other, True)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
        await asyncio.sleep(0.15)
        held = store.units(sid) == []
        sess.leave(other)
        await wait_for(lambda: len(store.units(sid)) == 1)
        sess.close()
        return held, sess

    held, sess = asyncio.run(main())
    assert held and sess.speaking == set() and len(sess.clients) == 1


def test_speech_start_and_stop_without_pending_thought_do_nothing(store, sid):
    async def main():
        sess, ws = start(store, sid, FakeAI(), None)
        sess.set_speaking(ws, True)
        sess.set_speaking(ws, False)
        sess.leave(ws)
        timer = sess.idle_timer
        sess.close()
        return timer

    assert asyncio.run(main()) is None


def test_requested_flush_waits_for_the_phrases_already_sent(store, sid):
    """Stop is pressed while the last phrase is still being transcribed: it belongs to the same thought."""
    gate, entered = threading.Event(), threading.Event()
    ai = FakeAI([heard("Xulosa qilib", speaker="new", complete=False),
                 gated(heard("aytganda", speaker="S1", complete=False), gate, entered),
                 heard("keyingi", speaker="S1", complete=False)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        try:
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
            await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
            await asyncio.to_thread(entered.wait, 5)
            sess.request_flush()
            await asyncio.sleep(0.05)
            assert store.units(sid) == [] and sess.unit_queue.empty() and sess.flush_requested
        finally:
            gate.set()
        await drain(sess)
        units = [u["source"] for u in store.units(sid)]
        # the request is used up: later phrases wait for their own end of thought
        await sess.add_audio(3, make_wav(2.0), T0 + 6, origin=ws)
        await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
        sess.close()
        return units, sess

    units, sess = asyncio.run(main())
    assert units == ["Xulosa qilib aytganda"]
    assert not sess.flush_requested and [p["text"] for p in sess.pending] == ["keyingi"]


def test_requested_flush_with_nothing_queued_is_immediate(store, sid):
    ai = FakeAI([heard("Gap", speaker="new", complete=False)])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
        sess.request_flush()
        assert sess.pending == [] and sess.unit_queue.qsize() == 1
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert [u["source"] for u in store.units(sid)] == ["Gap"]


# ---------------------------------------------------------------- speaker rules


def test_short_clip_never_makes_a_new_speaker_by_voice_alone(store, sid):
    """Short replies of a known person often score low against their voiceprint."""
    seed_speaker(store, sid, "S1", centroid=E1, samples=4)
    seed_speaker(store, sid, "S2", centroid=E2, samples=4)
    ai = FakeAI([heard("Rahmat", speaker="S1")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([toward(E3, E4, 0.9)]))
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert ws.of("heard")[0]["speaker"] == "S1" and list(sess.speakers) == ["S1", "S2"]
    assert ai.listen_calls[0]["references"] and ai.listen_calls[0]["identified"] is None
    assert sess.speakers["S1"]["samples"] == 4  # an unlike voice teaches the voiceprint nothing


def test_long_clip_unlike_everyone_but_not_clearly_follows_gemini(store, sid):
    seed_speaker(store, sid, "S1", centroid=E1, samples=4)
    ai = FakeAI([heard("Davom etamiz", speaker="S1"), heard("Men boshqa odamman", speaker="new")])
    embs = [toward(E1, E2, 0.25), toward(E1, E3, 0.25)]

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints(embs))
        for i in range(2):
            await sess.add_audio(i, make_wav(4.0), T0 + 5 * i, origin=ws)
            await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert [h["speaker"] for h in ws.of("heard")] == ["S1", "S2"]
    assert sess.speakers["S1"]["samples"] == 4  # followed, but too unlike to learn from


def test_voiceprint_decision_is_told_to_gemini(store, sid):
    seed_speaker(store, sid, "S1", label="Aziz", centroid=E1, samples=2)
    ai = FakeAI([heard("Davom etamiz", speaker="S1"), heard("Yangi odam", speaker="new")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([E1, E3]))
        for i in range(2):
            await sess.add_audio(i, make_wav(3.0), T0 + 5 * i, origin=ws)
            await drain(sess)
        sess.close()

    asyncio.run(main())
    assert [c["identified"] for c in ai.listen_calls] == ["S1", "new"]
    assert [c["references"] for c in ai.listen_calls] == [[], []]


def test_seminar_ai_listen_tells_gemini_what_the_voiceprints_decided():
    interp = FakeInterpreter(heard("x"))
    ai = SeminarAI(interp)
    ai.listen(b"RIFF", [], {"S1": 0.9}, [], {}, identified="S1")
    ai.listen(b"RIFF", [], {"S1": 0.1}, [], {}, identified="new")
    assert 'identified this speaker as S1: answer "S1"' in interp.calls[0]["system"]
    assert 'someone not heard before: answer "new"' in interp.calls[1]["system"]
    assert "Voiceprint similarity" not in interp.calls[0]["system"]


@pytest.mark.parametrize(
    "emb, gemini, name, expected_label",
    [
        (E1, "S1", "Prof. Karimov", "Prof. Karimov"),  # Gemini meant the person the voiceprint chose
        (E1, "new", "Prof. Karimov", "Spiker 1"),  # the moderator handing over the floor: a name for someone else
        (E1, "S2", "Prof. Karimov", "Spiker 1"),
    ],
    ids=["agrees", "gemini-says-new", "gemini-names-another"],
)
def test_speaker_name_only_for_the_person_gemini_meant(store, sid, emb, gemini, name, expected_label):
    seed_speaker(store, sid, "S1", centroid=E1, samples=3)
    seed_speaker(store, sid, "S2", centroid=E2, samples=3)
    ai = FakeAI([heard("So'zni professorga beraman", speaker=gemini, name=name)])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([emb]))
        await sess.add_audio(1, make_wav(3.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    assert list(sess.speakers) == ["S1", "S2"] and sess.speakers["S1"]["label"] == expected_label


def test_voiceprint_new_speaker_does_not_take_the_name_gemini_gave_someone_else(store, sid):
    seed_speaker(store, sid, "S1", label="Spiker 1", centroid=E1, samples=3)
    ai = FakeAI([heard("Savolim bor", speaker="S1", name="Aziz")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([E3]))
        await sess.add_audio(1, make_wav(3.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    assert [(sp["id"], sp["label"], sp["named"]) for sp in sess.speakers.values()] == [
        ("S1", "Spiker 1", False), ("S2", "Spiker 2", False)]


def test_voiceprints_learn_only_from_clips_of_two_seconds_or_more(store, sid):
    seed_speaker(store, sid, "S1", centroid=E1, samples=1)
    ai = FakeAI([heard("Ha", speaker="S1"), heard("Yo'q", speaker="new"), heard("Ha, albatta", speaker="S1")])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([E1, E3, E1]))
        await sess.add_audio(1, make_wav(1.5), T0, origin=ws)
        await sess.add_audio(2, make_wav(1.5), T0 + 3, origin=ws)
        await sess.add_audio(3, make_wav(2.0), T0 + 6, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    assert sess.speakers["S1"]["samples"] == 2  # only the 2 s clip
    assert sess.speakers["S2"]["centroid"] is None and sess.speakers["S2"]["samples"] == 0


def test_new_speaker_reference_is_cut_to_a_good_length(store, sid):
    ai = FakeAI([heard("Uzun gap", speaker="new")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(12.0), T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    s1 = sess.speakers["S1"]
    assert s1["ref_seconds"] == seminar.GOOD_REFERENCE_SECONDS[1]
    assert seminar.wav_seconds(s1["ref_wav"]) == pytest.approx(seminar.GOOD_REFERENCE_SECONDS[1])
    assert store.units(sid)[0]["start"] == pytest.approx(T0 - 12.0)  # the phrase itself is whole


def test_trim_wav_keeps_short_or_unreadable_clips():
    wav = make_wav(2.0)
    assert seminar.trim_wav(wav, 8.0) is wav
    assert seminar.trim_wav(b"RIFF....junk", 1.0) == b"RIFF....junk"
    assert seminar.wav_seconds(seminar.trim_wav(wav, 0.5)) == pytest.approx(0.5)


@pytest.mark.parametrize("voiceprints", [True, False], ids=["voiceprint-disagrees", "no-voiceprints"])
def test_reference_and_voice_change_only_on_confident_attribution(store, sid, voiceprints):
    seed_speaker(store, sid, "S1", centroid=E1 if voiceprints else None, ref_wav=b"R1", ref_seconds=1.5)
    seed_speaker(store, sid, "S2", centroid=E2 if voiceprints else None, ref_wav=b"R2", ref_seconds=1.5)
    wav = make_wav(5.0)
    ai = FakeAI([heard("Boshqa odamning gapi", speaker="S2", voice="ayol ovozi")])
    vp = FakeVoiceprints([toward(E1, E3, 0.5)]) if voiceprints else None

    async def main():
        sess, ws = start(store, sid, ai, vp)
        await sess.add_audio(1, wav, T0, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    s2 = sess.speakers["S2"]
    assert (s2["ref_wav"], s2["voice"]) == (b"R2", "")  # Gemini's pick wasn't confirmed by anything


def test_gemini_answer_with_a_merged_away_id_follows_the_merge(store, sid):
    seed_speaker(store, sid, "S1")
    seed_speaker(store, sid, "S2")
    ai = FakeAI([heard("birinchi", speaker="S2"), heard("ikkinchi", speaker="S2")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        await sess.merge_speakers("S2", "S1")
        recent = sess.recent_transcript()
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()
        return sess, ws, recent

    sess, ws, recent = asyncio.run(main())
    assert recent == ["[S1 Spiker 1] birinchi"]  # Gemini's context shows the merge at once
    assert list(sess.speakers) == ["S1"] and [h["speaker"] for h in ws.of("heard")] == ["S2", "S1"]


def test_merge_while_a_phrase_is_being_heard(store, sid):
    """The voiceprint matched S2 before the merge: no crash, no second Gemini call, S1 it is."""
    seed_speaker(store, sid, "S1", centroid=E1, samples=1)
    seed_speaker(store, sid, "S2", centroid=E2, samples=1)
    gate, entered = threading.Event(), threading.Event()
    ai = FakeAI([gated(heard("gap", speaker="S2"), gate, entered)])

    async def main():
        sess, ws = start(store, sid, ai, FakeVoiceprints([E2]))
        try:
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            await asyncio.to_thread(entered.wait, 5)
            await sess.merge_speakers("S2", "S1")
        finally:
            gate.set()
        await drain(sess)
        sess.close()
        return sess, ws

    sess, ws = asyncio.run(main())
    assert len(ai.listen_calls) == 1 and ws.of("heard")[0]["speaker"] == "S1"
    assert list(sess.speakers) == ["S1"] and [p["speaker"] for p in sess.pending + store.units(sid)] == ["S1"]


def test_recent_lines_show_current_labels(store, sid):
    seed_speaker(store, sid, "S1")
    ai = FakeAI([heard("salom", speaker="S1"), heard("davom", speaker="S1")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        await sess.rename_speaker("S1", "Aziz")
        await sess.add_audio(2, make_wav(2.0), T0 + 3, origin=ws)
        await drain(sess)
        sess.close()
        return sess

    sess = asyncio.run(main())
    assert ai.listen_calls[1]["recent"] == ["[S1 Aziz] salom"]
    assert ai.interpret_calls[1]["recent"] == ["[Aziz] T:salom"]


def test_references_rank_by_similarity_then_recency_and_keep_recent_speakers(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "MAX_REFERENCE_SPEAKERS", 3)
    for i in range(1, 8):  # Gemini-only: no similarities at all
        seed_speaker(store, sid, f"S{i}", ref_wav=f"R{i}".encode())
    # S1 spoke last in the recent transcript but long ago; S7 and S5 spoke most recently
    for spk, end in (("S1", T0 - 600), ("S3", T0 - 500), ("S5", T0 - 20), ("S7", T0 - 10)):
        store.add_segment(sid, {"start": end - 2, "end": end, "speaker": spk, "lang": "uz", "text": spk,
                                "complete": True})
    with closing(store._connect()) as db, db:  # S1's line is the newest in the transcript, S3 is older
        db.execute("UPDATE segments SET id = id + 100 WHERE speaker = 'S1'")
    monkeypatch.setattr(seminar, "RECENT_LINES", 1)
    ai = FakeAI([heard("x", speaker="S7")])

    async def main():
        sess, ws = start(store, sid, ai, None)
        await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
        await drain(sess)
        sess.close()

    asyncio.run(main())
    assert ai.listen_calls[0]["recent"] == ["[S1 Spiker 1] S1"]
    assert ai.listen_calls[0]["references"] == ["S7", "S5", "S1"]


def test_close_stops_everything_and_unload_checks(store, sid, monkeypatch):
    monkeypatch.setattr(seminar, "NOTES_FIRST_AFTER_UNITS", 1)
    seed_speaker(store, sid, "S1")
    store.add_segment(sid, {"start": 0, "end": 1, "speaker": "S1", "lang": "uz", "text": "x", "complete": False})
    ai = FakeAI([heard("a", speaker="S1", complete=False)])
    ai.notes_gate = threading.Event()

    async def main():
        sess, ws = start(store, sid, ai, None)
        try:
            sess.flush()
            await wait_for(lambda: sess.notes_task is not None)
            task = sess.notes_task
            await sess.add_audio(1, make_wav(2.0), T0, origin=ws)
            await wait_for(lambda: len(sess.pending) == 1 and sess.working == 0)
            sess.close()
            sess.set_speaking(ws, False)  # a late message after the close arms nothing
            await asyncio.sleep(0)
            return sess, task
        finally:
            ai.notes_gate.set()

    sess, task = asyncio.run(main())
    assert sess.closed and sess.idle_timer is None and task.cancelled()


# ---------------------------------------------------------------- build_report


def ts(hh, mm, ss=0):
    return datetime(2026, 10, 19, hh, mm, ss, tzinfo=TZ).timestamp()


def seed_report_units(store, sid):
    seed_speaker(store, sid, "S1", label="Aziz")
    seed_speaker(store, sid, "S2", label="Spiker 2")
    store.add_unit(sid, {"start": ts(10, 0), "end": ts(10, 0, 30), "speaker": "S1", "lang": "en", "target": "uz",
                         "source": "Hello everyone", "translation": "Hammaga salom.", "note": ""}, [])
    store.add_unit(sid, {"start": ts(10, 5), "end": ts(10, 6), "speaker": "S2", "lang": "ru", "target": "uz",
                         "source": "Привет", "translation": "", "note": ""}, [])  # untranslated
    store.add_unit(sid, {"start": ts(10, 3), "end": ts(10, 4), "speaker": "S1", "lang": "uz", "target": "uz",
                         "source": "rahmat", "translation": "Rahmat.", "note": ""}, [])  # cleaned-up Uzbek


def test_build_report_without_units_raises(store, sid):
    ai = FakeAI()
    assert not issubclass(seminar.ReportRangeError, ValueError)  # unlike a malformed model answer
    with pytest.raises(seminar.ReportRangeError, match="tarjima qilingan gap yo'q"):
        asyncio.run(build_report(store, ai, sid, 0.0, time.time(), TZ))
    seed_report_units(store, sid)
    with pytest.raises(seminar.ReportRangeError):
        asyncio.run(build_report(store, ai, sid, ts(11, 0), ts(12, 0), TZ))
    assert ai.report_calls == [] and store.reports(sid) == []


def test_build_report_clamps_period_and_sends_transcript(store, sid):
    seed_report_units(store, sid)
    ai = FakeAI()
    ai.report_result = make_report(title="Iqtisodiy seminar", decisions=["Qaror"])
    report_id = asyncio.run(build_report(store, ai, sid, ts(9, 0), ts(12, 0), TZ))
    row = store.report(report_id)
    assert (row["start"], row["end"]) == (ts(10, 0), ts(10, 6))  # the empty edges are cut off
    assert row["session_id"] == sid
    assert row["report"] == ai.report_result.model_dump()
    (call,) = ai.report_calls
    assert call["period"] == "2026-10-19 10:00 - 2026-10-19 10:06"
    assert call["speakers"] == "S1: Aziz\nS2: Spiker 2"
    assert call["transcript"].splitlines() == [
        "[10:00] S1 (Aziz): Hammaga salom.  <<en: Hello everyone>>",
        "[10:03] S1 (Aziz): Rahmat.",  # Uzbek kept in Uzbek: no need to repeat the source
        "[10:05] S2 (Spiker 2): Привет",  # an untranslated unit keeps its original words
    ]


def test_build_report_partial_range_keeps_requested_edges(store, sid):
    seed_report_units(store, sid)
    ai = FakeAI()
    report_id = asyncio.run(build_report(store, ai, sid, ts(10, 0, 10), ts(10, 5, 30), TZ))
    row = store.report(report_id)
    assert (row["start"], row["end"]) == (ts(10, 0, 10), ts(10, 5, 30))
    assert len(ai.report_calls[0]["transcript"].splitlines()) == 3
    assert ai.report_calls[0]["period"] == "2026-10-19 10:00 - 2026-10-19 10:05"


def test_build_report_failure_saves_nothing(store, sid):
    seed_report_units(store, sid)
    ai = FakeAI()
    ai.report_result = RuntimeError("gemini down")
    with pytest.raises(RuntimeError):
        asyncio.run(build_report(store, ai, sid, ts(9, 0), ts(12, 0), TZ))
    assert store.reports(sid) == []
