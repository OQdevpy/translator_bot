// node --test tests/
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  Downsampler,
  FRAME,
  SAMPLE_RATE,
  Segmenter,
  VAD,
  concat,
  encodeWav,
  packSegment,
  rms,
} from "../static/audio.js";
import {
  OUTBOX_MAX,
  Outbox,
  ReadQueue,
  captureStalled,
  findPendingReport,
  makeClientId,
  planResync,
  pollReport,
  tail,
  unionUnits,
} from "../static/live.js";

const FRAME_MS = (FRAME / SAMPLE_RATE) * 1000;

// Deterministic noise so the tests never flake.
function rng(seed) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}
const random = rng(42);

function noise(ms, amplitude) {
  const n = Math.round((ms / 1000) * SAMPLE_RATE);
  return Float32Array.from({ length: n }, () => (random() * 2 - 1) * amplitude);
}

// Speech-like signal: 180 Hz tone with syllable-rate amplitude modulation and short dips between "words".
function speech(ms, amplitude = 0.2) {
  const n = Math.round((ms / 1000) * SAMPLE_RATE);
  return Float32Array.from({ length: n }, (_, i) => {
    const t = i / SAMPLE_RATE;
    const syllable = 0.6 + 0.4 * Math.sin(2 * Math.PI * 4 * t);
    const word = t % 0.9 < 0.8 ? 1 : 0.15;
    return amplitude * syllable * word * Math.sin(2 * Math.PI * 180 * t);
  });
}

function run(signal, options) {
  const segments = [];
  const events = [];
  const seg = new Segmenter({
    options,
    onSegment: (samples, info) => segments.push({ ms: (samples.length / SAMPLE_RATE) * 1000, ...info }),
    onSpeech: (active) => events.push(active),
  });
  for (let i = 0; i + FRAME <= signal.length; i += FRAME) seg.push(signal.subarray(i, i + FRAME));
  return { seg, segments, events };
}

const join = (...parts) => concat(parts);

test("rms of a constant signal", () => {
  assert.equal(rms(new Float32Array([0.5, -0.5, 0.5, -0.5])), 0.5);
  assert.equal(rms(new Float32Array(0)), 0);
});

for (const rate of [48000, 44100, 32000, 22050, 16000, 8000]) {
  test(`downsampler ${rate} Hz -> 16 kHz keeps duration and level`, () => {
    const frames = [];
    const down = new Downsampler(rate, (f) => frames.push(f));
    const seconds = 2;
    const input = Float32Array.from({ length: rate * seconds }, (_, i) => 0.5 * Math.sin((2 * Math.PI * 200 * i) / rate));
    // feed in odd-sized chunks like an AudioWorklet would
    for (let i = 0; i < input.length; i += 2048) down.push(input.subarray(i, i + 2048));
    const out = frames.length * FRAME;
    assert.ok(Math.abs(out - SAMPLE_RATE * seconds) <= FRAME, `got ${out} samples`);
    assert.ok(frames.every((f) => f.length === FRAME));
    const level = rms(concat(frames));
    assert.ok(Math.abs(level - 0.5 / Math.SQRT2) < 0.03, `level ${level}`);
  });
}

test("silence and steady background noise produce no segments", () => {
  assert.equal(run(new Float32Array(SAMPLE_RATE * 5)).segments.length, 0);
  assert.equal(run(noise(10000, 0.02)).segments.length, 0);
});

test("one phrase between pauses becomes one segment with pre-roll and a short tail", () => {
  const { segments, events } = run(join(noise(1000, 0.002), speech(2000), noise(1500, 0.002)));
  assert.equal(segments.length, 1);
  assert.ok(segments[0].ms >= 2000 && segments[0].ms <= 2000 + VAD.prerollMs + VAD.tailMs + 100, `${segments[0].ms} ms`);
  assert.equal(segments[0].forced, false);
  assert.deepEqual(events, [true, false]);
});

test("two phrases separated by a pause are sent separately", () => {
  const { segments } = run(join(noise(500, 0.002), speech(1500), noise(1200, 0.002), speech(1800), noise(1500, 0.002)));
  assert.equal(segments.length, 2);
});

test("the dips between words do not split a phrase", () => {
  // speech() has a 100 ms dip every 900 ms
  const { segments } = run(join(noise(500, 0.002), speech(3500), noise(1500, 0.002)));
  assert.equal(segments.length, 1);
});

test("clicks and coughs shorter than the minimum are dropped", () => {
  const { segments } = run(join(noise(1000, 0.002), speech(150, 0.5), noise(2000, 0.002)));
  assert.equal(segments.length, 0);
});

test("non-stop speech is cut into segments no longer than maxMs, and nothing is lost", () => {
  const total = 40000;
  const signal = join(noise(500, 0.002), speech(total), noise(1500, 0.002));
  const { segments } = run(signal);
  assert.ok(segments.length >= 4, `${segments.length} segments`);
  assert.ok(segments.every((s) => s.ms <= VAD.maxMs), segments.map((s) => s.ms).join(","));
  const sent = segments.reduce((n, s) => n + s.ms, 0);
  assert.ok(sent >= total - 200, `sent ${sent} ms of ${total}`);
});

test("long speech is cut at a quiet moment, not mid-word", () => {
  const segments = [];
  const seg = new Segmenter({ onSegment: (samples, info) => segments.push({ samples, ...info }) });
  const signal = join(noise(300, 0.002), speech(30000), noise(1500, 0.002));
  for (let i = 0; i + FRAME <= signal.length; i += FRAME) seg.push(signal.subarray(i, i + FRAME));
  assert.ok(segments.filter((s) => s.forced).length >= 2);
  // a forced cut hands the quietest frame to the next segment: it must be a gap between words
  for (let i = 1; i < segments.length; i++) {
    if (!segments[i - 1].forced) continue;
    const level = rms(segments[i].samples.subarray(0, FRAME));
    assert.ok(level < 0.03, `cut at a loud frame (rms ${level.toFixed(3)})`);
  }
});

test("long phrases end on a shorter pause than short ones", () => {
  // after 9 s of speech a 300 ms pause ends the segment
  const long = run(join(noise(500, 0.002), speech(9000), noise(320, 0.002), speech(1000), noise(1500, 0.002)));
  assert.equal(long.segments.length, 2);
  // a 300 ms pause after a 1 s phrase does not
  const short = run(join(noise(500, 0.002), speech(1000), noise(320, 0.002), speech(1000), noise(1500, 0.002)));
  assert.equal(short.segments.length, 1);
});

test("adapts to a louder room: speech over background noise is still found", () => {
  const { segments } = run(join(noise(3000, 0.03), add(speech(2000, 0.3), noise(2000, 0.03)), noise(2000, 0.03)));
  assert.equal(segments.length, 1);
});

function add(a, b) {
  return a.map((v, i) => v + (b[i] ?? 0));
}

test("flush sends the phrase in progress when listening stops", () => {
  const { seg, segments } = run(join(noise(500, 0.002), speech(1500)));
  assert.equal(segments.length, 0);
  seg.flush();
  assert.equal(segments.length, 1);
  seg.flush(); // idempotent
  assert.equal(segments.length, 1);
});

test("requireSilence skips the tail of a phrase already under way", () => {
  const segments = [];
  const seg = new Segmenter({ onSegment: (s) => segments.push(s) });
  const feed = (signal) => {
    for (let i = 0; i + FRAME <= signal.length; i += FRAME) seg.push(signal.subarray(i, i + FRAME));
  };
  feed(noise(1000, 0.002));
  seg.requireSilence(300);
  feed(speech(1500)); // resumed mid-phrase: ignored
  assert.equal(seg.speaking, false);
  feed(noise(400, 0.002)); // the pause re-arms it
  feed(join(speech(1500), noise(1500, 0.002))); // the next phrase is captured
  assert.equal(segments.length, 1);
  assert.ok(segments[0].length / SAMPLE_RATE < 2.1);
});

test("speaking is true only while a phrase is open", () => {
  const { seg } = run(join(noise(500, 0.002), speech(1000)));
  assert.equal(seg.speaking, true);
  seg.flush();
  assert.equal(seg.speaking, false);
});

test("encodeWav writes a valid 16-bit mono header and clamps samples", () => {
  const wav = encodeWav(Float32Array.from([0, 1, -1, 2, -2, 0.5]), 16000);
  const view = new DataView(wav);
  const str = (o, n) => String.fromCharCode(...new Uint8Array(wav, o, n));
  assert.equal(str(0, 4), "RIFF");
  assert.equal(str(8, 4), "WAVE");
  assert.equal(str(36, 4), "data");
  assert.equal(view.getUint32(4, true), wav.byteLength - 8);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 12);
  const pcm = [0, 1, 2, 3, 4, 5].map((i) => view.getInt16(44 + i * 2, true));
  assert.deepEqual(pcm, [0, 32767, -32768, 32767, -32768, 16383]);
});

test("packSegment prefixes a big-endian id and the phrase's age in seconds", () => {
  const frame = packSegment(258, new Uint8Array([82, 73, 70, 70]).buffer, 2.5);
  const view = new DataView(frame.buffer);
  assert.equal(view.getUint32(0, false), 258);
  assert.equal(view.getFloat64(4, false), 2.5);
  assert.deepEqual([...frame.slice(12)], [82, 73, 70, 70]);
  assert.equal(new DataView(packSegment(1, new ArrayBuffer(0)).buffer).getFloat64(4, false), 0);
});

test("a short word followed by a sub-second pause stays with the next phrase", () => {
  // "Thank you." ... 0.9 s ... "Now a break." must not be sent as two pieces
  const { segments } = run(join(noise(500, 0.002), speech(700), noise(900, 0.002), speech(2000), noise(1500, 0.002)));
  assert.equal(segments.length, 1);
  // but a real pause still separates them
  const apart = run(join(noise(500, 0.002), speech(700), noise(1400, 0.002), speech(2000), noise(1500, 0.002)));
  assert.equal(apart.segments.length, 2);
});

// ---------------------------------------------------------------- live.js

const wav = (n) => new Uint8Array([n]).buffer;
const frameAge = (frame) => new DataView(frame.buffer).getFloat64(4, false);
const frameId = (frame) => new DataView(frame.buffer).getUint32(0, false);

test("client id is random and fits the server's [A-Za-z0-9_-]{1,40}", () => {
  const a = makeClientId();
  assert.match(a, /^[A-Za-z0-9_-]{1,40}$/);
  assert.notEqual(a, makeClientId());
});

test("outbox keeps phrases until the server acknowledges them", () => {
  const box = new Outbox();
  box.add({ id: 1, sessionId: 7, wav: wav(1), endedAt: 1000 });
  box.add({ id: 2, sessionId: 7, wav: wav(2), endedAt: 2000 });
  assert.equal(box.size, 2);
  assert.equal(box.ack(1), true);
  assert.equal(box.ack(1), false); // a second reply, or another page's id: nothing to do
  assert.equal(box.ack(99), false);
  assert.deepEqual(box.replay(7, 3000).map((i) => i.id), [2]);
});

test("outbox resends after a reconnect with the age recomputed at send time", () => {
  const box = new Outbox();
  const item = box.add({ id: 5, sessionId: 7, wav: wav(9), endedAt: 10_000 });
  assert.equal(frameAge(box.frame(item, 10_500)), 0.5);
  const [again] = box.replay(7, 40_000);
  const frame = box.frame(again, 40_000);
  assert.equal(frameId(frame), 5);
  assert.equal(frameAge(frame), 30);
  assert.deepEqual([...frame.slice(12)], [9]);
  assert.equal(frameAge(box.frame(item, 9_000)), 0); // a clock step back never gives a negative age
});

test("outbox drops phrases older than a minute and keeps at most the newest ten", () => {
  const box = new Outbox();
  for (let id = 1; id <= OUTBOX_MAX + 3; id++) box.add({ id, sessionId: 7, wav: wav(id), endedAt: id * 1000 });
  assert.equal(box.size, OUTBOX_MAX);
  assert.deepEqual(box.replay(7, 13_000).map((i) => i.id), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
  assert.deepEqual(box.replay(7, 70_500).map((i) => i.id), [11, 12, 13]);
  assert.equal(box.size, 3);
});

test("outbox never replays one seminar's phrases into another", () => {
  const box = new Outbox();
  box.add({ id: 1, sessionId: 7, wav: wav(1), endedAt: 1000 });
  box.add({ id: 2, sessionId: 8, wav: wav(2), endedAt: 1000 });
  assert.deepEqual(box.replay(8, 2000).map((i) => i.id), [2]);
  box.keepOnly(8); // switching seminars
  assert.deepEqual(box.items.map((i) => i.id), [2]);
  box.keepOnly(9);
  assert.equal(box.size, 0);
});

const unit = (id, start, speaker = "S1") => ({ id, start, end: start + 5, speaker });

test("re-sync appends units missed while offline when they are newer than the page", () => {
  const shown = [unit(1, 100), unit(2, 110)];
  const plan = planResync(shown, [unit(1, 100), unit(2, 110), unit(4, 130), unit(3, 120)]);
  assert.deepEqual(plan.missing.map((u) => u.id), [3, 4]);
  assert.equal(plan.rebuild, false);
  assert.deepEqual(planResync(shown, shown), { missing: [], rebuild: false });
  assert.deepEqual(planResync([], [unit(1, 100)]).missing.map((u) => u.id), [1]);
});

test("re-sync redraws when a missed unit belongs before what is shown", () => {
  // another device's phrase, resent late, was interpreted after ours
  const plan = planResync([unit(1, 100), unit(3, 130)], [unit(1, 100), unit(2, 110), unit(3, 130)]);
  assert.deepEqual(plan.missing.map((u) => u.id), [2]);
  assert.equal(plan.rebuild, true);
});

test("re-sync redraws when a merge on another device changed shown units' speaker", () => {
  const plan = planResync([unit(1, 100, "S3"), unit(2, 110, "S1")], [unit(1, 100, "S1"), unit(2, 110, "S1")]);
  assert.deepEqual(plan.missing, []);
  assert.equal(plan.rebuild, true);
});

test("redraw keeps units that arrived live after the fetch read the database", () => {
  const fetched = [unit(2, 110, "S1"), unit(3, 120, "S1")];
  const shown = [unit(1, 100, "S3"), unit(2, 110, "S3"), unit(4, 125, "S1")];
  // 1 is older than the fetched page and has the stale speaker: dropped; 4 arrived after the fetch: kept
  assert.deepEqual(unionUnits(fetched, shown).map((u) => [u.id, u.speaker]), [[2, "S1"], [3, "S1"], [4, "S1"]]);
  assert.deepEqual(unionUnits(fetched, shown, 2).map((u) => u.id), [3, 4]);
  assert.deepEqual(unionUnits([], [unit(5, 1)]).map((u) => u.id), [5]); // opening: only live units so far
});

test("the live strip shows the end of a long thought, cut at a word", () => {
  assert.equal(tail("short text", 40), "short text");
  const long = Array.from({ length: 60 }, (_, i) => `word${i}`).join(" ");
  const out = tail(long, 50);
  assert.ok(out.startsWith("…word"), out);
  assert.ok(out.endsWith("word59"));
  assert.ok(out.length <= 51);
  assert.equal(tail("x".repeat(100), 10), `…${"x".repeat(10)}`); // no space: plain cut
});

test("read-out keeps only the newest three texts", () => {
  const q = new ReadQueue();
  ["a", "b", "c", "d", "e"].forEach((t, i) => q.push(t, i));
  assert.equal(q.size, 3);
  assert.deepEqual([q.next(10), q.next(10), q.next(10), q.next(10)], ["c", "d", "e", undefined]);
});

test("read-out skips texts older than a minute", () => {
  const q = new ReadQueue();
  q.push("old", 0);
  q.push("new", 50_000);
  assert.equal(q.next(61_000), "new");
  q.push("x", 0);
  q.clear();
  assert.equal(q.next(1), undefined);
});

test("a pending report for about the same range is reused, not started again", () => {
  const reports = [
    { id: 1, status: "done", start: 0, end: 3600 },
    { id: 2, status: "pending", start: 0, end: 3600 },
    { id: 3, status: "pending", start: 1800, end: 3600 },
  ];
  assert.equal(findPendingReport(reports, 0, 3600)?.id, 2);
  assert.equal(findPendingReport(reports, 0, 3720)?.id, 2); // "whole seminar" pressed two minutes later
  assert.equal(findPendingReport(reports, 0, 7200), null);
  assert.equal(findPendingReport(reports, 600, 3600), null);
  assert.equal(findPendingReport([{ id: 1, status: "failed", start: 0, end: 3600 }], 0, 3600), null);
});

function statusSource(results) {
  const calls = [];
  const sleeps = [];
  return {
    calls,
    sleeps,
    get: async () => {
      const next = results[calls.length];
      calls.push(next);
      if (next instanceof Error) throw next;
      return next;
    },
    sleep: async (ms) => sleeps.push(ms),
  };
}

const httpError = (status) => Object.assign(new Error(`${status}`), { status });

test("report polling retries network blips and 5xx with backoff", async () => {
  const src = statusSource([
    { status: "pending" },
    new TypeError("Failed to fetch"),
    httpError(502),
    { status: "pending" },
    { status: "done" },
  ]);
  const status = await pollReport(src.get, { sleep: src.sleep });
  assert.equal(status.status, "done");
  assert.deepEqual(src.sleeps, [2000, 2000, 4000, 8000, 2000]);
});

test("report polling backoff is capped", async () => {
  const src = statusSource([...Array(8).fill(httpError(503)), { status: "failed", error: "x" }]);
  const status = await pollReport(src.get, { sleep: src.sleep, maxDelay: 10000 });
  assert.equal(status.status, "failed"); // a failed job ends the wait with its error
  assert.equal(Math.max(...src.sleeps), 10000);
});

test("report polling stops on 404", async () => {
  const src = statusSource([{ status: "pending" }, httpError(404), { status: "done" }]);
  await assert.rejects(pollReport(src.get, { sleep: src.sleep }), (err) => err.status === 404);
  assert.equal(src.calls.length, 2);
});

test("capture counts as stalled when the context stops, the track is muted or no audio arrives", () => {
  assert.equal(captureStalled({ state: "running", muted: false, silentMs: 100 }), false);
  assert.equal(captureStalled({ state: "interrupted", muted: false, silentMs: 0 }), true); // iOS call or Siri
  assert.equal(captureStalled({ state: "suspended", muted: false, silentMs: 0 }), true);
  assert.equal(captureStalled({ state: "running", muted: true, silentMs: 0 }), true);
  assert.equal(captureStalled({ state: "running", muted: undefined, silentMs: 3500 }), true);
});
