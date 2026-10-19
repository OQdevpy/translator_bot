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

test("packSegment prefixes a big-endian id and the capture time", () => {
  const frame = packSegment(258, new Uint8Array([82, 73, 70, 70]).buffer, 1760000000.5);
  const view = new DataView(frame.buffer);
  assert.equal(view.getUint32(0, false), 258);
  assert.equal(view.getFloat64(4, false), 1760000000.5);
  assert.deepEqual([...frame.slice(12)], [82, 73, 70, 70]);
});

test("a short word followed by a sub-second pause stays with the next phrase", () => {
  // "Thank you." ... 0.9 s ... "Now a break." must not be sent as two pieces
  const { segments } = run(join(noise(500, 0.002), speech(700), noise(900, 0.002), speech(2000), noise(1500, 0.002)));
  assert.equal(segments.length, 1);
  // but a real pause still separates them
  const apart = run(join(noise(500, 0.002), speech(700), noise(1400, 0.002), speech(2000), noise(1500, 0.002)));
  assert.equal(apart.segments.length, 2);
});
