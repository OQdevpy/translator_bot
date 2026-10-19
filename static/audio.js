// Audio helpers used by the page and by the Node tests: no DOM or Web Audio here.

export const SAMPLE_RATE = 16000;
export const FRAME = 320; // 20 ms at 16 kHz
const FRAME_MS = (FRAME / SAMPLE_RATE) * 1000;

export function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / (samples.length || 1));
}

/** Resamples any input rate to 16 kHz (box-filter average) and emits 20 ms frames. */
export class Downsampler {
  constructor(inputRate, onFrame) {
    this.ratio = inputRate / SAMPLE_RATE;
    this.onFrame = onFrame;
    this.pos = 0;
    this.sum = 0;
    this.count = 0;
    this.last = 0;
    this.frame = new Float32Array(FRAME);
    this.filled = 0;
  }

  push(samples) {
    for (let i = 0; i < samples.length; i++) {
      this.sum += samples[i];
      this.count++;
      this.pos += 1;
      while (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const value = this.count ? this.sum / this.count : this.last;
        this.last = value;
        this.sum = 0;
        this.count = 0;
        this.frame[this.filled++] = value;
        if (this.filled === FRAME) {
          this.onFrame(this.frame);
          this.frame = new Float32Array(FRAME);
          this.filled = 0;
        }
      }
    }
  }
}

export const VAD = {
  minThreshold: 0.008, // absolute speech floor (RMS, full scale = 1)
  noiseFactor: 3.5, // speech must be this many times louder than the background
  noiseWindowMs: 3000, // background level = quietest frame in this window
  startMs: 60, // continuous speech needed to open a segment
  prerollMs: 300, // audio kept from before the speech started
  minVoicedMs: 300, // segments with less speech are dropped (clicks, coughs)
  maxMs: 12000, // non-stop speech is cut at the quietest recent moment
  cutSearchMs: 2000, // how far back to look for that moment
  tailMs: 200, // silence kept after the last speech
};

/**
 * Voice activity detection: turns a stream of 20 ms frames into speech segments
 * cut at pauses. Short phrases wait for a longer pause, long speech is cut sooner,
 * so translations keep coming while someone talks.
 */
export class Segmenter {
  constructor({ onSegment = () => {}, onSpeech = () => {}, options = {} } = {}) {
    this.o = { ...VAD, ...options };
    this.onSegment = onSegment; // (Float32Array samples, {forced})
    this.onSpeech = onSpeech; // (speaking: boolean)
    this.levels = [];
    this.preroll = [];
    this.segment = null; // {frames, speech} while someone is speaking
    this.speechRun = 0;
    this.silenceMs = 0;
    this.level = 0;
    this.holdMs = 0; // while > 0, wait for this much silence before starting a segment
    this.quietMs = 0;
  }

  /** After a gap in the audio: skip the rest of a phrase that was already under way. */
  requireSilence(ms) {
    this.holdMs = ms;
    this.quietMs = 0;
    this.preroll = [];
    this.speechRun = 0;
  }

  get speaking() {
    return this.segment !== null;
  }

  get noise() {
    return this.levels.length ? Math.min(...this.levels) : 0;
  }

  threshold() {
    return Math.max(this.noise * this.o.noiseFactor, this.o.minThreshold);
  }

  endSilenceMs() {
    // a short word ("Thank you.") usually continues: give it longer so it isn't sent alone.
    // Measured up to the last speech, so the pause itself doesn't make the phrase look long.
    const ms = (this.segment.speech.lastIndexOf(true) + 1) * FRAME_MS;
    return ms < 1500 ? 1100 : ms < 4000 ? 700 : ms < 8000 ? 400 : 250;
  }

  push(frame) {
    const level = rms(frame);
    const speech = level > this.threshold();
    this.level = level;
    this.levels.push(level);
    if (this.levels.length > this.o.noiseWindowMs / FRAME_MS) this.levels.shift();

    if (!this.segment && this.holdMs > 0) {
      this.quietMs = speech ? 0 : this.quietMs + FRAME_MS;
      if (this.quietMs >= this.holdMs) this.holdMs = 0;
      return;
    }

    if (!this.segment) {
      this.preroll.push({ frame, speech });
      if (this.preroll.length > this.o.prerollMs / FRAME_MS) this.preroll.shift();
      this.speechRun = speech ? this.speechRun + FRAME_MS : 0;
      if (this.speechRun >= this.o.startMs) {
        this.segment = {
          frames: this.preroll.map((p) => p.frame),
          speech: this.preroll.map((p) => p.speech),
        };
        this.preroll = [];
        this.silenceMs = 0;
        this.onSpeech(true);
      }
      return;
    }

    this.segment.frames.push(frame);
    this.segment.speech.push(speech);
    this.silenceMs = speech ? 0 : this.silenceMs + FRAME_MS;
    if (this.silenceMs >= this.endSilenceMs()) this.finish();
    else if (this.segment.frames.length * FRAME_MS >= this.o.maxMs) this.splitLong();
  }

  /** Speaker paused: emit the segment and wait for new speech. */
  finish() {
    const { frames, speech } = this.segment;
    const lastSpeech = speech.lastIndexOf(true);
    const end = Math.min(frames.length, lastSpeech + 1 + Math.round(this.o.tailMs / FRAME_MS));
    this.segment = null;
    this.speechRun = 0;
    this.onSpeech(false);
    this.emit(frames.slice(0, end), speech.slice(0, end), false);
  }

  /** Non-stop speech: cut at the quietest recent frame and keep the rest for the next segment. */
  splitLong() {
    const { frames, speech } = this.segment;
    const from = Math.max(1, frames.length - Math.round(this.o.cutSearchMs / FRAME_MS));
    let cut = frames.length;
    let quietest = Infinity;
    for (let i = from; i < frames.length; i++) {
      const level = rms(frames[i]);
      if (level < quietest) {
        quietest = level;
        cut = i;
      }
    }
    this.segment = { frames: frames.slice(cut), speech: speech.slice(cut) };
    this.silenceMs = 0;
    this.emit(frames.slice(0, cut), speech.slice(0, cut), true);
  }

  emit(frames, speech, forced) {
    const voicedMs = speech.filter(Boolean).length * FRAME_MS;
    if (voicedMs >= this.o.minVoicedMs) this.onSegment(concat(frames), { forced });
  }

  /** Stop listening: send whatever is being said right now. */
  flush() {
    if (this.segment) this.finish();
    this.preroll = [];
  }
}

export function concat(frames) {
  const out = new Float32Array(frames.reduce((n, f) => n + f.length, 0));
  let offset = 0;
  for (const f of frames) {
    out.set(f, offset);
    offset += f.length;
  }
  return out;
}

/** 16-bit PCM mono WAV file. */
export function encodeWav(samples, sampleRate = SAMPLE_RATE) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

/**
 * WebSocket frame: 4-byte big-endian segment id, 8-byte big-endian age (seconds since the phrase
 * ended, taken at send time), WAV bytes. An age instead of a timestamp: the device clock may be off.
 */
export function packSegment(id, wav, age = 0) {
  const out = new Uint8Array(12 + wav.byteLength);
  const view = new DataView(out.buffer);
  view.setUint32(0, id, false);
  view.setFloat64(4, age, false);
  out.set(new Uint8Array(wav), 12);
  return out;
}
