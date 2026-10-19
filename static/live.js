// Connection and page-state helpers used by the page and by the Node tests: no DOM here.

import { packSegment } from "./audio.js";

export const OUTBOX_MAX = 10; // unacknowledged phrases kept for a resend
export const OUTBOX_MAX_AGE_MS = 60000; // older ones are too late for a live interpretation
export const READ_MAX = 3; // read-out keeps only the newest texts...
export const READ_MAX_AGE_MS = 60000; // ...that are still recent, so it never drifts minutes behind
export const STALL_MS = 3000; // no audio from the worklet for this long: capture stopped

/** Random id of this page load: the server deduplicates resent phrases per (client, phrase id). */
export function makeClientId() {
  const bytes = new Uint8Array(12);
  globalThis.crypto.getRandomValues(bytes);
  return [...bytes].map((b) => b.toString(36).padStart(2, "0")).join("");
}

/**
 * Phrases sent (or still waiting to be sent) that the server has not acknowledged yet.
 * A silently dead socket or a server restart loses whatever was in flight, so every
 * phrase stays here until heard/empty/error/dup for its id arrives, and is resent after a reconnect.
 */
export class Outbox {
  constructor({ max = OUTBOX_MAX, maxAgeMs = OUTBOX_MAX_AGE_MS } = {}) {
    this.max = max;
    this.maxAgeMs = maxAgeMs;
    this.items = []; // {id, sessionId, wav, endedAt (ms)}
  }

  get size() {
    return this.items.length;
  }

  add(item) {
    this.items.push(item);
    if (this.items.length > this.max) this.items.shift();
    return item;
  }

  /** True if the id was ours and still unacknowledged. */
  ack(id) {
    const i = this.items.findIndex((item) => item.id === id);
    if (i < 0) return false;
    this.items.splice(i, 1);
    return true;
  }

  /** Phrases to resend on a new socket: only that seminar's, and only while still recent. */
  replay(sessionId, now) {
    this.items = this.items.filter((item) => now - item.endedAt < this.maxAgeMs);
    return this.items.filter((item) => item.sessionId === sessionId);
  }

  /** Switching seminars: phrases recorded for another one must never land in this one. */
  keepOnly(sessionId) {
    this.items = this.items.filter((item) => item.sessionId === sessionId);
  }

  /** The age is taken at send time, so a resent phrase keeps its place in the timeline. */
  frame(item, now) {
    return packSegment(item.id, item.wav, Math.max(0, (now - item.endedAt) / 1000));
  }
}

export const unitOrder = (a, b) => a.start - b.start || a.id - b.id;

/**
 * What a re-sync after a reconnect must do with the fetched units: the ones the page lacks,
 * and whether they can simply be appended or the transcript must be redrawn (a missed unit
 * belongs before what is shown, or a merge on another device changed shown units' speakers).
 */
export function planResync(shown, fetched) {
  const byId = new Map(shown.map((u) => [u.id, u]));
  const missing = fetched.filter((u) => !byId.has(u.id)).sort(unitOrder);
  const newest = shown.reduce((max, u) => (!max || unitOrder(u, max) > 0 ? u : max), null);
  const rebuild =
    (newest !== null && missing.some((u) => unitOrder(u, newest) < 0)) ||
    fetched.some((u) => byId.has(u.id) && byId.get(u.id).speaker !== u.speaker);
  return { missing, rebuild };
}

/** Units to redraw: the fetched page plus units that arrived live after the fetch read the database. */
export function unionUnits(fetched, shown, limit = Infinity) {
  const last = fetched.reduce((max, u) => Math.max(max, u.id), -Infinity);
  const ids = new Set(fetched.map((u) => u.id));
  const later = shown.filter((u) => u.id > last && !ids.has(u.id));
  const all = [...fetched, ...later].sort(unitOrder);
  return all.slice(Math.max(0, all.length - limit));
}

/** The end of a long text (what is being said now), cut at a word boundary. */
export function tail(text, max) {
  if (text.length <= max) return text;
  const cut = text.slice(-max);
  const space = cut.indexOf(" ");
  return `…${space > 0 && space < max / 4 ? cut.slice(space + 1) : cut}`;
}

/** Read-out queue: drops texts the read-out can no longer catch up with. */
export class ReadQueue {
  constructor({ max = READ_MAX, maxAgeMs = READ_MAX_AGE_MS } = {}) {
    this.max = max;
    this.maxAgeMs = maxAgeMs;
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  push(text, now) {
    this.items.push({ text, at: now });
    if (this.items.length > this.max) this.items.splice(0, this.items.length - this.max);
  }

  /** The next text still worth reading, or undefined. */
  next(now) {
    this.items = this.items.filter((item) => now - item.at < this.maxAgeMs);
    return this.items.shift()?.text;
  }

  clear() {
    this.items = [];
  }
}

/** A report job already running for (about) this range: pressing again must not start a second one. */
export function findPendingReport(reports, start, end) {
  // "Whole seminar" ends at the current minute, so a press a few minutes later is still the same report
  return reports.find((r) => r.status === "pending" && Math.abs(r.start - start) < 60 && Math.abs(r.end - end) < 300) || null;
}

/**
 * Waits for a background report. Network blips, an app switch on a phone or a 5xx are retried
 * with backoff (the job keeps running on the server); only a 404 or a finished status ends it.
 */
export async function pollReport(getStatus, { sleep, interval = 2000, maxDelay = 30000 }) {
  let delay = interval;
  for (;;) {
    await sleep(delay);
    try {
      const status = await getStatus();
      if (status.status !== "pending") return status;
      delay = interval;
    } catch (err) {
      if (err?.status === 404) throw err;
      delay = Math.min(delay * 2, maxDelay);
    }
  }
}

/** iOS stops capture on calls, Siri or screen lock without ending the track. */
export function captureStalled({ state, muted, silentMs }) {
  return state !== "running" || Boolean(muted) || silentMs > STALL_MS;
}
