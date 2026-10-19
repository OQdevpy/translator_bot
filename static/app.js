import { Downsampler, SAMPLE_RATE, Segmenter, encodeWav } from "./audio.js";
import {
  Outbox,
  ReadQueue,
  captureStalled,
  findPendingReport,
  makeClientId,
  planResync,
  pollReport,
  tail,
  unionUnits,
} from "./live.js";

const $ = (id) => document.getElementById(id);
const ui = {
  sessionsBtn: $("sessions-btn"),
  title: $("session-title"),
  status: $("status"),
  source: $("source"),
  partner: $("partner"),
  showSrc: $("show-src"),
  tts: $("tts"),
  ttsLabel: $("tts-label"),
  context: $("context"),
  contextSummary: $("context-summary"),
  contextTerms: $("context-terms"),
  warning: $("warning"),
  transcript: $("transcript"),
  placeholder: $("placeholder"),
  live: $("live"),
  liveWho: $("live-who"),
  liveText: $("live-text"),
  liveBusy: $("live-busy"),
  toggle: $("toggle"),
  level: $("level"),
  hint: $("hint"),
  copy: $("copy"),
  reportBtn: $("report-btn"),
  logout: $("logout"),
  sessionsDialog: $("sessions-dialog"),
  sessionsList: $("sessions-list"),
  newSession: $("new-session"),
  newTitle: $("new-title"),
  reportDialog: $("report-dialog"),
  reportStart: $("report-start"),
  reportEnd: $("report-end"),
  reportMake: $("report-make"),
  reportStatus: $("report-status"),
  reportsList: $("reports-list"),
  speakerDialog: $("speaker-dialog"),
  speakerName: $("speaker-name"),
  speakerRename: $("speaker-rename"),
  speakerMerge: $("speaker-merge"),
  mergeInto: $("merge-into"),
};

const store = {
  get: (key) => {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set: (key, value) => {
    try { localStorage.setItem(key, value); } catch {}
  },
};

const COLORS = ["#2563eb", "#db2777", "#059669", "#d97706", "#7c3aed", "#0891b2", "#dc2626", "#4d7c0f"];
const MAX_TURNS_ON_PAGE = 300; // older parts stay in the report
const MAX_UNITS_KEPT = 1000; // units remembered for redraws and duplicate checks
const SAME_TURN_GAP_S = 300; // a speaker's units closer than this join one block
const LIVE_TEXT_CHARS = 240; // the live strip shows only the newest words
const PING_MS = 10000;
const PONG_TIMEOUT_MS = 25000; // no answer for this long: the socket is dead even if it still says OPEN
const CLIENT_ID = makeClientId();
const SILENCE = URL.createObjectURL(new Blob([encodeWav(new Float32Array(1600))], { type: "audio/wav" })); // 0.1 s
const TAP_HINT = "🔊 Ovozli o'qish ishlashi uchun sahifaga bir marta bosing.";
const STALL_WARNING =
  "⚠️ Mikrofondan ovoz kelmayapti (qo'ng'iroq, Siri yoki ekran qulfi bo'lishi mumkin). " +
  "Sahifani ochib qo'ying; tiklanmasa ■ To'xtatish, so'ng ▶ Boshlash tugmasini bosing.";

let config = null;
let session = null; // {id, title, created_at, partner, notes, reports, units_on_page}
const speakers = new Map(); // id -> {id, label, voice}
const shown = new Map(); // unit id -> unit, in page order
let lastTurn = null; // {el, speaker, end}
let ws = null;
let reconnectDelay = 1000;
let pingSentAt = null; // oldest unanswered ping on the current socket
let capture = null; // {ctx, stream, segmenter, paused, hearsOutput, heardAt, stalled, watchdog}
let seq = 0;
const outbox = new Outbox(); // phrases not acknowledged yet: resent after a reconnect
let flushOwed = null; // seminar whose stop could not send its flush while offline
let syncGen = 0; // only the newest re-sync applies its data
let syncing = null; // what live messages changed while that re-sync was fetching
let rebuildNext = false; // a requested redraw survives a re-sync that failed or was superseded
let choosing = null; // the seminar pick in progress
let deleting = null; // seminar this page is deleting: its 4404 is expected
let reportWait = null; // {id, sessionId}: the report whose progress is shown
const player = new Audio();
const readQueue = new ReadQueue();
let speaking = false;
let audioBlocked = false; // the browser refused playback: no clips are fetched until a tap
let unlockSeq = 0;
let wakeLock = null;
let editingSpeaker = null;

// ---------- api ----------

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (res.status === 401) {
    location.reload();
    throw new Error("login required");
  }
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// ---------- session ----------

async function openSession(id) {
  session = { id };
  outbox.keepOnly(id);
  speakers.clear();
  renderUnits([], false);
  showNotes(null);
  showPending([], 0);
  updateHint();
  // socket first, then the fetch: a unit stored in between still arrives live
  connect();
  await resync(true);
  store.set("session", String(id));
}

/**
 * Brings the page up to date: broadcasts reach only the sockets connected at that moment,
 * so after a reconnect (screen lock, Wi-Fi drop, server restart) whatever was said meanwhile is fetched.
 */
async function resync(rebuild = false) {
  const id = session.id;
  const gen = ++syncGen;
  const seen = (syncing = { pending: false, notes: false, speakers: new Map() });
  rebuildNext ||= rebuild;
  let data;
  try {
    data = await api(`/api/sessions/${id}`);
  } finally {
    if (syncing === seen) syncing = null;
  }
  if (gen !== syncGen || session.id !== id) return; // a newer re-sync or another seminar took over
  const { units, ...rest } = data;
  session = { ...session, ...rest };
  ui.title.textContent = data.title;
  document.title = `${data.title} · Jonli tarjimon`;
  ui.partner.value = data.partner;
  speakers.clear();
  data.speakers.forEach((sp) => speakers.set(sp.id, sp));
  seen.speakers.forEach((sp) => speakers.set(sp.id, sp)); // arrived after the fetch read them
  const plan = planResync([...shown.values()], units);
  if (rebuildNext || plan.rebuild) {
    rebuildNext = false;
    renderUnits(unionUnits(units, [...shown.values()], data.units_on_page), units.length >= data.units_on_page);
  } else {
    plan.missing.forEach((u) => addUnit(u, false));
  }
  relabelAll();
  if (!seen.notes) showNotes(data.notes);
  if (!seen.pending) showPending(data.pending, data.interpreting ?? 0);
}

function chooseInitialSession() {
  // the 4404 of a deleted seminar and its caller may both ask: only one pick may run
  choosing ??= pickSession().finally(() => (choosing = null));
  return choosing;
}

async function pickSession() {
  const saved = Number(store.get("session"));
  if (saved) {
    try {
      await openSession(saved);
      return;
    } catch {}
  }
  const list = await api("/api/sessions");
  const today = new Date().toDateString();
  const recent = list.find((s) => new Date(s.created_at * 1000).toDateString() === today);
  await openSession(recent ? recent.id : (await api("/api/sessions", { method: "POST", body: {} })).id);
}

function showNotes(notes) {
  const terms = notes?.glossary || [];
  ui.context.hidden = !notes?.summary && !terms.length;
  ui.contextSummary.textContent = notes?.summary || "";
  ui.contextTerms.replaceChildren(
    ...terms.map((t) => {
      const li = document.createElement("li");
      const b = document.createElement("b");
      b.textContent = t.term;
      li.append(b, ` — ${t.translation}${t.note ? ` (${t.note})` : ""}`);
      return li;
    }),
  );
}

// ---------- connection ----------

function setStatus(state, text) {
  ui.status.dataset.state = state;
  ui.status.textContent = text;
}

function connect() {
  if (ws) {
    ws.onclose = ws.onmessage = null;
    ws.close();
  }
  const sessionId = session.id;
  setStatus("connecting", "Ulanmoqda…");
  const socket = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?session=${sessionId}&client=${CLIENT_ID}`,
  );
  ws = socket;
  pingSentAt = null;
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    reconnectDelay = 1000;
    setStatus("online", "Ulangan");
    const now = Date.now();
    // the server answers "dup" for phrases it already has
    outbox.replay(sessionId, now).forEach((item) => socket.send(outbox.frame(item, now)));
    if (capture?.segmenter.speaking) sendJson({ type: "speech", active: true }, socket);
    if (flushOwed === sessionId && !capture) sendJson({ type: "flush" }, socket);
    flushOwed = null;
    updateHint();
    const sync = () =>
      resync().catch(() => setTimeout(() => ws === socket && socket.readyState === WebSocket.OPEN && sync(), 3000));
    sync();
  };
  socket.onmessage = (event) => {
    if (ws !== socket) return;
    pingSentAt = null; // anything from the server proves the socket is alive
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    handleMessage(msg);
  };
  socket.onclose = (event) => onClosed(socket, sessionId, event.code);
}

function onClosed(socket, sessionId, code) {
  if (ws !== socket) return;
  setStatus("offline", "Aloqa yo'q");
  if (code === 4401) {
    location.reload(); // session expired: show the login page
    return;
  }
  if (code === 4403) {
    showWarning(
      "Server ulanishni rad etdi: Origin va Host mos emas. nginx bo'lsa, proxy_set_header Host $http_host; qatorini qo'shing.",
    );
    return;
  }
  if (code === 4404) {
    if (sessionId === deleting) return; // this page deleted it: the delete handler picks the next seminar
    store.set("session", "");
    chooseInitialSession().catch((err) => showWarning(err.message));
    return;
  }
  setTimeout(() => {
    if (ws === socket) connect();
  }, reconnectDelay);
  reconnectDelay = Math.min(reconnectDelay * 2, 15000);
}

function heartbeat() {
  const socket = ws;
  if (socket?.readyState !== WebSocket.OPEN) return;
  if (pingSentAt !== null && Date.now() - pingSentAt > PONG_TIMEOUT_MS) {
    // Wi-Fi gone without a close: phrases would keep vanishing into this socket
    socket.onclose = socket.onmessage = null;
    socket.close();
    onClosed(socket, session.id, 1006);
    return;
  }
  pingSentAt ??= Date.now();
  socket.send(JSON.stringify({ type: "ping" }));
}

setInterval(heartbeat, PING_MS);

function sendJson(msg, socket = ws) {
  if (socket?.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(msg));
  return true;
}

function handleMessage(msg) {
  switch (msg.type) {
    case "heard":
    case "empty":
    case "dup":
      outbox.ack(msg.id);
      updateHint();
      break;
    case "pending":
      if (syncing) syncing.pending = true;
      showPending(msg.pending, msg.interpreting);
      break;
    case "unit":
      addUnit(msg.unit, true);
      break;
    case "speaker":
      syncing?.speakers.set(msg.speaker.id, msg.speaker);
      speakers.set(msg.speaker.id, msg.speaker);
      relabel(msg.speaker.id);
      break;
    case "merged":
      resync(true).catch(() => {}); // simplest way to regroup the blocks; a failed one is redone on the next sync
      break;
    case "notes":
      if (syncing) syncing.notes = true;
      showNotes(msg.notes);
      break;
    case "error":
      if (msg.id != null) outbox.ack(msg.id);
      updateHint();
      flashWarning(msg.message);
      break;
  }
}

// ---------- transcript ----------

function speakerIndex(id) {
  const n = parseInt(String(id).slice(1), 10);
  return Number.isFinite(n) ? n - 1 : 0;
}

function colorOf(id) {
  return COLORS[speakerIndex(id) % COLORS.length];
}

function labelOf(id) {
  return speakers.get(id)?.label || id;
}

function flagOf(code) {
  return config.langs[code]?.flag ?? "🌐";
}

function clock(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function nearBottom() {
  const t = ui.transcript;
  return t.scrollHeight - t.scrollTop - t.clientHeight < 160;
}

function newTurn(unit) {
  const turn = el("section", "turn");
  turn.dataset.speaker = unit.speaker;
  turn.style.setProperty("--c", colorOf(unit.speaker));
  const head = el("header");
  const who = el("button", "who", labelOf(unit.speaker));
  who.type = "button";
  who.addEventListener("click", () => editSpeaker(turn.dataset.speaker));
  head.append(who, el("time", null, clock(unit.start)));
  turn.append(head);
  ui.transcript.append(turn);
  const turns = ui.transcript.querySelectorAll(".turn");
  for (let i = 0; i < turns.length - MAX_TURNS_ON_PAGE; i++) turns[i].remove();
  return turn;
}

function addUnit(unit, live) {
  if (shown.has(unit.id)) return;
  shown.set(unit.id, unit);
  if (shown.size > MAX_UNITS_KEPT) shown.delete(shown.keys().next().value);
  const stick = nearBottom();
  if (!lastTurn || lastTurn.speaker !== unit.speaker || unit.start - lastTurn.end > SAME_TURN_GAP_S) {
    lastTurn = { el: newTurn(unit), speaker: unit.speaker, end: unit.end };
  }
  lastTurn.end = unit.end;
  const block = el("div", unit.translation ? "unit" : "unit untranslated");
  block.dataset.translation = unit.translation || unit.source;
  block.append(el("p", "tr", unit.translation || unit.source));
  if (!unit.translation) block.append(el("p", "note", "⚠️ Tarjima qilinmadi, asl matn ko'rsatilmoqda"));
  if (unit.note) block.append(el("p", "note", `ℹ️ ${unit.note}`));
  if (unit.translation && unit.lang !== unit.target) block.append(el("p", "src", `${flagOf(unit.lang)} ${unit.source}`));
  lastTurn.el.append(block);
  updatePlaceholder();
  if (stick) ui.transcript.scrollTop = ui.transcript.scrollHeight;
  if (live && ui.tts.checked && config.tts && unit.translation) speak(unit.translation);
}

/** Draws the transcript from scratch (opening a seminar, a merge, or units missed out of order). */
function renderUnits(units, earlier) {
  const stick = shown.size === 0 || nearBottom();
  shown.clear();
  lastTurn = null;
  ui.transcript.querySelectorAll(".turn, .earlier").forEach((node) => node.remove());
  // long seminars: only the newest part is on the page, all of it is in the reports
  if (earlier) ui.transcript.prepend(el("p", "earlier muted", "… avvalgi qismlar 📄 hisobotda"));
  units.forEach((u) => addUnit(u, false));
  updatePlaceholder();
  if (stick) ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

function relabel(id) {
  ui.transcript.querySelectorAll(`.turn[data-speaker="${CSS.escape(id)}"] .who`).forEach((b) => (b.textContent = labelOf(id)));
  if (ui.liveWho.dataset.speaker === id) ui.liveWho.textContent = labelOf(id);
}

function relabelAll() {
  ui.transcript.querySelectorAll(".turn").forEach((turn) => (turn.querySelector(".who").textContent = labelOf(turn.dataset.speaker)));
  if (ui.liveWho.dataset.speaker) ui.liveWho.textContent = labelOf(ui.liveWho.dataset.speaker);
}

function showPending(pending, interpreting) {
  const last = pending[pending.length - 1];
  ui.live.hidden = !last && !interpreting;
  if (last) {
    ui.liveWho.dataset.speaker = last.speaker;
    ui.liveWho.textContent = labelOf(last.speaker);
    ui.liveWho.style.setProperty("--c", colorOf(last.speaker));
    const text = pending.filter((p) => p.speaker === last.speaker).map((p) => p.text).join(" ");
    // what is being said now: on a phone only a couple of lines fit, so show the newest words
    ui.liveText.replaceChildren(el("span", null, `${flagOf(last.lang)} ${tail(text, LIVE_TEXT_CHARS)}`));
  } else {
    delete ui.liveWho.dataset.speaker;
    ui.liveWho.textContent = "";
    ui.liveText.replaceChildren();
  }
  ui.liveBusy.hidden = !interpreting;
}

function updatePlaceholder() {
  ui.placeholder.hidden = ui.transcript.querySelector(".turn") !== null;
}

function updateHint() {
  if (!capture || capture.paused) return;
  if (capture.stalled) ui.hint.textContent = "⚠️ Ovoz kelmayapti";
  else ui.hint.textContent = outbox.size ? `👂 Eshitilmoqda (${outbox.size})` : "Tinglanmoqda…";
}

function showWarning(text) {
  ui.warning.textContent = text;
  ui.warning.hidden = !text;
}

/** Clears the warning only if it still shows this text: another one may have replaced it. */
function clearWarning(text) {
  if (ui.warning.textContent === text) showWarning("");
}

function flashWarning(text) {
  showWarning(text);
  setTimeout(() => clearWarning(text), 6000);
}

// ---------- speakers ----------

function editSpeaker(id) {
  editingSpeaker = id;
  ui.speakerName.value = labelOf(id);
  ui.mergeInto.replaceChildren(
    ...[...speakers.values()].filter((sp) => sp.id !== id).map((sp) => new Option(`→ ${sp.label}`, sp.id)),
  );
  ui.speakerMerge.hidden = ui.mergeInto.options.length === 0;
  ui.speakerDialog.showModal();
}

ui.speakerRename.addEventListener("submit", async (event) => {
  event.preventDefault();
  const label = ui.speakerName.value.trim();
  if (!label) return;
  try {
    await api(`/api/sessions/${session.id}/speakers/${editingSpeaker}`, { method: "PATCH", body: { label } });
    speakers.set(editingSpeaker, { ...speakers.get(editingSpeaker), label });
    relabel(editingSpeaker);
    ui.speakerDialog.close();
  } catch (err) {
    showWarning(err.message);
  }
});

ui.speakerMerge.addEventListener("submit", async (event) => {
  event.preventDefault();
  const into = ui.mergeInto.value;
  if (!into || !confirm(`${labelOf(editingSpeaker)} → ${labelOf(into)}: birlashtirilsinmi?`)) return;
  try {
    await api(`/api/sessions/${session.id}/speakers/${editingSpeaker}/merge`, { method: "POST", body: { into } });
    ui.speakerDialog.close();
    await resync(true);
  } catch (err) {
    showWarning(err.message);
  }
});

// ---------- sessions dialog ----------

function duration(first, last) {
  if (!first) return "bo'sh";
  const minutes = Math.round((last - first) / 60);
  return minutes >= 60 ? `${Math.floor(minutes / 60)} soat ${minutes % 60} daq` : `${minutes} daq`;
}

async function switchSession(id) {
  if (capture) stop();
  try {
    await openSession(id);
  } catch (err) {
    showWarning(err.message);
  }
}

async function showSessions() {
  const list = await api("/api/sessions");
  ui.sessionsList.replaceChildren(
    ...list.map((s) => {
      const li = el("li", s.id === session?.id ? "current" : "");
      const open = el("button", "item");
      open.type = "button";
      open.append(
        el("b", null, s.title),
        el("span", "muted", `${new Date(s.created_at * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" })} · ${s.units} gap · ${duration(s.first, s.last)}`),
      );
      open.addEventListener("click", () => {
        ui.sessionsDialog.close();
        switchSession(s.id);
      });
      const rename = el("button", "icon small", "✏️");
      rename.type = "button";
      rename.title = "Nomini o'zgartirish";
      rename.addEventListener("click", async () => {
        const title = prompt("Seminar nomi", s.title)?.trim();
        if (!title) return;
        await api(`/api/sessions/${s.id}`, { method: "PATCH", body: { title } });
        if (s.id === session.id) ui.title.textContent = session.title = title;
        showSessions();
      });
      const del = el("button", "icon small", "🗑");
      del.type = "button";
      del.title = "O'chirish";
      del.addEventListener("click", () => deleteSession(s));
      li.append(open, rename, del);
      return li;
    }),
  );
  if (!ui.sessionsDialog.open) ui.sessionsDialog.showModal();
}

async function deleteSession(s) {
  if (!confirm(`«${s.title}» va uning barcha tarjimalari, hisobotlari o'chirilsinmi?`)) return;
  const current = s.id === session.id;
  if (current) deleting = s.id; // the server closes our socket with 4404 before it answers
  try {
    await api(`/api/sessions/${s.id}`, { method: "DELETE" });
  } catch (err) {
    deleting = null;
    if (current && ws?.readyState === WebSocket.CLOSED) connect();
    showWarning(err.message);
    return;
  }
  if (!current) {
    showSessions();
    return;
  }
  if (capture) stop();
  store.set("session", "");
  ui.sessionsDialog.close();
  try {
    await chooseInitialSession();
  } catch (err) {
    showWarning(err.message);
  } finally {
    deleting = null;
  }
}

ui.sessionsBtn.addEventListener("click", () => showSessions().catch((err) => showWarning(err.message)));

ui.newSession.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const created = await api("/api/sessions", { method: "POST", body: { title: ui.newTitle.value } });
    ui.newTitle.value = "";
    ui.sessionsDialog.close();
    await switchSession(created.id);
  } catch (err) {
    showWarning(err.message);
  }
});

// ---------- reports ----------

function toLocalInput(ts) {
  const d = new Date(ts * 1000);
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 16);
}

function fromLocalInput(value) {
  return value ? new Date(value).getTime() / 1000 : null;
}

function sessionSpan() {
  // the page shows only the newest units, so the whole seminar starts at its creation
  return { first: session.created_at, last: Date.now() / 1000 };
}

function setRange(preset) {
  const { first, last } = sessionSpan();
  const start = preset === "all" ? first : Math.max(first, last - Number(preset) * 60);
  ui.reportStart.value = toLocalInput(Math.floor(start / 60) * 60);
  ui.reportEnd.value = toLocalInput(Math.ceil(last / 60) * 60);
}

function renderReports(reports) {
  ui.reportsList.replaceChildren(
    ...reports.map((r) => {
      const li = el("li");
      const icon = r.status === "pending" ? "⏳" : r.status === "failed" ? "⚠️" : "📄";
      const a = el("a", "item", `${icon} ${clock(r.start)}–${clock(r.end)} · ${new Date(r.created_at * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}`);
      a.href = `/api/reports/${r.id}`;
      a.target = "_blank";
      const md = el("a", "icon small", "MD");
      md.href = `/api/reports/${r.id}.md`;
      md.title = "Markdown yuklab olish";
      md.setAttribute("download", "");
      li.append(a);
      if (r.status === "done") li.append(md);
      return li;
    }),
  );
  if (!reports.length) ui.reportsList.append(el("li", "muted", "Hali hisobot yo'q"));
}

async function refreshReports(sessionId) {
  try {
    const data = await api(`/api/sessions/${sessionId}`);
    if (session.id !== sessionId) return;
    session.reports = data.reports;
    renderReports(data.reports);
  } catch {}
}

/** Shows a report's progress until it is written (a whole day can take a minute or two). */
async function followReport(id, sessionId, text = "⏳ Hisobot tayyorlanmoqda… (uzun seminar uchun 1–2 daqiqa)") {
  ui.reportStatus.hidden = false;
  ui.reportStatus.textContent = text;
  if (reportWait?.id === id) return; // already being followed: that loop shows the result
  const job = (reportWait = { id, sessionId });
  const current = () => reportWait === job && session.id === sessionId;
  try {
    const status = await pollReport(() => api(`/api/reports/${id}/status`), {
      sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    });
    if (status.status === "failed") throw new Error(status.error || "Hisobot tayyorlab bo'lmadi");
    if (current()) {
      ui.reportStatus.textContent = "";
      const link = el("a", null, "✅ Tayyor — hisobotni ochish");
      link.href = `/api/reports/${id}`;
      link.target = "_blank";
      ui.reportStatus.append(link);
    }
  } catch (err) {
    if (current()) ui.reportStatus.textContent = `⚠️ ${err.message}`;
  } finally {
    if (reportWait === job) reportWait = null;
    refreshReports(sessionId);
  }
}

ui.reportBtn.addEventListener("click", async () => {
  try {
    const { units, ...rest } = await api(`/api/sessions/${session.id}`);
    session = { ...session, ...rest };
  } catch {}
  setRange("all");
  renderReports(session.reports || []);
  if (reportWait?.sessionId !== session.id) {
    ui.reportStatus.hidden = true;
    // a report still being written (e.g. started before a reload): show its progress, not an empty form
    const running = (session.reports || []).find((r) => r.status === "pending");
    if (running) followReport(running.id, session.id);
  }
  ui.reportDialog.showModal();
});

ui.reportDialog.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => setRange(b.dataset.preset)));

ui.reportMake.addEventListener("click", async () => {
  const start = fromLocalInput(ui.reportStart.value);
  const end = fromLocalInput(ui.reportEnd.value);
  ui.reportStatus.hidden = false;
  if (start == null || end == null || start >= end) {
    ui.reportStatus.textContent = "Vaqt oralig'ini to'g'ri tanlang";
    return;
  }
  const sessionId = session.id;
  // the same expensive report is already being written: follow it instead of starting a second one
  const running = findPendingReport(session.reports || [], start, end);
  if (running) {
    followReport(running.id, sessionId, "⏳ Bu oraliq uchun hisobot allaqachon tayyorlanmoqda…");
    return;
  }
  ui.reportMake.disabled = true;
  ui.reportStatus.textContent = "⏳ Hisobot tayyorlanmoqda… (uzun seminar uchun 1–2 daqiqa)";
  try {
    const { id } = await api(`/api/sessions/${sessionId}/reports`, { method: "POST", body: { start, end } });
    if (session.id === sessionId) {
      session.reports = [{ id, start, end, status: "pending", created_at: Date.now() / 1000 }, ...(session.reports || [])];
      renderReports(session.reports);
    }
    followReport(id, sessionId);
  } catch (err) {
    ui.reportStatus.textContent = `⚠️ ${err.message}`;
  } finally {
    ui.reportMake.disabled = false;
  }
});

// ---------- audio capture ----------

function onSegment(samples) {
  const item = outbox.add({ id: ++seq, sessionId: session?.id, wav: encodeWav(samples, SAMPLE_RATE), endedAt: Date.now() });
  // kept until the server acknowledges it: a socket can die with the phrase still in its buffer
  if (ws?.readyState === WebSocket.OPEN) ws.send(outbox.frame(item, Date.now()));
  updateHint();
}

function onSpeech(active) {
  ui.level.classList.toggle("speaking", active);
  // while someone is talking the server waits with interpreting an unfinished thought
  sendJson({ type: "speech", active });
}

async function openStream() {
  if (ui.source.value === "tab") {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: true,
      selfBrowserSurface: "exclude", // sharing this tab would include our own read-out
    });
    if (!stream.getAudioTracks().length) {
      stream.getTracks().forEach((t) => t.stop());
      throw new Error("Ovoz ulashilmadi: oynada «Tab ovozini ulashish» (Share tab audio) belgisini yoqing.");
    }
    return stream;
  }
  return navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
}

async function start() {
  ui.toggle.disabled = true;
  showWarning("");
  // Created inside the click so mobile browsers allow audio.
  const ctx = new AudioContext();
  unlockAudio();
  let stream = null;
  try {
    if (!ctx.audioWorklet) throw new Error("Brauzer eskirgan: AudioWorklet yo'q. Chrome, Edge, Firefox yoki Safari 14.5+ dan foydalaning.");
    stream = await openStream();
    await ctx.audioWorklet.addModule("/static/pcm-worklet.js");
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "capture");
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);

    const segmenter = new Segmenter({ onSegment, onSpeech });
    // A captured browser tab contains only that tab's audio, so our read-out can't leak into it.
    const surface = stream.getVideoTracks()[0]?.getSettings().displaySurface;
    const hearsOutput = ui.source.value !== "tab" || surface !== "browser";
    const c = { ctx, stream, segmenter, paused: false, hearsOutput, heardAt: performance.now(), stalled: false, watchdog: null };
    capture = c;
    const down = new Downsampler(ctx.sampleRate, (frame) => {
      if (capture === c && !c.paused) segmenter.push(frame);
    });
    node.port.onmessage = (event) => {
      c.heardAt = performance.now();
      down.push(event.data);
    };
    stream.getTracks().forEach((t) => t.addEventListener("ended", stop));
    // iOS suspends capture for a call, Siri or a locked screen and may not resume it by itself
    stream.getAudioTracks().forEach((t) => {
      t.addEventListener("mute", () => checkCapture(c));
      t.addEventListener("unmute", () => checkCapture(c));
    });
    await ctx.resume();
    c.heardAt = performance.now();
    ctx.onstatechange = () => checkCapture(c);
    c.watchdog = setInterval(() => checkCapture(c), 1000);

    ui.toggle.dataset.state = "running";
    ui.toggle.textContent = "■ To'xtatish";
    ui.source.disabled = true;
    updateHint();
    keepAwake();
    drawLevel();
  } catch (err) {
    stream?.getTracks().forEach((t) => t.stop());
    ctx.close();
    capture = null;
    showWarning(micError(err));
  } finally {
    ui.toggle.disabled = false;
  }
}

function stop() {
  if (!capture) return;
  const c = capture;
  const socket = ws; // the seminar being stopped, even if the page switches to another one right after
  c.segmenter.flush();
  capture = null;
  clearInterval(c.watchdog);
  c.stream.getTracks().forEach((t) => t.stop());
  c.ctx.close();
  clearWarning(STALL_WARNING);
  wakeLock?.release().catch(() => {});
  wakeLock = null;
  // interpret what is left: the server waits until the phrases sent so far are processed
  if (!sendJson({ type: "flush" }, socket)) flushOwed = session.id;
  ui.toggle.dataset.state = "idle";
  ui.toggle.textContent = "▶ Boshlash";
  ui.hint.textContent = "Tayyor";
  ui.source.disabled = false;
  ui.level.style.width = "0";
  ui.level.classList.remove("speaking");
}

/** Notices capture that stopped without an "ended" event, and keeps trying to resume it. */
function checkCapture(c) {
  if (capture !== c) return;
  if (c.ctx.state !== "running") c.ctx.resume().catch(() => {});
  if (c.paused) return; // muted for our read-out: no phrases expected
  const stalled = captureStalled({
    state: c.ctx.state,
    muted: c.stream.getAudioTracks()[0]?.muted,
    silentMs: performance.now() - c.heardAt,
  });
  if (stalled === c.stalled) return;
  c.stalled = stalled;
  if (stalled) {
    // end the phrase in progress: the server must not wait for speech that is not coming,
    // and the audio that comes back may start mid-word
    c.segmenter.flush();
    c.segmenter.requireSilence(300);
    showWarning(STALL_WARNING);
  } else {
    clearWarning(STALL_WARNING);
  }
  updateHint();
}

function micError(err) {
  if (err?.name === "NotAllowedError") return "Mikrofonga ruxsat berilmadi. Brauzer sozlamalaridan ruxsat bering.";
  if (err?.name === "NotFoundError") return "Mikrofon topilmadi.";
  return err?.message || String(err);
}

function drawLevel() {
  if (!capture) return;
  const level = capture.paused || capture.stalled ? 0 : capture.segmenter.level;
  const db = 20 * Math.log10(level + 1e-9);
  ui.level.style.width = `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;
  requestAnimationFrame(drawLevel);
}

async function keepAwake() {
  try {
    wakeLock = await navigator.wakeLock?.request("screen");
  } catch {}
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;
  if (capture) {
    keepAwake();
    if (capture.ctx.state !== "running") capture.ctx.resume().catch(() => {});
  }
  // back from a locked screen: reconnect now instead of waiting out the backoff,
  // and probe a socket that may only look open
  if (session && ws?.readyState === WebSocket.CLOSED) connect();
  else heartbeat();
});

window.addEventListener("beforeunload", (event) => {
  // a pull-to-refresh or a stray swipe would silently end the recording
  if (!capture) return;
  event.preventDefault();
  event.returnValue = "";
});

// ---------- voice output ----------

/** Phones start audio only inside a tap: play a silent clip and remember whether the browser allowed it. */
async function unlockAudio() {
  if (!player.paused) return; // already playing, so already allowed
  const n = ++unlockSeq;
  player.src = SILENCE;
  let ok = true;
  try {
    await player.play();
  } catch (err) {
    ok = err?.name !== "NotAllowedError";
  }
  if (n !== unlockSeq) return; // a newer attempt (the next event of the same tap) decides
  if (ok) {
    audioBlocked = false;
    clearWarning(TAP_HINT);
  } else if (ui.tts.checked) {
    blockAudio();
  }
}

function blockAudio() {
  // don't synthesize (and pay for) clips nobody can hear until the page is tapped
  audioBlocked = true;
  readQueue.clear();
  showWarning(TAP_HINT);
}

function onGesture() {
  if (capture && capture.ctx.state !== "running") capture.ctx.resume().catch(() => {});
  if (audioBlocked && ui.tts.checked) unlockAudio();
}

// which event counts as a user gesture differs between mouse, touch and browsers
["pointerdown", "pointerup", "click", "keydown"].forEach((type) => document.addEventListener(type, onGesture, true));

function speak(text) {
  if (audioBlocked) return;
  readQueue.push(text, Date.now());
  if (!speaking) playNext();
}

async function playNext() {
  const text = readQueue.next(Date.now());
  if (text === undefined) {
    speaking = false;
    return;
  }
  speaking = true;
  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`TTS ${res.status}`);
    const url = URL.createObjectURL(await res.blob());
    try {
      if (capture?.hearsOutput) {
        // Let whoever is talking finish, then mute the microphone while we speak
        // so our own voice is not translated again.
        await waitForPause(8000);
        if (capture && ui.tts.checked) {
          capture.segmenter.flush();
          capture.paused = true;
          ui.hint.textContent = "🔊 Tarjima o'qilmoqda, mikrofon pauzada";
        }
      }
      // no early return here: playNext() below must run to clear `speaking`
      if (ui.tts.checked && !audioBlocked) {
        player.src = url;
        await new Promise((resolve) => {
          // "pause" too: switching read-out off stops the clip without an "ended" event
          player.onended = player.onerror = player.onpause = resolve;
          player.play().catch((err) => {
            if (err?.name === "NotAllowedError") blockAudio();
            resolve();
          });
        });
      }
    } finally {
      URL.revokeObjectURL(url);
    }
  } catch (err) {
    console.warn(err);
  } finally {
    setTimeout(() => {
      // not while the next clip is already on its way
      if (capture && !speaking && capture.paused) {
        capture.segmenter.requireSilence(300); // don't send the tail of a phrase we only half heard
        capture.paused = false;
        updateHint();
      }
    }, 300);
  }
  playNext();
}

async function waitForPause(maxMs) {
  const until = performance.now() + maxMs;
  while (capture?.segmenter.speaking && performance.now() < until) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

// ---------- controls ----------

ui.toggle.addEventListener("click", () => (capture ? stop() : start()));

ui.partner.addEventListener("change", async () => {
  try {
    await api(`/api/sessions/${session.id}`, { method: "PATCH", body: { partner: ui.partner.value } });
  } catch (err) {
    showWarning(err.message);
  }
});

ui.showSrc.addEventListener("change", () => {
  document.body.classList.toggle("show-src", ui.showSrc.checked);
  store.set("showSrc", ui.showSrc.checked ? "1" : "0");
});

ui.tts.addEventListener("change", () => {
  store.set("tts", ui.tts.checked ? "1" : "0");
  if (ui.tts.checked) {
    unlockAudio(); // this tap is the moment a phone lets the page start audio
  } else {
    readQueue.clear();
    player.pause();
    clearWarning(TAP_HINT);
  }
});

ui.copy.addEventListener("click", async () => {
  const text = [...ui.transcript.querySelectorAll(".turn")]
    .map((turn) => {
      const who = turn.querySelector(".who").textContent;
      const time = turn.querySelector("time").textContent;
      const lines = [...turn.querySelectorAll(".unit")].map((u) => u.dataset.translation);
      return `[${time}] ${who}:\n${lines.join("\n")}`;
    })
    .join("\n\n");
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    ui.hint.textContent = "📋 Nusxalandi";
  } catch {
    showWarning("Nusxalab bo'lmadi");
  }
});

async function init() {
  config = await api("/api/config");
  ui.partner.add(new Option("O'zbekcha qolsin", "uz"));
  for (const code of config.partners) {
    const { flag, name } = config.langs[code];
    ui.partner.add(new Option(`${flag} ${name}ga tarjima`, code));
  }
  ui.ttsLabel.hidden = !config.tts;
  ui.tts.checked = config.tts && store.get("tts") === "1";
  // restored without a tap: works where autoplay is allowed, otherwise asks for one
  if (ui.tts.checked) unlockAudio();
  ui.showSrc.checked = store.get("showSrc") === "1";
  document.body.classList.toggle("show-src", ui.showSrc.checked);
  ui.logout.hidden = !config.auth;
  if (!navigator.mediaDevices?.getDisplayMedia) ui.source.querySelector('[value="tab"]').disabled = true;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    ui.toggle.disabled = true;
    showWarning(
      "Mikrofon faqat HTTPS orqali ishlaydi. Saytni https:// bilan oching " +
        "(serverda .env ga SSL_SELF_SIGNED=1 yozing yoki domen + sertifikat ulang).",
    );
  }
  await chooseInitialSession();
}

init().catch((err) => showWarning(err.message));
