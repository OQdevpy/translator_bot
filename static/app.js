import { Downsampler, SAMPLE_RATE, Segmenter, encodeWav, packSegment } from "./audio.js";

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
const SAME_TURN_GAP_S = 300; // a speaker's units closer than this join one block
const MAX_REPLAY = 5; // after a reconnect, only the newest unsent phrases are still worth sending
const MAX_REPLAY_AGE_MS = 30000;

let config = null;
let session = null; // {id, title, created_at, partner, notes, units, reports}
const speakers = new Map(); // id -> {id, label, voice}
const shownUnits = new Set();
let lastTurn = null; // {el, speaker, end}
let ws = null;
let reconnectDelay = 1000;
let capture = null; // {ctx, stream, segmenter, paused, hearsOutput}
let seq = 0;
const outbox = []; // phrases recorded while offline: {frame, at}
let listening = 0; // phrases sent and not answered yet
const player = new Audio();
const speechQueue = [];
let speaking = false;
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
    throw new Error(detail);
  }
  return res.json();
}

// ---------- session ----------

async function openSession(id) {
  const data = await api(`/api/sessions/${id}`);
  session = data;
  store.set("session", String(id));
  ui.title.textContent = data.title;
  document.title = `${data.title} · Jonli tarjimon`;
  ui.partner.value = data.partner;
  speakers.clear();
  data.speakers.forEach((sp) => speakers.set(sp.id, sp));
  shownUnits.clear();
  lastTurn = null;
  ui.transcript.querySelectorAll(".turn, .earlier").forEach((el) => el.remove());
  if (data.units.length >= data.units_on_page) {
    // long seminars: only the newest part is on the page, all of it is in the reports
    ui.transcript.prepend(el("p", "earlier muted", "… avvalgi qismlar 📄 hisobotda"));
  }
  data.units.forEach((u) => addUnit(u, false));
  showNotes(data.notes);
  showPending(data.pending, 0);
  updatePlaceholder();
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
  connect();
}

async function chooseInitialSession() {
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
    ws.onclose = null;
    ws.close();
  }
  const sessionId = session.id;
  setStatus("connecting", "Ulanmoqda…");
  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?session=${sessionId}`);
  ws = socket;
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    reconnectDelay = 1000;
    setStatus("online", "Ulangan");
    const fresh = outbox.splice(0).filter((item, i, all) => i >= all.length - MAX_REPLAY && Date.now() - item.at < MAX_REPLAY_AGE_MS);
    fresh.forEach((item) => send(item.frame));
  };
  socket.onmessage = (event) => handleMessage(JSON.parse(event.data));
  socket.onclose = (event) => {
    if (ws !== socket) return;
    setStatus("offline", "Aloqa yo'q");
    listening = 0;
    if (event.code === 4401) {
      location.reload(); // session expired: show the login page
      return;
    }
    if (event.code === 4403) {
      showWarning(
        "Server ulanishni rad etdi: Origin va Host mos emas. nginx bo'lsa, proxy_set_header Host $http_host; qatorini qo'shing.",
      );
      return;
    }
    if (event.code === 4404) {
      store.set("session", "");
      chooseInitialSession();
      return;
    }
    setTimeout(() => {
      if (ws === socket) connect();
    }, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 15000);
  };
}

function send(frame) {
  if (ws?.readyState !== WebSocket.OPEN) return false;
  ws.send(frame);
  listening++;
  updateHint();
  return true;
}

function handleMessage(msg) {
  switch (msg.type) {
    case "heard":
    case "empty":
      listening = Math.max(0, listening - 1);
      updateHint();
      break;
    case "pending":
      showPending(msg.pending, msg.interpreting);
      break;
    case "unit":
      addUnit(msg.unit, true);
      break;
    case "speaker":
      speakers.set(msg.speaker.id, msg.speaker);
      relabel(msg.speaker.id);
      break;
    case "merged":
      openSession(session.id); // simplest way to regroup the blocks
      break;
    case "notes":
      showNotes(msg.notes);
      break;
    case "error":
      if (msg.id != null) listening = Math.max(0, listening - 1);
      updateHint();
      showWarning(msg.message);
      setTimeout(() => showWarning(""), 6000);
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
  if (shownUnits.has(unit.id)) return;
  shownUnits.add(unit.id);
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

function relabel(id) {
  ui.transcript.querySelectorAll(`.turn[data-speaker="${CSS.escape(id)}"] .who`).forEach((b) => (b.textContent = labelOf(id)));
  if (ui.liveWho.dataset.speaker === id) ui.liveWho.textContent = labelOf(id);
}

function showPending(pending, interpreting) {
  const last = pending[pending.length - 1];
  ui.live.hidden = !last && !interpreting;
  if (last) {
    ui.liveWho.dataset.speaker = last.speaker;
    ui.liveWho.textContent = labelOf(last.speaker);
    ui.liveWho.style.setProperty("--c", colorOf(last.speaker));
    ui.liveText.textContent = `${flagOf(last.lang)} ${pending.filter((p) => p.speaker === last.speaker).map((p) => p.text).join(" ")}`;
  } else {
    ui.liveWho.textContent = "";
    ui.liveText.textContent = "";
  }
  ui.liveBusy.hidden = !interpreting;
}

function updatePlaceholder() {
  ui.placeholder.hidden = ui.transcript.querySelector(".turn") !== null;
}

function updateHint() {
  if (!capture) return;
  if (capture.paused) return;
  ui.hint.textContent = listening ? `👂 Eshitilmoqda (${listening})` : "Tinglanmoqda…";
}

function showWarning(text) {
  ui.warning.textContent = text;
  ui.warning.hidden = !text;
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
    await openSession(session.id);
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
      open.addEventListener("click", async () => {
        ui.sessionsDialog.close();
        if (capture) stop();
        await openSession(s.id);
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
      del.addEventListener("click", async () => {
        if (!confirm(`«${s.title}» va uning barcha tarjimalari, hisobotlari o'chirilsinmi?`)) return;
        await api(`/api/sessions/${s.id}`, { method: "DELETE" });
        if (s.id === session.id) {
          store.set("session", "");
          ui.sessionsDialog.close();
          await chooseInitialSession();
        } else {
          showSessions();
        }
      });
      li.append(open, rename, del);
      return li;
    }),
  );
  if (!ui.sessionsDialog.open) ui.sessionsDialog.showModal();
}

ui.sessionsBtn.addEventListener("click", () => showSessions().catch((err) => showWarning(err.message)));

ui.newSession.addEventListener("submit", async (event) => {
  event.preventDefault();
  const created = await api("/api/sessions", { method: "POST", body: { title: ui.newTitle.value } });
  ui.newTitle.value = "";
  ui.sessionsDialog.close();
  if (capture) stop();
  await openSession(created.id);
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
      const a = el("a", "item", `📄 ${clock(r.start)}–${clock(r.end)} · ${new Date(r.created_at * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}`);
      a.href = `/api/reports/${r.id}`;
      a.target = "_blank";
      const md = el("a", "icon small", "MD");
      md.href = `/api/reports/${r.id}.md`;
      md.title = "Markdown yuklab olish";
      md.setAttribute("download", "");
      li.append(a, md);
      return li;
    }),
  );
  if (!reports.length) ui.reportsList.append(el("li", "muted", "Hali hisobot yo'q"));
}

ui.reportBtn.addEventListener("click", async () => {
  try {
    session = { ...session, ...(await api(`/api/sessions/${session.id}`)) };
  } catch {}
  setRange("all");
  renderReports(session.reports || []);
  ui.reportStatus.hidden = true;
  ui.reportDialog.showModal();
});

ui.reportDialog.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => setRange(b.dataset.preset)));

ui.reportMake.addEventListener("click", async () => {
  const start = fromLocalInput(ui.reportStart.value);
  const end = fromLocalInput(ui.reportEnd.value);
  if (start == null || end == null || start >= end) {
    ui.reportStatus.hidden = false;
    ui.reportStatus.textContent = "Vaqt oralig'ini to'g'ri tanlang";
    return;
  }
  ui.reportMake.disabled = true;
  ui.reportStatus.hidden = false;
  ui.reportStatus.textContent = "⏳ Hisobot tayyorlanmoqda… (uzun seminar uchun 1–2 daqiqa)";
  try {
    const { id } = await api(`/api/sessions/${session.id}/reports`, { method: "POST", body: { start, end } });
    const data = await api(`/api/sessions/${session.id}`);
    session.reports = data.reports;
    renderReports(data.reports);
    ui.reportStatus.textContent = "";
    const link = el("a", null, "✅ Tayyor — hisobotni ochish");
    link.href = `/api/reports/${id}`;
    link.target = "_blank";
    ui.reportStatus.append(link);
  } catch (err) {
    ui.reportStatus.textContent = `⚠️ ${err.message}`;
  } finally {
    ui.reportMake.disabled = false;
  }
});

// ---------- audio capture ----------

function onSegment(samples) {
  const frame = packSegment(++seq, encodeWav(samples, SAMPLE_RATE), Date.now() / 1000);
  if (!send(frame)) {
    outbox.push({ frame, at: Date.now() });
    outbox.splice(0, Math.max(0, outbox.length - MAX_REPLAY)); // keep memory bounded while offline
  }
}

function onSpeech(active) {
  ui.level.classList.toggle("speaking", active);
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
  player.src = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=";
  player.play().catch(() => {});
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
    capture = { ctx, stream, segmenter, paused: false, hearsOutput };
    const down = new Downsampler(ctx.sampleRate, (frame) => {
      if (capture && !capture.paused) segmenter.push(frame);
    });
    node.port.onmessage = (event) => down.push(event.data);
    stream.getTracks().forEach((t) => t.addEventListener("ended", stop));
    await ctx.resume();

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
  c.segmenter.flush();
  capture = null;
  c.stream.getTracks().forEach((t) => t.stop());
  c.ctx.close();
  wakeLock?.release().catch(() => {});
  wakeLock = null;
  // interpret what was said up to now instead of waiting for more speech
  setTimeout(() => {
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "flush" }));
  }, 2500);
  ui.toggle.dataset.state = "idle";
  ui.toggle.textContent = "▶ Boshlash";
  ui.hint.textContent = "Tayyor";
  ui.source.disabled = false;
  ui.level.style.width = "0";
  ui.level.classList.remove("speaking");
}

function micError(err) {
  if (err?.name === "NotAllowedError") return "Mikrofonga ruxsat berilmadi. Brauzer sozlamalaridan ruxsat bering.";
  if (err?.name === "NotFoundError") return "Mikrofon topilmadi.";
  return err?.message || String(err);
}

function drawLevel() {
  if (!capture) return;
  const level = capture.paused ? 0 : capture.segmenter.level;
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
  if (capture && document.visibilityState === "visible") keepAwake();
});

// ---------- voice output ----------

function speak(text) {
  speechQueue.push(text);
  if (!speaking) playNext();
}

async function playNext() {
  const text = speechQueue.shift();
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
      if (ui.tts.checked) {
        player.src = url;
        await new Promise((resolve) => {
          // "pause" too: switching read-out off stops the clip without an "ended" event
          player.onended = player.onerror = player.onpause = resolve;
          player.play().catch(resolve);
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
  if (!ui.tts.checked) {
    speechQueue.length = 0;
    player.pause();
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
