import { Downsampler, SAMPLE_RATE, Segmenter, encodeWav, packSegment } from "./audio.js";

const $ = (id) => document.getElementById(id);
const ui = {
  status: $("status"),
  source: $("source"),
  partner: $("partner"),
  tts: $("tts"),
  ttsLabel: $("tts-label"),
  warning: $("warning"),
  transcript: $("transcript"),
  placeholder: $("placeholder"),
  toggle: $("toggle"),
  level: $("level"),
  hint: $("hint"),
  copy: $("copy"),
  clear: $("clear"),
  logout: $("logout"),
};

const store = {
  get: (key) => {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set: (key, value) => {
    try { localStorage.setItem(key, value); } catch {}
  },
};

let config = null;
let ws = null;
let reconnectDelay = 1000;
let capture = null; // {ctx, stream, segmenter, paused}
let seq = 0;
const pending = new Map(); // segment id -> {card, frame, sent}
const player = new Audio();
const speechQueue = [];
let speaking = false;
let wakeLock = null;
const MAX_REPLAY = 5; // after a reconnect, only the newest unsent segments are still worth translating
const MAX_REPLAY_AGE_MS = 30000;

// ---------- connection ----------

function setStatus(state, text) {
  ui.status.dataset.state = state;
  ui.status.textContent = text;
}

function connect() {
  setStatus("connecting", "Ulanmoqda…");
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    reconnectDelay = 1000;
    setStatus("online", "Ulangan");
    sendConfig();
    const unsent = [...pending].filter(([, item]) => !item.sent);
    unsent.forEach(([id, item], i) => {
      if (i < unsent.length - MAX_REPLAY || Date.now() - item.at > MAX_REPLAY_AGE_MS) {
        fail(id, "Aloqa yo'q edi");
      } else {
        item.replayed = true; // translate late, but don't read it aloud
        send(item);
      }
    });
  };
  ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
  ws.onclose = (event) => {
    setStatus("offline", "Aloqa yo'q");
    if (event.code === 4401) {
      location.reload(); // session expired: show the login page
      return;
    }
    if (event.code === 4403) {
      showWarning(
        "Server ulanishni rad etdi: Origin va Host mos emas. nginx bo'lsa, " +
          "proxy_set_header Host $http_host; qatorini qo'shing.",
      );
      return; // reconnecting would be rejected the same way
    }
    // Segments the server had not answered are lost with the connection.
    for (const [id, item] of pending) {
      if (item.sent) fail(id, "Aloqa uzildi");
    }
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 15000);
  };
}

function sendConfig() {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "config", partner: ui.partner.value }));
  }
}

function send(item) {
  if (ws?.readyState !== WebSocket.OPEN) return;
  ws.send(item.frame);
  item.sent = true;
}

function handleMessage(msg) {
  if (msg.type === "result") {
    const item = pending.get(msg.id);
    if (!item) return;
    pending.delete(msg.id);
    fillCard(item.card, msg);
    if (ui.tts.checked && config.tts && !item.replayed) speak(msg.translation);
  } else if (msg.type === "empty") {
    pending.get(msg.id)?.card.remove();
    pending.delete(msg.id);
    updatePlaceholder();
  } else if (msg.type === "error") {
    if (msg.id == null) showWarning(msg.message);
    else fail(msg.id, msg.message);
  }
}

// ---------- transcript ----------

function flagOf(code) {
  return config.langs[code]?.flag ?? "🌐";
}

function line(cls, flag, text) {
  const div = document.createElement("div");
  div.className = `line ${cls}`;
  const f = document.createElement("span");
  f.className = "flag";
  f.textContent = flag;
  const t = document.createElement("span");
  t.className = "text";
  t.textContent = text;
  div.append(f, t);
  return div;
}

function nearBottom() {
  const el = ui.transcript;
  return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
}

function addCard() {
  const stick = nearBottom();
  const card = document.createElement("article");
  card.className = "card pending";
  card.append(line("trans", "⏳", "Tarjima qilinmoqda"));
  ui.transcript.append(card);
  updatePlaceholder();
  if (stick) ui.transcript.scrollTop = ui.transcript.scrollHeight;
  return card;
}

function fillCard(card, msg) {
  const stick = nearBottom();
  card.className = "card";
  card.dataset.translation = msg.translation;
  card.replaceChildren(
    line("orig", flagOf(msg.source_lang), msg.text),
    line("trans", flagOf(msg.target), msg.translation),
  );
  if (stick) ui.transcript.scrollTop = ui.transcript.scrollHeight;
}

function fail(id, message) {
  const item = pending.get(id);
  if (!item) return;
  pending.delete(id);
  item.card.className = "card error";
  item.card.replaceChildren(line("trans", "⚠️", message));
}

function updatePlaceholder() {
  ui.placeholder.hidden = ui.transcript.querySelector(".card") !== null;
}

function showWarning(text) {
  ui.warning.textContent = text;
  ui.warning.hidden = !text;
}

// ---------- audio capture ----------

function onSegment(samples) {
  const id = ++seq;
  const item = {
    card: addCard(),
    frame: packSegment(id, encodeWav(samples, SAMPLE_RATE)),
    sent: false,
    at: Date.now(),
  };
  pending.set(id, item);
  send(item);
  if (!item.sent) {
    // offline: keep only the newest few in memory
    const unsent = [...pending].filter(([, it]) => !it.sent);
    unsent.slice(0, -MAX_REPLAY).forEach(([oldId]) => fail(oldId, "Aloqa yo'q edi"));
  }
}

function onSpeech(active) {
  ui.level.classList.toggle("speaking", active);
  if (capture) ui.hint.textContent = active ? "🎙️ Nutq eshitilmoqda…" : "Tinglanmoqda…";
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
    ui.hint.textContent = "Tinglanmoqda…";
    ui.source.disabled = true;
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
        ui.hint.textContent = "Tinglanmoqda…";
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

ui.partner.addEventListener("change", () => {
  store.set("partner", ui.partner.value);
  sendConfig();
});

ui.tts.addEventListener("change", () => {
  store.set("tts", ui.tts.checked ? "1" : "0");
  if (!ui.tts.checked) {
    speechQueue.length = 0;
    player.pause();
  }
});

ui.copy.addEventListener("click", async () => {
  const text = [...ui.transcript.querySelectorAll(".card[data-translation]")]
    .map((c) => c.dataset.translation)
    .join("\n");
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    ui.hint.textContent = "📋 Nusxalandi";
  } catch {
    showWarning("Nusxalab bo'lmadi");
  }
});

ui.clear.addEventListener("click", () => {
  ui.transcript.querySelectorAll(".card").forEach((c) => c.remove());
  pending.clear();
  updatePlaceholder();
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "reset" }));
});

async function init() {
  const res = await fetch("/api/config");
  if (res.status === 401) {
    location.reload();
    return;
  }
  config = await res.json();
  for (const code of config.partners) {
    const { flag, name } = config.langs[code];
    ui.partner.add(new Option(`${flag} ${name}ga`, code));
  }
  const saved = store.get("partner");
  ui.partner.value = config.partners.includes(saved) ? saved : config.default_partner;
  ui.ttsLabel.hidden = !config.tts;
  ui.tts.checked = config.tts && store.get("tts") === "1";
  ui.logout.hidden = !config.auth;
  if (!navigator.mediaDevices?.getDisplayMedia) ui.source.querySelector('[value="tab"]').disabled = true;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    ui.toggle.disabled = true;
    showWarning(
      "Mikrofon faqat HTTPS orqali ishlaydi. Saytni https:// bilan oching " +
        "(serverda .env ga SSL_SELF_SIGNED=1 yozing yoki domen + sertifikat ulang).",
    );
  }
  connect();
}

init();
