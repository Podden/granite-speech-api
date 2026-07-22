/* granite-speech transcription UI.
 *
 * Pipeline: decode audio locally (16 kHz mono via Web Audio API) -> WAV upload
 * (much smaller than video, no video data leaves the browser) -> NDJSON
 * streaming transcription with upload + inference progress.
 * Fallback: if the browser cannot decode the container, the original file is
 * uploaded and the server extracts the audio with ffmpeg.
 */
"use strict";

const $ = (sel) => document.querySelector(sel);

const els = {
  dropzone: $("#dropzone"),
  fileInput: $("#file-input"),
  dzIdle: $("#dz-idle"),
  dzFile: $("#dz-file"),
  waveform: $("#waveform"),
  fileName: $("#file-name"),
  fileInfo: $("#file-info"),
  fileError: $("#file-error"),
  btnReset: $("#btn-reset"),
  btnStart: $("#btn-start"),
  btnCancel: $("#btn-cancel"),
  speakerOpts: $("#speaker-opts"),
  cardStatus: $("#card-status"),
  cardResult: $("#card-result"),
  stepper: $("#stepper"),
  progressFill: $("#progress-fill"),
  statusText: $("#status-text"),
  statusDetail: $("#status-detail"),
  transcript: $("#transcript"),
  log: $("#log"),
  healthDot: $("#health-dot"),
  healthText: $("#health-text"),
};

const state = {
  file: null,          // original File
  uploadBlob: null,    // what actually gets uploaded (WAV or original)
  uploadName: null,
  duration: null,      // seconds
  peaks: null,         // waveform peaks for drawing
  maxSeconds: 600,     // refreshed from /health
  xhr: null,
  segments: [],
  partials: [],
  liveText: "",       // token stream of the chunk currently being decoded
  resultText: "",
  words: [],
  startedAt: 0,
  processingSecs: null,
  queue: [],
  finishCb: null,
  audioUrl: null,
  lastProgress: 0,
};

/* ── Logging ────────────────────────────────────────────── */

/* Das Protokoll puffert nur — gerendert wird erst, wenn es aufgeklappt ist.
   Zugeklappt kostet ein Log-Eintrag damit nichts außer einem Array-Push. */
const LOG_MAX = 4000;      // Ringpuffer
const LOG_DOM_MAX = 1500;  // so viele Zeilen landen höchstens im DOM
const logBuf = [];
let logDrawn = 0;
let logErrors = 0;
const logBox = $("#logbox");
const logBadge = $("#log-badge");

function logLine(e) {
  return `[${e.t}] ${e.msg}\n`;
}

function flushLog() {
  if (logDrawn > logBuf.length) logDrawn = 0; // Puffer wurde gekürzt
  if (logDrawn === 0) els.log.textContent = "";
  const start = Math.max(logDrawn, logBuf.length - LOG_DOM_MAX);
  const frag = document.createDocumentFragment();
  for (let i = start; i < logBuf.length; i++) {
    const e = logBuf[i];
    const span = document.createElement("span");
    if (e.cls) span.className = e.cls;
    span.textContent = logLine(e);
    frag.appendChild(span);
  }
  els.log.appendChild(frag);
  logDrawn = logBuf.length;
  els.log.scrollTop = els.log.scrollHeight;
}

function updateLogBadge() {
  logBadge.textContent =
    ` ${logBuf.length} Zeilen` + (logErrors ? ` · ${logErrors} Fehler` : "");
}

function log(msg, cls) {
  logBuf.push({ t: new Date().toLocaleTimeString("de-DE", { hour12: false }), msg, cls });
  if (logBuf.length > LOG_MAX) {
    logBuf.splice(0, logBuf.length - LOG_MAX);
    logDrawn = 0;
  }
  if (cls === "err") logErrors++;
  updateLogBadge();
  if (logBox.open) flushLog();
}

/* Rohdaten eines Schrittes — bewusst ungefiltert, nur längenbegrenzt. */
function logRaw(label, data, cls) {
  let s = typeof data === "string" ? data : JSON.stringify(data);
  if (s && s.length > 2000) s = `${s.slice(0, 2000)}… (${s.length} Zeichen)`;
  log(`${label}: ${s}`, cls);
}

logBox.addEventListener("toggle", () => { if (logBox.open) flushLog(); });

/* ── Health ─────────────────────────────────────────────── */

async function refreshHealth() {
  try {
    const res = await fetch("/health");
    const h = await res.json();
    els.healthDot.className = "dot ok";
    const device = h.device === "cuda" ? "GPU" : "CPU";
    let text = h.loaded_model
      ? `Bereit (${device}, Modell geladen)`
      : `Bereit (${device})`;
    const active = h.queue?.active_jobs || 0;
    if (active > 0) {
      text = `${active} Auftrag${active > 1 ? "e" : ""} in Arbeit (${device})`;
    }
    els.healthText.textContent = text;
    if (h.max_audio_seconds) state.maxSeconds = h.max_audio_seconds;
  } catch (err) {
    els.healthDot.className = "dot err";
    els.healthText.textContent = "Server nicht erreichbar";
    log(`Health-Check fehlgeschlagen: ${err.message}`, "err");
  }
}

/* ── Helpers ────────────────────────────────────────────── */

function fmtSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return Math.max(1, Math.round(bytes / 1024)) + " kB";
}

function fmtDuration(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${String(s).padStart(2, "0")} min`;
}

function fmtTs(t, sep) {
  if (t < 0) t = 0;
  const h = String(Math.floor(t / 3600)).padStart(2, "0");
  const m = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
  const s = String(Math.floor(t % 60)).padStart(2, "0");
  const ms = String(Math.round((t - Math.floor(t)) * 1000)).padStart(3, "0");
  return `${h}:${m}:${s}${sep}${ms}`;
}

/* ── File preparation ───────────────────────────────────── */

async function onFileSelected(file) {
  // Gleiche Datei wie das gespeicherte Transkript → Ergebnis stehen lassen,
  // damit Wiedergabe und Wortsprung nach einem Reload weitergehen.
  const keepResult = sameFile(sessionFileSig, file) &&
    (state.segments.length > 0 || !!state.resultText);
  if (keepResult) {
    els.cardStatus.hidden = true;
    els.btnCancel.hidden = false;
    log("Gleiche Datei wie das vorhandene Transkript — Ergebnis bleibt erhalten. " +
        "„Transkription starten“ überschreibt es.");
  } else {
    resetResult();
  }
  state.file = file;
  state.uploadBlob = null;
  state.duration = null;
  els.fileError.hidden = true;
  els.btnStart.disabled = true;

  els.dzIdle.hidden = true;
  els.dzFile.hidden = false;
  els.dropzone.classList.add("has-file");
  els.fileName.textContent = file.name;
  els.fileInfo.textContent = `${fmtSize(file.size)} — analysiere…`;
  log(`Datei gewählt: ${file.name} (${fmtSize(file.size)}, ${file.type || "unbekannter Typ"})`);

  const isVideo = /^video\//.test(file.type) || /\.(mp4|mkv|mov|webm|avi|m4v)$/i.test(file.name);

  try {
    const buf = await file.arrayBuffer();
    // Decoding into a 16 kHz context resamples for free — exactly what the
    // ASR model expects, and it shrinks the upload.
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const audio = await ctx.decodeAudioData(buf);
    ctx.close();

    state.duration = audio.duration;
    const mono = downmix(audio);
    state.peaks = computePeaks(mono, 600);
    drawWaveform(0);

    // Smallest upload wins: Opus (48 kbit/s, WebCodecs) → original audio
    // file → 16 kHz mono WAV as universal fallback.
    const base = file.name.replace(/\.[^.]+$/, "");
    const wav = encodeWav(mono, audio.sampleRate);
    let upload = { blob: wav, name: base + ".wav", label: "16 kHz mono WAV" };
    if (await oggOpusSupported(audio.sampleRate)) {
      try {
        const ogg = await encodeOggOpus(mono, audio.sampleRate);
        if (ogg.size < upload.blob.size) {
          upload = { blob: ogg, name: base + ".ogg", label: "Opus 48 kbit/s" };
        }
      } catch (err) {
        log(`Opus-Kompression fehlgeschlagen (${err.message}) — nutze WAV.`, "warn");
      }
    }
    if (!isVideo && file.size < upload.blob.size) {
      upload = { blob: file, name: file.name, label: "Original" };
    }
    state.uploadBlob = upload.blob;
    state.uploadName = upload.name;
    log(`Audio vorbereitet: ${fmtDuration(audio.duration)}, Upload als ${upload.label} ` +
        `(${fmtSize(upload.blob.size)} statt ${fmtSize(file.size)} — ` +
        `${Math.max(0, 100 - (upload.blob.size / file.size) * 100).toFixed(0)} % gespart)`);
    els.fileInfo.textContent =
      `${fmtDuration(audio.duration)} · Upload: ${fmtSize(state.uploadBlob.size)} (${upload.label})` +
      (isVideo ? ` · Tonspur aus ${fmtSize(file.size)} Video` : "");
  } catch (err) {
    log(`Browser-Dekodierung fehlgeschlagen (${err.message || err.name}) — Original wird hochgeladen, der Server extrahiert die Tonspur.`, "warn");
    state.uploadBlob = file;
    state.uploadName = file.name;
    state.peaks = null;
    drawWaveform(0);
    els.fileInfo.textContent = `${fmtSize(file.size)} · Dauer unbekannt (Server prüft)`;
  }

  setupPlayer();

  if (state.duration && state.duration > state.maxSeconds) {
    showFileError(
      `Die Aufnahme ist ${fmtDuration(state.duration)} lang — der Server akzeptiert maximal ` +
      `${fmtDuration(state.maxSeconds)}. Bitte die Datei vorher kürzen oder aufteilen.`
    );
    return;
  }
  els.btnStart.disabled = false;
}

function showFileError(msg) {
  els.fileError.textContent = msg;
  els.fileError.hidden = false;
  els.btnStart.disabled = true;
  log(msg, "err");
}

function downmix(audioBuffer) {
  const ch0 = audioBuffer.getChannelData(0);
  if (audioBuffer.numberOfChannels === 1) return ch0;
  const out = new Float32Array(ch0.length);
  for (let c = 0; c < audioBuffer.numberOfChannels; c++) {
    const data = audioBuffer.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += data[i];
  }
  const n = audioBuffer.numberOfChannels;
  for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}

function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const wstr = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); wstr(8, "WAVE");
  wstr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  wstr(36, "data"); v.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

/* ── Audio player + scrubbing ───────────────────────────── */

const audioEl = $("#audio-player");
const btnPlay = $("#btn-play");
const playTime = $("#play-time");
// Zweiter Satz Bedienelemente direkt am Transkript, damit man zum
// Pausieren nicht zur Waveform hochscrollen muss.
const btnPlayTr = $("#btn-play-tr");
const playTimeTr = $("#play-time-tr");
let playheadRaf = 0;

function setupPlayer() {
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  // The original file is the playback source — the browser just decoded it,
  // so it can play it (video files play their audio track in <audio>).
  const src = state.file;
  if (!src) return;
  state.audioUrl = URL.createObjectURL(src);
  audioEl.src = state.audioUrl;
  btnPlay.hidden = false;
  playTime.hidden = false;
  btnPlayTr.disabled = false;
  updatePlayTime();
}

function fmtClock(t) {
  if (!isFinite(t)) t = 0;
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function updatePlayTime() {
  const dur = state.duration || audioEl.duration || 0;
  const txt = `${fmtClock(audioEl.currentTime)} / ${fmtClock(dur)}`;
  playTime.textContent = txt;
  playTimeTr.textContent = txt;
}

function setPlayIcon(playing) {
  const icon = playing ? "&#10073;&#10073;" : "&#9654;";
  const label = playing ? "Pause" : "Abspielen";
  for (const b of [btnPlay, btnPlayTr]) {
    b.innerHTML = icon;
    b.setAttribute("aria-label", label);
  }
}

function togglePlay() {
  if (!audioEl.src) return;
  if (audioEl.paused) audioEl.play().catch(() => {}); else audioEl.pause();
}

let lastHeadX = -1;
function playheadLoop() {
  // Nur neu zeichnen, wenn der Playhead wirklich ein Pixel weiter ist.
  const dur = state.duration || audioEl.duration;
  const x = dur ? Math.round((audioEl.currentTime / dur) * els.waveform.width) : -1;
  if (x !== lastHeadX) {
    lastHeadX = x;
    drawWaveform(state.lastProgress || 0);
    updatePlayTime();
    highlightWordAt(audioEl.currentTime);
  }
  if (!audioEl.paused) playheadRaf = requestAnimationFrame(playheadLoop);
}

btnPlay.addEventListener("click", togglePlay);
btnPlayTr.addEventListener("click", togglePlay);
audioEl.addEventListener("play", () => {
  setPlayIcon(true);
  playheadRaf = requestAnimationFrame(playheadLoop);
});
audioEl.addEventListener("pause", () => {
  setPlayIcon(false);
  cancelAnimationFrame(playheadRaf);
  drawWaveform(state.lastProgress || 0);
  updatePlayTime();
});
audioEl.addEventListener("ended", () => audioEl.pause());
audioEl.addEventListener("timeupdate", () => {
  if (audioEl.paused) {
    drawWaveform(state.lastProgress || 0);
    updatePlayTime();
    highlightWordAt(audioEl.currentTime);
  }
});
audioEl.addEventListener("error", () => {
  btnPlay.hidden = true;
  playTime.hidden = true;
  btnPlayTr.disabled = true;
  log("Wiedergabe im Browser nicht möglich (Format).", "warn");
});

function seekFromEvent(e) {
  const dur = state.duration || audioEl.duration;
  if (!dur || !state.file) return;
  const rect = els.waveform.getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  audioEl.currentTime = frac * dur;
  drawWaveform(state.lastProgress || 0);
  updatePlayTime();
  highlightWordAt(audioEl.currentTime, true); // Sprung: einmal zur Textstelle
}

let scrubbing = false;
let scrubPlaying = false; // audio only plays while the pointer is held down

els.waveform.addEventListener("pointerdown", (e) => {
  scrubbing = true;
  els.waveform.setPointerCapture(e.pointerId);
  seekFromEvent(e);
  if (audioEl.paused && audioEl.src) {
    scrubPlaying = true;
    audioEl.play().catch(() => { scrubPlaying = false; });
  }
});
els.waveform.addEventListener("pointermove", (e) => { if (scrubbing) seekFromEvent(e); });
function endScrub() {
  scrubbing = false;
  if (scrubPlaying) {
    audioEl.pause();
    scrubPlaying = false;
  }
}
els.waveform.addEventListener("pointerup", endScrub);
els.waveform.addEventListener("pointercancel", endScrub);

/* Wort-Highlight beim Abspielen/Scrubben (nur mit Wort-Timestamps). */
let wordSpans = [];
let lastLitSpan = null;

// force: einmalig zur Stelle springen (z. B. nach Waveform-Klick), auch wenn
// das laufende Mitscrollen aus oder pausiert ist.
function highlightWordAt(t, force = false) {
  if (!wordSpans.length) return;
  // Binäre Suche — die lineare Suche lief pro Frame über alle Wörter.
  let lo = 0, hi = wordSpans.length - 1, lit = null;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    const s = wordSpans[m];
    if (t < s._start) hi = m - 1;
    else if (t >= s._end) lo = m + 1;
    else { lit = s; break; }
  }
  if (lit === lastLitSpan) {
    if (lit && force) scrollSpanIntoView(lit, true);
    return;
  }
  if (lastLitSpan) lastLitSpan.classList.remove("playing");
  if (lit) {
    lit.classList.add("playing");
    scrollSpanIntoView(lit, force);
  }
  lastLitSpan = lit;
}

/* Auto-Scroll ist optional (Checkbox, Default aus) und bewegt nur den
   Transkript-Kasten, nie die Seite. Eigenes Scrollen pausiert es kurz. */
const AUTOSCROLL_PAUSE_MS = 5000;
let autoScrollPausedUntil = 0;

function pauseAutoScroll() { autoScrollPausedUntil = performance.now() + AUTOSCROLL_PAUSE_MS; }
function autoScrollAllowed() {
  return $("#opt-autoscroll").checked && performance.now() >= autoScrollPausedUntil;
}

for (const ev of ["wheel", "touchmove"]) {
  window.addEventListener(ev, pauseAutoScroll, { passive: true });
}
els.transcript.addEventListener("keydown", pauseAutoScroll);

// Haken frisch gesetzt = klarer Wunsch: Pause aufheben und sofort nachziehen.
$("#opt-autoscroll").addEventListener("change", (e) => {
  if (!e.target.checked) return;
  autoScrollPausedUntil = 0;
  if (lastLitSpan) scrollSpanIntoView(lastLitSpan, true);
});

function scrollSpanIntoView(span, force = false) {
  if (!force && !autoScrollAllowed()) return;
  const el = els.transcript;
  const cr = el.getBoundingClientRect();
  const sr = span.getBoundingClientRect();
  if (sr.top < cr.top) el.scrollTop += sr.top - cr.top - 8;
  else if (sr.bottom > cr.bottom) el.scrollTop += sr.bottom - cr.bottom + 8;
}

/* Klick/Halten im Transkript. Maus bewegen bleibt Textmarkierung —
   erst 200 ms Halten ohne Bewegung schaltet auf Vorhören um. */
const TEXT_HOLD_MS = 200;
const TEXT_MOVE_PX = 4;
let textPress = null;

function wordSpanFromPoint(e) {
  const t = document.elementFromPoint(e.clientX, e.clientY);
  return t && t.classList && t.classList.contains("w") ? t : null;
}

function seekToSpan(span) {
  audioEl.currentTime = span._start;
  drawWaveform(state.lastProgress || 0);
  updatePlayTime();
  highlightWordAt(span._start);
}

function playFromSpan(span) {
  seekToSpan(span);
  if (audioEl.paused) audioEl.play().catch(() => {});
}

function cancelTextPress() {
  if (!textPress) return;
  clearTimeout(textPress.timer);
  textPress = null;
}

els.transcript.addEventListener("pointerdown", (e) => {
  cancelTextPress();
  const span = wordSpanFromPoint(e);
  if (!span || !audioEl.src) return;
  // Kein preventDefault: die Textmarkierung soll normal funktionieren.
  textPress = {
    span,
    x: e.clientX,
    y: e.clientY,
    wasPlaying: !audioEl.paused,
    held: false,
    timer: setTimeout(() => {
      if (!textPress) return;
      textPress.held = true;
      playFromSpan(textPress.span);
    }, TEXT_HOLD_MS),
  };
});

els.transcript.addEventListener("pointermove", (e) => {
  if (!textPress || textPress.held) return;
  // Bewegung vor Ablauf der Haltezeit → der Nutzer markiert Text.
  if (Math.abs(e.clientX - textPress.x) > TEXT_MOVE_PX ||
      Math.abs(e.clientY - textPress.y) > TEXT_MOVE_PX) cancelTextPress();
});

function endTextPress(e) {
  if (!textPress) return;
  if (textPress.held) {
    // Vorhören beendet: zurück in den Zustand von vor dem Halten.
    if (!textPress.wasPlaying) audioEl.pause();
  } else if (wordSpanFromPoint(e) === textPress.span) {
    // Kurzer Klick: ab dieser Stelle abspielen, egal ob Play oder Pause aktiv war.
    playFromSpan(textPress.span);
  }
  cancelTextPress();
}
els.transcript.addEventListener("pointerup", endTextPress);
els.transcript.addEventListener("pointercancel", cancelTextPress);

/* ── Waveform ───────────────────────────────────────────── */

function computePeaks(samples, buckets) {
  const peaks = new Float32Array(buckets);
  const per = Math.floor(samples.length / buckets) || 1;
  for (let b = 0; b < buckets; b++) {
    let max = 0;
    const start = b * per;
    for (let i = start; i < start + per && i < samples.length; i += 4) {
      const a = Math.abs(samples[i]);
      if (a > max) max = a;
    }
    peaks[b] = max;
  }
  const norm = Math.max(0.01, ...peaks);
  for (let b = 0; b < buckets; b++) peaks[b] /= norm;
  return peaks;
}

/* Farben einmal aus dem Stylesheet holen — getComputedStyle pro Balken
   erzwang bei jedem Frame einen Style-Recalc und machte das Scrollen zäh. */
let waveColors = null;
function getWaveColors() {
  if (waveColors) return waveColors;
  const css = getComputedStyle(document.documentElement);
  waveColors = {
    wave: css.getPropertyValue("--wave").trim(),
    done: css.getPropertyValue("--wave-done").trim(),
    played: css.getPropertyValue("--wave-played").trim(),
    head: css.getPropertyValue("--err").trim() || "#da1e28",
  };
  return waveColors;
}
window.matchMedia?.("(prefers-color-scheme: dark)")
  .addEventListener?.("change", () => { waveColors = null; });

function drawWaveform(progress) {
  state.lastProgress = progress || 0;
  const canvas = els.waveform;
  const ctx = canvas.getContext("2d");
  const col = getWaveColors();
  const W = canvas.width, H = canvas.height, mid = H / 2;
  ctx.clearRect(0, 0, W, H);
  if (!state.peaks) {
    ctx.fillStyle = col.wave;
    ctx.fillRect(0, mid - 1, W, 2);
  } else {
    const n = state.peaks.length;
    const bw = W / n;
    const doneX = W * state.lastProgress;
    const dur0 = state.duration || audioEl.duration;
    const playedX = dur0 && audioEl.currentTime > 0
      ? (audioEl.currentTime / dur0) * W : 0;
    let cur = null;
    for (let i = 0; i < n; i++) {
      const x = i * bw;
      const h = Math.max(2, state.peaks[i] * (H - 8));
      // Played portion (teal) wins over transcription progress (blue).
      const c = x < playedX ? col.played : (x < doneX ? col.done : col.wave);
      if (c !== cur) { ctx.fillStyle = c; cur = c; }
      ctx.fillRect(x, mid - h / 2, Math.max(1, bw - 1), h);
    }
  }
  // Playhead
  const dur = state.duration || audioEl.duration;
  if (dur && audioEl.currentTime > 0) {
    const x = (audioEl.currentTime / dur) * W;
    ctx.fillStyle = col.head;
    ctx.fillRect(x - 1, 0, 2, H);
  }
}

/* ── Status / stepper ───────────────────────────────────── */

function setStep(stepId, stateName) {
  for (const li of els.stepper.children) {
    if (li.dataset.step === stepId) li.className = stateName;
  }
}

function setPhase(stepId, text, detail) {
  for (const li of els.stepper.children) {
    if (li.dataset.step === stepId) li.className = "active";
    else if (li.className === "active") li.className = "done";
  }
  els.statusText.textContent = text;
  els.statusDetail.textContent = detail || "";
}

function setProgress(pct, indeterminate) {
  els.progressFill.classList.toggle("indeterminate", !!indeterminate);
  if (!indeterminate) els.progressFill.style.width = `${Math.min(100, pct)}%`;
}

/* ── Warteschlange (Server verarbeitet einen Auftrag nach dem anderen) ── */

let queueTimer = 0;

function startQueuePolling() {
  stopQueuePolling();
  queueTimer = setInterval(async () => {
    try {
      const h = await (await fetch("/health")).json();
      const q = h.queue;
      logRaw("< queue", q || h);
      if (!q || q.active_jobs <= 1) return;
      const rtf = q.rtf || 10;
      // Oldest jobs first; our own job is the newest → everything else is ahead.
      const jobs = [...q.jobs].sort((a, b) => b.elapsed - a.elapsed);
      const ahead = jobs.slice(0, -1);
      if (!ahead.length) return;
      const etaS = ahead.reduce(
        (s, j) => s + Math.max(0, j.duration - j.elapsed * rtf) / rtf, 0
      );
      setPhase("transcribe", "In der Warteschlange…",
        `${ahead.length} Auftrag${ahead.length > 1 ? "e" : ""} vor dir · ` +
        `geschätzt noch ~${fmtDuration(Math.max(5, etaS))}`);
    } catch { /* Header-Healthcheck meldet Ausfälle */ }
  }, 4000);
}

function stopQueuePolling() {
  if (queueTimer) clearInterval(queueTimer);
  queueTimer = 0;
}

/* ── Transcription ──────────────────────────────────────── */

function buildFormData() {
  const fd = new FormData();
  fd.append("file", state.uploadBlob, state.uploadName);
  fd.append("stream", "true");

  const model = $("#opt-model").value;
  if (model) fd.append("model", model);
  // Cohere/Qwen have no word timestamps → no speaker reconciliation possible.
  const noSpeakers = model === "cohere-transcribe" || model === "qwen3-asr";
  const multi = document.querySelector('input[name="speakers"]:checked').value === "multi";
  if (multi && noSpeakers) {
    log(`${model}: keine Sprecher-Erkennung möglich — Option wird ignoriert.`, "err");
  } else if (multi) {
    fd.append("speaker_attribution", "true");
    const n = $("#num-speakers").value;
    if (n) fd.append("num_speakers", n);
  }
  if ($("#opt-word-ts").checked && !noSpeakers) fd.append("timestamp_granularities[]", "word");
  const lang = $("#opt-language").value;
  if (lang) fd.append("language", lang);
  const translateTo = $("#opt-translate").value;
  if (translateTo) { fd.append("translate", "true"); fd.append("translate_to", translateTo); }
  const keywords = $("#opt-keywords").value.trim();
  if (keywords) fd.append("prompt", keywords);
  return fd;
}

function startTranscription() {
  if (state.queue.length > 1) {
    runBatch().catch((e) => log(`Batch-Fehler: ${e.message}`, "err"));
    return;
  }
  transcribeCurrent().catch(() => { /* UI already updated by fail() */ });
}

function transcribeCurrent() {
  return new Promise((resolve, reject) => {
    state.finishCb = { resolve, reject };
    beginTranscription();
  });
}

function beginTranscription() {
  if (!state.uploadBlob) return;
  resetResult();
  els.cardStatus.hidden = false;
  els.btnStart.disabled = true;
  for (const li of els.stepper.children) li.className = "";
  setStep("prepare", "done");
  setPhase("upload", "Datei wird hochgeladen…");
  setProgress(0, false);
  state.startedAt = Date.now();
  state.segments = [];
  state.partials = [];
  state.liveText = "";
  state.words = [];
  state.resultText = "";

  const fd = buildFormData();
  log(`Starte Upload: ${state.uploadName} (${fmtSize(state.uploadBlob.size)})`);
  const fields = [];
  for (const [k, v] of fd.entries()) {
    fields.push(`${k}=${v instanceof Blob ? `<blob ${v.size} B, ${v.type || "?"}>` : v}`);
  }
  logRaw("POST /v1/audio/transcriptions", fields.join(" · "));

  const xhr = new XMLHttpRequest();
  state.xhr = xhr;
  let parsedTo = 0;

  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = (e.loaded / e.total) * 100;
    setProgress(pct * 0.999, false);
    els.statusDetail.textContent = `${fmtSize(e.loaded)} / ${fmtSize(e.total)}`;
    if (e.loaded >= e.total) {
      setPhase("transcribe", "Warte auf Server…", "");
      setProgress(0, true);
      log("Upload abgeschlossen.");
      startQueuePolling();
    }
  };

  xhr.onreadystatechange = () => {
    if (xhr.readyState >= 3 && xhr.status === 200) {
      const chunk = xhr.responseText.substring(parsedTo);
      const lastNl = chunk.lastIndexOf("\n");
      if (lastNl >= 0) {
        parsedTo += lastNl + 1;
        for (const line of chunk.substring(0, lastNl).split("\n")) {
          if (line.trim()) handleEvent(line);
        }
      }
    }
    if (xhr.readyState === 4) onRequestDone(xhr);
  };

  xhr.onerror = () => {
    fail("Netzwerkfehler beim Upload — ist der Server erreichbar?");
  };

  xhr.open("POST", "/v1/audio/transcriptions");
  xhr.send(fd);
}

let deltaChars = 0;
let deltaCount = 0;

function flushDeltaLog() {
  if (!deltaCount) return;
  log(`< delta ×${deltaCount} (${deltaChars} Zeichen Token-Stream)`);
  deltaChars = 0;
  deltaCount = 0;
}

function handleEvent(line) {
  let ev;
  try { ev = JSON.parse(line); } catch { log(`Unlesbares Server-Event: ${line}`, "warn"); return; }
  // Rohes Event ins Protokoll — außer dem Token-Stream, der wird gebündelt.
  if (ev.type !== "delta") {
    flushDeltaLog();
    logRaw("< event", line, ev.type === "error" ? "err" : undefined);
  }
  switch (ev.type) {
    case "duration":
      if (!state.duration) state.duration = ev.duration;
      log(`Server bestätigt Audio: ${fmtDuration(ev.duration)}`);
      break;
    case "status": {
      stopQueuePolling();
      const short = (ev.model || "").split("/").pop();
      if (ev.stage === "loading_model") {
        setPhase("transcribe", "Modell wird geladen…",
          ev.cold ? `${short} — Kaltstart, kann 1–5 min dauern` : `Wechsel auf ${short}`);
        setProgress(0, true);
        log(`Modell wird ${ev.cold ? "kalt " : ""}geladen: ${ev.model}`);
      } else if (ev.stage === "model_ready") {
        setPhase("transcribe", "Server transkribiert…", short);
        log(`Modell bereit: ${ev.model}`);
      } else if (ev.stage === "diarizing") {
        setPhase("transcribe", "Sprecher werden erkannt…",
          ev.cold ? "Diarisierungs-Modell wird geladen" : "");
        log("Sprecher-Erkennung (pyannote) läuft…");
      } else if (ev.stage === "diarization_ready") {
        log(ev.engine === "pyannote"
          ? `Sprecher-Erkennung fertig: ${ev.speakers} Sprecher gefunden`
          : "Sprecher-Erkennung: Fallback auf Granite-Modell");
      }
      break;
    }
    case "progress": {
      const pct = ev.progress;
      if (pct > 0) {
        setProgress(pct, false);
        drawWaveform(pct / 100);
        const elapsed = (Date.now() - state.startedAt) / 1000;
        els.statusDetail.textContent = `${Math.round(pct)} % · ${Math.round(elapsed)} s`;
      }
      log(`Fortschritt: ${Math.round(pct)} %`);
      break;
    }
    case "delta":
      // Token-level live text of the chunk currently being decoded.
      state.liveText += ev.text;
      deltaChars += ev.text.length;
      deltaCount++;
      scheduleLiveRender();
      break;
    case "partial":
      // Chunk finished — its clean text replaces the raw token stream.
      state.liveText = "";
      state.partials.push(ev);
      if (els.cardResult.hidden) els.cardResult.hidden = false;
      renderTranscript();
      break;
    case "segment":
      state.segments.push(ev);
      appendSegment(ev);
      break;
    case "result":
      state.resultText = ev.text || "";
      if (ev.words) state.words = ev.words;
      break;
    case "error":
      fail(`Server-Fehler: ${ev.message}`);
      break;
    default:
      break; // Rohzeile steht bereits im Protokoll
  }
}

function onRequestDone(xhr) {
  state.xhr = null;
  stopQueuePolling();
  flushDeltaLog();
  logRaw("HTTP", `${xhr.status} ${xhr.statusText || ""} · ${xhr.responseText.length} B NDJSON`,
    xhr.status === 200 ? undefined : "err");
  if (xhr.status !== 200) {
    let msg = `HTTP ${xhr.status}`;
    try { msg = JSON.parse(xhr.responseText).detail || msg; } catch { /* keep */ }
    fail(msg);
    return;
  }
  if (!state.resultText && !state.segments.length) {
    fail("Der Server hat keine Transkription geliefert.");
    return;
  }
  const secs = Math.round((Date.now() - state.startedAt) / 1000);
  state.processingSecs = secs;
  setPhase("done", "Fertig.", `${secs} s gesamt`);
  setStep("done", "done");
  setProgress(100, false);
  drawWaveform(1);
  els.btnCancel.hidden = true;
  els.btnStart.disabled = false;
  els.cardResult.hidden = false;
  state.liveText = "";
  renderTranscript();
  renderSpeakerEditor();
  log(`Transkription abgeschlossen in ${secs} s (${state.segments.length} Segmente).`);
  saveSession();
  els.cardResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
  state.finishCb?.resolve();
  state.finishCb = null;
  // Auto-summary (configured in step 2) — single-file mode only.
  if ($("#sum-auto").checked && !els.cardResult.classList.contains("batch-mode")) {
    log("Starte automatische Zusammenfassung…");
    runSummary().catch((e) => log(`Zusammenfassung: ${e.message}`, "err"));
  }
}

function fail(msg) {
  if (state.xhr) { state.xhr.abort(); state.xhr = null; }
  stopQueuePolling();
  setProgress(0, false);
  for (const li of els.stepper.children) {
    if (li.className === "active") li.className = "error";
  }
  els.statusText.textContent = "Fehlgeschlagen";
  els.statusDetail.textContent = "";
  els.btnStart.disabled = false;
  els.btnCancel.hidden = true;
  log(msg, "err");
  showFileError(msg);
  state.finishCb?.reject(new Error(msg));
  state.finishCb = null;
}

function cancelTranscription() {
  if (state.xhr) { state.xhr.abort(); state.xhr = null; }
  stopQueuePolling();
  setProgress(0, false);
  els.cardStatus.hidden = true;
  els.btnStart.disabled = false;
  log("Abgebrochen.");
}

/* ── Result rendering & export ──────────────────────────── */

/* Deltas arrive at token rate — repaint at most ~8x/s. */
let liveRenderTimer = 0;

function scheduleLiveRender() {
  if (liveRenderTimer) return;
  liveRenderTimer = setTimeout(() => {
    liveRenderTimer = 0;
    if (els.cardResult.hidden) els.cardResult.hidden = false;
    renderTranscript();
  }, 120);
}

function speakerLabel(spk) {
  const custom = (state.speakerNames || {})[spk];
  if (custom) return custom;
  const m = /(\d+)/.exec(spk || "");
  return m ? `Sprecher ${parseInt(m[1], 10) + 1}` : spk;
}

/* Editable name per detected speaker — renames flow into the transcript
   rendering and all downloads (they resolve labels via speakerLabel()). */
function renderSpeakerEditor() {
  const box = $("#speaker-names");
  const speakers = [...new Set(state.segments.map((s) => s.speaker).filter(Boolean))];
  box.textContent = "";
  box.hidden = speakers.length === 0;
  if (!speakers.length) return;
  const title = document.createElement("span");
  title.className = "spk-edit-title";
  title.textContent = "Sprecher benennen:";
  box.appendChild(title);
  for (const spk of speakers) {
    const inp = document.createElement("input");
    inp.type = "text";
    inp.placeholder = speakerLabel(spk);
    inp.value = (state.speakerNames || {})[spk] || "";
    inp.addEventListener("input", () => {
      state.speakerNames = state.speakerNames || {};
      const v = inp.value.trim();
      if (v) state.speakerNames[spk] = v;
      else delete state.speakerNames[spk];
      renderTranscript();
      saveSession();
    });
    box.appendChild(inp);
  }
}

function appendSegment(seg) {
  if (els.cardResult.hidden) els.cardResult.hidden = false;
  renderTranscript();
}

function renderTranscript() {
  const el = els.transcript;
  el.textContent = "";
  wordSpans = [];
  lastLitSpan = null;
  const isLive = !state.segments.length && (state.partials.length > 0 || !!state.liveText);
  if (state.segments.length) {
    let lastSpeaker = null;
    for (const seg of state.segments) {
      if (seg.speaker && seg.speaker !== lastSpeaker) {
        const spk = document.createElement("span");
        spk.className = "spk";
        spk.textContent = `\n[${speakerLabel(seg.speaker)}] `;
        el.appendChild(spk);
        lastSpeaker = seg.speaker;
      }
      if (seg.words && seg.words.length) {
        // Word spans enable karaoke-style highlighting while playing/scrubbing.
        for (const w of seg.words) {
          const span = document.createElement("span");
          span.className = "w";
          span.textContent = w.word + " ";
          span._start = w.start;
          span._end = w.end;
          el.appendChild(span);
          wordSpans.push(span);
        }
      } else {
        el.appendChild(document.createTextNode((seg.text || "").trim() + " "));
      }
    }
  } else if (state.partials.length || state.liveText) {
    // Live preview while the server is still transcribing: finished chunks +
    // the raw token stream of the current chunk. Model tags become readable
    // labels; timestamp tags and silence markers are hidden.
    const text = (state.partials.map((p) => p.text).join(" ") + " " + state.liveText)
      .replace(/<\|[^|>]*\|>/g, "")
      .replace(/\[T:\d*\]?/g, "")
      .replace(/(^|\s)_(?=\s|$)/g, " ")
      .replace(/\[Speaker\s+(\d+)\]\s*:/g, (_, n) => `\n[Sprecher ${n}] `);
    el.textContent = text.replace(/[ \t]+/g, " ").trim() + " ▌";
  } else {
    el.textContent = state.resultText;
  }
  // Live-Vorschau folgt immer, das fertige Transkript nur mit Auto-Scroll.
  if (isLive || autoScrollAllowed()) el.scrollTop = el.scrollHeight;
}

function plainTextOf(segments, resultText) {
  if (!segments.length) return resultText || "";
  const parts = [];
  let lastSpeaker = null;
  for (const seg of segments) {
    let t = (seg.text || "").trim();
    if (!t) continue;
    if (seg.speaker && seg.speaker !== lastSpeaker) {
      parts.push(`\n[${speakerLabel(seg.speaker)}] ${t}`);
      lastSpeaker = seg.speaker;
    } else {
      parts.push(t);
    }
  }
  return parts.join(" ").trim();
}

function plainText() { return plainTextOf(state.segments, state.resultText); }

function toSrt(segments = state.segments) {
  const lines = [];
  segments.forEach((seg, i) => {
    let t = (seg.text || "").trim();
    if (seg.speaker) t = `[${speakerLabel(seg.speaker)}] ${t}`;
    lines.push(String(i + 1), `${fmtTs(seg.start, ",")} --> ${fmtTs(seg.end, ",")}`, t, "");
  });
  return lines.join("\n");
}

function toVtt(segments = state.segments) {
  const lines = ["WEBVTT", ""];
  for (const seg of segments) {
    let t = (seg.text || "").trim();
    if (seg.speaker) t = `<v ${speakerLabel(seg.speaker)}>${t}`;
    lines.push(`${fmtTs(seg.start, ".")} --> ${fmtTs(seg.end, ".")}`, t, "");
  }
  return lines.join("\n");
}

function buildExport(fmt, segments, text, filename, duration) {
  let content, mime = "text/plain";
  if (fmt === "txt") content = text;
  else if (fmt === "srt") content = toSrt(segments);
  else if (fmt === "vtt") { content = toVtt(segments); mime = "text/vtt"; }
  else {
    content = JSON.stringify({ text, duration, segments }, null, 2);
    mime = "application/json";
  }
  const base = (filename || "transkript").replace(/\.[^.]+$/, "");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = `${base}.${fmt}`;
  a.click();
  URL.revokeObjectURL(a.href);
  log(`Download: ${base}.${fmt}`);
}

function download(fmt) {
  buildExport(fmt, state.segments, plainText(), state.file?.name, state.duration);
}

function selectedFormats() {
  const fmts = Array.from(document.querySelectorAll(".fmt:checked")).map((c) => c.value);
  return fmts.length ? fmts : ["txt"];
}

/* Letztes Ergebnis übersteht einen Reload — nur der Text, die Audiodatei
   kann der Browser nicht wiederherstellen. */
const SESSION_KEY = "gsa:last-result";

// Signatur der Datei, zu der das gespeicherte Transkript gehört — wird die
// gleiche Datei nach einem Reload erneut gewählt, bleibt das Transkript stehen.
let sessionFileSig = null;

function fileSig(file) {
  if (!file) return null;
  return { name: file.name, size: file.size, lastModified: file.lastModified || 0 };
}

function sameFile(sig, file) {
  const b = fileSig(file);
  return !!sig && !!b && sig.name === b.name && sig.size === b.size &&
    sig.lastModified === b.lastModified;
}

function saveSession() {
  try {
    sessionFileSig = fileSig(state.file);
    const json = JSON.stringify({
      v: 1,
      savedAt: Date.now(),
      segments: state.segments,
      resultText: state.resultText,
      speakerNames: state.speakerNames || {},
      duration: state.duration,
      processingSecs: state.processingSecs,
      fileName: state.file?.name || state.uploadName || null,
      file: sessionFileSig,
    });
    if (json.length > 4_000_000) return; // localStorage-Quota nicht sprengen
    localStorage.setItem(SESSION_KEY, json);
  } catch { /* Quota / privater Modus — Persistenz ist optional */ }
}

function clearSession() {
  sessionFileSig = null;
  try { localStorage.removeItem(SESSION_KEY); } catch { /* egal */ }
}

function restoreSession() {
  let data = null;
  try { data = JSON.parse(localStorage.getItem(SESSION_KEY) || "null"); } catch { return; }
  if (!data || data.v !== 1) return;
  if (!data.segments?.length && !data.resultText) return;
  state.segments = data.segments || [];
  state.resultText = data.resultText || "";
  state.speakerNames = data.speakerNames || {};
  state.processingSecs = data.processingSecs ?? null;
  sessionFileSig = data.file || null;
  els.cardResult.hidden = false;
  renderTranscript();
  renderSpeakerEditor();
  const when = new Date(data.savedAt).toLocaleString("de-DE", { hour12: false });
  log(`Letztes Transkript wiederhergestellt (${data.fileName || "unbenannt"}, ${when}). ` +
      `Dieselbe Datei erneut wählen, dann bleibt es erhalten und Wiedergabe/Wortsprung gehen wieder.`);
}

function resetResult() {
  clearSession();
  els.cardResult.hidden = true;
  els.cardStatus.hidden = true;
  els.btnCancel.hidden = false;
  els.transcript.textContent = "";
  state.segments = [];
  state.partials = [];
  state.liveText = "";
  state.resultText = "";
  state.speakerNames = {};
  $("#speaker-names").hidden = true;
  els.fileError.hidden = true;
  sumOutput.hidden = true;
  renderSummary("", { plain: true });
  $("#sum-actions").hidden = true;
}

/* ── Zusammenfassung (Ollama) ───────────────────────────── */

const sumModel = $("#sum-model");
const btnSummarize = $("#btn-summarize");
const sumOutput = $("#summary-output");
let summarizing = false;
let sumMarkdown = ""; // Rohtext der Zusammenfassung (Quelle für Rendering & Copy)

/* ── Mini-Markdown → HTML (kein externes Paket nötig) ───── */

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function mdInline(s) {
  const codes = [];
  let t = escapeHtml(s).replace(/`([^`]+)`/g, (_, c) => `\u0001${codes.push(c) - 1}\u0001`);
  t = t
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_\w])_([^_\n]+)_/g, "$1<em>$2</em>")
    .replace(/~~([^~]+)~~/g, "<del>$1</del>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return t.replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${codes[i]}</code>`);
}

const RE_LIST = /^\s*([-*+]|\d+[.)])\s+(.*)$/;
const RE_TABLE_SEP = /^\s*\|?[\s:|-]*-[\s:|-]*$/;

function splitRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function renderMarkdown(md) {
  const lines = md.replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let para = [];
  const flushPara = () => {
    if (para.length) out.push(`<p>${mdInline(para.join(" "))}</p>`);
    para = [];
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flushPara();
      const body = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i++;
      out.push(`<pre><code>${escapeHtml(body.join("\n"))}</code></pre>`);
      continue;
    }
    if (!line.trim()) { flushPara(); i++; continue; }
    const h = line.match(/^\s*(#{1,6})\s+(.*)$/);
    if (h) {
      flushPara();
      const lvl = h[1].length;
      out.push(`<h${lvl}>${mdInline(h[2].replace(/\s*#+\s*$/, ""))}</h${lvl}>`);
      i++;
      continue;
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) { flushPara(); out.push("<hr>"); i++; continue; }
    if (line.includes("|") && i + 1 < lines.length && RE_TABLE_SEP.test(lines[i + 1])) {
      flushPara();
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(splitRow(lines[i++]));
      const th = head.map((c) => `<th>${mdInline(c)}</th>`).join("");
      const body = rows
        .map((r) => `<tr>${r.map((c) => `<td>${mdInline(c)}</td>`).join("")}</tr>`)
        .join("");
      out.push(`<table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`);
      continue;
    }
    if (/^\s*>/.test(line)) {
      flushPara();
      const body = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push(`<blockquote>${renderMarkdown(body.join("\n"))}</blockquote>`);
      continue;
    }
    const li = line.match(RE_LIST);
    if (li) {
      flushPara();
      const ordered = /\d/.test(li[1]);
      const tag = ordered ? "ol" : "ul";
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(RE_LIST);
        if (m) {
          if (/\d/.test(m[1]) !== ordered) break; // Listentyp wechselt
          items.push(m[2]);
          i++;
          continue;
        }
        // Eingerückte Folgezeile gehört zum letzten Punkt.
        if (items.length && lines[i].trim() && /^\s{2,}/.test(lines[i])) {
          items[items.length - 1] += ` ${lines[i].trim()}`;
          i++;
          continue;
        }
        break;
      }
      out.push(`<${tag}>${items.map((t) => `<li>${mdInline(t)}</li>`).join("")}</${tag}>`);
      continue;
    }
    para.push(line.trim());
    i++;
  }
  flushPara();
  return out.join("\n");
}

/* Kopiert formatiert (text/html) mit Markdown als Plaintext-Fallback, damit
   Word/Outlook die Formatierung übernehmen und Editoren den Rohtext. */
async function copyRich(html, text) {
  try {
    if (window.ClipboardItem && navigator.clipboard.write) {
      await navigator.clipboard.write([
        new ClipboardItem({
          "text/html": new Blob([html], { type: "text/html" }),
          "text/plain": new Blob([text], { type: "text/plain" }),
        }),
      ]);
      return;
    }
  } catch { /* fällt unten auf Plaintext zurück */ }
  await navigator.clipboard.writeText(text);
}

function renderSummary(md, { plain = false } = {}) {
  sumMarkdown = md;
  sumOutput.classList.toggle("md", !plain);
  if (plain) sumOutput.textContent = md;
  else sumOutput.innerHTML = renderMarkdown(md);
}

async function loadSummaryModels() {
  try {
    const res = await fetch("/v1/summary/models");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    sumModel.textContent = "";
    for (const m of data.models) {
      const o = document.createElement("option");
      o.value = m.name;
      o.textContent = m.parameter_size ? `${m.name} (${m.parameter_size})` : m.name;
      sumModel.appendChild(o);
    }
    if (data.models.length) {
      sumModel.disabled = false;
      btnSummarize.disabled = false;
      log(`Ollama verbunden: ${data.models.length} Modelle für Zusammenfassungen.`);
    } else {
      sumModel.options[0] = new Option("Keine Modelle gefunden", "");
    }
  } catch (err) {
    sumModel.textContent = "";
    sumModel.appendChild(new Option("Ollama nicht erreichbar", ""));
    log(`Zusammenfassung nicht verfügbar: ${err.message}`, "warn");
  }
}

async function runSummary() {
  const text = plainText();
  if (!text.trim()) { log("Kein Transkript zum Zusammenfassen vorhanden.", "warn"); return; }
  if (!sumModel.value || summarizing) return;
  summarizing = true;
  $("#summary-box").open = true;
  btnSummarize.disabled = true;
  btnSummarize.textContent = "Fasst zusammen…";
  sumOutput.hidden = false;
  renderSummary("", { plain: true });
  $("#sum-actions").hidden = true;
  $("#sum-stats").textContent = "";
  let raw = "";
  let thinkChars = 0;
  log(`Zusammenfassung gestartet (${sumModel.value}, ${text.length} Zeichen Transkript).`);
  renderSummary("(Modell wird geladen…)", { plain: true });
  try {
    const res = await fetch("/v1/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        model: sumModel.value,
        system_prompt: $("#sum-system").value.trim() || null,
      }),
    });
    if (!res.ok || !res.body) {
      let msg = `HTTP ${res.status}`;
      try { msg = (await res.json()).detail || msg; } catch { /* keep */ }
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let failed = null;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let ev;
        try { ev = JSON.parse(line); } catch { continue; }
        if (ev.type === "delta") {
          raw += ev.text;
          // Reasoning models wrap their thinking in <think>…</think> — hide it.
          const visible = raw.replace(/<think>[\s\S]*?(<\/think>|$)/, "").trimStart();
          // Während des Streams Rohtext zeigen (halbe Markdown-Zeilen sähen
          // sonst zappelig aus), gerendert wird am Ende.
          renderSummary(visible || "(Modell denkt nach…)", { plain: true });
          sumOutput.scrollTop = sumOutput.scrollHeight;
        } else if (ev.type === "thinking") {
          thinkChars += ev.text.length;
          if (!raw) renderSummary(`(Modell denkt nach… ${thinkChars} Zeichen)`, { plain: true });
        } else if (ev.type === "done") {
          renderSummary(raw.replace(/<think>[\s\S]*?(<\/think>|$)/, "").trim());
          $("#sum-actions").hidden = false;
          $("#sum-stats").textContent =
            (ev.duration_seconds ? `${ev.duration_seconds} s · ` : "") + (ev.model || "");
          log(`Zusammenfassung fertig (${ev.eval_count ?? "?"} Tokens, ${ev.duration_seconds ?? "?"} s).`);
        } else if (ev.type === "error") {
          failed = ev.message;
        }
      }
    }
    if (failed) throw new Error(failed);
  } catch (err) {
    sumOutput.hidden = false;
    renderSummary(`${sumMarkdown}\n\n[Fehler: ${err.message}]`, { plain: true });
    log(`Zusammenfassung fehlgeschlagen: ${err.message}`, "err");
  } finally {
    summarizing = false;
    btnSummarize.disabled = !sumModel.value;
    btnSummarize.textContent = "Zusammenfassen";
  }
}

/* ── Aktions-Trace (für nachvollziehbare Bug-Reports) ───── */

const trace = [];

function traceEvent(kind, detail) {
  trace.push({ t: new Date().toISOString(), [kind === "change" ? "change" : "click"]: detail });
  if (trace.length > 300) trace.shift();
}

document.addEventListener("click", (e) => {
  const el = e.target.closest("button, label.radio-card, summary, .dropzone");
  if (!el) return;
  traceEvent("click", el.id || (el.textContent || "").trim().slice(0, 40) || el.className);
}, true);

document.addEventListener("change", (e) => {
  const el = e.target;
  if (!el.id && !el.name) return;
  if (el.id === "fb-comment") return; // Kommentartext nicht in den Trace
  const val = el.type === "checkbox" ? el.checked
    : el.type === "file" ? "(Datei gewählt)"
    : String(el.value).slice(0, 60);
  traceEvent("change", `${el.id || el.name}=${val}`);
}, true);

/* ── Feedback ───────────────────────────────────────────── */

let fbRating = null;

async function sendFeedback() {
  const comment = $("#fb-comment").value.trim();
  if (!fbRating && !comment) {
    log("Feedback: bitte Bewertung anklicken oder Text eingeben.", "warn");
    return;
  }
  const settingsUsed = {
    speakers: document.querySelector('input[name="speakers"]:checked').value,
    language: $("#opt-language").value || null,
    model: $("#opt-model").value || null,
    word_timestamps: $("#opt-word-ts").checked,
    translate_to: $("#opt-translate").value || null,
    summary_auto: $("#sum-auto").checked,
    summary_model: $("#sum-model").value || null,
  };
  const body = {
    rating: fbRating,
    category: $("#fb-category").value,
    comment: comment || null,
    context: {
      filename: state.file?.name || null,
      duration_seconds: state.duration ? Math.round(state.duration) : null,
      upload_bytes: state.uploadBlob?.size || null,
      processing_seconds: state.processingSecs ?? null,
      segments: state.segments.length,
      status: els.statusText.textContent.trim() || null,
      settings: settingsUsed,
      ui_log: logBuf.slice(-200).map((e) => `[${e.t}] ${e.msg}`),
      actions: trace.slice(-200),
      browser: {
        user_agent: navigator.userAgent,
        viewport: `${window.innerWidth}x${window.innerHeight}`,
      },
    },
  };
  try {
    const res = await fetch("/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    $("#fb-thanks").hidden = false;
    $("#fb-comment").value = "";
    fbRating = null;
    document.querySelectorAll(".fb-rate").forEach((b) => b.classList.remove("selected"));
    setTimeout(() => { $("#fb-thanks").hidden = true; }, 4000);
    log(`Feedback gesendet (${body.category}${body.rating ? ", " + body.rating : ""}).`);
  } catch (err) {
    log(`Feedback senden fehlgeschlagen: ${err.message}`, "err");
  }
}

function resetFile() {
  resetResult();
  audioEl.pause();
  audioEl.removeAttribute("src");
  if (state.audioUrl) { URL.revokeObjectURL(state.audioUrl); state.audioUrl = null; }
  btnPlay.hidden = true;
  playTime.hidden = true;
  btnPlayTr.disabled = true;
  state.file = null;
  state.uploadBlob = null;
  state.peaks = null;
  state.duration = null;
  state.queue = [];
  renderBatchList();
  els.dzIdle.hidden = false;
  els.dzFile.hidden = true;
  els.dropzone.classList.remove("has-file");
  els.fileInput.value = "";
  els.btnStart.disabled = true;
}

/* ── Batch (mehrere Dateien nacheinander) ───────────────── */

function onFilesSelected(files) {
  if (!files.length) return;
  if (files.length === 1) {
    state.queue = [];
    renderBatchList();
    els.cardResult.classList.remove("batch-mode");
    clearBatchResults();
    onFileSelected(files[0]);
    return;
  }
  resetFile();
  els.cardResult.classList.add("batch-mode");
  state.queue = files.map((f) => ({ file: f, status: "wartet", result: null }));
  els.dzIdle.hidden = true;
  els.dzFile.hidden = false;
  els.dropzone.classList.add("has-file");
  els.fileName.textContent = `${files.length} Dateien`;
  const total = files.reduce((s, f) => s + f.size, 0);
  els.fileInfo.textContent = `${fmtSize(total)} gesamt — werden nacheinander verarbeitet`;
  state.peaks = null;
  drawWaveform(0);
  renderBatchList();
  els.btnStart.disabled = false;
  log(`${files.length} Dateien ausgewählt (Batch).`);
}

function renderBatchList() {
  const ul = $("#batch-list");
  if (!state.queue.length) { ul.hidden = true; ul.textContent = ""; return; }
  ul.hidden = false;
  ul.textContent = "";
  const icons = { wartet: "○", "läuft": "◐", fertig: "✓", fehler: "✗" };
  for (const item of state.queue) {
    const li = document.createElement("li");
    li.className = `bstatus-${item.status === "läuft" ? "laeuft" : item.status}`;
    const ic = document.createElement("span"); ic.className = "b-ic"; ic.textContent = icons[item.status];
    const nm = document.createElement("span"); nm.className = "b-name"; nm.textContent = item.file.name;
    const st = document.createElement("span"); st.className = "b-st mono muted";
    st.textContent = `${fmtSize(item.file.size)} · ${item.status}`;
    li.append(ic, nm, st);
    ul.appendChild(li);
  }
}

async function runBatch() {
  const queue = state.queue;
  els.cardResult.classList.add("batch-mode");
  clearBatchResults();
  for (const item of queue) {
    item.status = "läuft";
    renderBatchList();
    try {
      await onFileSelected(item.file);
      state.queue = queue; // onFileSelected leaves the queue alone, keep reference
      els.btnStart.disabled = true;
      if (!state.uploadBlob) throw new Error("Datei konnte nicht gelesen werden");
      if (state.duration && state.duration > state.maxSeconds) throw new Error("zu lang für den Server");
      await transcribeCurrent();
      item.result = {
        segments: [...state.segments],
        text: plainText(),
        duration: state.duration,
        name: item.file.name,
      };
      item.status = "fertig";
      appendBatchResult(item);
    } catch (e) {
      item.status = "fehler";
      log(`${item.file.name}: ${e.message}`, "err");
    }
    renderBatchList();
  }
  els.btnStart.disabled = false;
  els.transcript.textContent = "";
  const ok = queue.filter((i) => i.status === "fertig").length;
  setPhase("done", "Alle Dateien verarbeitet.", `${ok}/${queue.length} erfolgreich`);
  setStep("done", "done");
  log(`Batch abgeschlossen: ${ok}/${queue.length} erfolgreich.`);
}

function clearBatchResults() { $("#batch-results").textContent = ""; }

function appendBatchResult(item) {
  els.cardResult.hidden = false;
  const d = document.createElement("details");
  d.className = "batch-result";
  d.open = true;
  const s = document.createElement("summary");
  s.textContent = `✓ ${item.result.name}`;
  const body = document.createElement("div");
  body.className = "transcript";
  body.textContent = item.result.text;
  const acts = document.createElement("div");
  acts.className = "result-actions";
  const sel = document.createElement("button");
  sel.type = "button";
  sel.className = "btn-ghost";
  sel.textContent = "Auswahl herunterladen";
  sel.addEventListener("click", () => {
    for (const fmt of selectedFormats()) {
      buildExport(fmt, item.result.segments, item.result.text, item.result.name, item.result.duration);
    }
  });
  acts.appendChild(sel);
  for (const fmt of ["txt", "srt", "vtt", "json"]) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn-ghost";
    b.textContent = "." + fmt;
    b.addEventListener("click", () =>
      buildExport(fmt, item.result.segments, item.result.text, item.result.name, item.result.duration)
    );
    acts.appendChild(b);
  }
  d.append(s, body, acts);
  $("#batch-results").appendChild(d);
}

/* ── Wiring ─────────────────────────────────────────────── */

els.dropzone.addEventListener("click", (e) => {
  if (e.target.closest("#btn-reset")) return;
  if (!state.file) els.fileInput.click();
});
els.dropzone.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && !state.file) { e.preventDefault(); els.fileInput.click(); }
});
els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files.length) onFilesSelected(Array.from(els.fileInput.files));
});
["dragover", "dragenter"].forEach((t) =>
  els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((t) =>
  els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.remove("dragover"); })
);
els.dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) onFilesSelected(Array.from(e.dataTransfer.files));
});

document.querySelectorAll('input[name="speakers"]').forEach((r) =>
  r.addEventListener("change", () => {
    els.speakerOpts.hidden =
      document.querySelector('input[name="speakers"]:checked').value !== "multi";
  })
);

/* ── Model capabilities: only offer what the selected model can do ── */
const LANGS_COHERE = ["de", "en", "fr", "es", "it", "pt", "nl", "pl", "el", "zh", "ja", "ko", "vi", "ar"];
const LANGS_GRANITE = ["de", "en", "fr", "es", "pt"];
const MODEL_CAPS = {
  // wordTs: false | true | "always"; langs: null = alle (+ Autodetect);
  // needsLang: Sprachangabe Pflicht (kein Autodetect)
  "fusion": { wordTs: "always", keywords: false, multi: true, langs: LANGS_COHERE, needsLang: true },
  "": { wordTs: true, keywords: true, multi: true, langs: [...LANGS_GRANITE, "ja"] },
  "granite-speech-4.1-2b-plus": { wordTs: true, keywords: true, multi: true, langs: LANGS_GRANITE },
  "granite-speech-4.1-2b-nar": { wordTs: false, keywords: true, multi: false, langs: LANGS_GRANITE },
  "cohere-transcribe": { wordTs: false, keywords: false, multi: false, langs: LANGS_COHERE, needsLang: true },
  "qwen3-asr": { wordTs: false, keywords: false, multi: false, langs: null },
};

function applyModelCaps() {
  const caps = MODEL_CAPS[$("#opt-model").value] || MODEL_CAPS[""];
  const ts = $("#opt-word-ts");
  if (caps.wordTs === "always") {
    ts.checked = true;
    ts.disabled = true;
  } else {
    ts.disabled = !caps.wordTs;
    if (!caps.wordTs) ts.checked = false;
  }
  const kw = $("#opt-keywords");
  kw.disabled = !caps.keywords;
  kw.placeholder = caps.keywords
    ? "z. B. Kubernetes, VR Bits, Leipzig"
    : "vom gewählten Modell nicht unterstützt";
  const langSel = $("#opt-language");
  for (const opt of langSel.options) {
    opt.disabled = opt.value === ""
      ? !!caps.needsLang
      : !!caps.langs && !caps.langs.includes(opt.value);
  }
  if (langSel.selectedOptions[0]?.disabled) {
    langSel.value = !caps.langs || caps.langs.includes("de") ? "de" : caps.langs[0];
  }
  const multiRadio = document.querySelector('input[name="speakers"][value="multi"]');
  multiRadio.disabled = !caps.multi;
  if (!caps.multi && multiRadio.checked) {
    document.querySelector('input[name="speakers"][value="single"]').checked = true;
    els.speakerOpts.hidden = true;
  }
}
$("#opt-model").addEventListener("change", applyModelCaps);
applyModelCaps();

els.btnReset.addEventListener("click", resetFile);
els.btnStart.addEventListener("click", startTranscription);
els.btnCancel.addEventListener("click", cancelTranscription);
$("#btn-copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(plainText());
  $("#btn-copy").textContent = "Kopiert ✓";
  setTimeout(() => { $("#btn-copy").textContent = "Text kopieren"; }, 1500);
});
document.querySelectorAll(".dl").forEach((b) =>
  b.addEventListener("click", () => download(b.dataset.fmt))
);
$("#btn-dl-selected").addEventListener("click", () => {
  for (const fmt of selectedFormats()) download(fmt);
});
document.querySelectorAll(".fb-rate").forEach((b) =>
  b.addEventListener("click", () => {
    fbRating = fbRating === b.dataset.rating ? null : b.dataset.rating;
    document.querySelectorAll(".fb-rate").forEach((x) =>
      x.classList.toggle("selected", x.dataset.rating === fbRating)
    );
    $("#fb-thanks").hidden = true;
  })
);
$("#fb-send").addEventListener("click", sendFeedback);
btnSummarize.addEventListener("click", () => {
  runSummary().catch((e) => log(`Zusammenfassung: ${e.message}`, "err"));
});
$("#btn-sum-copy").addEventListener("click", async () => {
  await copyRich(renderMarkdown(sumMarkdown), sumMarkdown);
  $("#btn-sum-copy").textContent = "Kopiert ✓";
  setTimeout(() => { $("#btn-sum-copy").textContent = "Zusammenfassung kopieren"; }, 1500);
});
loadSummaryModels();

refreshHealth();
setInterval(refreshHealth, 30000);
log("UI bereit.");
restoreSession();
