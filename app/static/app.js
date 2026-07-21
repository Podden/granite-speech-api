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
  resultText: "",
  words: [],
  startedAt: 0,
};

/* ── Logging ────────────────────────────────────────────── */

function log(msg, cls) {
  const t = new Date().toLocaleTimeString("de-DE", { hour12: false });
  const line = document.createElement("span");
  if (cls) line.className = cls;
  line.textContent = `[${t}] ${msg}\n`;
  els.log.appendChild(line);
  els.log.scrollTop = els.log.scrollHeight;
}

/* ── Health ─────────────────────────────────────────────── */

async function refreshHealth() {
  try {
    const res = await fetch("/health");
    const h = await res.json();
    els.healthDot.className = "dot ok";
    const device = h.device === "cuda" ? "GPU" : "CPU";
    els.healthText.textContent = h.loaded_model
      ? `Bereit (${device}, Modell geladen)`
      : `Bereit (${device})`;
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
  resetResult();
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

    const wav = encodeWav(mono, audio.sampleRate);
    // For plain audio files the original can be smaller than 16-bit WAV
    // (e.g. opus/mp3) — upload whichever is smaller. Videos always get
    // stripped to WAV.
    if (!isVideo && file.size < wav.size) {
      state.uploadBlob = file;
      state.uploadName = file.name;
      log(`Original (${fmtSize(file.size)}) ist kleiner als konvertiertes WAV (${fmtSize(wav.size)}) — Original wird hochgeladen.`);
    } else {
      state.uploadBlob = wav;
      state.uploadName = file.name.replace(/\.[^.]+$/, "") + ".wav";
      log(`Audio extrahiert: ${fmtDuration(audio.duration)}, 16 kHz mono, ${fmtSize(wav.size)}` +
          (isVideo ? ` (statt ${fmtSize(file.size)} Video — ${(100 - (wav.size / file.size) * 100).toFixed(0)} % gespart)` : ""));
    }
    els.fileInfo.textContent =
      `${fmtDuration(audio.duration)} · Upload: ${fmtSize(state.uploadBlob.size)}` +
      (isVideo ? ` (Tonspur aus ${fmtSize(file.size)} Video)` : "");
  } catch (err) {
    log(`Browser-Dekodierung fehlgeschlagen (${err.message || err.name}) — Original wird hochgeladen, der Server extrahiert die Tonspur.`, "warn");
    state.uploadBlob = file;
    state.uploadName = file.name;
    state.peaks = null;
    drawWaveform(0);
    els.fileInfo.textContent = `${fmtSize(file.size)} · Dauer unbekannt (Server prüft)`;
  }

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

function drawWaveform(progress) {
  const canvas = els.waveform;
  const ctx = canvas.getContext("2d");
  const css = getComputedStyle(document.documentElement);
  const W = canvas.width, H = canvas.height, mid = H / 2;
  ctx.clearRect(0, 0, W, H);
  if (!state.peaks) {
    ctx.fillStyle = css.getPropertyValue("--wave").trim();
    ctx.fillRect(0, mid - 1, W, 2);
    return;
  }
  const n = state.peaks.length;
  const bw = W / n;
  const doneX = W * (progress || 0);
  for (let i = 0; i < n; i++) {
    const x = i * bw;
    const h = Math.max(2, state.peaks[i] * (H - 8));
    ctx.fillStyle = css.getPropertyValue(x < doneX ? "--wave-done" : "--wave").trim();
    ctx.fillRect(x, mid - h / 2, Math.max(1, bw - 1), h);
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

/* ── Transcription ──────────────────────────────────────── */

function buildFormData() {
  const fd = new FormData();
  fd.append("file", state.uploadBlob, state.uploadName);
  fd.append("stream", "true");

  const multi = document.querySelector('input[name="speakers"]:checked').value === "multi";
  if (multi) {
    fd.append("speaker_attribution", "true");
    const min = $("#min-speakers").value, max = $("#max-speakers").value;
    if (min) fd.append("min_speakers", min);
    if (max) fd.append("max_speakers", max);
  }
  if ($("#opt-word-ts").checked) fd.append("timestamp_granularities[]", "word");
  const lang = $("#opt-language").value;
  if (lang) fd.append("language", lang);
  const model = $("#opt-model").value;
  if (model) fd.append("model", model);
  const translateTo = $("#opt-translate").value;
  if (translateTo) { fd.append("translate", "true"); fd.append("translate_to", translateTo); }
  const keywords = $("#opt-keywords").value.trim();
  if (keywords) fd.append("prompt", keywords);
  return fd;
}

function startTranscription() {
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
  state.words = [];
  state.resultText = "";

  const fd = buildFormData();
  log(`Starte Upload: ${state.uploadName} (${fmtSize(state.uploadBlob.size)})`);

  const xhr = new XMLHttpRequest();
  state.xhr = xhr;
  let parsedTo = 0;

  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = (e.loaded / e.total) * 100;
    setProgress(pct * 0.999, false);
    els.statusDetail.textContent = `${fmtSize(e.loaded)} / ${fmtSize(e.total)}`;
    if (e.loaded >= e.total) {
      setPhase("transcribe", "Server transkribiert…", "");
      setProgress(0, true);
      log("Upload abgeschlossen — Modell wird geladen falls nötig (kann beim ersten Mal dauern).");
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

function handleEvent(line) {
  let ev;
  try { ev = JSON.parse(line); } catch { log(`Unlesbares Server-Event: ${line}`, "warn"); return; }
  switch (ev.type) {
    case "duration":
      if (!state.duration) state.duration = ev.duration;
      log(`Server bestätigt Audio: ${fmtDuration(ev.duration)}`);
      break;
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
      log(`Event: ${line}`);
  }
}

function onRequestDone(xhr) {
  state.xhr = null;
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
  setPhase("done", "Fertig.", `${secs} s gesamt`);
  setStep("done", "done");
  setProgress(100, false);
  drawWaveform(1);
  els.btnCancel.hidden = true;
  els.btnStart.disabled = false;
  els.cardResult.hidden = false;
  renderTranscript();
  log(`Transkription abgeschlossen in ${secs} s (${state.segments.length} Segmente).`);
  els.cardResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function fail(msg) {
  if (state.xhr) { state.xhr.abort(); state.xhr = null; }
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
}

function cancelTranscription() {
  if (state.xhr) { state.xhr.abort(); state.xhr = null; }
  setProgress(0, false);
  els.cardStatus.hidden = true;
  els.btnStart.disabled = false;
  log("Abgebrochen.");
}

/* ── Result rendering & export ──────────────────────────── */

function speakerLabel(spk) {
  const m = /(\d+)/.exec(spk || "");
  return m ? `Sprecher ${parseInt(m[1], 10) + 1}` : spk;
}

function appendSegment(seg) {
  if (els.cardResult.hidden) els.cardResult.hidden = false;
  renderTranscript();
}

function renderTranscript() {
  const el = els.transcript;
  el.textContent = "";
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
      el.appendChild(document.createTextNode((seg.text || "").trim() + " "));
    }
  } else {
    el.textContent = state.resultText;
  }
  el.scrollTop = el.scrollHeight;
}

function plainText() {
  if (!state.segments.length) return state.resultText;
  const parts = [];
  let lastSpeaker = null;
  for (const seg of state.segments) {
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

function toSrt() {
  const lines = [];
  state.segments.forEach((seg, i) => {
    let t = (seg.text || "").trim();
    if (seg.speaker) t = `[${speakerLabel(seg.speaker)}] ${t}`;
    lines.push(String(i + 1), `${fmtTs(seg.start, ",")} --> ${fmtTs(seg.end, ",")}`, t, "");
  });
  return lines.join("\n");
}

function toVtt() {
  const lines = ["WEBVTT", ""];
  for (const seg of state.segments) {
    let t = (seg.text || "").trim();
    if (seg.speaker) t = `<v ${speakerLabel(seg.speaker)}>${t}`;
    lines.push(`${fmtTs(seg.start, ".")} --> ${fmtTs(seg.end, ".")}`, t, "");
  }
  return lines.join("\n");
}

function download(fmt) {
  let content, mime = "text/plain";
  if (fmt === "txt") content = plainText();
  else if (fmt === "srt") content = toSrt();
  else if (fmt === "vtt") { content = toVtt(); mime = "text/vtt"; }
  else {
    content = JSON.stringify({
      text: plainText(),
      duration: state.duration,
      segments: state.segments,
    }, null, 2);
    mime = "application/json";
  }
  const base = (state.file?.name || "transkript").replace(/\.[^.]+$/, "");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = `${base}.${fmt}`;
  a.click();
  URL.revokeObjectURL(a.href);
  log(`Download: ${base}.${fmt}`);
}

function resetResult() {
  els.cardResult.hidden = true;
  els.cardStatus.hidden = true;
  els.btnCancel.hidden = false;
  els.transcript.textContent = "";
  state.segments = [];
  state.resultText = "";
  els.fileError.hidden = true;
}

function resetFile() {
  resetResult();
  state.file = null;
  state.uploadBlob = null;
  state.peaks = null;
  state.duration = null;
  els.dzIdle.hidden = false;
  els.dzFile.hidden = true;
  els.dropzone.classList.remove("has-file");
  els.fileInput.value = "";
  els.btnStart.disabled = true;
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
  if (els.fileInput.files.length) onFileSelected(els.fileInput.files[0]);
});
["dragover", "dragenter"].forEach((t) =>
  els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((t) =>
  els.dropzone.addEventListener(t, (e) => { e.preventDefault(); els.dropzone.classList.remove("dragover"); })
);
els.dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) onFileSelected(e.dataTransfer.files[0]);
});

document.querySelectorAll('input[name="speakers"]').forEach((r) =>
  r.addEventListener("change", () => {
    els.speakerOpts.hidden =
      document.querySelector('input[name="speakers"]:checked').value !== "multi";
  })
);

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

refreshHealth();
setInterval(refreshHealth, 30000);
log("UI bereit.");
