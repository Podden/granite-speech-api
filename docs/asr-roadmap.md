# ASR/Diarisierung — Optionen & Vorschlag (Juli 2026)

> **Status:** Stufe 1 (pyannote-Stage + Reconciliation) und Stufe 2
> (Registry-Öffnung, Cohere- & Qwen-Backends) sind implementiert.
> Qwen läuft über das native transformers-`qwen3_asr` (Checkpoint
> `Qwen/Qwen3-ASR-1.7B-hf`), NICHT über das `qwen-asr`-Package — das pinnt
> `transformers==4.57.6` und bricht Granite. pyannote community-1 und
> Cohere Transcribe sind **gated**: GRANITE_HF_TOKEN nötig + Bedingungen
> auf HF akzeptieren. Stufe 3 (Bench) steht aus.

Use case: aufgezeichnete Meetings (OBS), deutsch, teils englisch, **kein Realtime**.
Problem heute: Granite Speech 4.1 plus transkribiert gut, aber die
Sprecher-Zuordnung (SAA-Pass) ist schwach.

## Kernaussage

Sprecher-Trennung und Transkription sind **zwei getrennte Probleme**. Kein
offenes ASR-Modell (auch Granite nicht) macht Diarisierung nebenbei gut.
Der Industriestandard ist die WhisperX-Pipeline:

```
Audio → VAD/Diarisierung (pyannote)  ─┐
     → ASR mit Wort-Timestamps       ─┴→ Reconciliation → [Sprecher N] Text
```

Das heißt: **eine Diarisierungs-Stufe vor/neben dem ASR-Backend einziehen**,
statt das ASR-Modell zu wechseln. Modellwechsel ist danach eine
unabhängige, austauschbare Entscheidung.

## 1. Diarisierung (die eigentliche Lücke)

| Option | Lizenz | Bewertung |
| --- | --- | --- |
| **pyannote `speaker-diarization-community-1`** (pyannote.audio 4.x) | CC-BY-4.0, kommerziell nutzbar, HF-Token nötig (Gated) | **Empfehlung.** Ablöser von 3.1, deutlich besseres Speaker-Counting/-Assignment. Liefert zusätzlich `exclusive_speaker_diarization` — genau dafür gebaut, unpräzise ASR-Timestamps sauber auf Sprecher zu mappen. Offline-Nutzung offiziell unterstützt (wichtig für cave2). ~31 s pro Stunde Audio auf GPU. |
| pyannote `speaker-diarization-3.1` | MIT, Gated | Nur wenn community-1 aus Lizenz-/Token-Gründen ausfällt. AMI DER ~12–14 %, schlechter beim Zählen der Sprecher. |
| NVIDIA NeMo Sortformer / MSDD | NVIDIA-Terms | End-to-end, overlap-aware, GPU-freundlich, vergleichbar mit pyannote auf AMI. Zieht aber die komplette NeMo-Toolchain ins Image — für uns zu schwer. |
| Granite SAA (Status quo) | — | Bleibt als Fallback, wenn Diarisierung deaktiviert ist. |

Sprachneutral: Diarisierung arbeitet auf Sprecher-Embeddings, deutsch/englisch
ist egal. Ein pyannote-Lauf kostet ~500 MB VRAM — passt neben Granite.

**Zusätzlicher Hebel, unabhängig vom Modell:** OBS nimmt in der Regel
Mehrspur auf (Mikro pro Person / Desktop-Audio getrennt). Wenn die Rohspuren
existieren, ist "Diarisierung" trivial und 100 % korrekt: pro Spur
transkribieren, per Timestamp mergen. Das schlägt jedes Modell. Lohnt sich zu
prüfen, bevor wir ML dagegen werfen.

## 2. ASR-Modelle — Bewertung für deutsche Meetings

| Modell | Lizenz | Deutsch | Urteil |
| --- | --- | --- | --- |
| **Granite Speech 4.1 plus** (heute) | Apache 2.0 | gut (verifiziert) | Bleibt Default. Bekannte Limits: ~9 min Chunks, Wort-Timestamps nur ~3,5 min, Repetition-Loops. |
| **Cohere Transcribe 03-2026** (2B, Conformer Enc-Dec) | **Apache 2.0** | ja, 1 von 14 Sprachen, explizit auf Named Entities/Verbatim optimiert | **Top-Kandidat.** Long-form ist im Feature-Extractor eingebaut (auto-chunking + Reassembly über `audio_chunk_index`), Beispiel im Model Card ist ein 55-min Earnings-Call. Reines `transformers`-API — passt 1:1 in unser Backend-Interface, kein neuer Serving-Stack. |
| **Qwen3-ASR-1.7B** | Apache 2.0 (Qwen) | ja, 52 Sprachen, sehr gute Language-ID (98,7 % Fleurs) | **Zweiter Kandidat.** Separater `Qwen3-ForcedAligner-0.6B` liefert Wort-Timestamps — das ist genau das, was die Reconciliation mit pyannote braucht, und besser als Granites 3,5-min-Limit. Braucht das `qwen-asr`-Package (transformers- oder vLLM-Backend). |
| **Whisper large-v3 / turbo** (via faster-whisper/CTranslate2) | MIT | solide, aber halluziniert bei Stille | Referenz-Baseline. Schnell, extrem gut dokumentiert, Wort-Timestamps out of the box. Sinnvoll als Vergleichsmaßstab, ggf. als schnelle CPU-/Bulk-Option. |
| Nemotron 3.5 ASR streaming 0.6B | OpenMDW-1.1 | "transcription-ready" Tier (de-DE) | Streaming-Architektur (FastConformer Cache-Aware RNNT), 80 ms–1,12 s Chunks. Für Batch-Meetings kein Vorteil gegenüber einem Offline-Modell. Nur relevant, falls doch mal Live-Transkription kommt. |
| GigaAM-Multilingual | — | **kein Deutsch** | Raus. Stark auf Russisch/Kasachisch/Kirgisisch/Usbekisch, Englisch nur mittelmäßig (CV WER 21,5 vs. Whisper 20,0). |
| Voxtral Mini 4B Realtime 2602 | Apache 2.0 | 13 Sprachen | Realtime-Modell (<500 ms Delay, konfigurierbar 80 ms–2,4 s), vLLM empfohlen. Für unseren Batch-Fall die falsche Baustelle — falls Voxtral, dann die Offline-Variante. |
| Voxtral 4B TTS 2603 | CC-BY-NC (Voices) | — | **Text-to-Speech**, kein ASR. Für den Use case irrelevant (und die Referenz-Stimmen sind non-commercial). |

## 3. Gibt es fertige Unified-Backends zum Klauen?

Kurz: kein Projekt, das man 1:1 übernehmen will — aber Code zum Abschauen.

- **WhisperX** — der De-facto-Standard für ASR+Diarisierung+Alignment. Genau die
  Reconciliation-Logik (Wort-Timestamps → pyannote-Turns → Sprecher-Labels),
  die wir brauchen. Ist aber Whisper-fixiert, kein Modell-Plugin-System.
- **speaches / faster-whisper-server** — OpenAI-kompatibler Server, mehrere
  Whisper-Modelle, Model-Hot-Swap. Konzeptionell das, was wir schon gebaut haben
  (Registry + `/v1/audio/transcriptions`), nur ohne Granite.
- **NVIDIA NeMo** — echtes Multi-Modell-Framework inkl. Diarisierung, aber
  schweres Framework-Commitment.
- **achetronic/parakeet** — Whisper-kompatibler Go-Server für Parakeet TDT
  (ONNX). Single-Modell, kein Vorbild für Multi-Backend.

Fazit: Unser `ASRBackend`-Interface plus Registry ist bereits die richtige
Abstraktion. Migration auf ein Fremdprojekt würde Granite und die gesamte
Streaming-/Chunking-Logik wegwerfen. **Wir bauen weiter, aber übernehmen
WhisperX' Reconciliation-Ansatz.**

## 4. Vorschlag — Umsetzung in 3 Stufen

### Stufe 1 — Diarisierung als eigene Pipeline-Stufe (größter Effekt)
- Neues Modul `app/diarization.py`: pyannote community-1, eigenes Lazy-Loading
  + Idle-Unload analog zur Backend-Registry, Modell offline im Image/Volume.
- Neues Request-Flag `diarize: bool` (+ optional `num_speakers` /
  `min_speakers`/`max_speakers` — bei Meetings kennt man die Teilnehmerzahl oft).
- Reconciliation: Wort-Timestamps des ASR-Backends gegen die
  `exclusive_speaker_diarization`-Turns mappen → `TranscriptionSegment.speaker`.
  Fallback ohne Wort-Timestamps: Segment-Mitte gegen Turns matchen.
- Granite-SAA-Pass wird übersprungen, wenn `diarize=true` → spart sogar Zeit.
- UI: Checkbox "Sprecher erkennen" + optional Teilnehmerzahl; Rendering für
  `[Sprecher N]` existiert bereits.

### Stufe 2 — Backend-Registry für mehrere Modellfamilien öffnen
- `ASRBackend` um `capabilities` erweitern (`word_timestamps`, `streaming`,
  `native_diarization`, `languages`, `max_audio_seconds`).
- Registry: Mapping `model_id → Backend-Klasse` statt `if is_nar`.
- Erstes Fremd-Backend: **Cohere Transcribe** (reines transformers-API,
  eingebautes Long-form-Chunking → wenig neuer Code).
- `/v1/models` liefert Capabilities mit; UI blendet Optionen entsprechend aus.

### Stufe 3 — Messen statt raten
- Bench-Harness analog zum Opus-/Normalisierungs-Experiment: ein 5–10-min
  deutsches Meeting mit manuell korrigiertem Referenz-Transkript.
- Metriken: WER gegen Referenz + DER/Speaker-Confusion für die Diarisierung.
- Kandidaten: Granite plus (Baseline), Cohere Transcribe, Qwen3-ASR-1.7B,
  Whisper large-v3 (faster-whisper).
- Danach erst entscheiden, ob ein zweites ASR-Modell dauerhaft bleibt —
  jedes Modell im Image kostet VRAM auf der geteilten A6000.

## Nicht empfohlen

- GigaAM (kein Deutsch), Voxtral TTS (falscher Task), Nemotron Streaming
  (Realtime-Architektur ohne Nutzen für Batch).
- Diarisierung im ASR-Modell suchen. Granite/Whisper/Cohere sind alle
  schwach darin — das ist ein Architekturproblem, kein Modellproblem.
