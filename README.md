# granite-speech-api

OpenAI-compatible HTTP audio-transcription API powered by **IBM Granite Speech 4.1**.

Drop-in replacement for OpenAI's `POST /v1/audio/transcriptions` plus extras for
**speaker attribution** and **word-level timestamps** (WhisperX-style),
served natively by `granite-speech-4.1-2b-plus`.

---

## Features

- **Browser upload UI** at `/` (served from `app/static/`): drag & drop audio/video
  (multiple files = sequential batch), client-side audio extraction from video
  (Web Audio API, no video bytes uploaded), single/multi-speaker mode, live
  transcript build-up + model-load status, audio playback with waveform
  scrubbing and word-level karaoke highlighting, txt/srt/vtt/json export,
  log panel, always-visible feedback widget (rating/category/comment + UI log
  and click trace for reproducible bug reports, stored as JSONL via `/v1/feedback`).
- **OpenAI-compatible** `POST /v1/audio/transcriptions` (multipart-form, all standard fields).
- **Long-audio chunking**: audio beyond the model limits (9 min ASR/SAA, 3.5 min
  word timestamps) is split at quiet points; SAA uses the model-card incremental
  scheme (cumulative audio + `prefix_text`) to keep speaker numbering stable.
  Degenerate repetition loops are detected and collapsed.
- **Three Granite Speech 4.1 backends** behind one HTTP surface:
  - `granite-speech-4.1-2b` — default, autoregressive, supports AST + Japanese
  - `granite-speech-4.1-2b-plus` — adds speaker labels + per-word timestamps
  - `granite-speech-4.1-2b-nar` — non-autoregressive, fastest throughput
- **Auto-upgrade**: requests asking for `speaker_attribution` or
  `timestamp_granularities[]=word` automatically swap to the `-plus` model.
- **Auto-download** of model weights on first request (cached to a Docker volume).
- **Hot-swap loader** keeps a single model in VRAM at a time; switching `model`
  in a request swaps cleanly.
- **Response formats**: `json`, `text`, `srt`, `vtt`, `verbose_json`.
- **NDJSON streaming** (`stream=true`) for clients like Vibe / sona.
- **WhisperX-compat fields** (`diarize`, `hf_token`, `batch_size`, `compute_type`)
  silently accepted so existing pipelines keep working.
- **Robust audio loader**: `libsndfile` first, `audioread` + `ffmpeg` fallback for
  m4a/aac and other containers.
- **Docker Compose** out of the box for CUDA, ROCm and CPU.

---

## Quickstart

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/Podden/granite-speech-api.git
cd granite-speech-api

# Pick one:
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d   # NVIDIA
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d   # AMD (Linux)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml  up -d   # CPU only

# Sanity check
curl http://localhost:8000/health
```

The first transcription request triggers the HuggingFace download
(~5 GB per model) into the named volume `hf-cache`; subsequent restarts reuse it.

### Option B — Local install

Requires Python ≥ 3.11 and **`ffmpeg` on `PATH`** for non-PCM audio formats
(m4a/AAC, some MP3 edge-cases). Without ffmpeg you can still transcribe wav,
flac, ogg/opus and most mp3 — but uploads of `.m4a` from iPhones / browsers
will fail.

- Linux: `apt install ffmpeg` / `dnf install ffmpeg` / `pacman -S ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: `winget install Gyan.FFmpeg` (or `choco install ffmpeg`),
  then re-open your shell so `PATH` is refreshed.

The Docker images already bundle ffmpeg — use Docker if you don't want to
deal with this.

```bash
git clone https://github.com/Podden/granite-speech-api.git
cd granite-speech-api

uv venv --python 3.12
uv pip install -e .

# Pick the matching torch wheel for your hardware:
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121   # CUDA
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.0 # ROCm
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu     # CPU

cp .env.example .env
granite-speech-api    # or: uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Models & feature matrix

| Feature                        | `2b` (default) | `2b-plus`        | `2b-nar`     | Cohere Transcribe | Qwen3-ASR    |
| ------------------------------ | :------------: | :--------------: | :----------: | :---------------: | :----------: |
| Plain ASR                      | ✅             | ✅               | ✅           | ✅                | ✅           |
| Punctuation + capitalization   | ✅             | ✅ (plain ASR)   | ✅           | ✅ (best)         | ✅           |
| Word-level timestamps          | ❌             | ✅ via `[T:N]`   | ❌           | ❌                | ❌ ¹         |
| Punctuation **with** word timestamps | ❌       | ❌ ²             | ❌           | ❌                | ❌           |
| Speaker attribution (pyannote) | via auto-upgrade | ✅             | ❌           | ❌ (400)          | ❌ (400)     |
| Speaker attribution (native SAA) | ❌           | ✅ `[Speaker N]:`| ❌           | ❌                | ❌           |
| Keyword biasing (`prompt`)     | ✅             | ✅               | ✅           | ❌                | ❌           |
| Speech translation (AST)       | ✅ 7 langs     | ❌               | ❌           | ❌                | ❌           |
| Language auto-detect           | ❌             | ❌               | ❌           | ❌ (code required)| ✅ 52 langs  |
| German conversational ASR      | ❌ translates! | ✅               | untested     | ✅                | ✅           |
| Live token deltas (`delta`)    | ✅             | ✅               | ❌           | ❌                | ❌           |
| Long audio                     | chunked 8 min  | chunked 8 min / 3 min (TS) | chunked | built-in chunking | chunked 5 min |
| Gated (needs `GRANITE_HF_TOKEN`) | no           | no               | no           | **yes**           | no           |
| Relative speed (5-min clip)    | ~35 s          | ~35–75 s         | fastest (claimed) | **~10 s**    | ~40 s        |

¹ Word timestamps would need the separate `Qwen3-ForcedAligner-0.6B-hf` (not integrated yet).
² Verified empirically (July 2026): the `[T:N]` timestamp task always emits lowercase,
  punctuation-free output — prompt variants asking for punctuation have zero effect.
  Consequence: pyannote-diarized transcripts are lowercase/unpunctuated too, since they
  are built from the word-timestamp pass.

**Granite languages** (ASR): English, French, German, Spanish, Portuguese
(Japanese also supported by `2b`). **Cohere**: 14 languages (en, fr, de, it, es,
pt, el, nl, pl, zh, ja, ko, vi, ar). **Qwen**: 52 languages/dialects.

> **NAR notes:** the NAR backend is implemented but only smoke-tested. It uses
> a different transformers API (`AutoModel` + `AutoFeatureExtractor`,
> `trust_remote_code=True`) and prefers `flash_attention_2` on CUDA — install
> `flash-attn` for best throughput, otherwise it falls back to PyTorch SDPA.
> NAR does not produce punctuation, AST, timestamps or speaker labels.

### Additional ASR backends (evaluation)

Beyond Granite, the registry dispatches to two more model families (same
hot-swap + idle-unload semantics; one model in VRAM at a time):

- **`CohereLabs/cohere-transcribe-03-2026`** (alias `cohere-transcribe`) —
  2B conformer encoder-decoder, 14 languages, long-form chunking built into
  the feature extractor. Plain ASR only.
- **`Qwen/Qwen3-ASR-1.7B`** (alias `qwen3-asr`) — 52 languages with automatic
  language identification (via the `qwen-asr` package). Long audio is chunked
  at quiet points server-side. Plain ASR only.

Neither produces word timestamps yet, so speaker attribution is rejected for
them (400).

### Speaker diarization (pyannote)

`speaker_attribution=true` now runs a dedicated diarization stage by default:
[pyannote `speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
produces speaker turns, the Granite `-plus` word-timestamp pass produces word
timings, and the two are reconciled into speaker-labelled segments. This
replaces the Granite SAA pass (which remains available via
`diarization_engine=granite` and as automatic fallback).

Requirements: the pipeline is **gated** — set `GRANITE_HF_TOKEN` to a
HuggingFace token whose account accepted the model conditions. The pipeline
(~500 MB VRAM) is lazy-loaded and idle-unloaded like the ASR models.

---

## API

### `POST /v1/audio/transcriptions`

Multipart form. Everything except `file` is optional.

| Field                          | Type     | Notes                                                                                              |
| ------------------------------ | -------- | -------------------------------------------------------------------------------------------------- |
| `file`                         | file     | Audio (mp3, wav, ogg, m4a, flac, …) or video (audio track extracted via ffmpeg)                    |
| `model`                        | string   | Granite model id; auto-upgrades to `-plus` for rich features                                       |
| `language`                     | string   | ISO-639-1 hint. Non-English hints auto-upgrade to `-plus` (the base model tends to translate non-English speech to English instead of transcribing) |
| `response_format`              | string   | `json` (default), `text`, `srt`, `vtt`, `verbose_json`                                             |
| `timestamp_granularities[]`    | string   | `segment` (default), `word` (forces `-plus`)                                                       |
| `prompt`                       | string   | Comma-separated keywords for biased ASR                                                            |
| `translate`, `translate_to`    | bool/str | AST (`-2b` only): `english`/`french`/`german`/`spanish`/`portuguese`/`japanese`/`italian`/`mandarin` |
| `speaker_attribution`          | bool     | Forces `-plus`, adds `[Speaker N]:` to segments                                                    |
| `min_speakers`/`max_speakers`  | int      | Reserved (advisory)                                                                                |
| `stream`                       | bool     | Emit NDJSON event stream                                                                           |
| `diarize`/`hf_token`/`batch_size`/`compute_type` | — | WhisperX aliases, accepted for drop-in compatibility                                       |

#### Examples

```bash
# Plain transcription
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.wav

# Verbose JSON with word-level timestamps + speaker labels (auto-loads -plus)
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F response_format=verbose_json \
  -F speaker_attribution=true \
  -F 'timestamp_granularities[]=word'

# SRT subtitles
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@lecture.mp3 \
  -F response_format=srt

# Speech translation to English
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@german.wav \
  -F translate=true -F translate_to=english

# Keyword biasing
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@call.wav \
  -F 'prompt=Kubernetes, GitOps, ArgoCD'

# NDJSON streaming
curl -s --no-buffer http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F stream=true
```

#### `verbose_json` shape

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 42.3,
  "text": "[SPEAKER_00] Hello world. [SPEAKER_01] How are you?",
  "segments": [
    {
      "id": 0, "start": 0.0, "end": 1.5,
      "speaker": "SPEAKER_00",
      "text": "Hello world.",
      "words": [
        { "word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00" },
        { "word": "world.", "start": 0.5, "end": 1.5, "speaker": "SPEAKER_00" }
      ]
    }
  ]
}
```

#### NDJSON stream events

```jsonl
{"type":"duration","duration":42.3}
{"type":"status","stage":"loading_model","model":"ibm-granite/granite-speech-4.1-2b","cold":true}
{"type":"status","stage":"model_ready","model":"ibm-granite/granite-speech-4.1-2b"}
{"type":"progress","progress":0}
{"type":"delta","text":" Hello"}
{"type":"delta","text":" world."}
{"type":"partial","text":"Hello world.","start":0.0,"end":1.5}
{"type":"progress","progress":50.0}
{"type":"segment","start":0.0,"end":1.5,"text":"Hello","speaker":"SPEAKER_00","words":[...]}
{"type":"result","text":"Hello world. How are you?","language":"en"}
{"type":"progress","progress":100}
```

`status` reports (cold) model loading, `delta` streams raw decoded text pieces
token-by-token while a chunk is being generated (`TextIteratorStreamer`;
AR models only — the NAR model decodes in one shot), `partial` streams each
finished chunk's clean text, `progress` is real per-chunk progress. Concurrent
requests are serialized per model; requesting a different model hot-swaps after
the running inference finishes.

> `delta` text is the raw model output — for word-timestamp or speaker requests
> it contains `[T:NNN]` / `[Speaker N]:` tags; clients should treat it as a
> preview and rely on `partial`/`segment` for parsed results.

### `POST /v1/feedback` / `GET /v1/feedback`

Minimal user-feedback capture for the browser UI (thumbs up/down + optional
comment + request context). Entries are appended to a JSONL file
(`GRANITE_FEEDBACK_FILE`, a named volume in Docker); `GET /v1/feedback?limit=50`
returns the newest entries. No auth — intended for internal/LAN use only.

### `GET /v1/models`

Lists the three available Granite Speech 4.1 model ids.

### `POST /v1/models/load` / `POST /v1/models/unload`

Manual hot-swap control:

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"path": "ibm-granite/granite-speech-4.1-2b-plus"}'
```

### `GET /health`

Reports loaded model and device:

```json
{
  "status": "ok",
  "loaded_model": "ibm-granite/granite-speech-4.1-2b",
  "idle_seconds": 12.4,
  "idle_unload_seconds": 600,
  "device": "cuda",
  "default_model": "ibm-granite/granite-speech-4.1-2b",
  "available_models": [
    "ibm-granite/granite-speech-4.1-2b",
    "ibm-granite/granite-speech-4.1-2b-plus",
    "ibm-granite/granite-speech-4.1-2b-nar"
  ]
}
```

---

## Configuration

All settings via env vars (or `.env` file). See [.env.example](.env.example).

### Auto-unload

The API auto-unloads the model after `GRANITE_IDLE_UNLOAD_SECONDS` of inactivity
(default: 600s = 10 min). The next request triggers a fresh load — weights are
still cached on disk in `HF_HOME`, so reload is fast (no re-download). To keep
the model resident forever, set `GRANITE_IDLE_UNLOAD_SECONDS=-1`.

| Env var                      | Default                                  | Notes                                            |
| ---------------------------- | ---------------------------------------- | ------------------------------------------------ |
| `GRANITE_API_HOST`           | `0.0.0.0`                                |                                                  |
| `GRANITE_API_PORT`           | `8000`                                   |                                                  |
| `GRANITE_DEFAULT_MODEL`      | `ibm-granite/granite-speech-4.1-2b`      | Model loaded when none specified                 |
| `GRANITE_DEVICE`             | `auto`                                   | `auto` / `cuda` / `cuda:0` / `rocm` / `cpu`      |
| `GRANITE_DTYPE`              | `bfloat16`                               | `bfloat16` / `float16` / `float32`               |
| `GRANITE_MAX_AUDIO_SECONDS`  | `600`                                    | Reject longer audio with HTTP 413                |
| `GRANITE_REPETITION_PENALTY` | `1.0`                                    | `generate()` repetition penalty (1.0 = off); chunking already prevents most loops |
| `GRANITE_IDLE_UNLOAD_SECONDS`| `600`                                    | Auto-unload model after N seconds idle. `-1` disables. |
| `GRANITE_IDLE_CHECK_INTERVAL`| `30`                                     | How often the idle monitor wakes up (seconds)    |
| `GRANITE_CORS_ORIGINS`       | `*`                                      | Comma-separated, or `*`                          |
| `HF_HOME`                    | `/data/hf-cache` (in container)          | Where weights are cached                         |
| `HUGGING_FACE_HUB_TOKEN`     | —                                        | For private/gated models                         |

---

## Docker Compose details

The base `docker-compose.yml` is GPU-agnostic and mounts a named volume
`hf-cache` so model weights persist across restarts. Combine it with one
runtime overlay:

- `docker-compose.cuda.yml` — NVIDIA Container Toolkit (`--gpus all` style),
  `GRANITE_DEVICE=cuda`. **Untested** — feedback welcome.
- `docker-compose.rocm.yml` — AMD ROCm via `/dev/dxg` (ROCDXG path, requires
  Adrenalin 26.2.2+ and Docker inside WSL on Windows). **Untested** — feedback welcome.
- `docker-compose.cpu.yml` — no GPU, `float32` inference. Verified working.

Override torch wheels at build time:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml \
  build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
```

> **GPU support status:** GPU inference paths (CUDA and ROCm) are implemented
> but have not been end-to-end tested. CPU inference works and was verified with
> all three models. If you get GPU inference working, please open an issue or PR
> with your setup details.

---

## Repository layout

```
granite-speech-api/
├── app/
│   ├── main.py                 # FastAPI app, lifespan, CORS, static UI, run()
│   ├── config.py               # pydantic-settings (GRANITE_* env)
│   ├── audio.py                # libsndfile + audioread/ffmpeg → mono 16 kHz
│   ├── static/                 # browser upload UI (vanilla JS, IBM Plex)
│   ├── api/
│   │   ├── transcriptions.py   # POST /v1/audio/transcriptions
│   │   ├── models.py           # GET /v1/models, POST /v1/models/{load,unload}
│   │   └── health.py           # GET /health
│   ├── backends/
│   │   ├── base.py             # ASRBackend ABC + transcribe_stream() default
│   │   ├── granite.py          # AR backend (2b, 2b-plus) + output parsers
│   │   ├── granite_nar.py      # NAR backend (2b-nar)
│   │   └── registry.py         # Hot-swap registry (1 model active)
│   └── schema/
│       ├── request.py          # TranscriptionRequest dataclass
│       └── response.py         # OpenAI verbose_json models
├── tests/
│   └── test_api.py             # Smoke + parser tests (no HF download)
├── Dockerfile
├── docker-compose.yml          # GPU-agnostic base
├── docker-compose.cuda.yml     # NVIDIA overlay
├── docker-compose.rocm.yml     # AMD ROCm overlay (Linux only)
├── docker-compose.cpu.yml      # CPU-only overlay
├── pyproject.toml
├── .env.example
├── implementation-plan.md
└── README.md
```

---

## Development

```bash
uv pip install -e ".[dev]"
pytest -q
ruff check .
```

Smoke tests cover the request shape, OpenAI-compatibility model normalization,
and the timestamp / speaker output parsers — they do **not** download any
HF weights, so they run fast on plain CI.

---

## Backlog

- [ ] Bearer-token auth (env `API_KEY`) for direct-internet exposure.
- [ ] vLLM-backed alternative path for high-throughput production deployments.
- [ ] Time-aligned speaker segments in SAA-only mode (currently linearly
  distributed over the audio duration when no word-timestamps are requested).
- [ ] `flash-attn` install layer in a `Dockerfile.cuda-flash` variant for NAR.

---

## License

Apache-2.0 — see model cards on Hugging Face for the underlying weights.

## Credits

- [IBM Granite Speech 4.1](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — the model family powering this API.
- [HuggingFace Transformers](https://github.com/huggingface/transformers).
- [FastAPI](https://fastapi.tiangolo.com/).
