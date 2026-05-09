# granite-speech-api

OpenAI-compatible HTTP audio-transcription API powered by **IBM Granite Speech 4.1**.

Drop-in replacement for OpenAI's `POST /v1/audio/transcriptions` plus extras for
**speaker attribution** and **word-level timestamps** (WhisperX-style),
served natively by `granite-speech-4.1-2b-plus`.

---

## Features

- **OpenAI-compatible** `POST /v1/audio/transcriptions` (multipart-form, all standard fields).
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

| Feature                       | `2b` (default) | `2b-plus`       | `2b-nar`     |
| ----------------------------- | :------------: | :-------------: | :----------: |
| Plain ASR                     | ✅             | ✅              | ✅           |
| Punctuation + Capitalization  | ✅             | ❌              | ✅           |
| Word-level timestamps         | ❌             | ✅ via `[T:N]`  | ❌           |
| Speaker attribution (SAA)     | ❌             | ✅ `[Speaker N]:`| ❌           |
| Keyword biasing               | ✅             | ✅              | ✅           |
| Speech translation (AST)      | ✅ 7 langs     | ❌              | ❌           |
| Japanese ASR                  | ✅             | ❌              | ❌           |
| Throughput                    | medium         | medium          | **fastest**  |

**Supported languages** (ASR): English, French, German, Spanish, Portuguese
(Japanese also supported by `2b`).

> **NAR notes:** the NAR backend is implemented but only smoke-tested. It uses
> a different transformers API (`AutoModel` + `AutoFeatureExtractor`,
> `trust_remote_code=True`) and prefers `flash_attention_2` on CUDA — install
> `flash-attn` for best throughput, otherwise it falls back to PyTorch SDPA.
> NAR does not produce punctuation, AST, timestamps or speaker labels.

---

## API

### `POST /v1/audio/transcriptions`

Multipart form. Everything except `file` is optional.

| Field                          | Type     | Notes                                                                                              |
| ------------------------------ | -------- | -------------------------------------------------------------------------------------------------- |
| `file`                         | file     | Audio (mp3, wav, ogg, m4a, flac, …)                                                                |
| `model`                        | string   | Granite model id; auto-upgrades to `-plus` for rich features                                       |
| `language`                     | string   | ISO-639-1 hint, echoed back in `verbose_json` (Granite does not detect language itself)            |
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
{"type":"progress","progress":0}
{"type":"segment","start":0.0,"end":1.5,"text":"Hello","speaker":"SPEAKER_00"}
{"type":"result","text":"Hello world. How are you?","language":"en"}
{"type":"progress","progress":100}
```

> Streaming is currently a wrapper around the synchronous `transcribe()` —
> Granite's `generate()` is non-streaming. Token-level streaming via
> `TextIteratorStreamer` is on the backlog.

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
  `GRANITE_DEVICE=cuda`.
- `docker-compose.rocm.yml` — Linux only. Mounts `/dev/kfd` + `/dev/dri`,
  joins `video` + `render` groups, sets `HSA_OVERRIDE_GFX_VERSION` (override
  with `HSA_OVERRIDE_GFX_VERSION=11.5.1 docker compose ... up`).
- `docker-compose.cpu.yml` — no GPU, `float32` inference.

Override torch wheels at build time:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml \
  build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
```

### ROCm auf Windows / WSL2

Ab **Adrenalin 26.2.2 + ROCm 7.2.1** nutzt AMD den **ROCDXG**-Pfad (kein
`/dev/kfd` mehr, kein `amdgpu-install --usecase=wsl`). Das Gerät heißt jetzt
`/dev/dxg` und kommt direkt vom Windows-Treiber.

**Voraussetzungen auf Windows-Seite (einmalig):**

- Adrenalin-Treiber ≥ 26.2.2 installiert ✓
- Windows SDK installiert (z.B. über Visual Studio Installer oder direkt von Microsoft) — wird zum Bauen von librocdxg in WSL benötigt. Standardpfad: `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\`

**Einrichten in WSL Ubuntu 24.04** (im WSL-Terminal ausführen):

```bash
# --- Schritt 1: ROCm-Repo einrichten und ROCm installieren ---
sudo apt update && sudo apt install -y wget gnupg2 ca-certificates

# ROCm GPG-Key (für ROCm 7.2 / amdgpu 6.4)
sudo mkdir -p /etc/apt/keyrings
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

# ROCm-Paketquelle (noble = Ubuntu 24.04)
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
  https://repo.radeon.com/rocm/apt/6.4.1 noble main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list

sudo apt update
sudo apt install -y rocm-dev rocminfo rocm-smi-lib

# --- Schritt 2: User zu render+video Gruppe hinzufügen ---
sudo usermod -aG render,video $USER

# --- Schritt 3: Build-Abhängigkeiten für librocdxg ---
sudo apt install -y cmake build-essential git

# --- Schritt 4: librocdxg bauen und installieren ---
# librocdxg ist die Brücke zwischen ROCm-Runtime und /dev/dxg (Windows DXCore)
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg

WIN_SDK="/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0"
mkdir build && cd build
cmake .. -DWIN_SDK="${WIN_SDK}/shared"
make -j$(nproc)
sudo make install
cd ~/

# --- Schritt 5: Verifizieren ---
export HSA_ENABLE_DXG_DETECTION=1
export PATH="$PATH:/opt/rocm/bin"
rocminfo | grep -A5 "Agent 2"  # muss deine GPU anzeigen

# Wenn rocminfo die GPU zeigt, in .bashrc/profile dauerhaft setzen:
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
echo 'export PATH="$PATH:/opt/rocm/bin"' >> ~/.bashrc
```

**Docker in WSL** (nicht Docker Desktop — das kann `/dev/dxg` nicht weiterreichen):

```bash
# Docker direkt in WSL installieren:
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
# WSL-Session neu starten, dann:

docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d
```

> **Docker Desktop vs. Docker in WSL:** Docker Desktop unter Windows nutzt eine
> eigene LinuxKit-VM und kann `/dev/dxg` nicht weiterreichen. Für ROCm-GPU-Zugriff
> muss Docker *innerhalb* der WSL-Instanz installiert werden.

> **Strix Halo (8060S):** Seit librocdxg v1.2.0 offiziell unterstützt (Ryzen AI Max-Serie).
> `HSA_OVERRIDE_GFX_VERSION` ist mit dem neuen ROCDXG-Pfad **nicht mehr nötig**.

---

## Repository layout

```
granite-speech-api/
├── app/
│   ├── main.py                 # FastAPI app, lifespan, CORS, run()
│   ├── config.py               # pydantic-settings (GRANITE_* env)
│   ├── audio.py                # libsndfile + audioread/ffmpeg → mono 16 kHz
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

- [ ] Chunking for audio > 9 min ASR/SAA, > 5 min timestamps, with
  `prefix_text`-based incremental decoding.
- [ ] Real token-level streaming via `TextIteratorStreamer`.
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
