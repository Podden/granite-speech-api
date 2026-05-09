# granite-speech-api

OpenAI-compatible HTTP audio transcription API powered by **IBM Granite Speech 4.1**.

Designed as a drop-in replacement for the OpenAI `/v1/audio/transcriptions`
endpoint with extras for **speaker attribution** and **word-level timestamps**
(WhisperX-style), backed natively by `granite-speech-4.1-2b-plus`.

## Models

| Model | Use for |
| --- | --- |
| `ibm-granite/granite-speech-4.1-2b` (default) | ASR + AST, 6 languages + Japanese, keyword biasing, punctuation + capitalization |
| `ibm-granite/granite-speech-4.1-2b-plus` | Speaker-attributed ASR + word-level timestamps |
| `ibm-granite/granite-speech-4.1-2b-nar` | Higher throughput, non-autoregressive |

The server **auto-upgrades** to `…-plus` when a request asks for
`speaker_attribution=true` or `timestamp_granularities[]=word`.

Only one model is held in VRAM at a time (hot-swap on `model` change).

## Install

```bash
# Create env (Python ≥ 3.11)
uv venv && source .venv/bin/activate   # or python -m venv

# CUDA 12.1 (default)
uv pip install -e .
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# ROCm 6.0 (AMD, e.g. Strix Halo)
uv pip install -e .
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# CPU
uv pip install -e .
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
cp .env.example .env
granite-speech-api    # or:  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /v1/audio/transcriptions`

Multipart-form, OpenAI-compatible.

| Field | Type | Notes |
| --- | --- | --- |
| `file` | file | Audio (mp3, wav, ogg, m4a, flac, …) |
| `model` | string | Granite model id (auto-upgrades to `-plus` for rich features) |
| `language` | string | ISO 639-1 hint (optional) |
| `response_format` | string | `json` (default) \| `text` \| `srt` \| `vtt` \| `verbose_json` |
| `timestamp_granularities[]` | string | `segment` \| `word` (forces `-plus`) |
| `prompt` | string | Keyword-biasing list |
| `translate` | bool | Translate to `translate_to` (AST mode, 2b only) |
| `translate_to` | string | `english`/`french`/`german`/`spanish`/`portuguese`/`japanese`/`italian`/`mandarin` |
| `speaker_attribution` | bool | Forces `-plus`. Returns `[Speaker N]:` segments |
| `min_speakers` / `max_speakers` | int | Reserved for compat (currently advisory) |
| `stream` | bool | Emit ndjson event stream |
| `diarize` / `hf_token` / `batch_size` / `compute_type` | — | WhisperX-compat aliases (silently accepted) |

### `verbose_json`

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 42.3,
  "text": "[SPEAKER_00] Hello world. [SPEAKER_01] How are you?",
  "segments": [
    {
      "id": 0,
      "start": 0.0, "end": 1.5,
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

### ndjson stream (`stream=true`)

```jsonl
{"type":"duration","duration":42.3}
{"type":"progress","progress":0}
{"type":"segment","start":0.0,"end":1.5,"text":"Hello world.","speaker":"SPEAKER_00"}
{"type":"result","text":"Hello world. How are you?","language":"en"}
{"type":"progress","progress":100}
```

### `GET /v1/models`

Lists the three Granite Speech 4.1 model ids.

### `POST /v1/models/load` / `POST /v1/models/unload`

Manual hot-swap control.

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"path":"ibm-granite/granite-speech-4.1-2b-plus"}'
```

### `GET /health`

Reports loaded model and device.

## Docker

```bash
# CUDA
docker build -t granite-speech-api .

# ROCm
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm6.0 \
             -t granite-speech-api:rocm .

# CPU
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
             -t granite-speech-api:cpu .
```

## Compatibility matrix

| Feature | 2b | 2b-plus | 2b-nar |
| --- | --- | --- | --- |
| ASR (plain) | ✅ | ✅ | ✅ |
| Punctuation + Capitalization | ✅ | ❌ | ✅ |
| Word-level timestamps | ❌ | ✅ via `[T:N]` | ❌ |
| Speaker attribution | ❌ | ✅ via `[Speaker N]:` | ❌ |
| Keyword biasing | ✅ | ✅ | ✅ |
| Speech translation (AST) | ✅ 7 langs | ❌ | ✅ |
| Japanese ASR | ✅ | ❌ | ✅ |

## License

Apache-2.0
