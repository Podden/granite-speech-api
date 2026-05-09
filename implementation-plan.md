# Plan: granite-speech-api

> **Status:** Initial scaffolding implemented. See [README.md](README.md) for usage.

## Ziel

FastAPI-Server mit **OpenAI Audio API-kompatibler** HTTP-Schnittstelle,
ausschließlich gestützt auf **IBM Granite Speech 4.1**. Drop-in-kompatibel mit
Vibe / sona und mit WhisperX-Style Word-Timestamps + Speaker-Diarization als
Request-Felder (nativ vom 2b-plus-Modell beliefert, kein WhisperX-Backend).

---

## Modell-Strategie

| Modell | Use case | Auto-Trigger |
|---|---|---|
| `ibm-granite/granite-speech-4.1-2b` | Default. ASR (multi-lingual + JP), AST, KWB, Punctuation. | Default |
| `ibm-granite/granite-speech-4.1-2b-plus` | Speaker-attributed ASR + Word-Level Timestamps via `[T:N]` / `[Speaker N]:` | `speaker_attribution=true` ODER `timestamp_granularities[]=word` |
| `ibm-granite/granite-speech-4.1-2b-nar` | Higher throughput, non-autoregressive | Nur explizit per `model` |

**Hot-Swap-Loader**: Nur ein Modell gleichzeitig im VRAM. Wechsel bei neuem
`model`-Wert im Request, geschützt durch `asyncio.Lock`. HF-Calls laufen via
`run_in_executor` aus dem Event-Loop heraus.

**Auto-Download**: Modelle werden beim ersten Bedarf automatisch über
`AutoProcessor.from_pretrained` / `AutoModelForSpeechSeq2Seq.from_pretrained`
geladen. HuggingFace-Cache ist konfigurierbar via `HF_HOME` (Docker:
named volume `hf-cache:/data/hf-cache`). Kein Pre-Download nötig.

---

## Repo-Struktur (Stand)

```
granite-speech-api/
├── pyproject.toml
├── README.md
├── Dockerfile
├── docker-compose.yml          # CUDA default
├── docker-compose.rocm.yml     # ROCm overlay
├── docker-compose.cpu.yml      # CPU overlay
├── .env.example
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, CORS, run()
│   ├── config.py               # pydantic-settings (GRANITE_* env)
│   ├── audio.py                # soundfile + torchaudio decode → mono 16kHz
│   ├── api/
│   │   ├── __init__.py         # api_router aggregator
│   │   ├── transcriptions.py   # POST /v1/audio/transcriptions
│   │   ├── models.py           # GET /v1/models, POST /v1/models/{load,unload}
│   │   └── health.py           # GET /health
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py             # ASRBackend ABC + transcribe_stream() default
│   │   ├── granite.py          # GraniteBackend + Output-Parser
│   │   └── registry.py         # Hot-swap registry (1 Modell aktiv)
│   └── schema/
│       ├── __init__.py
│       ├── request.py          # TranscriptionRequest dataclass
│       └── response.py         # OpenAI verbose_json Modelle
└── tests/
    └── test_api.py             # Smoke + Parser-Tests (kein HF-Download)
```

---

## API

### `POST /v1/audio/transcriptions`

Multipart-Form. Felder folgen OpenAI; WhisperX-Aliase werden für Drop-in-Kompat
akzeptiert (silent no-op wenn nicht relevant für Granite).

| Feld | Typ | Bemerkung |
|---|---|---|
| `file` | file | mp3/wav/ogg/m4a/flac/… (alles was libsndfile kann) |
| `model` | string | Granite model id; Auto-Upgrade auf `-plus` |
| `language` | string | ISO 639-1 Hint |
| `response_format` | string | `json` / `text` / `srt` / `vtt` / `verbose_json` |
| `timestamp_granularities[]` | string | `segment` (default) / `word` (forciert `-plus`) |
| `prompt` | string | Keyword-Biasing-Liste |
| `translate` / `translate_to` | bool / string | AST: `english`, `french`, `german`, `spanish`, `portuguese`, `japanese`, `italian`, `mandarin` |
| `speaker_attribution` | bool | Forciert `-plus`, returned `[Speaker N]:` Segmente |
| `min_speakers` / `max_speakers` | int | Reserved für spätere Diarization-Tunings |
| `stream` | bool | ndjson-Stream ein |
| `diarize` / `hf_token` / `batch_size` / `compute_type` | — | WhisperX-Aliase, akzeptiert |

#### `verbose_json` (OpenAI-kompatibel + speaker)

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
        { "word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00" }
      ]
    }
  ]
}
```

#### ndjson-Stream

```jsonl
{"type":"duration","duration":42.3}
{"type":"progress","progress":0}
{"type":"segment","start":0.0,"end":1.5,"text":"Hello","speaker":"SPEAKER_00"}
{"type":"result","text":"...","language":"en"}
{"type":"progress","progress":100}
```

> Aktuell ist Streaming ein Wrapper um `transcribe()`, weil Granite-`generate`
> non-streaming ist. Token-Level-Streaming via `TextIteratorStreamer` ist Backlog.

### `GET /v1/models`, `POST /v1/models/load`, `POST /v1/models/unload`

Manuelles Hot-Swap-Steuern. `/health` reportet aktuell geladenes Modell.

---

## Granite-Pipeline-Details

```python
# Base ASR (default)
"<|audio|> transcribe the speech with proper punctuation and capitalization."

# Keyword-biased ASR
"<|audio|> transcribe the speech to text. Keywords: <kw1>, <kw2>"

# AST (2b)
"<|audio|> translate the speech to <Language> with proper punctuation and capitalization."

# Word timestamps (2b-plus)
"<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag..."

# Speaker attribution (2b-plus)
"<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: ..."
```

**Combined Word-TS + Speaker-Attribution**: Zwei separate Forward-Passes,
danach Sequential-Merge der Word-Streams (Speaker-Token wird auf Position-im-Turn
zu Position-im-TS-Stream gemappt). Best-effort, kein perfektes Alignment.

**Timestamp-Decoding**: `[T:NNN]` ist centiseconds modulo 1000 (10s rollover).
Unwrap-Logik in `_parse_word_timestamps` addiert 10s-Offsets bei Rollover-Detect.

---

## Deployment

### Lokal

```powershell
uv venv; .venv\Scripts\Activate.ps1
uv pip install -e .
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
granite-speech-api
```

### Docker Compose

```bash
# CUDA (default)
docker compose up -d

# ROCm (Strix Halo / AMD)
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d

# CPU
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

`hf-cache` ist als named volume gemounted → Modelle werden beim ersten Request
automatisch heruntergeladen und bleiben persistent über Container-Neustarts.

Optional: `HUGGING_FACE_HUB_TOKEN` als env setzen falls private/gated Modelle.

---

## Kompatibilitäts-Matrix

| Feature | 2b | 2b-plus | 2b-nar |
|---|---|---|---|
| ASR (plain) | ✅ | ✅ | ✅ |
| Punctuation + Capitalization | ✅ | ❌ | ✅ |
| Word-Level Timestamps | ❌ | ✅ via `[T:N]` | ❌ |
| Speaker Attribution | ❌ | ✅ via `[Speaker N]:` | ❌ |
| Keyword Biasing | ✅ | ✅ | ✅ |
| Speech Translation (AST) | ✅ 7 langs | ❌ | ✅ |
| Japanese ASR | ✅ | ❌ | ✅ |
| ndjson Streaming | ✅ wrapper | ✅ wrapper | ✅ wrapper |
| sona-kompatibel | ✅ | ✅ | ✅ |

---

## Backlog

- [ ] Chunking für Audios > 9 min (ASR/SAA) bzw. > 5 min (TS) mit
  `prefix_text`-basiertem incremental decoding
- [ ] Echtes Token-Streaming via `TextIteratorStreamer` statt Batch-Wrapper
- [ ] Bearer-Auth (env `API_KEY`) wenn ohne Reverse Proxy betrieben
- [ ] vLLM-Backend als Alternativpfad (Production / Throughput)
- [ ] Granite-Speech-Plus: zeitlich präzisere Speaker-Segmente im SAA-Only-Modus
- [ ] OpenAPI-Examples für Vibe/sona-Validierung
- [ ] CI: ruff + pytest
