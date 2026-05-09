# Plan: granite-speech-api — Neues Repo

## Ziel

FastAPI-Server, der eine **OpenAI Audio API-kompatible** HTTP-Schnittstelle exponiert und
folgende Backends unterstützt:

- **Granite Speech 4.1** (2b & 2b-plus) — Primärbackend, nativ word-timestamps + speaker attribution

Vollständig kompatibel mit Vibe's sona-HTTP-API (Drop-in-Ersatz) und WhisperX

---

## Repo-Struktur

```
granite-speech-api/
├── pyproject.toml
├── README.md
├── .env.example
├── Dockerfile
├── app/
│   ├── main.py                 # FastAPI app entrypoint
│   ├── config.py               # Settings via pydantic-settings
│   ├── api/
│   │   ├── routes.py           # Route-Registrierung
│   │   ├── transcriptions.py   # POST /v1/audio/transcriptions
│   │   ├── models.py           # GET /v1/models, POST /v1/models/load
│   │   └── health.py           # GET /health
│   ├── backends/
│   │   ├── base.py             # Abstrakte Backend-Klasse
│   │   └── granite.py          # Granite Speech 4.1 Backend
│   └── schema/
│       ├── request.py          # TranscriptionRequest
│       └── response.py         # TranscriptionResponse (OpenAI-kompatibel)
```

---

## API-Spezifikation

### `POST /v1/audio/transcriptions`

Multipart-Form, OpenAI-kompatibel:

| Field | Typ | Beschreibung |
|---|---|---|
| `file` | file | Audio-Datei (mp3, wav, ogg, m4a, …) |
| `model` | string | `"ibm-granite/granite-speech-4.1-2b"`, `"...-plus"`, `"whisperx"` |
| `language` | string | ISO 639-1, optional (auto-detect wenn leer) |
| `response_format` | string | `"json"` \| `"text"` \| `"srt"` \| `"vtt"` \| `"verbose_json"` |
| `timestamp_granularities[]` | string[] | `"segment"` \| `"word"` |
| `stream` | bool | ndjson-Stream (sona-Extension) |
| `prompt` | string | Initiales Prompt / Keyword-Biasing |
| `translate` | bool | Output nach Englisch übersetzen (AST) |
| — | — | **Granite-Extensions** |
| `speaker_attribution` | bool | Speaker-Labels (nur plus-Modell) |
| `translate_to` | string | AST-Zielsprache: `en/de/fr/es/pt/ja/zh` |
| — | — | **WhisperX-Extensions** |
| `hf_token` | string | HuggingFace Token für pyannote Diarization |
| `min_speakers` | int | Mindestanzahl Speaker |
| `max_speakers` | int | Maximalanzahl Speaker |
| `batch_size` | int | Batch-Größe für faster-whisper (default 16) |
| `compute_type` | string | `"float16"` \| `"int8"` \| `"float32"` |

### `verbose_json` Response-Format (OpenAI-kompatibel)

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 42.3,
  "text": "Hello world, how are you?",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.4,
      "text": "Hello world, how are you?",
      "speaker": "SPEAKER_00",
      "words": [
        { "word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.99, "speaker": "SPEAKER_00" },
        { "word": "world,", "start": 0.6, "end": 1.1, "probability": 0.98, "speaker": "SPEAKER_00" }
      ]
    }
  ]
}
```

### ndjson-Stream (sona-kompatibel)

```jsonl
{"type": "progress", "progress": 15}
{"type": "segment", "start": 0.0, "end": 3.4, "text": "Hello world", "speaker": 0}
{"type": "result", "text": "Hello world"}
```

### `GET /v1/models`

```json
{
  "object": "list",
  "data": [
    { "id": "ibm-granite/granite-speech-4.1-2b", "object": "model" },
    { "id": "ibm-granite/granite-speech-4.1-2b-plus", "object": "model" },
    { "id": "whisperx/large-v3", "object": "model" }
  ]
}
```

### `POST /v1/models/load`

```json
{ "path": "ibm-granite/granite-speech-4.1-2b", "device": "cuda" }
```

---

## Backend-Abstraktion

```python
# app/backends/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator
from app.schema.response import TranscriptionSegment

class ASRBackend(ABC):
    @abstractmethod
    async def load(self, model_id: str, device: str) -> None: ...

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None,
        word_timestamps: bool,
        speaker_attribution: bool,
        translate: bool,
        prompt: str | None,
        **kwargs,
    ) -> list[TranscriptionSegment]: ...

    async def transcribe_stream(self, ...) -> AsyncIterator[dict]:
        # Default: wraps transcribe() und emittiert progress + segments
        ...
```

---

## WhisperX-Pipeline im Backend

```python
# app/backends/whisperx.py
import whisperx

class WhisperXBackend(ASRBackend):
    async def load(self, model_id: str, device: str):
        # model_id z.B. "whisperx/large-v3" → extrahiere "large-v3"
        whisper_model = model_id.split("/")[-1]
        self._asr = whisperx.load_model(whisper_model, device, compute_type="float16")
        self._device = device

    async def transcribe(self, audio_bytes, language, word_timestamps,
                         speaker_attribution, translate, prompt,
                         hf_token=None, min_speakers=None, max_speakers=None, **kwargs):
        audio = whisperx.load_audio(audio_bytes)  # numpy array

        # Schritt 1: ASR
        result = self._asr.transcribe(audio, batch_size=16, language=language)

        # Schritt 2: Forced Alignment → Word-Level Timestamps
        if word_timestamps:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self._device
            )
            result = whisperx.align(result["segments"], model_a, metadata, audio, self._device)

        # Schritt 3: Speaker Diarization (optional)
        if speaker_attribution and hf_token:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(token=hf_token, device=self._device)
            diarize_segments = diarize_model(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)

        return self._to_segments(result)
```

---

## Granite-Pipeline im Backend

```python
# app/backends/granite.py
import torch, torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

class GraniteBackend(ASRBackend):
    async def load(self, model_id: str, device: str):
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16
        )
        self._model_id = model_id

    async def transcribe(self, audio_bytes, language, word_timestamps,
                         speaker_attribution, translate, prompt, **kwargs):
        wav = self._load_audio(audio_bytes)  # → mono 16kHz tensor

        # Prompt-Auswahl je nach Modus
        task_prompt = self._build_prompt(
            language=language,
            word_timestamps=word_timestamps,
            speaker_attribution=speaker_attribution,
            translate=translate,
            prompt=prompt,
        )

        inputs = self._processor(task_prompt, wav, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(**inputs, max_new_tokens=2000, do_sample=False)
        text = self._processor.tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:],
                                                skip_special_tokens=True)

        # Parse Granite-Output:
        # - Plain ASR → einfaches Segment
        # - word timestamps → "[T:N]" Tags parsen
        # - speaker attribution → "[Speaker N]:" Tags parsen
        return self._parse_output(text, word_timestamps, speaker_attribution)

    def _build_prompt(self, language, word_timestamps, speaker_attribution, translate, prompt):
        # Granite prompt table:
        if speaker_attribution:
            return "<|audio|> Speaker attribution: Transcribe and denote who is speaking..."
        if word_timestamps:
            return "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag..."
        if translate:
            return f"<|audio|> translate the speech to {translate} with proper punctuation."
        if prompt:
            return f"<|audio|> transcribe the speech to text. Keywords: {prompt}"
        return "<|audio|> transcribe the speech with proper punctuation and capitalization."
```

---

## pyproject.toml

```toml
[project]
name = "granite-speech-api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "python-multipart>=0.0.9",
    "sse-starlette>=2.0",
    "pydantic-settings>=2.0",
    "torch>=2.7",
    "torchaudio>=2.7",
    "transformers>=4.52.1",
    "accelerate>=0.30",
    # Optional: WhisperX backend
    # "whisperx>=3.8",
    # Optional: vLLM backend (Production)
    # "vllm>=0.8",
]

[project.optional-dependencies]
whisperx = ["whisperx>=3.8"]
vllm = ["vllm>=0.8"]
dev = ["pytest", "httpx", "pytest-asyncio"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Kompatibilitäts-Matrix

| Feature | Granite 2b | Granite 2b-plus | WhisperX |
|---|---|---|---|
| ASR (plain) | ✅ | ✅ | ✅ |
| Punctuation / Capitalization | ✅ | ❌ | ✅ via Whisper |
| Word-Level Timestamps | ❌ nativ | ✅ via `[T:N]` | ✅ via wav2vec2 |
| Speaker Diarization | ❌ | ✅ `[Speaker N]:` | ✅ via pyannote |
| Keyword Biasing | ✅ | ✅ | ❌ |
| Speech Translation (AST) | ✅ 7 Sprachen | ❌ | ✅ via Whisper |
| Japanese ASR | ✅ | ❌ | ✅ |
| ndjson Streaming | ✅ | ✅ | ❌ (implementierbar) |
| sona-kompatibel | ✅ | ✅ | ✅ |
| GPU-Anforderung | ~5GB VRAM | ~5GB VRAM | ~8GB VRAM (large-v2) |

---

## Erste Schritte im neuen Repo

1. `uv init --python 3.11 granite-speech-api`
2. `uv add fastapi uvicorn python-multipart transformers torch torchaudio accelerate`
3. FastAPI-Skeleton mit `/health` und `/v1/audio/transcriptions` (stub)
4. `GraniteBackend` implementieren, mit einfachem ASR testen
5. `verbose_json` Response-Format implementieren (OpenAI-kompatibel)
6. ndjson-Streaming hinzufügen (sona-Kompatibilität)
7. `WhisperXBackend` als optionales Extra implementieren
8. Dockerfile für einfaches Deployment
9. Vibe: sona-URL auf lokalen granite-speech-api Server umzeigen → validieren

---

## Abgrenzung zu sona

| | sona (Go + whisper.cpp) | granite-speech-api (Python) |
|---|---|---|
| Modellformat | GGML `.bin` | Safetensors / HuggingFace |
| Zielplattform | Desktop-Sidecar | Server / Cloud |
| GPU | CUDA / Metal / CPU | CUDA / CPU |
| Deployment | Tauri-Bundle | Docker / Bare-Metal |
| Streaming | ✅ ndjson | ✅ ndjson (geplant) |
