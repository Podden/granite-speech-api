"""Backend for Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026).

2B conformer encoder-decoder, plain punctuated ASR (no word timestamps, no
speaker attribution — the pipeline's forced aligner and diarizer add those).
Audio is cut into ≤5 min windows at quiet points; within a window the model's
own feature extractor chunks further and ``processor.decode`` reassembles.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

import torch

from app.audio import TARGET_SR, load_audio_bytes
from app.backends.base import ASRBackend
from app.backends.granite import _collapse_repeats, _plan_windows, window_limits
from app.config import settings
from app.schema import TranscriptionRequest, TranscriptionSegment

log = logging.getLogger(__name__)

MAX_CHUNK_SECONDS = 300.0
TARGET_CHUNK_SECONDS = 240.0

# Prompt-style phrase the model occasionally emits at the start of a chunk
# (seen on German talk-show audio) — never actual speech.
_ARTIFACT_RE = re.compile(r"\s*Input transcript corrected:\s*", re.IGNORECASE)

# Languages the model was trained on (2-letter codes).
SUPPORTED_LANGUAGES = {
    "en", "fr", "de", "it", "es", "pt", "el", "nl", "pl", "zh", "ja", "ko", "vi", "ar",
}


def _to_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


class CohereTranscribeBackend(ASRBackend):
    def __init__(self, dtype: str = "bfloat16") -> None:
        self.model_id: str = ""
        self._processor: Any = None
        self._model: Any = None
        self._dtype = _to_torch_dtype(dtype)
        self._lock = asyncio.Lock()

    async def load(self, model_id: str, device: str) -> None:
        if self.model_id == model_id and self._model is not None:
            return
        await self.unload()
        log.info("Loading Cohere Transcribe %s on %s (%s)", model_id, device, self._dtype)
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        def _load() -> tuple[Any, Any]:
            proc = AutoProcessor.from_pretrained(model_id)
            mdl = CohereAsrForConditionalGeneration.from_pretrained(
                model_id, device_map=device, dtype=self._dtype
            )
            mdl.eval()
            return proc, mdl

        loop = asyncio.get_running_loop()
        self._processor, self._model = await loop.run_in_executor(None, _load)
        self.model_id = model_id
        log.info("Cohere Transcribe %s ready", model_id)

    async def unload(self) -> None:
        if self._model is None:
            return
        log.info("Unloading Cohere Transcribe %s", self.model_id)
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self.model_id = ""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    async def transcribe(
        self,
        req: TranscriptionRequest,
        progress_cb: Callable[[float], None] | None = None,
        partial_cb: Callable[[str, float, float], None] | None = None,
        delta_cb: Callable[[str], None] | None = None,  # noqa: ARG002 — no token stream
    ) -> tuple[list[TranscriptionSegment], str | None]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        wav, duration = load_audio_bytes(req.audio_bytes)
        # The processor requires a language code; default to English when no
        # hint is given (the UI always sends one).
        lang = (req.language or "en").strip().lower()[:2]
        if lang not in SUPPORTED_LANGUAGES:
            log.warning("Cohere Transcribe: unsupported language %r, using 'en'", lang)
            lang = "en"

        processor = self._processor
        model = self._model

        gen_extra: dict[str, Any] = {}
        if settings.repetition_penalty != 1.0:
            gen_extra["repetition_penalty"] = settings.repetition_penalty
        if settings.no_repeat_ngram_size > 0:
            gen_extra["no_repeat_ngram_size"] = settings.no_repeat_ngram_size

        def _infer(audio) -> str:
            inputs = processor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt", language=lang
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                # max_new_tokens is per chunk (feature extractor auto-chunks
                # long audio) — 512 is generous for the ~30s chunk size.
                outputs = model.generate(**inputs, max_new_tokens=512, **gen_extra)
            decoded = processor.decode(outputs, skip_special_tokens=True)
            # decode() reassembles chunked long-form audio and returns one
            # string per input audio (list) — we always pass exactly one.
            if isinstance(decoded, (list, tuple)):
                decoded = " ".join(str(part) for part in decoded)
            return _collapse_repeats(_ARTIFACT_RE.sub(" ", decoded))

        # Windows at quiet points give live partials/progress and keep each
        # piece within the forced aligner's 5-minute limit.
        windows = _plan_windows(
            wav, duration, *window_limits(req, MAX_CHUNK_SECONDS, TARGET_CHUNK_SECONDS)
        )
        segments: list[TranscriptionSegment] = []
        done = 0.0
        loop = asyncio.get_running_loop()
        async with self._lock:
            for t0, t1 in windows:
                piece = wav[0, int(t0 * TARGET_SR): int(t1 * TARGET_SR)].numpy()
                text = (await loop.run_in_executor(None, _infer, piece)).strip()
                if text:
                    segments.append(
                        TranscriptionSegment(
                            id=len(segments), start=round(t0, 3), end=round(t1, 3),
                            text=text,
                        )
                    )
                    if partial_cb:
                        partial_cb(text, round(t0, 3), round(t1, 3))
                done += t1 - t0
                if progress_cb and duration > 0:
                    progress_cb(min(99.0, done / duration * 100.0))

        if not segments:
            segments = [TranscriptionSegment(id=0, start=0.0, end=duration, text="")]
        return segments, lang
