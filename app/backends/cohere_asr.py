"""Backend for Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026).

2B conformer encoder-decoder, plain ASR only (no word timestamps, no speaker
attribution). Long-form audio is handled by the model's own feature extractor,
which chunks the waveform and reassembles the per-chunk transcriptions via
``audio_chunk_index`` inside ``processor.decode``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import torch

from app.audio import TARGET_SR, load_audio_bytes
from app.backends.base import ASRBackend
from app.schema import TranscriptionRequest, TranscriptionSegment

log = logging.getLogger(__name__)

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
        audio = wav[0].numpy()
        # The processor requires a language code; default to English when no
        # hint is given (the UI always sends one).
        lang = (req.language or "en").strip().lower()[:2]
        if lang not in SUPPORTED_LANGUAGES:
            log.warning("Cohere Transcribe: unsupported language %r, using 'en'", lang)
            lang = "en"

        processor = self._processor
        model = self._model

        def _infer() -> str:
            inputs = processor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt", language=lang
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                # max_new_tokens is per chunk (feature extractor auto-chunks
                # long audio) — 512 is generous for the ~30s chunk size.
                outputs = model.generate(**inputs, max_new_tokens=512)
            return processor.decode(outputs, skip_special_tokens=True)

        async with self._lock:
            if progress_cb:
                progress_cb(5.0)
            loop = asyncio.get_running_loop()
            text = (await loop.run_in_executor(None, _infer)).strip()

        if partial_cb and text:
            partial_cb(text, 0.0, duration)
        if progress_cb:
            progress_cb(99.0)
        segments = [
            TranscriptionSegment(id=0, start=0.0, end=round(duration, 3), text=text)
        ]
        return segments, lang
