"""Backend for granite-speech-5.0-470m-turboctc (English-only CTC, transformers-native).

A 470M conformer with greedy CTC decoding: no LLM, so no hallucinated text and
very high throughput. Output is normalized: lowercase, no punctuation, "i am"
for "I'm". English only; no AST / SAA / word timestamps / keyword biasing. Long audio is cut at
quiet points and the windows are decoded in batches.

The ``-nc`` sibling (more training data) is CC-BY-NC-SA, so it is not offered.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import torch

from app.audio import TARGET_SR, load_audio_bytes
from app.backends.base import ASRBackend
from app.backends.granite import _plan_windows
from app.lazy import free_gpu_memory
from app.schema import TranscriptionRequest, TranscriptionSegment

log = logging.getLogger(__name__)

GRANITE_TURBOCTC = "ibm-granite/granite-speech-5.0-470m-turboctc"
# ponytail: same 60 s windows as the base 2b; the card names no limit. Measure
# longer windows before raising this.
MAX_CHUNK_SECONDS = 75.0
TARGET_CHUNK_SECONDS = 60.0
BATCH_SIZE = 16


def _to_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


class TurboCTCBackend(ASRBackend):
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
        log.info("Loading TurboCTC %s on %s (%s)", model_id, device, self._dtype)
        from transformers import AutoModelForCTC, AutoProcessor

        def _load() -> tuple[Any, Any]:
            proc = AutoProcessor.from_pretrained(model_id)
            mdl = AutoModelForCTC.from_pretrained(model_id, device_map=device, dtype=self._dtype)
            mdl.eval()
            return proc, mdl

        loop = asyncio.get_running_loop()
        self._processor, self._model = await loop.run_in_executor(None, _load)
        self.model_id = model_id
        log.info("TurboCTC %s ready", model_id)

    async def unload(self) -> None:
        if self._model is None:
            return
        log.info("Unloading TurboCTC %s", self.model_id)
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self.model_id = ""
        free_gpu_memory()

    async def transcribe(
        self,
        req: TranscriptionRequest,
        progress_cb: Callable[[float], None] | None = None,
        partial_cb: Callable[[str, float, float], None] | None = None,
        delta_cb: Callable[[str], None] | None = None,  # noqa: ARG002 — CTC, no token stream
    ) -> tuple[list[TranscriptionSegment], str | None]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        wav, duration = load_audio_bytes(req.audio_bytes)
        cap = min(req.max_window_seconds or MAX_CHUNK_SECONDS, MAX_CHUNK_SECONDS)
        target = TARGET_CHUNK_SECONDS if cap == MAX_CHUNK_SECONDS else cap * 0.8
        windows = _plan_windows(wav, duration, cap, target)
        processor, model = self._processor, self._model
        sr = processor.feature_extractor.sampling_rate
        loop = asyncio.get_running_loop()

        def _infer(pieces: list) -> list[str]:
            inputs = processor(pieces, sampling_rate=sr, device=model.device)
            inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                out = model.generate(**inputs)
            return processor.batch_decode(out, skip_special_tokens=True)

        segments: list[TranscriptionSegment] = []
        async with self._lock:
            for i in range(0, len(windows), BATCH_SIZE):
                batch = windows[i: i + BATCH_SIZE]
                pieces = [wav[0, int(t0 * TARGET_SR): int(t1 * TARGET_SR)].numpy() for t0, t1 in batch]
                texts = await loop.run_in_executor(None, _infer, pieces)
                for (t0, t1), text in zip(batch, texts, strict=True):
                    text = text.strip()
                    if not text:
                        continue
                    segments.append(TranscriptionSegment(
                        id=len(segments), start=round(t0, 3), end=round(t1, 3), text=text,
                    ))
                    if partial_cb:
                        partial_cb(text, round(t0, 3), round(t1, 3))
                if progress_cb and duration > 0:
                    progress_cb(min(99.0, batch[-1][1] / duration * 100.0))

        if not segments:
            segments = [TranscriptionSegment(id=0, start=0.0, end=duration, text="")]
        return segments, "en"
