"""Backend for Qwen3-ASR-1.7B via native transformers support (qwen3_asr arch).

Plain ASR with built-in language identification (52 languages). Long audio is
chunked at quiet points with the same window planner as the Granite backend,
so token budgets stay bounded. No word timestamps yet (would need the separate
Qwen3-ForcedAligner-0.6B-hf), hence no speaker attribution.

Note: the transformers-converted checkpoint lives under the ``-hf`` suffix
(``Qwen/Qwen3-ASR-1.7B-hf``); the original repo only works with Qwen's own
``qwen-asr`` package, which hard-pins an old transformers incompatible with
Granite Speech 4.1.
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
from app.schema import TranscriptionRequest, TranscriptionSegment

log = logging.getLogger(__name__)

# Chunking bounds (the model handles a few minutes comfortably; keep windows
# well below that and cut at quiet points).
MAX_CHUNK_SECONDS = 300.0
TARGET_CHUNK_SECONDS = 240.0

# Detected-language full names → ISO codes (subset relevant for us).
_LANG_CODES = {
    "german": "de",
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "polish": "pl",
    "russian": "ru",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
}


def _to_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


class QwenASRBackend(ASRBackend):
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
        log.info("Loading Qwen3-ASR %s on %s (%s)", model_id, device, self._dtype)
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        def _load() -> tuple[Any, Any]:
            proc = AutoProcessor.from_pretrained(model_id)
            mdl = Qwen3ASRForConditionalGeneration.from_pretrained(
                model_id, device_map=device, dtype=self._dtype
            )
            mdl.eval()
            return proc, mdl

        loop = asyncio.get_running_loop()
        self._processor, self._model = await loop.run_in_executor(None, _load)
        self.model_id = model_id
        log.info("Qwen3-ASR %s ready", model_id)

    async def unload(self) -> None:
        if self._model is None:
            return
        log.info("Unloading Qwen3-ASR %s", self.model_id)
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
        # apply_transcription_request accepts ISO codes and full names alike.
        lang_hint = (req.language or "").strip().lower()[:2] or None
        processor = self._processor
        model = self._model
        loop = asyncio.get_running_loop()

        windows = _plan_windows(wav, duration, MAX_CHUNK_SECONDS, TARGET_CHUNK_SECONDS)
        segments: list[TranscriptionSegment] = []
        detected: str | None = None
        done = 0.0

        async with self._lock:
            for t0, t1 in windows:
                piece = wav[0, int(t0 * TARGET_SR): int(t1 * TARGET_SR)].numpy()
                budget = int((t1 - t0) * 15) + 256

                def _infer(chunk=piece, max_new=budget) -> dict:
                    inputs = processor.apply_transcription_request(
                        audio=chunk, language=lang_hint
                    ).to(model.device, model.dtype)
                    with torch.inference_mode():
                        out = model.generate(
                            **inputs, max_new_tokens=max_new, do_sample=False
                        )
                    gen = out[:, inputs["input_ids"].shape[1]:]
                    return processor.decode(gen, return_format="parsed")[0]

                parsed = await loop.run_in_executor(None, _infer)
                text = (parsed.get("transcription") or "").strip()
                if detected is None and parsed.get("language"):
                    detected = str(parsed["language"])
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
        lang_code = req.language
        if not lang_code and detected:
            lang_code = _LANG_CODES.get(detected.lower(), detected)
        return segments, lang_code
