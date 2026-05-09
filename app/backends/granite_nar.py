"""Backend for the non-autoregressive granite-speech-4.1-2b-nar.

This model has a different API than the AR family:
- AutoModel + AutoFeatureExtractor (not Seq2SeqLM + Processor)
- trust_remote_code=True
- Prefers flash_attention_2; falls back to sdpa for non-CUDA / non-flash-attn setups
- No chat template, no `<|audio|>` token. Just feeds waveforms to .generate()
- Output is `output.text_preds[0]`
- Supports only ASR (no AST / no SAA / no word timestamps / no KWB)
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from typing import Any

import torch

from app.audio import load_audio_bytes
from app.backends.base import ASRBackend
from app.schema import TranscriptionRequest, TranscriptionSegment

log = logging.getLogger(__name__)


GRANITE_NAR_ID = "ibm-granite/granite-speech-4.1-2b-nar"


def _to_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def _pick_attn_impl(device: str) -> str:
    has_flash = importlib.util.find_spec("flash_attn") is not None
    if device.startswith("cuda") and has_flash:
        return "flash_attention_2"
    return "sdpa"


class GraniteNARBackend(ASRBackend):
    def __init__(self, dtype: str = "bfloat16") -> None:
        self.model_id: str = ""
        self._extractor: Any = None
        self._model: Any = None
        self._device: str = "cpu"
        self._dtype = _to_torch_dtype(dtype)
        self._lock = asyncio.Lock()

    async def load(self, model_id: str, device: str) -> None:
        if self.model_id == model_id and self._model is not None:
            return
        await self.unload()
        attn_impl = _pick_attn_impl(device)
        log.info(
            "Loading NAR model %s on %s (%s, attn=%s)",
            model_id, device, self._dtype, attn_impl,
        )
        from transformers import AutoFeatureExtractor, AutoModel

        def _load() -> tuple[Any, Any]:
            mdl = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                device_map=device,
                dtype=self._dtype,
            )
            mdl.eval()
            ext = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
            return mdl, ext

        loop = asyncio.get_running_loop()
        mdl, ext = await loop.run_in_executor(None, _load)
        self._model = mdl
        self._extractor = ext
        self._device = device
        self.model_id = model_id
        log.info("NAR model %s ready", model_id)

    async def unload(self) -> None:
        if self._model is None:
            return
        log.info("Unloading NAR model %s", self.model_id)
        del self._model
        del self._extractor
        self._model = None
        self._extractor = None
        self.model_id = ""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    async def transcribe(
        self, req: TranscriptionRequest
    ) -> tuple[list[TranscriptionSegment], str | None]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        wav, duration = load_audio_bytes(req.audio_bytes)
        # NAR feature extractor expects 1-D mono waveform.
        waveform = wav.squeeze(0)

        async with self._lock:
            loop = asyncio.get_running_loop()

            def _infer() -> str:
                inputs = self._extractor([waveform], device=self._device)
                with torch.inference_mode():
                    output = self._model.generate(**inputs)
                preds = getattr(output, "text_preds", None)
                if not preds:
                    raise RuntimeError("NAR model returned no text_preds")
                return preds[0]

            text = await loop.run_in_executor(None, _infer)

        if req.word_timestamps or req.speaker_attribution:
            log.warning(
                "NAR backend cannot produce word timestamps or speaker labels — "
                "returning plain text. Use the -plus model for those features."
            )

        return [
            TranscriptionSegment(id=0, start=0.0, end=duration, text=text.strip())
        ], req.language
