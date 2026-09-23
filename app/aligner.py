"""Forced-alignment stage (Qwen3-ForcedAligner-0.6B).

Maps any transcriber's text back onto the audio, yielding word-level
timestamps. This is what lets text-only transcribers (Cohere, Qwen3-ASR,
Granite base/NAR) take part in word navigation and speaker diarization.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch

from app.audio import TARGET_SR
from app.config import settings
from app.lazy import LazyModel
from app.schema import TranscriptionSegment, TranscriptionWord

log = logging.getLogger(__name__)

# The aligner handles up to 5 min per call; transcribers are asked to keep
# their windows below this when alignment is planned (req.max_window_seconds).
MAX_ALIGN_SECONDS = 300.0

# ISO code → full name (the aligner wants names).
LANG_NAMES = {
    "de": "German",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "el": "Greek",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "ar": "Arabic",
    "ru": "Russian",
}


def _norm_token(s: str) -> str:
    return re.sub(r"[\W_]+", "", s, flags=re.UNICODE).lower()


def _restore_original_tokens(stamps: list[dict], text: str) -> list[str]:
    """Map aligner word items (punctuation-stripped) back to the original tokens.

    Walks both sequences in lock-step; when the aligner's normalized word is
    found within the next few original tokens, the original (punctuated,
    capitalized) token is used. Mismatches fall back to the aligner's text so
    timestamps never get lost.
    """
    orig = text.split()
    out: list[str] = []
    oi = 0
    for item in stamps:
        target = _norm_token(str(item.get("text", "")))
        matched = None
        for j in range(oi, min(oi + 3, len(orig))):
            cand = _norm_token(orig[j])
            if target and (target == cand or target in cand):
                matched = j
                break
        if matched is not None:
            out.append(orig[matched])
            oi = matched + 1
        else:
            out.append(str(item.get("text", "")))
    return out


class ForcedAligner(LazyModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = settings.aligner_model

    def _load(self) -> tuple[Any, Any]:
        from transformers import AutoModelForTokenClassification, AutoProcessor

        dtype = {"float16": torch.float16, "float32": torch.float32}.get(
            settings.dtype, torch.bfloat16
        )
        proc = AutoProcessor.from_pretrained(self.name)
        mdl = AutoModelForTokenClassification.from_pretrained(
            self.name, device_map=settings.resolved_device(), dtype=dtype
        )
        mdl.eval()
        return proc, mdl

    async def align(
        self,
        wav: torch.Tensor,
        segments: list[TranscriptionSegment],
        language: str | None,
    ) -> list[TranscriptionWord]:
        """Word timestamps for the text of each segment within its time span."""
        lang = (language or "de").strip().lower()[:2]
        lang_name = LANG_NAMES.get(lang, "German")

        def _align(model: tuple[Any, Any]) -> list[TranscriptionWord]:
            proc, mdl = model
            words: list[TranscriptionWord] = []
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                if seg.end - seg.start > MAX_ALIGN_SECONDS + 1:
                    log.warning(
                        "Aligner: segment %.0f-%.0fs exceeds %.0fs — timestamps may drift",
                        seg.start, seg.end, MAX_ALIGN_SECONDS,
                    )
                chunk = wav[0, int(seg.start * TARGET_SR): int(seg.end * TARGET_SR)].numpy()
                inputs, word_lists = proc.prepare_forced_aligner_inputs(
                    audio=chunk, transcript=text, language=lang_name,
                )
                inputs = inputs.to(mdl.device, mdl.dtype)
                with torch.inference_mode():
                    out = mdl(**inputs)
                stamps = proc.decode_forced_alignment(
                    logits=out.logits,
                    input_ids=inputs["input_ids"],
                    word_lists=word_lists,
                    timestamp_token_id=mdl.config.timestamp_token_id,
                )[0]
                originals = _restore_original_tokens(stamps, text)
                words.extend(
                    TranscriptionWord(
                        word=word,
                        start=round(float(item["start_time"]) + seg.start, 3),
                        end=round(float(item["end_time"]) + seg.start, 3),
                        probability=1.0,
                    )
                    for item, word in zip(stamps, originals)
                )
            return words

        return await self.run(_align)


aligner = ForcedAligner()
