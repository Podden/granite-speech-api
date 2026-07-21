"""Fusion pipeline backend — best-of-everything multi-pass transcription.

Per audio window (cut at quiet points):
    1. Cohere Transcribe produces punctuated, capitalized text.
    2. Qwen3-ForcedAligner-0.6B maps that exact text back onto the audio,
       yielding word-level timestamps (model-agnostic forced alignment).
Combined with the pyannote diarization stage (turns arrive via
``req.diarization_turns``) this yields speaker-labelled segments whose text
keeps Cohere's punctuation — the combination no single model offers.

Both models stay loaded together (~7 GB bf16) and are dropped as one unit by
the registry's idle unload.
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
from app.backends.cohere_asr import SUPPORTED_LANGUAGES as COHERE_LANGUAGES
from app.backends.granite import _plan_windows, _speaker_runs_to_segments
from app.schema import TranscriptionRequest, TranscriptionSegment, TranscriptionWord

log = logging.getLogger(__name__)

ASR_MODEL = "CohereLabs/cohere-transcribe-03-2026"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

MAX_CHUNK_SECONDS = 300.0
TARGET_CHUNK_SECONDS = 240.0

# ISO code → full name (the aligner wants names, Cohere wants codes).
_LANG_NAMES = {
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


def _to_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


class FusionBackend(ASRBackend):
    def __init__(self, dtype: str = "bfloat16") -> None:
        self.model_id: str = ""
        self._asr_processor: Any = None
        self._asr_model: Any = None
        self._aligner_processor: Any = None
        self._aligner_model: Any = None
        self._dtype = _to_torch_dtype(dtype)
        self._lock = asyncio.Lock()

    async def load(self, model_id: str, device: str) -> None:
        if self.model_id == model_id and self._asr_model is not None:
            return
        await self.unload()
        log.info("Loading fusion pipeline (%s + %s) on %s", ASR_MODEL, ALIGNER_MODEL, device)
        from transformers import (
            AutoModelForTokenClassification,
            AutoProcessor,
            CohereAsrForConditionalGeneration,
        )

        def _load() -> tuple[Any, Any, Any, Any]:
            asr_proc = AutoProcessor.from_pretrained(ASR_MODEL)
            asr_mdl = CohereAsrForConditionalGeneration.from_pretrained(
                ASR_MODEL, device_map=device, dtype=self._dtype
            )
            asr_mdl.eval()
            al_proc = AutoProcessor.from_pretrained(ALIGNER_MODEL)
            al_mdl = AutoModelForTokenClassification.from_pretrained(
                ALIGNER_MODEL, device_map=device, dtype=self._dtype
            )
            al_mdl.eval()
            return asr_proc, asr_mdl, al_proc, al_mdl

        loop = asyncio.get_running_loop()
        (
            self._asr_processor,
            self._asr_model,
            self._aligner_processor,
            self._aligner_model,
        ) = await loop.run_in_executor(None, _load)
        self.model_id = model_id
        log.info("Fusion pipeline ready")

    async def unload(self) -> None:
        if self._asr_model is None:
            return
        log.info("Unloading fusion pipeline")
        del self._asr_model
        del self._aligner_model
        self._asr_model = None
        self._asr_processor = None
        self._aligner_model = None
        self._aligner_processor = None
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
        if self._asr_model is None:
            raise RuntimeError("Model not loaded")

        wav, duration = load_audio_bytes(req.audio_bytes)
        lang = (req.language or "").strip().lower()[:2]
        if lang not in COHERE_LANGUAGES:
            if lang:
                log.warning("Fusion: unsupported language %r, assuming 'de'", lang)
            lang = "de"
        lang_name = _LANG_NAMES[lang]

        asr_proc, asr_model = self._asr_processor, self._asr_model
        al_proc, al_model = self._aligner_processor, self._aligner_model
        loop = asyncio.get_running_loop()

        windows = _plan_windows(wav, duration, MAX_CHUNK_SECONDS, TARGET_CHUNK_SECONDS)
        words: list[TranscriptionWord] = []
        done = 0.0

        async with self._lock:
            for t0, t1 in windows:
                piece = wav[0, int(t0 * TARGET_SR): int(t1 * TARGET_SR)].numpy()

                def _window(chunk=piece, offset=t0) -> tuple[str, list[TranscriptionWord]]:
                    # Pass 1: Cohere — punctuated text (auto-chunks internally).
                    inputs = asr_proc(
                        chunk, sampling_rate=TARGET_SR, return_tensors="pt",
                        language=lang,
                    ).to(asr_model.device, dtype=asr_model.dtype)
                    with torch.inference_mode():
                        out = asr_model.generate(**inputs, max_new_tokens=512)
                    decoded = asr_proc.decode(out, skip_special_tokens=True)
                    if isinstance(decoded, (list, tuple)):
                        decoded = " ".join(str(p) for p in decoded)
                    text = decoded.strip()
                    if not text:
                        return "", []

                    # Pass 2: forced alignment of that exact text.
                    al_inputs, word_lists = al_proc.prepare_forced_aligner_inputs(
                        audio=chunk, transcript=text, language=lang_name,
                    )
                    al_inputs = al_inputs.to(al_model.device, al_model.dtype)
                    with torch.inference_mode():
                        al_out = al_model(**al_inputs)
                    stamps = al_proc.decode_forced_alignment(
                        logits=al_out.logits,
                        input_ids=al_inputs["input_ids"],
                        word_lists=word_lists,
                        timestamp_token_id=al_model.config.timestamp_token_id,
                    )[0]
                    # The aligner strips punctuation from its word items — map
                    # the timestamps back onto the original (punctuated) tokens.
                    originals = _restore_original_tokens(stamps, text)
                    chunk_words = [
                        TranscriptionWord(
                            word=word,
                            start=round(float(item["start_time"]) + offset, 3),
                            end=round(float(item["end_time"]) + offset, 3),
                            probability=1.0,
                        )
                        for item, word in zip(stamps, originals)
                    ]
                    return text, chunk_words

                text, chunk_words = await loop.run_in_executor(None, _window)
                words.extend(chunk_words)
                if partial_cb and text:
                    partial_cb(text, round(t0, 3), round(t1, 3))
                done += t1 - t0
                if progress_cb and duration > 0:
                    progress_cb(min(99.0, done / duration * 100.0))

        if not words:
            return [TranscriptionSegment(id=0, start=0.0, end=duration, text="")], lang

        turns = req.diarization_turns
        if turns:
            from app.diarization import assign_speakers

            assign_speakers(words, turns)
            segments = _speaker_runs_to_segments(words)
        else:
            if req.speaker_attribution:
                log.warning("Fusion: no diarization turns — returning unlabelled text")
            segments = [
                TranscriptionSegment(
                    id=0, start=words[0].start, end=words[-1].end,
                    text=" ".join(w.word for w in words), words=words,
                )
            ]
        return segments, lang
