"""Composable transcription pipeline: diarizer → transcriber → aligner.

Every stage is picked independently per request:

- **transcriber** (`model`): any ASR backend from the registry.
- **aligner**: where word timestamps come from — `native` (Granite 2b-plus
  emits them itself), `qwen` (Qwen3 forced aligner on the transcriber's text,
  works for every transcriber), `none`, or `auto` (native when available,
  otherwise qwen — only when words are actually needed).
- **diarizer**: `pyannote` / `nemotron` produce speaker turns that are
  reconciled with the word timestamps; `granite` is the 2b-plus model's own
  speaker-attribution pass; `auto` = configured default engine, falling back
  to granite when the external engine fails and the transcriber supports it.

`plan()` validates the combination up front (HTTP 400 on impossible ones);
`Pipeline.diarize()` runs before the transcriber, `Pipeline.post()` after it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fastapi import HTTPException

from app.aligner import MAX_ALIGN_SECONDS, aligner as forced_aligner
from app.audio import load_audio_bytes
from app.backends.catalog import has_native_words
from app.config import settings
from app.diarization import Turn, diarizer, label_words
from app.schema import TranscriptionRequest, TranscriptionSegment, TranscriptionWord

log = logging.getLogger(__name__)


@dataclass
class Pipeline:
    req: TranscriptionRequest
    diarizer: str | None  # external engine ("pyannote" | "nemotron") or None
    aligner: str | None  # "qwen" | "native" | None
    granite_fallback: bool = False  # auto mode: fall back to granite SAA on failure
    turns: list[Turn] | None = field(default=None, repr=False)

    async def diarize(self) -> str | None:
        """Run the external diarizer (if any). Returns the engine actually used."""
        if self.diarizer is None:
            return "granite" if self.req.speaker_attribution else None
        try:
            self.turns = await diarizer.diarize(
                self.req.audio_bytes,
                engine=self.diarizer,
                num_speakers=self.req.num_speakers,
                min_speakers=self.req.min_speakers,
                max_speakers=self.req.max_speakers,
            )
            return self.diarizer
        except Exception as exc:
            if not self.granite_fallback:
                raise HTTPException(
                    status_code=502, detail=f"{self.diarizer} diarization failed: {exc}"
                ) from exc
            log.warning("%s diarization failed (%s) — falling back to granite SAA",
                        self.diarizer, exc)
            self.diarizer = None
            self.turns = None
            # Granite SAA needs the backend's own speaker pass back on.
            self.req.speaker_attribution = True
            return "granite"

    async def post(
        self, segments: list[TranscriptionSegment], language: str | None
    ) -> list[TranscriptionSegment]:
        """Alignment + speaker reconciliation on the transcriber's segments."""
        if self.aligner == "qwen":
            wav, _ = load_audio_bytes(self.req.audio_bytes)
            words = await forced_aligner.align(wav, segments, language or self.req.language)
            if not self.turns:
                _attach_words(segments, words)
                return segments
        elif self.turns:
            words = [w for s in segments for w in (s.words or [])]
        else:
            return segments
        if not words:
            return segments
        return label_words(words, self.turns) or segments

    @property
    def needs_post(self) -> bool:
        return self.aligner == "qwen" or bool(self.diarizer)


def _attach_words(
    segments: list[TranscriptionSegment], words: list[TranscriptionWord]
) -> None:
    """Give each segment the aligned words that fall into its time span."""
    i = 0
    for seg in segments:
        own: list[TranscriptionWord] = []
        while i < len(words) and words[i].start < seg.end:
            own.append(words[i])
            i += 1
        seg.words = own or None
    if i < len(words) and segments:  # rounding at the last boundary
        segments[-1].words = (segments[-1].words or []) + words[i:]


def plan(
    req: TranscriptionRequest,
    model_id: str,
    *,
    diarization_engine: str,
    aligner: str,
) -> Pipeline:
    """Resolve and validate the stage combination; adjusts `req` for the backend.

    Raises HTTP 400 for combinations that cannot work.
    """
    native = has_native_words(model_id)
    speakers = req.speaker_attribution

    if req.translate:
        # Translated text can't be aligned to (or diarized against) the audio.
        if speakers or aligner == "qwen":
            log.info("Translation: alignment/diarization stages skipped")
        req.speaker_attribution = False
        return Pipeline(req, diarizer=None, aligner=None)

    # --- diarizer ---
    ext: str | None = None
    fallback = False
    if speakers:
        if diarization_engine == "granite":
            if not native:
                raise HTTPException(
                    status_code=400,
                    detail=f"diarization_engine=granite needs granite-speech-4.1-2b-plus, "
                    f"not {model_id}. Use pyannote or nemotron.",
                )
        elif diarization_engine == "auto":
            ext = settings.default_diarization_engine
            fallback = native
        else:
            ext = diarization_engine

    # --- aligner ---
    need_words = req.word_timestamps or ext is not None
    if aligner == "native" and not native:
        raise HTTPException(
            status_code=400,
            detail=f"{model_id} has no native word timestamps — use aligner=qwen or auto.",
        )
    if aligner == "none" and ext is not None and not native:
        raise HTTPException(
            status_code=400,
            detail=f"{ext} diarization needs word timestamps, but {model_id} has none "
            "and aligner=none. Use aligner=qwen or auto.",
        )
    if aligner == "qwen":
        resolved: str | None = "qwen"
    elif aligner == "none":
        resolved = "native" if (ext and native) else None
    elif need_words:  # auto / native
        resolved = "native" if native else "qwen"
    else:
        resolved = None

    # --- backend request flags ---
    if ext is not None:
        req.speaker_attribution = False  # external turns replace the SAA pass
    req.word_timestamps = resolved == "native"
    if resolved == "qwen":
        req.max_window_seconds = MAX_ALIGN_SECONDS
    return Pipeline(req, diarizer=ext, aligner=resolved, granite_fallback=fallback)
