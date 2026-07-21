from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptionRequest:
    """Normalized form of an /v1/audio/transcriptions request."""

    audio_bytes: bytes
    filename: str
    model: str
    language: str | None
    response_format: str  # "json" | "text" | "srt" | "vtt" | "verbose_json"
    word_timestamps: bool
    segment_timestamps: bool
    speaker_attribution: bool
    translate: bool
    translate_to: str | None
    prompt: str | None
    stream: bool
    min_speakers: int | None
    max_speakers: int | None
    num_speakers: int | None = None
    # Speaker turns from the pyannote stage (list[app.diarization.Turn]).
    # When set, backends skip their own speaker attribution and label words
    # against these turns instead.
    diarization_turns: list | None = None
