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
    # Already-decoded transcript of the *same* audio's beginning (incremental /
    # live use): the AR Granite models then decode only the continuation.
    prefix_text: str | None = None
    # Upper bound for transcriber windows (seconds). Set by the pipeline when a
    # forced-alignment stage follows, which handles at most 5 min per call.
    max_window_seconds: float | None = None
