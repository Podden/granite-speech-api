from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0
    speaker: str | None = None


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[TranscriptionWord] | None = None
    # Whisper-style fields kept for OpenAI verbose_json compatibility.
    seek: int = 0
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible verbose_json response."""

    task: str = "transcribe"
    language: str | None = None
    duration: float | None = None
    text: str
    segments: list[TranscriptionSegment] | None = None
    words: list[TranscriptionWord] | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "ibm-granite"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
