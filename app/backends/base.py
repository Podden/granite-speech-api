from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.schema import TranscriptionRequest, TranscriptionSegment


class ASRBackend(ABC):
    """Abstract speech-recognition backend."""

    model_id: str

    @abstractmethod
    async def load(self, model_id: str, device: str) -> None: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    async def transcribe(
        self, req: TranscriptionRequest
    ) -> tuple[list[TranscriptionSegment], str | None]:
        """Run inference. Returns (segments, detected_language)."""

    async def transcribe_stream(
        self, req: TranscriptionRequest
    ) -> AsyncIterator[dict]:
        """Default streaming wrapper: emits progress + final segments + result.

        Backends with native streaming may override.
        """
        yield {"type": "progress", "progress": 0}
        segments, language = await self.transcribe(req)
        for seg in segments:
            yield {
                "type": "segment",
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker,
            }
        yield {
            "type": "result",
            "text": " ".join(s.text for s in segments).strip(),
            "language": language,
        }
        yield {"type": "progress", "progress": 100}
