from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable

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
        self,
        req: TranscriptionRequest,
        progress_cb: Callable[[float], None] | None = None,
    ) -> tuple[list[TranscriptionSegment], str | None]:
        """Run inference. Returns (segments, detected_language).

        `progress_cb` (optional) receives percentages in [0, 100] as chunks
        complete; backends without chunking may ignore it.
        """

    async def transcribe_stream(
        self, req: TranscriptionRequest
    ) -> AsyncIterator[dict]:
        """Default streaming wrapper: emits progress + final segments + result.

        Progress events are forwarded live from `transcribe()` chunk progress.
        Backends with native streaming may override.
        """
        yield {"type": "progress", "progress": 0}

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[float] = asyncio.Queue()

        def on_progress(pct: float) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, pct)

        task = asyncio.create_task(self.transcribe(req, progress_cb=on_progress))
        while not task.done():
            getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {getter, task}, return_when=asyncio.FIRST_COMPLETED
            )
            if getter in done:
                yield {"type": "progress", "progress": round(getter.result(), 1)}
            else:
                getter.cancel()
        while not queue.empty():
            yield {"type": "progress", "progress": round(queue.get_nowait(), 1)}

        segments, language = await task
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
