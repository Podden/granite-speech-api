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
        partial_cb: Callable[[str, float, float], None] | None = None,
    ) -> tuple[list[TranscriptionSegment], str | None]:
        """Run inference. Returns (segments, detected_language).

        `progress_cb` (optional) receives percentages in [0, 100] as chunks
        complete. `partial_cb` (optional) receives (text, start, end) of each
        finished chunk for live display. Backends without chunking may ignore
        both.
        """

    async def transcribe_stream(
        self, req: TranscriptionRequest
    ) -> AsyncIterator[dict]:
        """Default streaming wrapper: emits progress + partial chunk texts +
        final segments + result. Progress/partial events are forwarded live
        from `transcribe()`. Backends with native streaming may override.
        """
        yield {"type": "progress", "progress": 0}

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict] = asyncio.Queue()

        def on_progress(pct: float) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "progress", "progress": round(pct, 1)}
            )

        def on_partial(text: str, start: float, end: float) -> None:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "partial", "text": text, "start": start, "end": end},
            )

        task = asyncio.create_task(
            self.transcribe(req, progress_cb=on_progress, partial_cb=on_partial)
        )
        while not task.done():
            getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {getter, task}, return_when=asyncio.FIRST_COMPLETED
            )
            if getter in done:
                yield getter.result()
            else:
                getter.cancel()
        while not queue.empty():
            yield queue.get_nowait()

        segments, language = await task
        for seg in segments:
            ev: dict = {
                "type": "segment",
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker,
            }
            if seg.words:
                ev["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in seg.words
                ]
            yield ev
        yield {
            "type": "result",
            "text": " ".join(s.text for s in segments).strip(),
            "language": language,
        }
        yield {"type": "progress", "progress": 100}
