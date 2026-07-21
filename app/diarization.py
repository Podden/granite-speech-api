"""Speaker diarization via pyannote (community-1) with idle auto-unload.

Runs as a separate pipeline stage in front of the ASR backend: the diarizer
produces speaker turns, the ASR backend produces word timestamps, and
``assign_speakers`` reconciles the two into speaker-labelled words.

The pyannote pipeline (~500 MB VRAM) is lazy-loaded on first use and unloaded
again after the same idle TTL as the ASR registry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.audio import TARGET_SR, load_audio_bytes
from app.config import settings
from app.schema import TranscriptionWord

log = logging.getLogger(__name__)


@dataclass
class Turn:
    """One diarization turn: `speaker` talks from `start` to `end` (seconds)."""

    start: float
    end: float
    speaker: str


class Diarizer:
    """Lazy-loaded pyannote pipeline with registry-style idle unload."""

    def __init__(self) -> None:
        self._pipeline = None
        self._lock = asyncio.Lock()
        self._last_used: float = 0.0
        self._idle_task: asyncio.Task[None] | None = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    async def diarize(
        self,
        audio_bytes: bytes,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[Turn]:
        """Return speaker turns for `audio_bytes`, sorted by start time."""
        async with self._lock:
            await self._ensure_loaded()
            self._last_used = time.monotonic()
            pipeline = self._pipeline

            def _run() -> list[Turn]:
                wav, _ = load_audio_bytes(audio_bytes)
                kwargs: dict = {}
                if num_speakers:
                    kwargs["num_speakers"] = num_speakers
                else:
                    if min_speakers:
                        kwargs["min_speakers"] = min_speakers
                    if max_speakers:
                        kwargs["max_speakers"] = max_speakers
                out = pipeline({"waveform": wav, "sample_rate": TARGET_SR}, **kwargs)
                # community-1 returns an output object with the *exclusive*
                # diarization (non-overlapping, built for reconciliation with
                # ASR timestamps); 3.1 returns a plain Annotation.
                ann = getattr(out, "exclusive_speaker_diarization", None)
                if ann is None:
                    ann = getattr(out, "speaker_diarization", out)
                turns = [
                    Turn(float(seg.start), float(seg.end), str(label))
                    for seg, _, label in ann.itertracks(yield_label=True)
                ]
                turns.sort(key=lambda t: t.start)
                return turns

            loop = asyncio.get_running_loop()
            turns = await loop.run_in_executor(None, _run)
            self._last_used = time.monotonic()
            log.info(
                "Diarization: %d turns, %d speakers",
                len(turns), len({t.speaker for t in turns}),
            )
            return turns

    async def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        model_id = settings.diarization_model
        token = settings.hf_token or None
        device = settings.resolved_device()
        log.info("Loading diarization pipeline %s on %s", model_id, device)

        def _load():
            import torch
            from pyannote.audio import Pipeline

            try:
                pipe = Pipeline.from_pretrained(model_id, token=token)
            except TypeError:  # pyannote.audio < 4 uses use_auth_token
                pipe = Pipeline.from_pretrained(model_id, use_auth_token=token)
            if pipe is None:
                raise RuntimeError(
                    f"Could not load {model_id} — gated model: set GRANITE_HF_TOKEN "
                    "and accept the model conditions on huggingface.co"
                )
            pipe.to(torch.device(device))
            return pipe

        loop = asyncio.get_running_loop()
        self._pipeline = await loop.run_in_executor(None, _load)
        log.info("Diarization pipeline %s ready", model_id)

    async def unload(self) -> str | None:
        async with self._lock:
            if self._pipeline is None:
                return None
            log.info("Unloading diarization pipeline")
            self._pipeline = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            return settings.diarization_model

    # ----- idle-unload background loop (mirrors BackendRegistry) -----

    def start_idle_monitor(self) -> None:
        if settings.idle_unload_seconds <= 0 or self._idle_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._idle_task = loop.create_task(self._idle_loop(), name="diarizer-idle-unload")

    async def _idle_loop(self) -> None:
        ttl = settings.idle_unload_seconds
        interval = max(1, settings.idle_check_interval)
        try:
            while True:
                await asyncio.sleep(interval)
                if self._pipeline is None:
                    continue
                idle = time.monotonic() - self._last_used
                if idle < ttl:
                    continue
                log.info("Auto-unloading diarization pipeline after %.0fs idle", idle)
                await self.unload()
        except asyncio.CancelledError:
            raise

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._idle_task = None
        await self.unload()


def assign_speakers(words: list[TranscriptionWord], turns: list[Turn]) -> None:
    """Label each word with the speaker of the best-overlapping turn.

    Words and turns must be time-sorted. Words that overlap no turn (silence
    padding, boundary drift) get the nearest turn by midpoint distance.
    """
    if not turns:
        return
    lo = 0
    for w in words:
        # Advance past turns that end before this word starts.
        while lo + 1 < len(turns) and turns[lo].end <= w.start:
            lo += 1
        best: Turn | None = None
        best_overlap = 0.0
        j = lo
        while j < len(turns) and turns[j].start < w.end:
            overlap = min(turns[j].end, w.end) - max(turns[j].start, w.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = turns[j]
            j += 1
        if best is None:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: abs((t.start + t.end) / 2 - mid))
        w.speaker = best.speaker


diarizer = Diarizer()
