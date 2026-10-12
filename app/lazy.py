"""Lazy-loaded pipeline stage model with registry-style idle unload.

Base for the stages that live next to the ASR registry (diarizers, forced
aligner): loaded on first use, dropped again after the same idle TTL.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from app import vram
from app.config import settings

log = logging.getLogger(__name__)

T = TypeVar("T")


def free_gpu_memory() -> None:
    """Hand the VRAM of a just-dropped model back to the driver.

    Models sit in reference cycles (NeMo always); an idle server allocates
    nothing, so the cyclic GC never runs and ``empty_cache()`` alone frees
    nothing — the process keeps holding the VRAM with no model loaded.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class LazyModel:
    """Subclasses implement ``_load()`` (blocking, returns the model object)."""

    name: str = "model"
    vram_mb: int = 2500  # lease at the VRAM broker while loaded (subclasses override with their measured value)

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._last_used: float = 0.0
        self._idle_task: asyncio.Task[None] | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def _load(self) -> Any:  # pragma: no cover — heavy, model-specific
        raise NotImplementedError

    async def run(self, fn: Callable[[Any], T]) -> T:
        """Run blocking ``fn(model)`` in the executor, loading the model first."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self._model is None:
                log.info("Loading %s", self.name)
                await vram.reserve(self.name, self.vram_mb)
                try:
                    self._model = await loop.run_in_executor(None, self._load)
                except BaseException:
                    await vram.drop(self.name)
                    raise
                log.info("%s ready", self.name)
            self._last_used = time.monotonic()
            model = self._model
            try:
                return await loop.run_in_executor(None, fn, model)
            finally:
                self._last_used = time.monotonic()

    async def warm(self) -> None:
        """Load the model now (no-op when loaded) and reset its idle timer."""
        await self.run(lambda _model: None)

    async def unload(self) -> str | None:
        async with self._lock:
            if self._model is None:
                return None
            log.info("Unloading %s", self.name)
            self._model = None
            free_gpu_memory()
        await vram.drop(self.name)
        return self.name

    def start_idle_monitor(self) -> None:
        if settings.idle_unload_seconds <= 0 or self._idle_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._idle_task = loop.create_task(self._idle_loop(), name=f"{self.name}-idle-unload")

    async def _idle_loop(self) -> None:
        ttl = settings.idle_unload_seconds
        interval = max(1, settings.idle_check_interval)
        while True:
            await asyncio.sleep(interval)
            if self._model is None or self._lock.locked():
                continue
            idle = time.monotonic() - self._last_used
            if idle >= ttl:
                log.info("Auto-unloading %s after %.0fs idle", self.name, idle)
                await self.unload()

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._idle_task = None
        await self.unload()
