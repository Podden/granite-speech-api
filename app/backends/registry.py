"""Hot-swap registry — keeps a single Granite backend alive at a time.

Dispatches to the AR backend (granite-speech-4.1-2b / -plus) or NAR backend
(granite-speech-4.1-2b-nar) based on the requested model id. Optionally
auto-unloads the model after a configurable idle TTL.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.backends.base import ASRBackend
from app.backends.granite import GraniteBackend, normalize_model_id
from app.backends.granite_nar import GRANITE_NAR_ID, GraniteNARBackend
from app.config import settings

log = logging.getLogger(__name__)


def _is_nar(model_id: str) -> bool:
    return model_id == GRANITE_NAR_ID


class BackendRegistry:
    def __init__(self) -> None:
        self._backend: ASRBackend | None = None
        self._lock = asyncio.Lock()
        self._last_used: float = 0.0
        self._idle_task: asyncio.Task[None] | None = None

    @property
    def loaded_model(self) -> str | None:
        return self._backend.model_id if self._backend else None

    @property
    def idle_seconds(self) -> float | None:
        if self._backend is None:
            return None
        return max(0.0, time.monotonic() - self._last_used)

    async def get(self, *, model: str | None, want_plus_features: bool) -> ASRBackend:
        target = normalize_model_id(model, want_plus_features=want_plus_features)
        async with self._lock:
            wants_nar = _is_nar(target)
            current_is_nar = isinstance(self._backend, GraniteNARBackend)
            if self._backend is not None and wants_nar != current_is_nar:
                await self._backend.unload()
                self._backend = None

            if self._backend is None:
                self._backend = (
                    GraniteNARBackend(dtype=settings.dtype)
                    if wants_nar
                    else GraniteBackend(dtype=settings.dtype)
                )

            if self._backend.model_id != target:
                await self._backend.load(target, settings.resolved_device())
            self._last_used = time.monotonic()
            return self._backend

    def touch(self) -> None:
        """Mark the model as just-used (call after a successful inference)."""
        if self._backend is not None:
            self._last_used = time.monotonic()

    async def unload(self) -> str | None:
        """Force unload, returning the previously loaded model id (if any)."""
        async with self._lock:
            if self._backend is None:
                return None
            prev = self._backend.model_id
            await self._backend.unload()
            self._backend = None
            return prev

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._idle_task = None
        await self.unload()

    # ----- idle-unload background loop -----

    def start_idle_monitor(self) -> None:
        if settings.idle_unload_seconds <= 0:
            log.info("Idle auto-unload disabled (GRANITE_IDLE_UNLOAD_SECONDS<=0)")
            return
        if self._idle_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._idle_task = loop.create_task(self._idle_loop(), name="idle-unload")
        log.info(
            "Idle auto-unload enabled: ttl=%ss, check every %ss",
            settings.idle_unload_seconds,
            settings.idle_check_interval,
        )

    async def _idle_loop(self) -> None:
        ttl = settings.idle_unload_seconds
        interval = max(1, settings.idle_check_interval)
        try:
            while True:
                await asyncio.sleep(interval)
                idle = self.idle_seconds
                if idle is None:
                    continue
                if idle >= ttl:
                    log.info(
                        "Auto-unloading %s after %.0fs idle (ttl=%ss)",
                        self.loaded_model, idle, ttl,
                    )
                    await self.unload()
        except asyncio.CancelledError:
            raise


registry = BackendRegistry()
