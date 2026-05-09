"""Hot-swap registry — keeps a single Granite backend alive at a time."""

from __future__ import annotations

import asyncio
import logging

from app.backends.base import ASRBackend
from app.backends.granite import GraniteBackend, normalize_model_id
from app.config import settings

log = logging.getLogger(__name__)


class BackendRegistry:
    def __init__(self) -> None:
        self._backend: ASRBackend | None = None
        self._lock = asyncio.Lock()

    async def get(self, *, model: str | None, want_plus_features: bool) -> ASRBackend:
        target = normalize_model_id(model, want_plus_features=want_plus_features)
        async with self._lock:
            if self._backend is None:
                self._backend = GraniteBackend(dtype=settings.dtype)
            if self._backend.model_id != target:
                await self._backend.load(target, settings.resolved_device())
            return self._backend

    async def preload_default(self) -> None:
        try:
            await self.get(model=settings.default_model, want_plus_features=False)
        except Exception:  # noqa: BLE001
            log.exception("Failed to preload default model — continuing without it.")

    async def shutdown(self) -> None:
        if self._backend is not None:
            await self._backend.unload()
            self._backend = None


registry = BackendRegistry()
