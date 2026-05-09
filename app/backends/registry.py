"""Hot-swap registry — keeps a single Granite backend alive at a time.

Dispatches to the AR backend (granite-speech-4.1-2b / -plus) or NAR backend
(granite-speech-4.1-2b-nar) based on the requested model id.
"""

from __future__ import annotations

import asyncio
import logging

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
            return self._backend

    async def shutdown(self) -> None:
        if self._backend is not None:
            await self._backend.unload()
            self._backend = None


registry = BackendRegistry()
