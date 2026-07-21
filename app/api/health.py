from __future__ import annotations

from fastapi import APIRouter

from app.backends import registry
from app.backends.granite import GRANITE_BASE, GRANITE_NAR, GRANITE_PLUS
from app.config import settings
from app.jobs import tracker

router = APIRouter()


@router.get("/health", summary="Health & model status")
async def health() -> dict:
    """Report service status, the currently loaded model and idle-unload state."""
    idle = registry.idle_seconds
    return {
        "status": "ok",
        "loaded_model": registry.loaded_model,
        "idle_seconds": None if idle is None else round(idle, 1),
        "idle_unload_seconds": settings.idle_unload_seconds,
        "device": settings.resolved_device(),
        "default_model": settings.default_model,
        "max_audio_seconds": settings.max_audio_seconds,
        "queue": tracker.snapshot(),
        "available_models": [GRANITE_BASE, GRANITE_PLUS, GRANITE_NAR],
    }
