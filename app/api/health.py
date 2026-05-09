from __future__ import annotations

from fastapi import APIRouter

from app.backends import registry
from app.backends.granite import GRANITE_BASE, GRANITE_NAR, GRANITE_PLUS
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    idle = registry.idle_seconds
    return {
        "status": "ok",
        "loaded_model": registry.loaded_model,
        "idle_seconds": None if idle is None else round(idle, 1),
        "idle_unload_seconds": settings.idle_unload_seconds,
        "device": settings.resolved_device(),
        "default_model": settings.default_model,
        "available_models": [GRANITE_BASE, GRANITE_PLUS, GRANITE_NAR],
    }
