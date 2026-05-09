from __future__ import annotations

from fastapi import APIRouter

from app.backends import registry
from app.backends.granite import GRANITE_BASE, GRANITE_NAR, GRANITE_PLUS
from app.config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    backend = registry._backend  # internal access fine here
    return {
        "status": "ok",
        "loaded_model": backend.model_id if backend else None,
        "device": settings.resolved_device(),
        "default_model": settings.default_model,
        "available_models": [GRANITE_BASE, GRANITE_PLUS, GRANITE_NAR],
    }
