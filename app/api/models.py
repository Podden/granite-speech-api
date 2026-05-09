from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.backends import registry
from app.backends.granite import GRANITE_BASE, GRANITE_NAR, GRANITE_PLUS
from app.config import settings
from app.schema import ModelInfo, ModelList

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(
        data=[
            ModelInfo(id=GRANITE_BASE),
            ModelInfo(id=GRANITE_PLUS),
            ModelInfo(id=GRANITE_NAR),
        ]
    )


class LoadRequest(BaseModel):
    path: str
    device: str | None = None


@router.post("/v1/models/load")
async def load_model(req: LoadRequest) -> dict:
    backend = await registry.get(model=req.path, want_plus_features=False)
    return {"loaded_model": backend.model_id, "device": settings.resolved_device()}


@router.post("/v1/models/unload")
async def unload_model() -> dict:
    prev = await registry.unload()
    return {"unloaded_model": prev, "loaded_model": None}
