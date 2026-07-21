from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.backends import registry
from app.backends.catalog import COHERE_TRANSCRIBE, FUSION, QWEN3_ASR
from app.backends.granite import GRANITE_BASE, GRANITE_NAR, GRANITE_PLUS
from app.config import settings
from app.schema import ModelInfo, ModelList

router = APIRouter()


@router.get("/v1/models", response_model=ModelList, summary="List available models")
async def list_models() -> ModelList:
    """List the model ids accepted by `/v1/audio/transcriptions`."""
    return ModelList(
        data=[
            ModelInfo(id=GRANITE_BASE),
            ModelInfo(id=GRANITE_PLUS),
            ModelInfo(id=GRANITE_NAR),
            ModelInfo(id=COHERE_TRANSCRIBE, owned_by="cohere"),
            ModelInfo(id=QWEN3_ASR, owned_by="qwen"),
            ModelInfo(id=FUSION, owned_by="pipeline"),
        ]
    )


class LoadRequest(BaseModel):
    path: str = Field(
        description="Granite model id to load (see `GET /v1/models`).",
        examples=["ibm-granite/granite-speech-4.1-2b-plus"],
    )
    device: str | None = Field(
        default=None,
        description="Reserved; the device is taken from server config (`GRANITE_DEVICE`).",
    )


@router.post("/v1/models/load", summary="Preload a model")
async def load_model(req: LoadRequest) -> dict:
    """Load a model into VRAM ahead of time (hot-swaps the currently loaded one)."""
    backend = await registry.get(model=req.path, want_plus_features=False)
    return {"loaded_model": backend.model_id, "device": settings.resolved_device()}


@router.post("/v1/models/unload", summary="Unload the current model")
async def unload_model() -> dict:
    """Free VRAM/RAM by unloading the currently loaded model."""
    prev = await registry.unload()
    return {"unloaded_model": prev, "loaded_model": None}
