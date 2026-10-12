from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import vram
from app.aligner import aligner
from app.backends import registry
from app.backends.catalog import COHERE_TRANSCRIBE, FUSION, QWEN3_ASR
from app.backends.granite import GRANITE_BASE, GRANITE_NAR, GRANITE_PLUS
from app.backends.turboctc import GRANITE_TURBOCTC
from app.config import settings
from app.diarization import diarizer
from app.jobs import tracker
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
            ModelInfo(id=GRANITE_TURBOCTC),
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
async def unload_model(if_idle: bool = False) -> dict:
    """Free VRAM/RAM: unload the ASR model, the diarizers and the forced aligner.

    With `if_idle=true` (used by the VRAM broker to make room) nothing is touched and 409 is returned while a job
    runs or a model is loading.
    """
    if if_idle:
        if tracker.snapshot()["active_jobs"] > 0 or registry.busy or diarizer.busy or aligner.busy:
            raise HTTPException(409, "granite-speech-api is busy")
        await vram.release_all()  # before the unloads: the broker waits for this answer while holding its admission lock
    prev = await registry.unload()
    others = await diarizer.unload() + [n for n in [await aligner.unload()] if n]
    return {"unloaded_model": prev, "unloaded_stages": others, "loaded_model": None}
