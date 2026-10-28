"""Idle unload must free models stuck in reference cycles (NeMo): an idle
server allocates nothing, so the cyclic GC never reclaims their VRAM."""

from __future__ import annotations

import gc
import weakref

from app.lazy import LazyModel


class _CyclicModel:
    def __init__(self) -> None:
        self.me = self


class _Stage(LazyModel):
    def _load(self) -> _CyclicModel:
        return _CyclicModel()


async def test_unload_collects_cyclic_model() -> None:
    stage = _Stage()
    ref = await stage.run(weakref.ref)
    gc.disable()  # idle server: no allocations, so no automatic collection
    try:
        await stage.unload()
        assert ref() is None
    finally:
        gc.enable()


class _LeakyStage(LazyModel):
    """Simulates two interleaved from_pretrained() calls leaving bf16 as default."""

    def _load(self) -> object:
        import torch

        torch.set_default_dtype(torch.bfloat16)
        return object()


async def test_load_resets_leaked_default_dtype() -> None:
    import torch

    try:
        await _LeakyStage().warm()
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(torch.float32)
