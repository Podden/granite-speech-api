"""Model-swap race in BackendRegistry: a request for another model must not
unload the backend while a request on the current one is still running."""

from __future__ import annotations

import asyncio
import importlib

from app.backends.catalog import COHERE_TRANSCRIBE, QWEN3_ASR

# `app.backends.registry` as an attribute is the instance, not the module.
registry_mod = importlib.import_module("app.backends.registry")


class _Fake:
    def __init__(self, dtype: str) -> None:
        self.model_id = ""
        self.unloaded = False

    async def load(self, model_id: str, device: str) -> None:
        self.model_id = model_id

    async def unload(self) -> None:
        self.unloaded = True
        self.model_id = ""


class _FakeA(_Fake):
    pass


class _FakeB(_Fake):
    pass


async def test_swap_waits_for_running_request(monkeypatch) -> None:
    monkeypatch.setattr(
        registry_mod,
        "_backend_class",
        lambda mid: _FakeA if mid == COHERE_TRANSCRIBE else _FakeB,
    )
    reg = registry_mod.BackendRegistry()

    first = await reg.acquire(model=COHERE_TRANSCRIBE, want_plus_features=False)
    second_task = asyncio.create_task(reg.acquire(model=QWEN3_ASR, want_plus_features=False))
    await asyncio.sleep(0.05)
    assert not second_task.done()
    assert not first.unloaded and first.model_id == COHERE_TRANSCRIBE

    # Same model may join while the first request still runs.
    same = await asyncio.wait_for(
        reg.acquire(model=COHERE_TRANSCRIBE, want_plus_features=False), timeout=1
    )
    assert same is first
    await reg.release()
    await asyncio.sleep(0.05)
    assert not second_task.done()

    await reg.release()
    second = await asyncio.wait_for(second_task, timeout=1)
    assert first.unloaded
    assert isinstance(second, _FakeB) and second.model_id == QWEN3_ASR
    await reg.release()
