"""VRAM broker lease: one lease for ASR model + diarizer + aligner, re-negotiated on every change, released when empty."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app import vram
from app.config import settings
from app.lazy import LazyModel


@pytest.fixture
def broker(monkeypatch):
    calls: list[tuple] = []
    answer = {"ok": True}

    async def lease(mb: int, wait_s: float) -> bool:
        calls.append(("lease", mb, wait_s))
        return answer["ok"]

    async def release() -> None:
        calls.append(("release",))

    monkeypatch.setattr(settings, "vram_broker_sock", "/x/broker.sock")
    monkeypatch.setattr(vram, "_lease", lease)
    monkeypatch.setattr(vram, "_release", release)
    vram._held.clear()
    yield calls, answer
    vram._held.clear()


async def test_lease_grows_and_shrinks_with_the_loaded_parts(broker) -> None:
    calls, _ = broker
    await vram.reserve("asr", 5500)
    await vram.reserve("aligner", 2200)
    await vram.drop("asr")
    await vram.drop("aligner")
    assert [c[:2] for c in calls] == [("lease", 5500), ("lease", 7700), ("lease", 2200), ("release",)]
    assert calls[2][2] == 0  # shrinking never waits


async def test_swap_replaces_the_size_of_a_part(broker) -> None:
    calls, _ = broker
    await vram.reserve("asr", 5500)
    await vram.reserve("asr", 2000)
    assert calls[-1][:2] == ("lease", 2000)


async def test_no_room_is_503_and_nothing_is_held(broker) -> None:
    _, answer = broker
    answer["ok"] = False
    with pytest.raises(HTTPException) as e:
        await vram.reserve("asr", 5500)
    assert e.value.status_code == 503 and e.value.headers["Retry-After"] == "30"
    assert vram._held == {}


async def test_disabled_without_socket(monkeypatch) -> None:
    monkeypatch.setattr(settings, "vram_broker_sock", "")
    await vram.reserve("asr", 5500)  # would try to reach a broker if it were on
    assert vram._held == {}


async def test_release_all_is_one_call(broker) -> None:
    calls, _ = broker
    await vram.reserve("asr", 5500)
    await vram.reserve("aligner", 2200)
    calls.clear()
    await vram.release_all()
    await vram.drop("asr")
    await vram.drop("aligner")
    assert calls == [("release",)]


class _Stage(LazyModel):
    name = "stage"
    vram_mb = 1234

    def _load(self):
        return object()


async def test_lazy_model_leases_while_loaded(broker) -> None:
    calls, _ = broker
    stage = _Stage()
    await stage.warm()
    assert vram._held == {"stage": 1234}
    await stage.unload()
    assert vram._held == {} and calls[-1] == ("release",)


def test_broker_unload_is_409_while_busy_and_frees_everything_when_idle(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app.backends import registry
    from app.main import app

    client = TestClient(app)
    monkeypatch.setattr(registry, "_active_requests", 1)
    assert client.post("/v1/models/unload", params={"if_idle": "true"}).status_code == 409
    monkeypatch.setattr(registry, "_active_requests", 0)
    r = client.post("/v1/models/unload", params={"if_idle": "true"})
    assert r.status_code == 200 and r.json()["loaded_model"] is None
