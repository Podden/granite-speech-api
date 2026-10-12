"""Lease GPU memory at the cave2 VRAM broker before a model is loaded.

One lease ("granite") covers everything this process keeps on the GPU: the ASR model, a diarizer and the forced
aligner. It is re-negotiated whenever a part is added or dropped and renewed every minute. The broker evicts an idle
granite through ``POST /v1/models/unload?if_idle=true`` (409 while a job runs).

Fail-open: without ``GRANITE_VRAM_BROKER_SOCK`` (or with the broker down) nothing changes. The broker is only
reachable through its unix socket, so the compose file has to mount it into the container.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import HTTPException

from app.config import settings

log = logging.getLogger(__name__)

HOLDER = "granite"
RENEW_SECONDS = 60.0
_held: dict[str, int] = {}  # part -> MB
_lock = asyncio.Lock()
_renew_task: asyncio.Task[None] | None = None


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, transport=httpx.AsyncHTTPTransport(uds=settings.vram_broker_sock))


async def _lease(mb: int, wait_s: float) -> bool:
    """True = go ahead (granted, or broker unavailable). False = no room within wait_s."""
    body = dict(
        holder=HOLDER, gpu=settings.vram_gpu, mb=mb, priority=settings.vram_priority, ttl_s=180, wait_s=wait_s,
        unload_url=f"{settings.vram_public_url}/v1/models/unload?if_idle=true",
    )
    try:
        async with _client(wait_s + 20) as c:
            r = await c.post("http://broker/lease", json=body)
    except httpx.TransportError as e:
        log.warning("VRAM broker unreachable (%s): continuing without lease", e)
        return True
    if r.status_code == 409:
        return False
    if r.status_code != 200:
        log.warning("VRAM broker answered %s %s: continuing without lease", r.status_code, r.text[:200])
    return True


async def _release() -> None:
    try:
        async with _client(10) as c:
            await c.delete("http://broker/lease", params=dict(holder=HOLDER, gpu=settings.vram_gpu))
    except httpx.TransportError as e:
        log.warning("VRAM broker unreachable on release: %s", e)


def asr_mb(model_id: str) -> int:
    """Lease size of an ASR model; unknown ids get the default (measured on cave2, see docker.md)."""
    return settings.vram_asr_mb.get(model_id, settings.vram_asr_default_mb)


async def reserve(part: str, mb: int) -> None:
    """Make room for ``part`` before it is loaded. 503 (+ Retry-After) when the GPU stays busy for vram_wait_seconds."""
    if not settings.vram_broker_sock:
        return
    async with _lock:
        total = sum(_held.values()) - _held.get(part, 0) + mb
        if not await _lease(total, settings.vram_wait_seconds):
            raise HTTPException(503, f"no GPU memory for {part} within {settings.vram_wait_seconds:.0f}s (VRAM broker)",
                                headers={"Retry-After": "30"})
        _held[part] = mb


async def drop(part: str) -> None:
    """``part`` left the GPU: shrink the lease, release it when nothing is left."""
    if not settings.vram_broker_sock:
        return
    async with _lock:
        if _held.pop(part, None) is None:
            return
        if _held:
            await _lease(sum(_held.values()), 0)
        else:
            await _release()


async def release_all() -> None:
    """Give the whole lease back at once. Used when the broker itself asks us to unload: shrinking step by step would
    queue behind the broker's own eviction (it holds its admission lock while it waits for our answer)."""
    if not settings.vram_broker_sock:
        return
    async with _lock:
        if _held:
            _held.clear()
            await _release()


async def _renew_loop() -> None:
    while True:
        await asyncio.sleep(RENEW_SECONDS)
        async with _lock:
            if _held:
                await _lease(sum(_held.values()), 0)  # same size = renew


def start() -> None:
    global _renew_task
    if settings.vram_broker_sock and _renew_task is None:
        _renew_task = asyncio.get_running_loop().create_task(_renew_loop(), name="vram-renew")
        log.info("VRAM broker lease enabled (%s, priority %s)", settings.vram_broker_sock, settings.vram_priority)


async def shutdown() -> None:
    global _renew_task
    if _renew_task is not None:
        _renew_task.cancel()
        _renew_task = None
    if settings.vram_broker_sock and _held:
        _held.clear()
        await _release()
