"""Early upload: send the audio while the user still picks options.

`POST /v1/uploads` reads the file(s) and keeps them **in RAM only, for as long
as that request stays open**: the response is an NDJSON stream that announces
the `upload_id` and then pings. Closing the tab / aborting the request, or
`HOLD_IDLE_SECONDS` without use, drops the audio — nothing touches the disk.
`/v1/audio/transcriptions` takes `upload_id` instead of `file`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)
router = APIRouter()

HOLD_IDLE_SECONDS = 30 * 60
PING_SECONDS = 15  # also how fast a closed tab is noticed (disconnect shows on send)


@dataclass
class Held:
    parts: list[bytes]
    filename: str
    last_used: float = field(default_factory=time.monotonic)


_held: dict[str, Held] = {}


def take(upload_id: str) -> Held:
    """Audio of a held upload (stays held, idle timer reset) or HTTP 404."""
    held = _held.get(upload_id)
    if held is None:
        raise HTTPException(status_code=404, detail="Upload not held (anymore) — send the file again")
    held.last_used = time.monotonic()
    return held


async def _hold_stream(upload_id: str, held: Held):
    try:
        yield json.dumps({"type": "held", "upload_id": upload_id}) + "\n"
        while time.monotonic() - held.last_used < HOLD_IDLE_SECONDS:
            await asyncio.sleep(PING_SECONDS)
            yield json.dumps({"type": "ping"}) + "\n"
        yield json.dumps({"type": "expired"}) + "\n"
    finally:
        _held.pop(upload_id, None)


@router.post("/v1/uploads", summary="Hold audio in memory for a later transcription")
async def hold_upload(
    file: Annotated[list[UploadFile], File(description="Audio file(s), as for /v1/audio/transcriptions.")],
) -> StreamingResponse:
    parts = [await f.read() for f in file]
    if not parts or not all(parts):
        raise HTTPException(status_code=400, detail="Empty audio file")
    upload_id = secrets.token_urlsafe(16)
    held = Held(parts=parts, filename=file[0].filename or "audio")
    _held[upload_id] = held
    return StreamingResponse(_hold_stream(upload_id, held), media_type="application/x-ndjson")
