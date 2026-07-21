"""User feedback — append-only JSONL, no auth (internal office use)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter()


class FeedbackIn(BaseModel):
    rating: Literal["good", "bad"] | None = Field(
        default=None,
        description="Optional thumbs verdict (mainly for transcription results).",
    )
    category: Literal["transcription", "ui", "feature", "bug", "other"] = Field(
        default="other",
        description="What the feedback is about.",
    )
    comment: str | None = Field(
        default=None, max_length=4000,
        description="Free-text feedback (what was good/bad, what is missing).",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Client context for debugging: file/settings/timings plus the UI "
            "log and click trace for reproducible bug reports."
        ),
    )


@router.post("/v1/feedback", status_code=201, summary="Submit feedback")
async def submit_feedback(fb: FeedbackIn) -> dict:
    """Append a feedback entry to the JSONL file (`GRANITE_FEEDBACK_FILE`)."""
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fb.model_dump()}
    path = Path(settings.feedback_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "ok"}


@router.get("/v1/feedback", summary="List feedback")
async def list_feedback(
    limit: int = Query(default=50, ge=1, le=1000, description="Newest N entries."),
) -> list[dict]:
    """Return the newest feedback entries (newest last)."""
    path = Path(settings.feedback_file)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
