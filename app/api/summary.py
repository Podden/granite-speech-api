"""Transcript summarization via a self-hosted Ollama instance.

Proxies Ollama's /api/chat streaming endpoint as NDJSON delta events so the
browser UI can stream the summary token-by-token. Uses urllib (stdlib) in an
executor thread instead of adding an async HTTP client dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.request
from collections.abc import AsyncIterator, Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein Assistent, der Gesprächs-Transkripte zusammenfasst. "
    "Fasse das Transkript strukturiert zusammen: zuerst ein Überblick in "
    "3-5 Sätzen, dann die wichtigsten Punkte als Stichpunkte, zuletzt offene "
    "Aufgaben und Entscheidungen, falls vorhanden. Antworte auf Deutsch."
)

# Embedding/reranker models are not chat-capable; vision families waste VRAM
# on a text-only task.
_EXCLUDE_NAME_RE = re.compile(
    r"embed|rerank|minilm|sentence-transformers|coder", re.IGNORECASE
)

# Summaries need a capable general-purpose model — hide everything below this.
_MIN_PARAMS_B = 20.0


class SummaryIn(BaseModel):
    text: str = Field(min_length=1, description="Transcript text to summarize.")
    model: str = Field(
        min_length=1,
        description="Ollama model name (see `GET /v1/summary/models`).",
    )
    system_prompt: str | None = Field(
        default=None,
        max_length=8000,
        description="Custom system prompt; server default (German) when empty.",
    )


def _is_text_model(m: dict) -> bool:
    details = m.get("details") or {}
    families = {str(details.get("family") or "")}
    families |= {str(f) for f in details.get("families") or []}
    families = {f.lower() for f in families}
    if any("bert" in f or "vl" in f or "clip" in f for f in families):
        return False
    return not _EXCLUDE_NAME_RE.search(m.get("name", ""))


def _param_billions(m: dict) -> float | None:
    ps = str((m.get("details") or {}).get("parameter_size") or "")
    match = re.match(r"([\d.]+)\s*([BM])", ps, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000 if match.group(2).upper() == "M" else value


def _big_enough(m: dict) -> bool:
    billions = _param_billions(m)
    if billions is not None:
        return billions >= _MIN_PARAMS_B
    # No parameter_size in the tag — approximate via blob size (q4-ish 20B).
    return (m.get("size") or 0) >= 12_000_000_000


def _filter_models(tags: dict) -> list[dict]:
    """Ollama /api/tags payload -> capable text models (>=20B), largest first."""
    models = [
        m for m in tags.get("models", []) if _is_text_model(m) and _big_enough(m)
    ]
    models.sort(key=lambda m: -(m.get("size") or 0))
    return [
        {
            "name": m["name"],
            "parameter_size": (m.get("details") or {}).get("parameter_size"),
            "size_bytes": m.get("size"),
        }
        for m in models
    ]


def _fetch_tags() -> dict:
    with urllib.request.urlopen(f"{settings.ollama_url}/api/tags", timeout=10) as resp:
        return json.load(resp)


def _open_chat_stream(payload: dict) -> Iterator[bytes]:
    """Blocking: POST to Ollama /api/chat, yield raw NDJSON response lines."""
    req = urllib.request.Request(
        f"{settings.ollama_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        yield from resp


@router.get(
    "/v1/summary/models",
    summary="List Ollama models usable for summarization",
    response_description="Chat-capable text models, largest first.",
)
async def list_summary_models() -> dict:
    loop = asyncio.get_running_loop()
    try:
        tags = await loop.run_in_executor(None, _fetch_tags)
    except Exception as exc:  # noqa: BLE001 — surface upstream failure as 502
        raise HTTPException(
            status_code=502, detail=f"Ollama nicht erreichbar ({settings.ollama_url}): {exc}"
        ) from exc
    models = _filter_models(tags)
    return {"models": models, "default_system_prompt": DEFAULT_SYSTEM_PROMPT}


@router.post(
    "/v1/summary",
    summary="Summarize a transcript (streaming)",
    response_description=(
        "NDJSON stream: `delta` events with text pieces, then `done` "
        "(or `error`)."
    ),
)
async def summarize(body: SummaryIn) -> StreamingResponse:
    system = (body.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    payload = {
        "model": body.model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": body.text},
        ],
    }
    return StreamingResponse(_ndjson_events(payload), media_type="application/x-ndjson")


async def _ndjson_events(payload: dict) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    _done = object()

    def _worker() -> None:
        try:
            for raw in _open_chat_stream(payload):
                loop.call_soon_threadsafe(queue.put_nowait, raw)
        except Exception as exc:  # noqa: BLE001 — forwarded as error event
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _done)

    loop.run_in_executor(None, _worker)
    log.info("Summary via Ollama model %s (%d chars input)",
             payload["model"], len(payload["messages"][1]["content"]))

    while True:
        item = await queue.get()
        if item is _done:
            break
        if isinstance(item, Exception):
            log.warning("Ollama summary stream failed: %s", item)
            yield json.dumps({"type": "error", "message": str(item)}) + "\n"
            continue
        try:
            chunk = json.loads(item)
        except ValueError:
            continue
        if chunk.get("error"):
            yield json.dumps({"type": "error", "message": chunk["error"]}) + "\n"
            continue
        msg = chunk.get("message") or {}
        # Reasoning models stream their chain-of-thought separately — forward
        # it as `thinking` so clients can show activity before the answer.
        if msg.get("thinking"):
            yield json.dumps({"type": "thinking", "text": msg["thinking"]}) + "\n"
        if msg.get("content"):
            yield json.dumps({"type": "delta", "text": msg["content"]}) + "\n"
        if chunk.get("done"):
            yield json.dumps({
                "type": "done",
                "model": chunk.get("model"),
                "eval_count": chunk.get("eval_count"),
                "duration_seconds": round((chunk.get("total_duration") or 0) / 1e9, 1),
            }) + "\n"
