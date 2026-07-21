from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from app.api import api_router
from app.backends import registry
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Raise Starlette's multipart part-size limit to match our max_upload_bytes.
# In Starlette 0.52.x `Request.form()` has its OWN keyword-only default
# `max_part_size=1 MB` which it passes explicitly to MultiPartParser. FastAPI 0.132
# calls `await request.form()` with no args, so this 1 MB default always wins and
# any multipart upload >1 MB throws MultiPartException -> 413, regardless of our
# ContentSizeLimitMiddleware. Patching MultiPartParser's default is a no-op because
# Request.form overrides it — so we patch Request.form's keyword default directly.
from app.config import settings as _settings  # noqa: E402 (circular-safe, settings is immutable)
if StarletteRequest.form.__kwdefaults__ is not None:
    StarletteRequest.form.__kwdefaults__["max_part_size"] = _settings.max_upload_bytes


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Lazy-load on first request — keeps cold start fast and avoids OOM
    # when no transcription is ever requested.
    registry.start_idle_monitor()
    yield
    await registry.shutdown()


app = FastAPI(
    title="granite-speech-api",
    version="0.1.0",
    description=(
        "OpenAI-compatible audio transcription API powered by IBM Granite "
        "Speech 4.1. Drop-in compatible with WhisperX-style word-level "
        "timestamps and speaker diarization (via the 2b-plus model)."
    ),
    lifespan=lifespan,
)

class ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > settings.max_upload_bytes:
                return Response(
                    content=f"Request body too large (max {settings.max_upload_bytes} bytes)",
                    status_code=413,
                )
        return await call_next(request)


app.add_middleware(ContentSizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Browser upload UI (served at /ui, / redirects there).
app.mount(
    "/ui",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="ui",
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


def run() -> None:
    """Console entrypoint."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    run()
