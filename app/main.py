from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.backends import registry
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


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
