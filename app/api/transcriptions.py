from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.audio import get_duration
from app.backends import registry
from app.config import settings
from app.schema import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
)

log = logging.getLogger(__name__)
router = APIRouter()


def _parse_granularities(raw: str | None) -> tuple[bool, bool]:
    """Return (segment_timestamps, word_timestamps) flags.

    Accepts JSON array string, comma-separated string, or repeated form values
    that FastAPI joins via multiple `timestamp_granularities[]` entries.
    """
    if not raw:
        return True, False
    parts: list[str] = []
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parts = [str(x).lower() for x in json.loads(raw)]
        except Exception:
            parts = []
    if not parts:
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return ("segment" in parts) or not parts, "word" in parts


def _bool(v: str | bool | None) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@router.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: str | None = Form(default=None, alias="timestamp_granularities[]"),
    prompt: str | None = Form(default=None),
    stream: str | None = Form(default=None),
    translate: str | None = Form(default=None),
    translate_to: str | None = Form(default=None),
    speaker_attribution: str | None = Form(default=None),
    min_speakers: int | None = Form(default=None),
    max_speakers: int | None = Form(default=None),
    # WhisperX-compat aliases (currently no-op or mapped):
    diarize: str | None = Form(default=None),
    hf_token: str | None = Form(default=None),  # noqa: ARG001 — accepted for compat
    batch_size: int | None = Form(default=None),  # noqa: ARG001
    compute_type: str | None = Form(default=None),  # noqa: ARG001
) -> Any:
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        duration = get_duration(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {exc}") from exc

    if duration > settings.max_audio_seconds:
        raise HTTPException(
            status_code=413,
            detail=f"Audio too long: {duration:.1f}s > {settings.max_audio_seconds}s",
        )

    seg_ts, word_ts = _parse_granularities(timestamp_granularities)
    do_diarize = _bool(speaker_attribution) or _bool(diarize)
    do_translate = _bool(translate) or bool(translate_to)
    do_stream = _bool(stream)

    want_plus = word_ts or do_diarize

    req = TranscriptionRequest(
        audio_bytes=audio_bytes,
        filename=file.filename or "audio",
        model=model or settings.default_model,
        language=language,
        response_format=response_format,
        word_timestamps=word_ts,
        segment_timestamps=seg_ts,
        speaker_attribution=do_diarize,
        translate=do_translate,
        translate_to=translate_to,
        prompt=prompt,
        stream=do_stream,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    backend = await registry.get(model=req.model, want_plus_features=want_plus)

    if do_stream:
        return StreamingResponse(
            _ndjson_stream(backend, req, duration),
            media_type="application/x-ndjson",
        )

    segments, detected_lang = await backend.transcribe(req)
    text = _join_segments(segments).strip()

    fmt = response_format.lower()
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "srt":
        return PlainTextResponse(_to_srt(segments), media_type="text/plain")
    if fmt == "vtt":
        return PlainTextResponse(_to_vtt(segments), media_type="text/vtt")
    if fmt == "verbose_json":
        return TranscriptionResponse(
            task="translate" if do_translate else "transcribe",
            language=detected_lang or language,
            duration=duration,
            text=text,
            segments=segments,
            words=_collect_words(segments) if word_ts else None,
        )
    # default "json"
    return {"text": text}


# ----- helpers -----


def _join_segments(segments: list[TranscriptionSegment]) -> str:
    chunks: list[str] = []
    for s in segments:
        chunk = s.text.strip()
        if s.speaker:
            chunk = f"[{s.speaker}] {chunk}"
        chunks.append(chunk)
    return " ".join(chunks)


def _collect_words(segments: list[TranscriptionSegment]) -> list:
    out = []
    for s in segments:
        if s.words:
            out.extend(s.words)
    return out


def _format_ts(t: float, sep: str = ",") -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _to_srt(segments: list[TranscriptionSegment]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        text = seg.text.strip()
        if seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(str(i))
        lines.append(f"{_format_ts(seg.start)} --> {_format_ts(seg.end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _to_vtt(segments: list[TranscriptionSegment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = seg.text.strip()
        if seg.speaker:
            text = f"<v {seg.speaker}>{text}"
        lines.append(f"{_format_ts(seg.start, '.')} --> {_format_ts(seg.end, '.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


async def _ndjson_stream(backend, req: TranscriptionRequest, duration: float):
    yield json.dumps({"type": "duration", "duration": duration}) + "\n"
    async for event in backend.transcribe_stream(req):
        yield json.dumps(event) + "\n"
