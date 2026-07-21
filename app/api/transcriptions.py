from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.audio import get_duration
from app.backends import registry
from app.backends.granite import normalize_model_id
from app.config import settings
from app.jobs import tracker
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


TranslateTarget = Literal[
    "english",
    "french",
    "german",
    "spanish",
    "portuguese",
    "japanese",
    "italian",
    "mandarin",
]


@router.post(
    "/v1/audio/transcriptions",
    summary="Transcribe audio",
    response_description=(
        "Transcription in the requested `response_format`; NDJSON event stream "
        "when `stream=true`."
    ),
)
async def transcribe(
    file: Annotated[
        UploadFile,
        File(description="Audio (or video) file to transcribe: wav, mp3, ogg/opus, flac, m4a, mp4, …"),
    ],
    model: Annotated[
        str | None,
        Form(
            description=(
                "Granite model id (see `GET /v1/models`). Defaults to the server's "
                "default model. Requests using `speaker_attribution` or word "
                "timestamps auto-upgrade to `granite-speech-4.1-2b-plus`."
            ),
        ),
    ] = None,
    language: Annotated[
        str | None,
        Form(
            description=(
                "ISO-639-1 hint of the spoken language (e.g. `de`, `en`). Echoed back "
                "in `verbose_json`; Granite does not auto-detect the language."
            ),
            examples=["de"],
        ),
    ] = None,
    response_format: Annotated[
        Literal["json", "text", "srt", "vtt", "verbose_json"],
        Form(description="Output format. `verbose_json` includes segments, timestamps and speakers."),
    ] = "json",
    timestamp_granularities: Annotated[
        str | None,
        Form(
            alias="timestamp_granularities[]",
            description=(
                "Timestamp detail: `segment` (default) and/or `word`. Accepts a JSON "
                'array (`["segment","word"]`) or comma-separated string. `word` '
                "auto-upgrades to the `-plus` model."
            ),
            examples=["segment,word"],
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        Form(
            description="Comma-separated keywords for biased ASR (improves recognition of names/jargon).",
            examples=["Kubernetes, GitOps, ArgoCD"],
        ),
    ] = None,
    stream: Annotated[
        bool,
        Form(description="Stream the result as NDJSON events (`duration`, `progress`, `segment`, `result`)."),
    ] = False,
    translate: Annotated[
        bool,
        Form(description="Translate speech instead of transcribing (AST, `-2b` model only)."),
    ] = False,
    translate_to: Annotated[
        TranslateTarget | None,
        Form(description="Target language for speech translation. Implies `translate=true`."),
    ] = None,
    speaker_attribution: Annotated[
        bool,
        Form(
            description=(
                "Label segments with `[Speaker N]` (speaker diarization). "
                "Auto-upgrades to the `-plus` model."
            ),
        ),
    ] = False,
    min_speakers: Annotated[
        int | None,
        Form(ge=1, description="Advisory lower bound on the number of speakers (currently reserved)."),
    ] = None,
    max_speakers: Annotated[
        int | None,
        Form(ge=1, description="Advisory upper bound on the number of speakers (currently reserved)."),
    ] = None,
    # WhisperX-compat aliases (currently no-op or mapped):
    diarize: Annotated[
        bool,
        Form(description="WhisperX-compat alias for `speaker_attribution`."),
    ] = False,
    hf_token: Annotated[  # noqa: ARG001 — accepted for compat
        str | None,
        Form(description="WhisperX-compat, ignored (no HuggingFace token needed for diarization)."),
    ] = None,
    batch_size: Annotated[  # noqa: ARG001
        int | None,
        Form(description="WhisperX-compat, ignored."),
    ] = None,
    compute_type: Annotated[  # noqa: ARG001
        str | None,
        Form(description="WhisperX-compat, ignored."),
    ] = None,
) -> Any:
    """OpenAI-compatible transcription endpoint.

    Multipart form; everything except `file` is optional. Extras over OpenAI:
    speaker attribution, word timestamps, speech translation and NDJSON streaming.
    """
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
    do_diarize = speaker_attribution or diarize
    do_translate = translate or bool(translate_to)
    do_stream = stream

    # The base 2b model tends to translate non-English speech to English
    # instead of transcribing ("wrong-language hallucination"); the -plus
    # model transcribes the spoken language faithfully. AST still needs base.
    non_english = bool(language) and language.strip().lower() not in {"en", "english"}
    want_plus = word_ts or do_diarize or (non_english and not do_translate)

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

    if do_stream:
        # Model acquisition happens inside the stream so the client sees
        # loading status (cold start after idle-unload can take a while).
        return StreamingResponse(
            _ndjson_stream(req, duration, want_plus),
            media_type="application/x-ndjson",
        )

    job_id = tracker.enter(duration)
    backend = await registry.acquire(model=req.model, want_plus_features=want_plus)
    try:
        segments, detected_lang = await backend.transcribe(req)
    finally:
        await registry.release()
        tracker.exit(job_id)
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


async def _ndjson_stream(req: TranscriptionRequest, duration: float, want_plus: bool):
    backend = None
    job_id = tracker.enter(duration)
    try:
        yield json.dumps({"type": "duration", "duration": duration}) + "\n"

        target = normalize_model_id(req.model, want_plus_features=want_plus)
        loaded = registry.loaded_model
        if loaded != target:
            yield json.dumps({
                "type": "status",
                "stage": "loading_model",
                "model": target,
                "cold": loaded is None,
            }) + "\n"
        backend = await registry.acquire(model=req.model, want_plus_features=want_plus)
        yield json.dumps({
            "type": "status", "stage": "model_ready", "model": backend.model_id,
        }) + "\n"

        async for event in backend.transcribe_stream(req):
            yield json.dumps(event) + "\n"
    except Exception as exc:  # noqa: BLE001 — surface to the client, stream is already 200
        log.exception("Streaming transcription failed")
        yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
    finally:
        if backend is not None:
            await registry.release()
        tracker.exit(job_id)
