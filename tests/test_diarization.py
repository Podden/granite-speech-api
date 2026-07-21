"""Tests for the pyannote diarization stage and multi-family model catalog."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backends.catalog import (
    COHERE_TRANSCRIBE,
    QWEN3_ASR,
    is_granite,
    resolve_model_id,
)
from app.diarization import Turn, assign_speakers
from app.schema import TranscriptionWord


def _silence_wav(seconds: float = 1.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(int(seconds * sr), dtype=np.int16).tobytes())
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def _w(word: str, start: float, end: float) -> TranscriptionWord:
    return TranscriptionWord(word=word, start=start, end=end, probability=1.0)


def test_assign_speakers_by_overlap() -> None:
    words = [_w("hallo", 0.0, 0.5), _w("welt", 0.6, 1.0), _w("moin", 2.1, 2.6)]
    turns = [Turn(0.0, 1.1, "SPEAKER_00"), Turn(2.0, 3.0, "SPEAKER_01")]
    assign_speakers(words, turns)
    assert [w.speaker for w in words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]


def test_assign_speakers_nearest_when_no_overlap() -> None:
    # Word sits in a gap between turns — nearest midpoint wins.
    words = [_w("äh", 1.4, 1.5)]
    turns = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(1.6, 3.0, "SPEAKER_01")]
    assign_speakers(words, turns)
    assert words[0].speaker == "SPEAKER_01"


def test_assign_speakers_prefers_larger_overlap() -> None:
    words = [_w("überlapp", 0.8, 1.4)]
    turns = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(1.0, 2.0, "SPEAKER_01")]
    assign_speakers(words, turns)
    assert words[0].speaker == "SPEAKER_01"  # 0.4s overlap beats 0.2s


def test_speaker_runs_to_segments() -> None:
    from app.backends.granite import _speaker_runs_to_segments

    words = [_w("a", 0.0, 0.2), _w("b", 0.3, 0.5), _w("c", 1.0, 1.2)]
    words[0].speaker = "SPEAKER_00"
    words[1].speaker = "SPEAKER_00"
    words[2].speaker = "SPEAKER_01"
    segs = _speaker_runs_to_segments(words)
    assert len(segs) == 2
    assert segs[0].text == "a b" and segs[0].speaker == "SPEAKER_00"
    assert segs[1].text == "c" and segs[1].speaker == "SPEAKER_01"
    assert segs[0].words and len(segs[0].words) == 2


def test_fusion_resolution_and_capabilities() -> None:
    from app.backends.catalog import FUSION, supports_diarization

    assert resolve_model_id("fusion", want_plus_features=False) == FUSION
    assert resolve_model_id("auto", want_plus_features=True) == FUSION
    assert supports_diarization(FUSION)
    assert supports_diarization(resolve_model_id(None, want_plus_features=True))
    assert not supports_diarization(COHERE_TRANSCRIBE)
    assert not supports_diarization(QWEN3_ASR)


def test_resolve_model_id_external_aliases() -> None:
    assert resolve_model_id("cohere-transcribe", want_plus_features=True) == COHERE_TRANSCRIBE
    assert resolve_model_id("qwen3-asr", want_plus_features=False) == QWEN3_ASR
    assert resolve_model_id("Qwen/Qwen3-ASR-1.7B", want_plus_features=False) == QWEN3_ASR
    assert not is_granite(COHERE_TRANSCRIBE)


def test_resolve_model_id_granite_fallback() -> None:
    assert resolve_model_id(None, want_plus_features=False).endswith("granite-speech-4.1-2b")
    assert resolve_model_id("granite-speech-4.1-2b", want_plus_features=True).endswith("-plus")
    assert is_granite(resolve_model_id("whisper-1", want_plus_features=False))


def test_diarize_rejected_for_models_without_word_timestamps(client: TestClient) -> None:
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _silence_wav(), "audio/wav")},
        data={"model": "cohere-transcribe", "speaker_attribution": "true"},
    )
    assert r.status_code == 400
    assert "speaker attribution" in r.json()["detail"]


async def test_maybe_diarize_auto_falls_back(monkeypatch) -> None:
    import app.api.transcriptions as tr
    from app.schema import TranscriptionRequest

    async def _boom(*a, **kw):  # noqa: ANN002, ANN003
        raise RuntimeError("no token")

    monkeypatch.setattr(tr.diarizer, "diarize", _boom)
    req = TranscriptionRequest(
        audio_bytes=b"x", filename="a", model="m", language=None,
        response_format="json", word_timestamps=False, segment_timestamps=True,
        speaker_attribution=True, translate=False, translate_to=None,
        prompt=None, stream=False, min_speakers=None, max_speakers=None,
    )
    assert await tr._maybe_diarize(req, "auto") == "granite"
    assert req.diarization_turns is None

    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        await tr._maybe_diarize(req, "pyannote")


async def test_maybe_diarize_attaches_turns(monkeypatch) -> None:
    import app.api.transcriptions as tr
    from app.schema import TranscriptionRequest

    turns = [Turn(0.0, 1.0, "SPEAKER_00")]

    async def _fake(audio_bytes, num_speakers=None, min_speakers=None, max_speakers=None):
        assert num_speakers == 3
        return turns

    monkeypatch.setattr(tr.diarizer, "diarize", _fake)
    req = TranscriptionRequest(
        audio_bytes=b"x", filename="a", model="m", language=None,
        response_format="json", word_timestamps=False, segment_timestamps=True,
        speaker_attribution=True, translate=False, translate_to=None,
        prompt=None, stream=False, min_speakers=None, max_speakers=None,
        num_speakers=3,
    )
    assert await tr._maybe_diarize(req, "auto") == "pyannote"
    assert req.diarization_turns == turns
