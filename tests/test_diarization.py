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


def test_snap_speakers_to_sentences() -> None:
    from app.diarization import snap_speakers_to_sentences as _snap_speakers_to_sentences

    # One sentence with a single flickered word in the middle + a short
    # standalone interjection that must keep its own speaker.
    words = [
        _w("Das", 0.0, 0.2), _w("ist", 0.2, 0.4), _w("ein", 0.4, 0.6),
        _w("Test.", 0.6, 1.0), _w("Genau.", 1.1, 1.4),
    ]
    for w in words[:4]:
        w.speaker = "SPEAKER_00"
    words[2].speaker = "SPEAKER_01"  # flicker
    words[4].speaker = "SPEAKER_01"  # real interjection
    _snap_speakers_to_sentences(words)
    assert [w.speaker for w in words] == ["SPEAKER_00"] * 4 + ["SPEAKER_01"]


def test_fusion_is_a_preset() -> None:
    from app.backends.catalog import has_native_words, preset_aligner

    assert resolve_model_id("fusion", want_plus_features=False) == COHERE_TRANSCRIBE
    assert resolve_model_id("auto", want_plus_features=True) == COHERE_TRANSCRIBE
    assert preset_aligner("fusion") == "qwen"
    assert preset_aligner("cohere-transcribe") is None
    assert has_native_words(resolve_model_id(None, want_plus_features=True))
    assert not has_native_words(COHERE_TRANSCRIBE)
    assert not has_native_words(QWEN3_ASR)


def test_resolve_model_id_external_aliases() -> None:
    assert resolve_model_id("cohere-transcribe", want_plus_features=True) == COHERE_TRANSCRIBE
    assert resolve_model_id("qwen3-asr", want_plus_features=False) == QWEN3_ASR
    assert resolve_model_id("Qwen/Qwen3-ASR-1.7B", want_plus_features=False) == QWEN3_ASR
    assert not is_granite(COHERE_TRANSCRIBE)


def test_resolve_model_id_granite_fallback() -> None:
    assert resolve_model_id(None, want_plus_features=False).endswith("granite-speech-4.1-2b")
    assert resolve_model_id("granite-speech-4.1-2b", want_plus_features=True).endswith("-plus")
    assert is_granite(resolve_model_id("whisper-1", want_plus_features=False))


def test_granite_engine_rejected_for_other_transcribers(client: TestClient) -> None:
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _silence_wav(), "audio/wav")},
        data={
            "model": "cohere-transcribe",
            "speaker_attribution": "true",
            "diarization_engine": "granite",
        },
    )
    assert r.status_code == 400
    assert "2b-plus" in r.json()["detail"]


def _req(**kw):
    from app.schema import TranscriptionRequest

    base = dict(
        audio_bytes=b"x", filename="a", model="m", language="de",
        response_format="json", word_timestamps=False, segment_timestamps=True,
        speaker_attribution=False, translate=False, translate_to=None,
        prompt=None, stream=False, min_speakers=None, max_speakers=None,
    )
    base.update(kw)
    return TranscriptionRequest(**base)


@pytest.mark.parametrize(
    ("model", "speakers", "word_ts", "engine", "aligner", "want"),
    [
        # text-only transcriber + speakers → qwen aligner + external turns
        (COHERE_TRANSCRIBE, True, False, "nemotron", "auto", ("nemotron", "qwen")),
        # plus has native words → no aligner model needed
        ("ibm-granite/granite-speech-4.1-2b-plus", True, False, "pyannote", "auto",
         ("pyannote", "native")),
        # granite SAA: no external diarizer
        ("ibm-granite/granite-speech-4.1-2b-plus", True, False, "granite", "auto",
         (None, None)),
        # nothing requested → no stages
        (QWEN3_ASR, False, False, "auto", "auto", (None, None)),
        # word timestamps only → aligner
        (QWEN3_ASR, False, True, "auto", "auto", (None, "qwen")),
        # forced qwen on plus
        ("ibm-granite/granite-speech-4.1-2b-plus", False, False, "auto", "qwen",
         (None, "qwen")),
    ],
)
def test_plan_combinations(model, speakers, word_ts, engine, aligner, want) -> None:
    from app.pipeline import plan

    req = _req(speaker_attribution=speakers, word_timestamps=word_ts)
    p = plan(req, model, diarization_engine=engine, aligner=aligner)
    assert (p.diarizer, p.aligner) == want
    if p.diarizer:
        assert not req.speaker_attribution  # backend must not run its own SAA
    assert req.word_timestamps == (p.aligner == "native")
    assert (req.max_window_seconds is not None) == (p.aligner == "qwen")


@pytest.mark.parametrize(
    ("model", "kw"),
    [
        (COHERE_TRANSCRIBE, {"aligner": "native", "diarization_engine": "auto"}),
        (COHERE_TRANSCRIBE, {"aligner": "none", "diarization_engine": "pyannote"}),
    ],
)
def test_plan_rejects_impossible(model, kw) -> None:
    from fastapi import HTTPException

    from app.pipeline import plan

    with pytest.raises(HTTPException):
        plan(_req(speaker_attribution=True), model, **kw)


async def test_pipeline_diarize_falls_back_to_granite(monkeypatch) -> None:
    import app.pipeline as pl
    from fastapi import HTTPException

    async def _boom(*a, **kw):  # noqa: ANN002, ANN003
        raise RuntimeError("no token")

    monkeypatch.setattr(pl.diarizer, "diarize", _boom)
    plus = "ibm-granite/granite-speech-4.1-2b-plus"
    req = _req(speaker_attribution=True)
    p = pl.plan(req, plus, diarization_engine="auto", aligner="auto")
    assert await p.diarize() == "granite"
    assert req.speaker_attribution and p.turns is None and not p.needs_post

    p = pl.plan(_req(speaker_attribution=True), plus,
                diarization_engine="nemotron", aligner="auto")
    with pytest.raises(HTTPException):
        await p.diarize()


async def test_pipeline_post_aligns_and_labels(monkeypatch) -> None:
    import app.pipeline as pl
    from app.schema import TranscriptionSegment

    turns = [Turn(0.0, 1.0, "speaker_0"), Turn(1.0, 2.0, "speaker_1")]

    async def _fake_diarize(audio_bytes, engine, **kw):  # noqa: ANN003
        assert engine == "nemotron" and kw["num_speakers"] == 2
        return turns

    async def _fake_align(wav, segments, language):
        assert language == "de"
        return [_w("Hallo.", 0.1, 0.5), _w("Tschüss.", 1.2, 1.6)]

    monkeypatch.setattr(pl.diarizer, "diarize", _fake_diarize)
    monkeypatch.setattr(pl.forced_aligner, "align", _fake_align)
    monkeypatch.setattr(pl, "load_audio_bytes", lambda b: (None, 2.0))

    req = _req(speaker_attribution=True, num_speakers=2)
    p = pl.plan(req, COHERE_TRANSCRIBE, diarization_engine="nemotron", aligner="auto")
    assert await p.diarize() == "nemotron"
    segs = await p.post([TranscriptionSegment(id=0, start=0, end=2, text="Hallo. Tschüss.")], "de")
    assert [(s.speaker, s.text) for s in segs] == [
        ("speaker_0", "Hallo."), ("speaker_1", "Tschüss."),
    ]


def test_parse_nemo_segment() -> None:
    from app.diarization import _parse_nemo_segment

    assert _parse_nemo_segment("1.800 2.100 speaker_1") == Turn(1.8, 2.1, "speaker_1")
    assert _parse_nemo_segment("0.5, 1.5, speaker_0") == Turn(0.5, 1.5, "speaker_0")
    assert _parse_nemo_segment("garbage") is None
