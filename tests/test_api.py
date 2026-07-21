"""Smoke tests — do not require the actual Granite model weights."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _silence_wav(seconds: float = 1.0, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(seconds * sr)
        w.writeframes(np.zeros(n, dtype=np.int16).tobytes())
    return buf.getvalue()


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "ibm-granite/granite-speech-4.1-2b" in body["available_models"]


def test_models_list(client: TestClient) -> None:
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "ibm-granite/granite-speech-4.1-2b" in ids
    assert "ibm-granite/granite-speech-4.1-2b-plus" in ids


def test_normalize_model_id_auto_upgrade() -> None:
    from app.backends.granite import GRANITE_BASE, GRANITE_PLUS, normalize_model_id

    assert normalize_model_id(None, want_plus_features=False) == GRANITE_BASE
    assert normalize_model_id("whisper-1", want_plus_features=False) == GRANITE_BASE
    assert normalize_model_id("gpt-4o-transcribe", want_plus_features=False) == GRANITE_BASE
    assert normalize_model_id(None, want_plus_features=True) == GRANITE_PLUS
    assert normalize_model_id("whisper-1", want_plus_features=True) == GRANITE_PLUS
    assert normalize_model_id("granite-speech-4.1-2b", want_plus_features=True) == GRANITE_PLUS
    assert (
        normalize_model_id("ibm-granite/granite-speech-4.1-2b-plus", want_plus_features=False)
        == GRANITE_PLUS
    )


def test_parse_word_timestamps() -> None:
    from app.backends.granite import _parse_word_timestamps

    out = _parse_word_timestamps("hello [T:45] world [T:82]")
    assert len(out) == 1
    seg = out[0]
    assert seg.words is not None and len(seg.words) == 2
    assert seg.words[0].word == "hello"
    assert seg.words[0].end == pytest.approx(0.45)
    assert seg.words[1].end == pytest.approx(0.82)


def test_parse_word_timestamps_rollover() -> None:
    from app.backends.granite import _parse_word_timestamps

    # Tags wrap modulo 1000 cs (=10s). After "a"@8.50s a "[T:30]" must
    # decode to 10.30s (added rollover), not 0.30s.
    out = _parse_word_timestamps("a [T:850] b [T:30]")
    assert out[0].words[0].end == pytest.approx(8.5)
    assert out[0].words[1].end == pytest.approx(10.3)


def test_parse_speaker_segments() -> None:
    from app.backends.granite import _parse_speaker_segments

    txt = "[Speaker 1]: hello there [Speaker 2]: hi back"
    segs = _parse_speaker_segments(txt, duration=2.0)
    assert len(segs) == 2
    assert segs[0].speaker == "SPEAKER_00"
    assert segs[1].speaker == "SPEAKER_01"


def test_merge_words_and_speakers() -> None:
    from app.backends.granite import _merge_words_and_speakers

    ts = "hello [T:50] world [T:100] hi [T:150]"
    spk = "[Speaker 1]: hello world [Speaker 2]: hi"
    segs = _merge_words_and_speakers(ts, spk)
    # Expect 2 contiguous speaker segments.
    speakers = [s.speaker for s in segs]
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert segs[0].words[0].word == "hello"
    assert segs[1].words[0].word == "hi"


def test_transcriptions_rejects_empty(client: TestClient) -> None:
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400


def test_feedback_roundtrip(client: TestClient, tmp_path, monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "feedback_file", str(tmp_path / "fb.jsonl"))
    r = client.post(
        "/v1/feedback",
        json={"rating": "good", "comment": "top", "context": {"filename": "a.mp3"}},
    )
    assert r.status_code == 201
    r = client.post("/v1/feedback", json={"rating": "bad"})
    assert r.status_code == 201

    r = client.get("/v1/feedback")
    assert r.status_code == 200
    entries = r.json()
    assert len(entries) == 2
    assert entries[0]["rating"] == "good"
    assert entries[0]["comment"] == "top"
    assert entries[1]["rating"] == "bad"
    assert "ts" in entries[0]

    r = client.get("/v1/feedback", params={"limit": 1})
    assert [e["rating"] for e in r.json()] == ["bad"]


def test_feedback_rejects_invalid_rating(client: TestClient) -> None:
    r = client.post("/v1/feedback", json={"rating": "meh"})
    assert r.status_code == 422


def test_feedback_without_rating(client: TestClient, tmp_path, monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "feedback_file", str(tmp_path / "fb.jsonl"))
    r = client.post(
        "/v1/feedback",
        json={"category": "feature", "comment": "Dark-Mode-Toggle bitte"},
    )
    assert r.status_code == 201
    entry = client.get("/v1/feedback").json()[0]
    assert entry["rating"] is None
    assert entry["category"] == "feature"
