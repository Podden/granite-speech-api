"""Unit tests for chunking/parsing helpers in app.backends.granite and the
default ASRBackend.transcribe_stream wrapper in app.backends.base.

Smoke-style: no model weights are downloaded/loaded.
"""

from __future__ import annotations

import torch

import pytest


# ----- _plan_windows -----


def test_plan_windows_short_returns_single_window() -> None:
    from app.backends.granite import _plan_windows

    wav = torch.zeros((1, int(5.0 * 16000)))
    windows = _plan_windows(wav, duration=5.0, limit=10.0, target=8.0)
    assert windows == [(0.0, 5.0)]


def test_plan_windows_at_exact_limit_returns_single_window() -> None:
    from app.backends.granite import _plan_windows

    wav = torch.zeros((1, int(10.0 * 16000)))
    windows = _plan_windows(wav, duration=10.0, limit=10.0, target=8.0)
    assert windows == [(0.0, 10.0)]


def test_plan_windows_splits_and_snaps_to_quiet_points() -> None:
    from app.backends.granite import _plan_windows
    from app.audio import TARGET_SR

    duration = 60.0
    n_samples = int(duration * TARGET_SR)
    torch.manual_seed(0)
    wav = torch.rand((1, n_samples)) * 0.5 + 0.5  # loud noise everywhere

    # Nominal boundaries land at t=20 and t=40 (n=ceil(60/20)=3, step=20).
    # Carve quiet (near-zero) patches near each nominal boundary, within the
    # +-10s search radius used by _plan_windows/_quiet_point.
    def carve_quiet(center: float, width: float = 0.5) -> None:
        lo = int((center - width / 2) * TARGET_SR)
        hi = int((center + width / 2) * TARGET_SR)
        wav[0, lo:hi] = 0.0

    carve_quiet(15.0)
    carve_quiet(45.0)

    windows = _plan_windows(wav, duration=duration, limit=20.0, target=20.0)

    assert len(windows) == 3
    assert windows[0][0] == 0.0
    assert windows[-1][1] == duration
    # Boundaries should have snapped away from the naive 20/40 split points,
    # towards the carved-out quiet regions.
    b1 = windows[0][1]
    b2 = windows[1][1]
    assert b1 == pytest.approx(15.0, abs=0.5)
    assert b2 == pytest.approx(45.0, abs=0.5)
    assert windows[1][0] == b1
    assert windows[2][0] == b2


def test_plan_windows_drops_degenerate_tiny_windows() -> None:
    from app.backends.granite import _plan_windows

    # duration just barely above limit: n = ceil(20.0001/20) = 2, step ~ 10.0.
    wav = torch.zeros((1, int(20.0001 * 16000)))
    windows = _plan_windows(wav, duration=20.0001, limit=20.0, target=20.0)
    # Every window must be non-degenerate (> 0.05s).
    for a, b in windows:
        assert b - a > 0.05


# ----- _quiet_point -----


def test_quiet_point_returns_t_unchanged_when_window_too_small() -> None:
    from app.backends.granite import _quiet_point

    # Very short tensor: (hi - lo) // frame < 2 regardless of radius.
    wav = torch.rand((1, 100))
    result = _quiet_point(wav, t=0.0, radius=10.0)
    assert result == 0.0


def test_quiet_point_finds_low_energy_region() -> None:
    from app.backends.granite import _quiet_point
    from app.audio import TARGET_SR

    duration = 10.0
    n_samples = int(duration * TARGET_SR)
    torch.manual_seed(1)
    wav = torch.rand((1, n_samples)) * 0.5 + 0.5  # loud everywhere

    quiet_center = 5.0
    width = 0.5
    lo = int((quiet_center - width / 2) * TARGET_SR)
    hi = int((quiet_center + width / 2) * TARGET_SR)
    wav[0, lo:hi] = 0.0

    result = _quiet_point(wav, t=5.0, radius=3.0)
    assert result == pytest.approx(quiet_center, abs=0.2)


# ----- _collapse_repeats -----


def test_collapse_repeats_leaves_normal_text_untouched() -> None:
    from app.backends.granite import _collapse_repeats

    text = "the quick brown fox jumps over the lazy dog"
    assert _collapse_repeats(text) == text


def test_collapse_repeats_leaves_short_runs_untouched() -> None:
    from app.backends.granite import _collapse_repeats

    # Exactly max_run (4) repeats — must NOT be collapsed.
    text = "word word word word"
    assert _collapse_repeats(text, max_run=4) == text


def test_collapse_repeats_collapses_long_1gram_run() -> None:
    from app.backends.granite import _collapse_repeats

    text = "word word word word word word"  # 6 reps, > max_run(4)
    assert _collapse_repeats(text) == "word"


def test_collapse_repeats_collapses_long_2gram_run() -> None:
    from app.backends.granite import _collapse_repeats

    text = "foo bar " * 6
    assert _collapse_repeats(text.strip()) == "foo bar"


def test_collapse_repeats_collapses_timestamp_tag_loop() -> None:
    from app.backends.granite import _collapse_repeats

    text = "foo [T:10] " * 6
    assert _collapse_repeats(text.strip()) == "foo [T:10]"


def test_collapse_repeats_custom_max_run() -> None:
    from app.backends.granite import _collapse_repeats

    text = "hi hi hi"  # 3 reps
    assert _collapse_repeats(text, max_run=2) == "hi"
    assert _collapse_repeats(text, max_run=3) == text


# ----- _token_budget -----


def test_token_budget_floor() -> None:
    from app.backends.granite import _token_budget

    assert _token_budget(0.0, "asr") == 256
    assert _token_budget(0.0, "saa") == 256
    assert _token_budget(0.0, "ts") == 256
    # Small durations should still hit the floor.
    assert _token_budget(1.0, "asr") == 256


def test_token_budget_scales_with_seconds() -> None:
    from app.backends.granite import _token_budget

    low = _token_budget(50.0, "asr")
    high = _token_budget(200.0, "asr")
    assert high > low


def test_token_budget_rate_ordering_ts_gt_saa_gt_asr() -> None:
    from app.backends.granite import _token_budget

    seconds = 100.0
    asr = _token_budget(seconds, "asr")
    saa = _token_budget(seconds, "saa")
    ts = _token_budget(seconds, "ts")
    assert ts > saa > asr


def test_token_budget_unknown_task_raises() -> None:
    from app.backends.granite import _token_budget

    with pytest.raises(KeyError):
        _token_budget(10.0, "bogus")


# ----- _parse_words -----


def test_parse_words_time_offset_with_rollover() -> None:
    from app.backends.granite import _parse_words

    # Same tag sequence as test_parse_word_timestamps_rollover but shifted by
    # a 100s time_offset — rollover handling must still fire relative to the
    # already-offset `last_end`.
    words = _parse_words("a [T:850] b [T:30]", time_offset=100.0)
    assert len(words) == 2
    assert words[0].word == "a"
    assert words[0].start == pytest.approx(100.0)
    assert words[0].end == pytest.approx(108.5)
    assert words[1].word == "b"
    assert words[1].start == pytest.approx(108.5)
    assert words[1].end == pytest.approx(110.3)


def test_parse_words_no_offset_matches_default() -> None:
    from app.backends.granite import _parse_words

    words = _parse_words("hello [T:45] world [T:82]")
    assert words[0].end == pytest.approx(0.45)
    assert words[1].end == pytest.approx(0.82)


def test_parse_words_empty_text() -> None:
    from app.backends.granite import _parse_words

    assert _parse_words("") == []


def test_parse_words_skips_silence_marker() -> None:
    from app.backends.granite import _parse_words

    words = _parse_words("hello [T:45] _ [T:60] world [T:82]")
    assert [w.word for w in words] == ["hello", "world"]


# ----- _segments_from_words -----


def test_segments_from_words_empty() -> None:
    from app.backends.granite import _segments_from_words

    segs = _segments_from_words([], fallback_end=12.5)
    assert len(segs) == 1
    assert segs[0].start == 0.0
    assert segs[0].end == 12.5
    assert segs[0].text == ""
    assert segs[0].words is None


def test_segments_from_words_non_empty_spans_first_to_last() -> None:
    from app.backends.granite import _segments_from_words
    from app.schema import TranscriptionWord

    words = [
        TranscriptionWord(word="hello", start=1.0, end=1.5),
        TranscriptionWord(word="world", start=1.5, end=2.2),
    ]
    segs = _segments_from_words(words, fallback_end=99.0)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.start == 1.0
    assert seg.end == 2.2
    assert seg.text == "hello world"
    assert seg.words == words


# ----- ASRBackend.transcribe_stream default wrapper -----


def _make_fake_backend():
    from app.backends.base import ASRBackend
    from app.schema import TranscriptionSegment

    class FakeBackend(ASRBackend):
        """Minimal ASRBackend subclass driving the default transcribe_stream."""

        model_id = "fake"

        async def load(self, model_id: str, device: str) -> None:
            pass

        async def unload(self) -> None:
            pass

        async def transcribe(self, req, progress_cb=None, partial_cb=None):
            if progress_cb:
                progress_cb(50.0)
            if partial_cb:
                partial_cb("hi", 0.0, 1.0)
            return (
                [TranscriptionSegment(id=0, start=0, end=1, text="hi")],
                "de",
            )

    return FakeBackend()


def _make_request():
    from app.schema import TranscriptionRequest

    return TranscriptionRequest(
        audio_bytes=b"",
        filename="a.wav",
        model="fake",
        language=None,
        response_format="json",
        word_timestamps=False,
        segment_timestamps=False,
        speaker_attribution=False,
        translate=False,
        translate_to=None,
        prompt=None,
        stream=True,
        min_speakers=None,
        max_speakers=None,
    )


async def test_transcribe_stream_default_wrapper_event_sequence() -> None:
    backend = _make_fake_backend()
    req = _make_request()

    events = [event async for event in backend.transcribe_stream(req)]

    assert events[0] == {"type": "progress", "progress": 0}

    progress_values = [e["progress"] for e in events if e["type"] == "progress"]
    assert 50.0 in progress_values
    assert progress_values[0] == 0
    assert progress_values[-1] == 100

    segment_events = [e for e in events if e["type"] == "segment"]
    assert len(segment_events) == 1
    assert segment_events[0]["text"] == "hi"
    assert segment_events[0]["start"] == 0
    assert segment_events[0]["end"] == 1

    result_events = [e for e in events if e["type"] == "result"]
    assert len(result_events) == 1
    assert result_events[0]["text"] == "hi"
    assert result_events[0]["language"] == "de"

    partial_events = [e for e in events if e["type"] == "partial"]
    assert partial_events == [{"type": "partial", "text": "hi", "start": 0.0, "end": 1.0}]
    # Partials must arrive before the final segment events.
    assert events.index(partial_events[0]) < events.index(segment_events[0])

    assert events[-1] == {"type": "progress", "progress": 100}
