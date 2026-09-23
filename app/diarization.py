"""Speaker diarization stage: pyannote (community-1) or NVIDIA Nemotron 3.

Runs as a separate pipeline stage in front of the ASR backend: the diarizer
produces speaker turns, the transcriber (+ optional forced aligner) produces
word timestamps, and ``label_words`` reconciles the two into speaker-labelled
segments.

Each engine is lazy-loaded on first use and unloaded again after the same idle
TTL as the ASR registry.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.audio import TARGET_SR, load_audio_bytes
from app.config import settings
from app.lazy import LazyModel
from app.schema import TranscriptionSegment, TranscriptionWord

log = logging.getLogger(__name__)

ENGINES = ("pyannote", "nemotron")


@dataclass
class Turn:
    """One diarization turn: `speaker` talks from `start` to `end` (seconds)."""

    start: float
    end: float
    speaker: str


class PyannoteDiarizer(LazyModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = settings.diarization_model

    def _load(self) -> Any:
        import torch
        from pyannote.audio import Pipeline

        token = settings.hf_token or None
        try:
            pipe = Pipeline.from_pretrained(self.name, token=token)
        except TypeError:  # pyannote.audio < 4 uses use_auth_token
            pipe = Pipeline.from_pretrained(self.name, use_auth_token=token)
        if pipe is None:
            raise RuntimeError(
                f"Could not load {self.name} — gated model: set GRANITE_HF_TOKEN "
                "and accept the model conditions on huggingface.co"
            )
        pipe.to(torch.device(settings.resolved_device()))
        return pipe

    async def diarize(
        self,
        audio_bytes: bytes,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[Turn]:
        def _run(pipeline: Any) -> list[Turn]:
            wav, _ = load_audio_bytes(audio_bytes)
            kwargs: dict = {}
            if num_speakers:
                kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers:
                    kwargs["min_speakers"] = min_speakers
                if max_speakers:
                    kwargs["max_speakers"] = max_speakers
            out = pipeline({"waveform": wav, "sample_rate": TARGET_SR}, **kwargs)
            # community-1 returns an output object with the *exclusive*
            # diarization (non-overlapping, built for reconciliation with
            # ASR timestamps); 3.1 returns a plain Annotation.
            ann = getattr(out, "exclusive_speaker_diarization", None)
            if ann is None:
                ann = getattr(out, "speaker_diarization", out)
            return [
                Turn(float(seg.start), float(seg.end), str(label))
                for seg, _, label in ann.itertracks(yield_label=True)
            ]

        return await self.run(_run)


# Offline-style configuration from the model card (units: 80 ms frames).
_NEMOTRON_OFFLINE = {
    "spkcache_len": 264,
    "fifo_len": 40,
    "chunk_len": 340,
    "chunk_right_context": 40,
    "spkcache_update_period": 300,
}
NEMOTRON_MAX_SPEAKERS = 8


def _parse_nemo_segment(seg: Any) -> Turn | None:
    """NeMo returns 'start end speaker_N' strings (or tuples) per segment."""
    parts = seg if isinstance(seg, (list, tuple)) else re.split(r"[,\s]+", str(seg).strip())
    if len(parts) < 3:
        return None
    try:
        return Turn(float(parts[0]), float(parts[1]), str(parts[2]))
    except ValueError:
        return None


def _use_device() -> None:
    """Make the configured GPU current for this (executor) thread.

    NeMo allocates some tensors on the *current* CUDA device instead of the
    model's; with GRANITE_DEVICE=cuda:1 that mixes devices → illegal memory
    access. The current device is per-thread, so call this in every job.
    """
    import torch

    device = torch.device(settings.resolved_device())
    if device.type == "cuda" and torch.cuda.is_available():
        # Plain "cuda" has no index; set_device() needs one → default GPU 0.
        torch.cuda.set_device(device.index or 0)


class NemotronDiarizer(LazyModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = settings.nemotron_diarization_model

    def _load(self) -> Any:
        from nemo.collections.asr.models import SortformerEncLabelModel

        _use_device()
        model = SortformerEncLabelModel.from_pretrained(
            self.name, map_location=settings.resolved_device()
        )
        model.eval()
        for key, value in _NEMOTRON_OFFLINE.items():
            setattr(model.sortformer_modules, key, value)
        model._check_streaming_parameters()
        return model

    async def diarize(
        self,
        audio_bytes: bytes,
        num_speakers: int | None = None,
        min_speakers: int | None = None,  # noqa: ARG002 — Sortformer takes no bounds
        max_speakers: int | None = None,  # noqa: ARG002
    ) -> list[Turn]:
        if num_speakers and num_speakers > NEMOTRON_MAX_SPEAKERS:
            log.warning(
                "Nemotron supports at most %d speakers (requested %d) — "
                "extra speakers will be merged or missed",
                NEMOTRON_MAX_SPEAKERS, num_speakers,
            )

        def _run(model: Any) -> list[Turn]:
            import torch

            _use_device()
            wav, _ = load_audio_bytes(audio_bytes)
            with torch.inference_mode():
                out = model.diarize(
                    audio=[wav[0].numpy()], batch_size=1, sample_rate=TARGET_SR,
                )
            turns = [_parse_nemo_segment(s) for s in out[0]]
            return [t for t in turns if t is not None]

        return await self.run(_run)


class Diarizer:
    """Engine dispatcher (pyannote | nemotron) sharing one lifecycle."""

    def __init__(self) -> None:
        self.engines: dict[str, PyannoteDiarizer | NemotronDiarizer] = {
            "pyannote": PyannoteDiarizer(),
            "nemotron": NemotronDiarizer(),
        }

    def loaded(self, engine: str) -> bool:
        return self.engines[engine].loaded

    async def diarize(
        self,
        audio_bytes: bytes,
        engine: str = "pyannote",
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[Turn]:
        """Return speaker turns for `audio_bytes`, sorted by start time."""
        turns = await self.engines[engine].diarize(
            audio_bytes,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        turns.sort(key=lambda t: t.start)
        log.info(
            "Diarization (%s): %d turns, %d speakers",
            engine, len(turns), len({t.speaker for t in turns}),
        )
        return turns

    def start_idle_monitor(self) -> None:
        for e in self.engines.values():
            e.start_idle_monitor()

    async def shutdown(self) -> None:
        for e in self.engines.values():
            await e.shutdown()


def assign_speakers(words: list[TranscriptionWord], turns: list[Turn]) -> None:
    """Label each word with the speaker of the best-overlapping turn.

    Words and turns must be time-sorted. Words that overlap no turn (silence
    padding, boundary drift) get the nearest turn by midpoint distance.
    Overlapping turns (Nemotron reports crosstalk) are fine: the turn with the
    largest overlap wins.
    """
    if not turns:
        return
    lo = 0
    for w in words:
        # Advance past turns that end before this word starts.
        while lo + 1 < len(turns) and turns[lo].end <= w.start:
            lo += 1
        best: Turn | None = None
        best_overlap = 0.0
        j = lo
        while j < len(turns) and turns[j].start < w.end:
            overlap = min(turns[j].end, w.end) - max(turns[j].start, w.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = turns[j]
            j += 1
        if best is None:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: abs((t.start + t.end) / 2 - mid))
        w.speaker = best.speaker


# A word that ends a sentence (possibly followed by closing quotes/brackets).
_SENT_END_RE = re.compile(r"[.!?…]['\")\]»]*$")


def snap_speakers_to_sentences(
    words: list[TranscriptionWord], max_sentence_words: int = 40
) -> None:
    """Majority-vote the speaker per sentence.

    Word-level turn assignment flickers on backchannels and boundary jitter,
    producing speaker switches mid-sentence. For punctuated text every word in
    a sentence votes (weighted by duration) and the winner labels the whole
    sentence. Runaway sentences without punctuation are capped so crosstalk
    can't be swallowed entirely.
    """
    sent: list[TranscriptionWord] = []

    def flush() -> None:
        if not sent:
            return
        durations: dict[str, float] = {}
        for w in sent:
            if w.speaker:
                durations[w.speaker] = durations.get(w.speaker, 0.0) + max(
                    0.0, w.end - w.start
                )
        if durations:
            best = max(durations, key=durations.get)  # type: ignore[arg-type]
            for w in sent:
                w.speaker = best
        sent.clear()

    for w in words:
        sent.append(w)
        if _SENT_END_RE.search(w.word) or len(sent) >= max_sentence_words:
            flush()
    flush()


def label_words(
    words: list[TranscriptionWord], turns: list[Turn]
) -> list[TranscriptionSegment]:
    """Assign speakers from `turns` and group words into per-speaker segments."""
    from app.backends.granite import _speaker_runs_to_segments

    assign_speakers(words, turns)
    # Sentence snapping needs punctuation; unpunctuated output (Granite word
    # pass) would degrade to fixed 40-word blocks that swallow short turns.
    if any(_SENT_END_RE.search(w.word) for w in words):
        snap_speakers_to_sentences(words)
    return _speaker_runs_to_segments(words)


diarizer = Diarizer()
