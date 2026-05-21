from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import torch

from app.audio import load_audio_bytes
from app.backends.base import ASRBackend
from app.schema import TranscriptionRequest, TranscriptionSegment, TranscriptionWord

log = logging.getLogger(__name__)


GRANITE_BASE = "ibm-granite/granite-speech-4.1-2b"
GRANITE_PLUS = "ibm-granite/granite-speech-4.1-2b-plus"
GRANITE_NAR = "ibm-granite/granite-speech-4.1-2b-nar"

GRANITE_MODELS = {GRANITE_BASE, GRANITE_PLUS, GRANITE_NAR}

OPENAI_TRANSCRIPTION_MODEL_ALIASES = {
    "whisper-1": GRANITE_BASE,
    "gpt-4o-transcribe": GRANITE_BASE,
    "gpt-4o-mini-transcribe": GRANITE_BASE,
}

# Languages the AST mode of the base/2b model supports.
AST_LANGUAGES = {
    "english": "English",
    "en": "English",
    "french": "French",
    "fr": "French",
    "german": "German",
    "de": "German",
    "spanish": "Spanish",
    "es": "Spanish",
    "portuguese": "Portuguese",
    "pt": "Portuguese",
    "japanese": "Japanese",
    "ja": "Japanese",
    "italian": "Italian",
    "it": "Italian",
    "mandarin": "Mandarin",
    "zh": "Mandarin",
}

PLUS_SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)

# Regex for word-timestamp tags emitted by the plus model: "[T:NNN]"
_TS_RE = re.compile(r"\[T:(\d+)\]")
# Regex for speaker turn markers: "[Speaker N]:"
_SPK_RE = re.compile(r"\[Speaker\s+(\d+)\]\s*:")


def normalize_model_id(model: str | None, *, want_plus_features: bool) -> str:
    """Return a fully-qualified Granite model id given a user-supplied name.

    Auto-upgrades the base 2b model to 2b-plus when the request needs features
    (speaker attribution / word timestamps) that only the plus model supports.
    Accepts short aliases ("granite-speech-4.1-2b-plus") and falls back to the
    default base model for unknown values.
    """
    if not model:
        model = GRANITE_BASE
    else:
        model = model.strip()

    model = OPENAI_TRANSCRIPTION_MODEL_ALIASES.get(model.lower(), model)

    # Allow shorthand without the org prefix.
    if "/" not in model:
        model = f"ibm-granite/{model}"

    if model not in GRANITE_MODELS:
        log.warning("Unknown model %s — falling back to %s", model, GRANITE_BASE)
        model = GRANITE_BASE

    if want_plus_features and model == GRANITE_BASE:
        log.info("Auto-upgrading %s -> %s for requested rich-transcription features",
                 model, GRANITE_PLUS)
        model = GRANITE_PLUS

    return model


def _to_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


class GraniteBackend(ASRBackend):
    """Backend for IBM Granite Speech 4.1 family."""

    def __init__(self, dtype: str = "bfloat16") -> None:
        self.model_id: str = ""
        self._processor: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: str = "cpu"
        self._dtype = _to_torch_dtype(dtype)
        self._lock = asyncio.Lock()

    # ----- Loading -----

    async def load(self, model_id: str, device: str) -> None:
        if self.model_id == model_id and self._model is not None:
            return
        await self.unload()
        log.info("Loading Granite model %s on %s (%s)", model_id, device, self._dtype)
        # Heavy import — defer.
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        def _load() -> tuple[Any, Any]:
            proc = AutoProcessor.from_pretrained(model_id)
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                device_map=device,
                dtype=self._dtype,
            )
            mdl.eval()
            return proc, mdl

        loop = asyncio.get_running_loop()
        proc, mdl = await loop.run_in_executor(None, _load)
        self._processor = proc
        self._tokenizer = proc.tokenizer
        self._model = mdl
        self._device = device
        self.model_id = model_id
        log.info("Granite model %s ready", model_id)

    async def unload(self) -> None:
        if self._model is None:
            return
        log.info("Unloading Granite model %s", self.model_id)
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._tokenizer = None
        self.model_id = ""
        try:
            import torch as _t

            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ----- Inference -----

    async def transcribe(
        self, req: TranscriptionRequest
    ) -> tuple[list[TranscriptionSegment], str | None]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        wav, duration = load_audio_bytes(req.audio_bytes)

        is_plus = self.model_id == GRANITE_PLUS

        want_words = req.word_timestamps and is_plus
        want_speakers = req.speaker_attribution and is_plus

        async with self._lock:
            if want_words and want_speakers:
                # Two-pass: word-timestamps + speakers, merge by sequential alignment.
                ts_text = await self._run(wav, self._build_prompt(
                    word_timestamps=True, speaker_attribution=False, req=req,
                ), max_new_tokens=10000)
                spk_text = await self._run(wav, self._build_prompt(
                    word_timestamps=False, speaker_attribution=True, req=req,
                ), max_new_tokens=4000)
                segments = _merge_words_and_speakers(ts_text, spk_text)
            elif want_words:
                ts_text = await self._run(wav, self._build_prompt(
                    word_timestamps=True, speaker_attribution=False, req=req,
                ), max_new_tokens=10000)
                segments = _parse_word_timestamps(ts_text)
            elif want_speakers:
                spk_text = await self._run(wav, self._build_prompt(
                    word_timestamps=False, speaker_attribution=True, req=req,
                ), max_new_tokens=4000)
                segments = _parse_speaker_segments(spk_text, duration)
            else:
                text = await self._run(wav, self._build_prompt(
                    word_timestamps=False, speaker_attribution=False, req=req,
                ), max_new_tokens=2000)
                segments = [
                    TranscriptionSegment(
                        id=0, start=0.0, end=duration, text=text.strip()
                    )
                ]

        language = req.language or _guess_lang(req.translate_to)
        return segments, language

    # ----- Internals -----

    def _build_prompt(
        self,
        *,
        word_timestamps: bool,
        speaker_attribution: bool,
        req: TranscriptionRequest,
    ) -> str:
        is_plus = self.model_id == GRANITE_PLUS
        keywords = (req.prompt or "").strip()

        if speaker_attribution and is_plus:
            return (
                "<|audio|> Speaker attribution: Transcribe and denote who is "
                "speaking by adding [Speaker 1]: and [Speaker 2]: tags before "
                "speaker turns."
            )
        if word_timestamps and is_plus:
            return (
                "<|audio|> Timestamps: Transcribe the speech. After each word, "
                "add a timestamp tag showing the end time in centiseconds, e.g. "
                "hello [T:45] world [T:82]"
            )
        if req.translate and req.translate_to:
            lang = AST_LANGUAGES.get(req.translate_to.lower(), req.translate_to)
            base = f"<|audio|> translate the speech to {lang} with proper punctuation and capitalization."
            if keywords:
                base = (
                    f"<|audio|> translate the speech to {lang}. Keywords: {keywords}"
                )
            return base
        if keywords:
            return (
                f"<|audio|> transcribe the speech to text. Keywords: {keywords}"
            )
        # Default: punctuated + capitalized ASR.
        return "<|audio|> transcribe the speech with proper punctuation and capitalization."

    async def _run(self, wav: torch.Tensor, user_prompt: str, max_new_tokens: int) -> str:
        is_plus = self.model_id == GRANITE_PLUS
        chat: list[dict] = []
        if is_plus:
            chat.append({"role": "system", "content": PLUS_SYSTEM_PROMPT})
        chat.append({"role": "user", "content": user_prompt})
        prompt_text = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        loop = asyncio.get_running_loop()

        def _infer() -> str:
            inputs = self._processor(
                prompt_text, wav, device=self._device, return_tensors="pt"
            ).to(self._device)
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
            return self._tokenizer.decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )

        return await loop.run_in_executor(None, _infer)


# ----- Output parsing helpers -----


def _guess_lang(translate_to: str | None) -> str | None:
    if not translate_to:
        return None
    return translate_to[:2].lower()


def _decode_centiseconds(centi: int, last_end: float, offset: float) -> tuple[float, float]:
    """Unwrap a [T:NNN] centisecond tag (modulo 1000, =10s rollover)."""
    end = centi / 100.0
    while end + offset < last_end:
        offset += 10.0
    return end + offset, offset


def _parse_word_timestamps(text: str) -> list[TranscriptionSegment]:
    """Parse output of the word-timestamps prompt.

    The model emits e.g. `hello [T:45] world [T:82]`. Silences are encoded as `_`.
    """
    parts = _TS_RE.split(text)
    # parts = [word0, ts0, word1, ts1, ..., trailing]
    words: list[TranscriptionWord] = []
    last_end = 0.0
    offset = 0.0
    word_start = 0.0
    for word, ts in zip(parts[::2], parts[1::2]):
        token = word.strip()
        if not token:
            last_end_tmp, offset = _decode_centiseconds(int(ts), last_end, offset)
            word_start = last_end_tmp
            last_end = last_end_tmp
            continue
        end, offset = _decode_centiseconds(int(ts), last_end, offset)
        # Skip pure-silence markers from the words array.
        if token != "_":
            words.append(
                TranscriptionWord(word=token, start=word_start, end=end, probability=1.0)
            )
        word_start = end
        last_end = end

    if not words:
        return [TranscriptionSegment(id=0, start=0.0, end=0.0, text=text.strip())]

    seg_text = " ".join(w.word for w in words)
    return [
        TranscriptionSegment(
            id=0,
            start=words[0].start,
            end=words[-1].end,
            text=seg_text,
            words=words,
        )
    ]


def _parse_speaker_segments(text: str, duration: float) -> list[TranscriptionSegment]:
    """Parse output of the speaker-attribution prompt: `[Speaker N]: text...`."""
    # Split on speaker tags but keep them.
    parts = re.split(r"(\[Speaker\s+\d+\]\s*:)", text)
    # parts may start with leading text before any tag — treat as Speaker 1 fallback.
    segments: list[TranscriptionSegment] = []
    current_speaker: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        chunk = " ".join(b.strip() for b in buf if b.strip())
        if chunk:
            segments.append(
                TranscriptionSegment(
                    id=len(segments), start=0.0, end=0.0, text=chunk,
                    speaker=current_speaker,
                )
            )
        buf.clear()

    for part in parts:
        if not part:
            continue
        m = _SPK_RE.match(part)
        if m:
            flush()
            current_speaker = f"SPEAKER_{int(m.group(1)) - 1:02d}"
        else:
            buf.append(part)
    flush()

    # Linearly distribute timing across segments based on word counts.
    if segments and duration > 0:
        total_words = sum(len(s.text.split()) for s in segments) or 1
        cursor = 0.0
        for s in segments:
            words = max(1, len(s.text.split()))
            length = duration * words / total_words
            s.start = round(cursor, 3)
            s.end = round(cursor + length, 3)
            cursor += length

    if not segments:
        segments = [TranscriptionSegment(id=0, start=0.0, end=duration, text=text.strip())]
    return segments


def _merge_words_and_speakers(ts_text: str, spk_text: str) -> list[TranscriptionSegment]:
    """Combine outputs from a word-timestamp pass and a speaker-attribution pass.

    Strategy: parse word-timestamps to get [(word, start, end)] and parse
    speaker turns to get [(speaker, [tokens])]. Then walk the speaker turn
    word stream and the timestamp word stream in lock-step, assigning each
    timestamp word the speaker label of the corresponding turn position.
    Mismatches are absorbed by advancing the timestamp index without
    advancing speakers — best-effort, but produces a coherent transcript.
    """
    ts_segments = _parse_word_timestamps(ts_text)
    if not ts_segments or not ts_segments[0].words:
        return ts_segments

    ts_words = ts_segments[0].words
    duration = ts_segments[0].end

    # Build speaker turns: list of (speaker_label, list_of_tokens).
    parts = re.split(r"(\[Speaker\s+\d+\]\s*:)", spk_text)
    turns: list[tuple[str, list[str]]] = []
    current = "SPEAKER_00"
    buf: list[str] = []

    def push_turn() -> None:
        if buf:
            turns.append((current, [t.lower() for t in buf if t]))
            buf.clear()

    for part in parts:
        if not part:
            continue
        m = _SPK_RE.match(part)
        if m:
            push_turn()
            current = f"SPEAKER_{int(m.group(1)) - 1:02d}"
        else:
            buf.extend(re.findall(r"\w+", part))
    push_turn()

    # Walk both streams; assign speakers to ts_words.
    if turns:
        turn_idx = 0
        token_idx = 0  # index inside the current turn's tokens
        for w in ts_words:
            wt = re.sub(r"\W+", "", w.word).lower()
            if not wt:
                w.speaker = turns[turn_idx][0]
                continue
            # advance to a turn that still has tokens
            while turn_idx < len(turns) and token_idx >= len(turns[turn_idx][1]):
                turn_idx += 1
                token_idx = 0
            if turn_idx >= len(turns):
                w.speaker = turns[-1][0]
                continue
            w.speaker = turns[turn_idx][0]
            # advance one token inside the turn (allow imperfect match)
            token_idx += 1

    # Build per-speaker segments from contiguous runs of words with the same speaker.
    segments: list[TranscriptionSegment] = []
    if not ts_words:
        return ts_segments

    run: list[TranscriptionWord] = []

    def flush_run() -> None:
        if not run:
            return
        spk = run[0].speaker
        segments.append(
            TranscriptionSegment(
                id=len(segments),
                start=run[0].start,
                end=run[-1].end,
                text=" ".join(w.word for w in run),
                speaker=spk,
                words=list(run),
            )
        )
        run.clear()

    for w in ts_words:
        if run and run[-1].speaker != w.speaker:
            flush_run()
        run.append(w)
    flush_run()

    if not segments:
        segments = ts_segments
        for s in segments:
            s.end = duration
    return segments
