from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Callable
from typing import Any

import torch

from app.audio import TARGET_SR, load_audio_bytes
from app.backends.base import ASRBackend
from app.config import settings
from app.schema import TranscriptionRequest, TranscriptionSegment, TranscriptionWord

log = logging.getLogger(__name__)

# IBM: the plus model "works well with audio segments up to 9 minutes long for
# ASR and SAA, and up to 3.5 minutes for timestamps". Longer inputs degenerate
# into repetition loops, so anything above these bounds is chunked.
MAX_ASR_SECONDS = 540.0
MAX_TS_SECONDS = 210.0
# Target chunk sizes when splitting (kept below the max so quiet-point search
# has room to move the boundary).
CHUNK_ASR_SECONDS = 480.0
CHUNK_TS_SECONDS = 180.0
# Segment size for incremental SAA decoding (model-card "Task 4" scheme:
# cumulative audio + prefix_text keeps speaker numbering stable).
SAA_INCREMENT_SECONDS = 240.0

# Matches any <|...|> special-token artifact the model occasionally emits as
# literal text (e.g. <|fim_middle|>). These are not real transcription and break
# downstream tiktoken-based consumers, so we strip them from decoded output.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


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
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer

        def _load() -> tuple[Any, Any, Any]:
            proc = AutoProcessor.from_pretrained(model_id)
            # proc.tokenizer may be None in newer transformers versions for
            # multi-modal processors — load explicitly to be safe.
            tok = (
                proc.tokenizer
                if getattr(proc, "tokenizer", None) is not None
                else AutoTokenizer.from_pretrained(model_id)
            )
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                device_map=device,
                dtype=self._dtype,
            )
            mdl.eval()
            return proc, tok, mdl

        loop = asyncio.get_running_loop()
        proc, tok, mdl = await loop.run_in_executor(None, _load)
        self._processor = proc
        self._tokenizer = tok
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
        self,
        req: TranscriptionRequest,
        progress_cb: Callable[[float], None] | None = None,
    ) -> tuple[list[TranscriptionSegment], str | None]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        wav, duration = load_audio_bytes(req.audio_bytes)

        is_plus = self.model_id == GRANITE_PLUS
        want_words = req.word_timestamps and is_plus
        want_speakers = req.speaker_attribution and is_plus

        # Progress accounting in processed audio-seconds across all passes.
        total_work = duration * (2 if (want_words and want_speakers) else 1)
        done_work = 0.0

        def tick(seconds: float) -> None:
            nonlocal done_work
            done_work += seconds
            if progress_cb and total_work > 0:
                progress_cb(min(99.0, done_work / total_work * 100.0))

        async with self._lock:
            if want_words and want_speakers:
                # Two-pass: word-timestamps + speakers, merge by sequential alignment.
                words = await self._words_pass(wav, duration, req, tick)
                spk_text = await self._saa_pass(wav, duration, req, tick)
                segments = _merge_words_and_speakers(words, spk_text)
            elif want_words:
                words = await self._words_pass(wav, duration, req, tick)
                segments = _segments_from_words(words, fallback_end=duration)
            elif want_speakers:
                spk_text = await self._saa_pass(wav, duration, req, tick)
                segments = _parse_speaker_segments(spk_text, duration)
            else:
                segments = await self._asr_pass(wav, duration, req, tick)

        language = req.language or _guess_lang(req.translate_to)
        return segments, language

    async def _asr_pass(
        self,
        wav: torch.Tensor,
        duration: float,
        req: TranscriptionRequest,
        tick: Callable[[float], None],
    ) -> list[TranscriptionSegment]:
        """Plain ASR / AST, chunked at quiet points when above the model limit."""
        prompt = self._build_prompt(
            word_timestamps=False, speaker_attribution=False, req=req,
        )
        windows = _plan_windows(wav, duration, MAX_ASR_SECONDS, CHUNK_ASR_SECONDS)
        segments: list[TranscriptionSegment] = []
        for t0, t1 in windows:
            piece = wav[:, int(t0 * TARGET_SR): int(t1 * TARGET_SR)]
            text = await self._run(
                piece, prompt, max_new_tokens=_token_budget(t1 - t0, "asr")
            )
            text = _collapse_repeats(text).strip()
            if text:
                segments.append(
                    TranscriptionSegment(
                        id=len(segments), start=round(t0, 3), end=round(t1, 3),
                        text=text,
                    )
                )
            tick(t1 - t0)
        if not segments:
            segments = [TranscriptionSegment(id=0, start=0.0, end=duration, text="")]
        return segments

    async def _words_pass(
        self,
        wav: torch.Tensor,
        duration: float,
        req: TranscriptionRequest,
        tick: Callable[[float], None],
    ) -> list[TranscriptionWord]:
        """Word-timestamp pass, chunked to the (much lower) timestamp limit."""
        prompt = self._build_prompt(
            word_timestamps=True, speaker_attribution=False, req=req,
        )
        windows = _plan_windows(wav, duration, MAX_TS_SECONDS, CHUNK_TS_SECONDS)
        words: list[TranscriptionWord] = []
        for t0, t1 in windows:
            piece = wav[:, int(t0 * TARGET_SR): int(t1 * TARGET_SR)]
            ts_text = await self._run(
                piece, prompt, max_new_tokens=_token_budget(t1 - t0, "ts")
            )
            words.extend(_parse_words(_collapse_repeats(ts_text), time_offset=t0))
            tick(t1 - t0)
        return words

    async def _saa_pass(
        self,
        wav: torch.Tensor,
        duration: float,
        req: TranscriptionRequest,
        tick: Callable[[float], None],
    ) -> str:
        """Speaker attribution via the model-card incremental decoding scheme.

        Audio is fed cumulatively (from the current context-block start) while
        `prefix_text` carries the transcript decoded so far, which keeps
        speaker numbering stable across increments. When the cumulative audio
        would exceed the model limit, the context block is reset.
        """
        prompt = self._build_prompt(
            word_timestamps=False, speaker_attribution=True, req=req,
        )
        if duration <= CHUNK_ASR_SECONDS:
            text = await self._run(
                wav, prompt, max_new_tokens=_token_budget(duration, "saa")
            )
            tick(duration)
            return _collapse_repeats(text)

        windows = _plan_windows(wav, duration, SAA_INCREMENT_SECONDS, SAA_INCREMENT_SECONDS)
        parts: list[str] = []
        block_start = 0.0
        prefix = ""
        for t0, t1 in windows:
            if t1 - block_start > MAX_ASR_SECONDS:
                log.info("SAA context block reset at %.1fs (model limit)", t0)
                block_start = t0
                prefix = ""
            piece = wav[:, int(block_start * TARGET_SR): int(t1 * TARGET_SR)]
            text = await self._run(
                piece,
                prompt,
                max_new_tokens=_token_budget(t1 - t0, "saa"),
                prefix_text=prefix or None,
            )
            text = _collapse_repeats(text).strip()
            parts.append(text)
            prefix = (prefix + " " + text).strip()
            tick(t1 - t0)
        return " ".join(p for p in parts if p)

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
                "<|audio|> Speaker attribution: Transcribe the speech and label "
                "each speaker turn with a [Speaker N]: tag (e.g. [Speaker 1]:, "
                "[Speaker 2]:, [Speaker 3]:), assigning a new number to each "
                "distinct speaker you hear."
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
        # Naming the spoken language stops the base model from spontaneously
        # translating to English instead of transcribing.
        lang_name = AST_LANGUAGES.get((req.language or "").lower())
        speech = f"the {lang_name} speech" if lang_name else "the speech"
        if keywords:
            return (
                f"<|audio|> transcribe {speech} to text. Keywords: {keywords}"
            )
        # Default: punctuated + capitalized ASR.
        return f"<|audio|> transcribe {speech} with proper punctuation and capitalization."

    async def _run(
        self,
        wav: torch.Tensor,
        user_prompt: str,
        max_new_tokens: int,
        prefix_text: str | None = None,
    ) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded — backend in inconsistent state")
        is_plus = self.model_id == GRANITE_PLUS
        chat: list[dict] = []
        if is_plus:
            chat.append({"role": "system", "content": PLUS_SYSTEM_PROMPT})
        chat.append({"role": "user", "content": user_prompt})
        # `prefix_text` (Granite chat-template kwarg) carries the already
        # decoded transcript so the model only decodes the continuation.
        extra = {"prefix_text": prefix_text} if prefix_text else {}
        prompt_text = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, **extra
        )

        loop = asyncio.get_running_loop()

        # Capture all model references as locals so that a concurrent unload()
        # setting self._tokenizer / _processor / _model to None cannot cause
        # AttributeError inside the executor thread.
        tokenizer = self._tokenizer
        processor = self._processor
        model = self._model
        device = self._device

        gen_extra: dict[str, Any] = {}
        if settings.repetition_penalty != 1.0:
            gen_extra["repetition_penalty"] = settings.repetition_penalty

        def _infer() -> str:
            inputs = processor(
                prompt_text, wav, device=device, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    **gen_extra,
                )
            new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
            decoded = tokenizer.decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )
            # skip_special_tokens drops registered specials, but Granite's FIM
            # tokens (<|fim_middle|>, …) come back as literal text — strip them.
            return _SPECIAL_TOKEN_RE.sub("", decoded)

        return await loop.run_in_executor(None, _infer)


# ----- Chunking helpers -----


def _plan_windows(
    wav: torch.Tensor, duration: float, limit: float, target: float
) -> list[tuple[float, float]]:
    """Split `duration` into windows <= `limit`, cutting at quiet points.

    Returns [(start, end), ...]; a single full-length window when the audio
    already fits the limit.
    """
    if duration <= limit:
        return [(0.0, duration)]
    n = math.ceil(duration / target)
    step = duration / n
    bounds = [0.0]
    bounds += [_quiet_point(wav, step * i, radius=10.0) for i in range(1, n)]
    bounds.append(duration)
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a > 0.05]


def _quiet_point(wav: torch.Tensor, t: float, radius: float) -> float:
    """Return the lowest-energy time within +-radius of `t` (avoid mid-word cuts)."""
    lo = max(0, int((t - radius) * TARGET_SR))
    hi = min(wav.shape[-1], int((t + radius) * TARGET_SR))
    frame = int(0.1 * TARGET_SR)
    n = (hi - lo) // frame
    if n < 2:
        return t
    window = wav[0, lo: lo + n * frame]
    energy = window.reshape(n, frame).abs().mean(dim=1)
    idx = int(energy.argmin())
    return (lo + idx * frame + frame // 2) / TARGET_SR


def _token_budget(seconds: float, task: str) -> int:
    """Upper bound on generated tokens for a chunk — also caps runaway loops."""
    rate = {"asr": 12, "saa": 15, "ts": 30}[task]
    return max(256, int(seconds * rate) + 128)


def _collapse_repeats(text: str, max_run: int = 4) -> str:
    """Collapse degenerate repetition loops (same 1-4 gram repeated endlessly)."""
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        collapsed = False
        for n in (1, 2, 3, 4):
            gram = tokens[i: i + n]
            if len(gram) < n:
                continue
            reps = 1
            while tokens[i + reps * n: i + (reps + 1) * n] == gram:
                reps += 1
            if reps > max_run:
                out.extend(gram)
                log.warning(
                    "Collapsed %d consecutive repetitions of %r in model output",
                    reps, " ".join(gram),
                )
                i += reps * n
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


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


def _parse_words(text: str, time_offset: float = 0.0) -> list[TranscriptionWord]:
    """Parse word-timestamp output (`hello [T:45] world [T:82]`) into words.

    Tags are centiseconds modulo 1000 relative to the chunk start;
    `time_offset` shifts them to absolute time. Silences are encoded as `_`.
    """
    parts = _TS_RE.split(text)
    # parts = [word0, ts0, word1, ts1, ..., trailing]
    words: list[TranscriptionWord] = []
    last_end = time_offset
    offset = time_offset
    word_start = time_offset
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
    return words


def _segments_from_words(
    words: list[TranscriptionWord], fallback_end: float
) -> list[TranscriptionSegment]:
    if not words:
        return [TranscriptionSegment(id=0, start=0.0, end=fallback_end, text="")]
    return [
        TranscriptionSegment(
            id=0,
            start=words[0].start,
            end=words[-1].end,
            text=" ".join(w.word for w in words),
            words=words,
        )
    ]


def _parse_word_timestamps(text: str) -> list[TranscriptionSegment]:
    """Parse a single word-timestamps output into one segment (test/back-compat)."""
    words = _parse_words(text)
    if not words:
        return [TranscriptionSegment(id=0, start=0.0, end=0.0, text=text.strip())]
    return _segments_from_words(words, fallback_end=words[-1].end)


def _parse_speaker_segments(
    text: str, duration: float, initial_speaker: str | None = None
) -> list[TranscriptionSegment]:
    """Parse output of the speaker-attribution prompt: `[Speaker N]: text...`.

    `initial_speaker` seeds the speaker for any leading text before the first
    tag (text continuing the previous chunk's final speaker turn).
    """
    # Split on speaker tags but keep them.
    parts = re.split(r"(\[Speaker\s+\d+\]\s*:)", text)
    # parts may start with leading text before any tag — treat as Speaker 1 fallback.
    segments: list[TranscriptionSegment] = []
    current_speaker: str | None = initial_speaker
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


def _merge_words_and_speakers(
    ts_words: list[TranscriptionWord] | str, spk_text: str
) -> list[TranscriptionSegment]:
    """Combine a word-timestamp pass (parsed words) with a speaker-attribution pass.

    Strategy: walk the speaker-turn word stream and the timestamp word stream
    in lock-step, assigning each timestamp word the speaker label of the
    corresponding turn position. Mismatches are absorbed by advancing the
    timestamp index without advancing speakers — best-effort, but produces a
    coherent transcript.
    """
    if isinstance(ts_words, str):  # convenience for raw single-chunk output
        ts_words = _parse_words(ts_words)
    if not ts_words:
        return [TranscriptionSegment(id=0, start=0.0, end=0.0, text=spk_text.strip())]

    duration = ts_words[-1].end

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
        segments = _segments_from_words(ts_words, fallback_end=duration)
    return segments
