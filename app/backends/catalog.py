"""Model catalog — maps user-supplied model ids to backend families.

Granite stays the default family (with its base→plus auto-upgrade); external
families (Cohere Transcribe, Qwen3-ASR) are matched by id/alias and passed
through unchanged.
"""

from __future__ import annotations

from app.backends.granite import GRANITE_MODELS, normalize_model_id as _granite_normalize

COHERE_TRANSCRIBE = "CohereLabs/cohere-transcribe-03-2026"
# The "-hf" checkpoint is the transformers-native conversion; the original
# repo requires Qwen's qwen-asr package (pins an incompatible transformers).
QWEN3_ASR = "Qwen/Qwen3-ASR-1.7B-hf"

# Multi-pass pipeline: Cohere text + Qwen forced-aligner timestamps
# (+ pyannote turns when speaker attribution is requested).
FUSION = "fusion"

EXTERNAL_MODELS = {COHERE_TRANSCRIBE, QWEN3_ASR, FUSION}

_EXTERNAL_ALIASES = {
    "fusion": FUSION,
    "auto": FUSION,
    "cohere-transcribe": COHERE_TRANSCRIBE,
    "cohere-transcribe-03-2026": COHERE_TRANSCRIBE,
    "coherelabs/cohere-transcribe-03-2026": COHERE_TRANSCRIBE,
    "qwen3-asr": QWEN3_ASR,
    "qwen3-asr-1.7b": QWEN3_ASR,
    "qwen/qwen3-asr-1.7b": QWEN3_ASR,
    "qwen/qwen3-asr-1.7b-hf": QWEN3_ASR,
}

ALL_MODELS = GRANITE_MODELS | EXTERNAL_MODELS


def resolve_model_id(model: str | None, *, want_plus_features: bool) -> str:
    """Return a fully-qualified model id for any supported family."""
    if model:
        ext = _EXTERNAL_ALIASES.get(model.strip().lower())
        if ext:
            return ext
    return _granite_normalize(model, want_plus_features=want_plus_features)


def is_granite(model_id: str) -> bool:
    return model_id in GRANITE_MODELS


def supports_diarization(model_id: str) -> bool:
    """Models that produce word timestamps to reconcile pyannote turns with."""
    return model_id in GRANITE_MODELS or model_id == FUSION
