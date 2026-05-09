"""Audio loading helpers — decode any common format to mono 16 kHz float tensor."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import torch
import torchaudio


TARGET_SR = 16_000


def load_audio_bytes(data: bytes) -> tuple[torch.Tensor, float]:
    """Decode `data` (any libsndfile-supported format) to mono 16 kHz tensor.

    Returns (tensor of shape [1, num_samples], duration_seconds).
    """
    try:
        wav_np, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc

    # shape [num_samples, num_channels] -> [num_channels, num_samples]
    wav = torch.from_numpy(wav_np.T.copy())

    # Mono down-mix
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample to 16 kHz
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    duration = wav.shape[-1] / TARGET_SR
    return wav, float(duration)


def get_duration(data: bytes) -> float:
    """Return audio duration in seconds without full decoding when possible."""
    try:
        info = sf.info(io.BytesIO(data))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        # fall back to full decode
        _, dur = load_audio_bytes(data)
        return dur
