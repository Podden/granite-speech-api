"""Audio loading helpers — decode any common format to mono 16 kHz float tensor.

Decoding chain:
    1. soundfile (libsndfile)        — wav, flac, ogg, opus, …
    2. audioread + ffmpeg fallback   — m4a/aac, edge-case mp3, anything else ffmpeg knows.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio


TARGET_SR = 16_000


def _try_audioread(data: bytes) -> tuple[np.ndarray, int]:
    """Decode arbitrary audio via audioread (which dispatches to ffmpeg / gstreamer)."""
    import audioread

    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(data)
        with audioread.audio_open(path) as src:
            sr = src.samplerate
            channels = src.channels
            chunks: list[np.ndarray] = []
            for buf in src:
                chunk = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    chunk = chunk.reshape(-1, channels)
                else:
                    chunk = chunk[:, None]
                chunks.append(chunk)
        if not chunks:
            raise ValueError("audioread produced no samples")
        wav = np.concatenate(chunks, axis=0)
        return wav, sr
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def load_audio_bytes(data: bytes) -> tuple[torch.Tensor, float]:
    """Decode `data` to a mono 16 kHz float32 tensor of shape [1, num_samples].

    Falls back from libsndfile to audioread+ffmpeg for formats libsndfile
    cannot read (e.g. AAC inside .m4a containers).
    """
    wav_np: np.ndarray | None = None
    sr = 0
    sf_err: Exception | None = None
    try:
        wav_np, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        sf_err = exc

    if wav_np is None:
        try:
            wav_np, sr = _try_audioread(data)
        except Exception as exc:
            raise ValueError(
                f"Could not decode audio (libsndfile: {sf_err}; audioread: {exc}). "
                "Make sure ffmpeg is installed for non-PCM formats."
            ) from exc

    wav = torch.from_numpy(wav_np.T.copy())

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

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
        _, dur = load_audio_bytes(data)
        return dur


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None
