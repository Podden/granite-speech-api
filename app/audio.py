"""Audio loading helpers — decode any common format to mono 16 kHz float tensor.

Decoding chain:
    1. soundfile (libsndfile)        — wav, flac, ogg, opus, …
    2. audioread + ffmpeg fallback   — m4a/aac, edge-case mp3, anything else ffmpeg knows.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.config import settings

log = logging.getLogger(__name__)

TARGET_SR = 16_000

# 500 ms frames, gain smoothed over 31 frames (~15 s), at most 10x (+20 dB) so
# room noise in long pauses is not blown up to speech level.
_DYNAUDNORM = "dynaudnorm=f=500:g=31:m=10:p=0.95"


def normalize_loudness(wav: torch.Tensor) -> torch.Tensor:
    """Dynamic loudness normalization of a mono 16 kHz [1, n] float tensor.

    Returns the input unchanged (with a warning) when ffmpeg fails, so a
    missing/broken ffmpeg degrades quality instead of failing the request.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "f32le", "-ar", str(TARGET_SR), "-ac", "1", "-i", "pipe:0",
                "-af", _DYNAUDNORM,
                "-f", "f32le", "-ar", str(TARGET_SR), "-ac", "1", "pipe:1",
            ],
            input=wav[0].contiguous().numpy().astype(np.float32).tobytes(),
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning("loudness normalization skipped: %s", exc)
        return wav
    out = torch.from_numpy(np.frombuffer(proc.stdout, dtype=np.float32).copy())
    n = wav.shape[-1]
    # dynaudnorm keeps the length; pad/trim guards timestamps against an off-by-a-few.
    if out.shape[0] < n:
        out = torch.nn.functional.pad(out, (0, n - out.shape[0]))
    return out[:n].unsqueeze(0)


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


def load_audio_bytes(data: bytes, normalize: bool | None = None) -> tuple[torch.Tensor, float]:
    """Decode `data` to a mono 16 kHz float32 tensor of shape [1, num_samples].

    Falls back from libsndfile to audioread+ffmpeg for formats libsndfile
    cannot read (e.g. AAC inside .m4a containers). `normalize` (default
    ``settings.normalize_audio``) applies `normalize_loudness`.
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

    if settings.normalize_audio if normalize is None else normalize:
        wav = normalize_loudness(wav)

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


def concat_audio(parts: list[bytes]) -> tuple[bytes, float]:
    """Decode `parts` and join them, in order, into one 16 kHz mono PCM16 WAV.

    Used when a recording was split into several files (e.g. an interrupted
    meeting): the pipeline then sees one continuous audio, so timestamps run
    through and diarization keeps speaker labels consistent across parts.
    """
    # Not normalized here: every consumer re-decodes (and normalizes) the joined WAV.
    wavs = [load_audio_bytes(p, normalize=False)[0] for p in parts]
    wav = torch.cat(wavs, dim=-1)
    buf = io.BytesIO()
    sf.write(buf, wav[0].numpy(), TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue(), wav.shape[-1] / TARGET_SR


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None
