"""Loudness normalization: a quiet stretch between loud ones gets lifted."""

from __future__ import annotations

import shutil

import pytest
import torch

from app.audio import TARGET_SR, normalize_loudness


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_quiet_middle_is_lifted_and_length_kept() -> None:
    t = torch.arange(60 * TARGET_SR) / TARGET_SR
    tone = torch.sin(2 * torch.pi * 220 * t)
    gain = torch.full_like(t, 0.5)
    gain[20 * TARGET_SR : 40 * TARGET_SR] = 0.02  # phone put down: ~-28 dB
    wav = (tone * gain).unsqueeze(0)

    out = normalize_loudness(wav)

    assert out.shape == wav.shape
    mid = slice(25 * TARGET_SR, 35 * TARGET_SR)
    before = wav[0, mid].abs().max()
    after = out[0, mid].abs().max()
    assert after > 5 * before
