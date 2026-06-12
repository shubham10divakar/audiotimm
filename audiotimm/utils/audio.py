from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def load_audio(
    path: str | Path,
    target_sr: int = 32000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load an audio file, resample, and optionally mix to mono.

    Args:
        path:      Path to any audio format supported by torchaudio
                   (wav, mp3, flac, ogg, m4a, …).
        target_sr: Target sample rate in Hz.
        mono:      If True, average all channels into one.

    Returns:
        ``(waveform, sample_rate)`` where *waveform* is a 1-D float32
        tensor of shape ``(samples,)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError:      If torchaudio cannot decode the file.
    """
    import torchaudio  # deferred so the package imports fast without torchaudio

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))  # (channels, samples)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sr
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0), target_sr  # (samples,), sr


def pad_or_trim(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Pad with zeros or trim a 1-D waveform to exactly *length* samples."""
    n = waveform.shape[-1]
    if n >= length:
        return waveform[..., :length]
    pad = length - n
    return torch.nn.functional.pad(waveform, (0, pad))
