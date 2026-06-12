"""PANNs (Pretrained Audio Neural Networks) — Wave M0 CNN taggers.

Implements CNN14 (32 kHz and 16 kHz variants) exactly matching the architecture
in qiuqiangkong/audioset_tagging_cnn so pretrained Zenodo checkpoints load
with strict=True and no key remapping.

Reference: Kong et al. (2020) "PANNs: Large-Scale Pretrained Audio Neural
Networks for Audio Pattern Recognition." IEEE/ACM TASLP.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from urllib.parse import unquote

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audiotimm.models._base import ModelAdapter
from audiotimm.utils.download import download_file, get_cache_dir

if TYPE_CHECKING:
    from audiotimm.core.registry import ModelSpec


# ---------------------------------------------------------------------------
# AudioSet 527-class labels (lazy, downloaded once and cached)
# ---------------------------------------------------------------------------

_LABELS: Optional[List[str]] = None
_LABELS_URL = (
    "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn"
    "/master/metadata/class_labels_indices.csv"
)


def get_audioset_labels() -> List[str]:
    global _LABELS
    if _LABELS is None:
        _LABELS = _download_labels()
    return _LABELS


def _download_labels() -> List[str]:
    dest = get_cache_dir() / "audioset_class_labels_indices.csv"
    download_file(_LABELS_URL, dest, desc="AudioSet class labels")
    labels: List[str] = [""] * 527
    with open(dest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels[int(row["index"])] = row["display_name"]
    return labels


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def _init_layer(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight)
        if getattr(layer, "bias", None) is not None:
            nn.init.zeros_(layer.bias)


def _init_bn(bn: nn.BatchNorm2d) -> None:
    nn.init.zeros_(bn.bias)
    nn.init.ones_(bn.weight)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(
        self,
        x: torch.Tensor,
        pool_size: tuple = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(
                x, kernel_size=pool_size
            )
        return x


class Cnn14(nn.Module):
    """PANNs CNN14 — 14-layer CNN AudioSet tagger.

    Architecture is an exact replica of the original so that Zenodo
    checkpoints load without remapping.  The spectrogram front-end uses
    torchlibrosa which also matches the original parameter layout.
    """

    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        fmin: int,
        fmax: int,
        classes_num: int,
    ) -> None:
        super().__init__()
        from torchlibrosa.augmentation import SpecAugmentation
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)
        _init_bn(self.bn0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Waveform tensor of shape ``(batch, samples)``.

        Returns:
            ``(clipwise_output, embedding)`` both of shape
            ``(batch, 527)`` and ``(batch, 2048)`` respectively.
        """
        x = self.spectrogram_extractor(x)   # (batch, 1, T, F)
        x = self.logmel_extractor(x)        # (batch, 1, T, mel)

        x = x.transpose(1, 3)              # (batch, mel, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)              # (batch, 1, T, mel)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)            # (batch, 2048, T')
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2                         # (batch, 2048)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))

        return clipwise_output, embedding


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

# Zoo-id → (Zenodo download URL, model config overrides)
_CHECKPOINT_MAP: dict[str, tuple[str, dict]] = {
    "panns-cnn14": (
        "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth",
        {},
    ),
    "panns-cnn14-16k": (
        "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
        {"sample_rate": 16000, "window_size": 512, "hop_size": 160, "fmax": 8000},
    ),
}

# Base config for the 32 kHz CNN14
_BASE_CONFIG = dict(
    sample_rate=32000,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527,
)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class PANNsAdapter(ModelAdapter):
    """Lazy-loading adapter for PANNs CNN14 checkpoints."""

    def __init__(self, spec: "ModelSpec", device: str = "cpu") -> None:
        self._spec = spec
        self._device = device
        self._model: Optional[Cnn14] = None
        self._labels: Optional[List[str]] = None

    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        url, overrides = _CHECKPOINT_MAP[self._spec.name]
        cfg = {**_BASE_CONFIG, **overrides}
        model = Cnn14(**cfg)

        # Derive local filename from the URL (handles %3D → = encoding)
        filename = unquote(url.split("/")[-1])
        cache_path = get_cache_dir() / filename
        download_file(url, cache_path, desc=f"{self._spec.name} weights")

        checkpoint = torch.load(
            str(cache_path), map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        model.to(self._device)
        model.eval()

        self._model = model
        self._labels = get_audioset_labels()

    # ------------------------------------------------------------------ #

    def predict(self, waveform: torch.Tensor) -> Dict[str, float]:
        self._ensure_loaded()
        with torch.no_grad():
            x = torch.as_tensor(waveform, dtype=torch.float32, device=self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)                  # (1, samples)
            clipwise, _ = self._model(x)
            scores = clipwise[0].cpu().numpy()      # (527,)
        return {self._labels[i]: float(scores[i]) for i in range(len(self._labels))}

    def embed(self, waveform: torch.Tensor) -> np.ndarray:
        self._ensure_loaded()
        with torch.no_grad():
            x = torch.as_tensor(waveform, dtype=torch.float32, device=self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            _, embedding = self._model(x)
            return embedding[0].cpu().numpy()       # (2048,)


# ---------------------------------------------------------------------------
# Register all PANNs models into the global registry at import time
# ---------------------------------------------------------------------------

def _register() -> None:
    from audiotimm.core.registry import ModelSpec, registry

    for name, (url, overrides) in _CHECKPOINT_MAP.items():
        sr = overrides.get("sample_rate", _BASE_CONFIG["sample_rate"])
        registry.register(
            ModelSpec(
                name=name,
                family="panns",
                adapter_factory=PANNsAdapter,
                checkpoint=url,
                sample_rate=sr,
                n_classes=527,
                embed_dim=2048,
                task="tagging",
                wave="M0",
                description=f"PANNs CNN14 AudioSet tagger ({sr // 1000} kHz). "
                            f"mAP ≈ {0.438 if '16k' in name else 0.431}",
            )
        )


_register()
