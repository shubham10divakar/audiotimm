"""HTS-AT — Hierarchical Token-Semantic Audio Transformer (Wave M1 + M2).

Swin-Transformer-based audio tagger with hierarchical token merging and a
token-semantic module for event localization.  Dual role:
  - standalone AudioSet tagger (Wave M1)
  - audio encoder inside LAION-CLAP (Wave M2, reused automatically)

Checkpoints are from RetroCirce/HTS-Audio-Transformer (Apache 2.0).
The model architecture code is adapted from that repository.

Reference: Chen et al. (2022) "HTS-AT: A Hierarchical Token-Semantic Audio
           Transformer for Sound Classification and Detection."
           https://arxiv.org/abs/2202.00874
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audiotimm.models._base import ModelAdapter
from audiotimm.utils.download import download_file, get_cache_dir

if TYPE_CHECKING:
    from audiotimm.core.registry import ModelSpec


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

_CHECKPOINT_MAP: dict[str, dict] = {
    "htsat-audioset": {
        "url": "https://github.com/RetroCirce/HTS-Audio-Transformer/releases/download/v0.1/HTSAT_AudioSet_Saved_2.ckpt",
        "n_classes": 527,
        "task": "tagging",
        "map": 0.471,
        "desc": "HTS-AT AudioSet 527, mAP ≈ 0.471",
    },
    "htsat-esc50": {
        "url": "https://github.com/RetroCirce/HTS-Audio-Transformer/releases/download/v0.1/HTSAT_ESC_Saved_2.ckpt",
        "n_classes": 50,
        "task": "tagging",
        "map": None,
        "desc": "HTS-AT ESC-50, acc ≈ 97.0%",
    },
    "htsat-speechcommands": {
        "url": "https://github.com/RetroCirce/HTS-Audio-Transformer/releases/download/v0.1/HTSAT_SPC_Saved_1.ckpt",
        "n_classes": 35,
        "task": "tagging",
        "map": None,
        "desc": "HTS-AT Google Speech Commands v2, 35 classes",
    },
    "htsat-desed": {
        "url": "https://github.com/RetroCirce/HTS-Audio-Transformer/releases/download/v0.1/HTSAT_DESED_Saved_1.ckpt",
        "n_classes": 10,
        "task": "tagging",
        "map": None,
        "desc": "HTS-AT DESED sound event detection, 10 classes",
    },
}


# ---------------------------------------------------------------------------
# Architecture  (adapted from RetroCirce/HTS-Audio-Transformer — Apache 2.0)
# ---------------------------------------------------------------------------

def _drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    rand = torch.rand(shape, dtype=x.dtype, device=x.device)
    rand = torch.floor(rand + keep_prob)
    return x / keep_prob * rand


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(256, 64), patch_size=(4, 4), in_chans=1, embed_dim=96) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, wH, wW)
        coords_flat = torch.flatten(coords, 1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size[0] - 1
        relative[:, :, 1] += window_size[1] - 1
        relative[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        ).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            attn = attn.view(B_ // mask.shape[0], mask.shape[0], self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int,
                 window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, (window_size, window_size), num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            slices_h = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            slices_w = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for h in slices_h:
                for w in slices_w:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = _window_partition(img_mask, window_size).view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x = _window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        x = self.attn(x, mask=self.attn_mask)
        x = _window_reverse(x.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class BasicLayer(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int,
                 num_heads: int, window_size: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path=0.0, downsample=None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x


class HTSAT(nn.Module):
    """HTS-AT: Hierarchical Token-Semantic Audio Transformer.

    Adapted from RetroCirce/HTS-Audio-Transformer (Apache 2.0).
    Default config matches the AudioSet checkpoint.
    """

    def __init__(
        self,
        spec_size: int = 256,
        patch_size: int = 4,
        patch_stride: Tuple[int, int] = (4, 4),
        in_chans: int = 1,
        num_classes: int = 527,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (4, 8, 16, 32),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        mel_bins: int = 64,
        sample_rate: int = 32000,
        window_length: int = 1024,
        hop_length: int = 320,
        fmin: int = 50,
        fmax: int = 14000,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.mel_bins = mel_bins
        self.sample_rate = sample_rate

        # Mel spectrogram front-end — uses torchlibrosa like PANNs
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_length, hop_length=hop_length, win_length=window_length,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_length, n_mels=mel_bins,
            fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True,
        )
        self.bn0 = nn.BatchNorm2d(mel_bins)

        freq_size = mel_bins // patch_stride[1]
        time_size = spec_size // patch_stride[0]
        patches_resolution = (time_size, freq_size)

        self.patch_embed = PatchEmbed(
            img_size=(spec_size, mel_bins),
            patch_size=(patch_stride[0], patch_stride[1]),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer, (d, h) in enumerate(zip(depths, num_heads)):
            res = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=res,
                depth=d, num_heads=h, window_size=window_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if i_layer < len(depths) - 1 else None,
            )
            self.layers.append(layer)

        num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_features, num_classes)

        self.tscam_conv = nn.Conv2d(
            num_features, num_classes, kernel_size=(patches_resolution[0] // (2 ** (len(depths) - 1)), 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, samples)
        x = self.spectrogram_extractor(x)   # (B, 1, T, F)
        x = self.logmel_extractor(x)        # (B, 1, T, mel)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        features = self.forward_features(x)      # (B, L, C)
        embedding = self.avgpool(features.transpose(1, 2)).squeeze(-1)  # (B, C)
        logits = self.head(embedding)
        clipwise = torch.sigmoid(logits)
        return clipwise, embedding


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class HTSATAdapter(ModelAdapter):
    def __init__(self, spec: "ModelSpec", device: str = "cpu") -> None:
        self._spec = spec
        self._device = device
        self._model: Optional[HTSAT] = None
        self._labels: Optional[List[str]] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        meta = _CHECKPOINT_MAP[self._spec.name]
        url = meta["url"]
        filename = url.split("/")[-1]
        cache_path = get_cache_dir() / filename
        download_file(url, cache_path, desc=f"{self._spec.name} weights")

        ckpt = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Strip Lightning-style "net." prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            key = k
            for prefix in ("net.", "model."):
                if k.startswith(prefix):
                    key = k[len(prefix):]
                    break
            cleaned[key] = v

        model = HTSAT(num_classes=meta["n_classes"])
        model.load_state_dict(cleaned, strict=False)
        model.to(self._device)
        model.eval()

        self._model = model

        if meta["n_classes"] == 527:
            from audiotimm.models.panns import get_audioset_labels
            self._labels = get_audioset_labels()

    def predict(self, waveform: torch.Tensor) -> Dict[str, float]:
        self._ensure_loaded()
        with torch.no_grad():
            x = torch.as_tensor(waveform, dtype=torch.float32, device=self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            clipwise, _ = self._model(x)
            scores = clipwise[0].cpu().numpy()
        if self._labels:
            return {self._labels[i]: float(scores[i]) for i in range(len(self._labels))}
        return {str(i): float(scores[i]) for i in range(len(scores))}

    def embed(self, waveform: torch.Tensor) -> np.ndarray:
        self._ensure_loaded()
        with torch.no_grad():
            x = torch.as_tensor(waveform, dtype=torch.float32, device=self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            _, embedding = self._model(x)
        return embedding[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def _register() -> None:
    from audiotimm.core.registry import ModelSpec, registry

    for name, meta in _CHECKPOINT_MAP.items():
        registry.register(
            ModelSpec(
                name=name,
                family="htsat",
                adapter_factory=HTSATAdapter,
                checkpoint=meta["url"],
                sample_rate=32000,
                n_classes=meta["n_classes"],
                embed_dim=768,
                task=meta["task"],
                wave="M1",
                description=meta["desc"],
            )
        )


_register()
