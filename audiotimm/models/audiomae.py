"""AudioMAE — Masked Autoencoders that Listen (Wave M1).

Self-supervised ViT-Base model pretrained with masked spectrogram modeling,
then fine-tuned on AudioSet.  Weights are loaded from HuggingFace mirrors
(gaunernst/vit_base_patch16_1024_128.audiomae_*) using the timm ViT API.

Reference: Huang et al. (2022) "Masked Autoencoders that Listen."
           NeurIPS 2022. https://arxiv.org/abs/2207.06405
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audiotimm.models._base import ModelAdapter
from audiotimm.utils.download import get_cache_dir

if TYPE_CHECKING:
    from audiotimm.core.registry import ModelSpec


# ---------------------------------------------------------------------------
# Checkpoint registry (HuggingFace mirrors by gaunernst)
# ---------------------------------------------------------------------------

_CHECKPOINT_MAP: dict[str, dict] = {
    "audiomae-base-as2m": {
        "hf_repo": "gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
        "n_classes": 0,
        "task": "embed",
        "map": None,
        "desc": "AudioMAE ViT-Base SSL pretrained on AudioSet-2M — embedding backbone",
    },
    "audiomae-base-as20k": {
        "hf_repo": "gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k",
        "n_classes": 527,
        "task": "tagging",
        "map": 0.370,
        "desc": "AudioMAE ViT-Base fine-tuned AS2M→AS20K, mAP ≈ 0.370",
    },
}


# ---------------------------------------------------------------------------
# ViT architecture matching AudioMAE's patch16 / 1024-frame / 128-mel layout
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (1024, 128),
                 patch_size: int = 16, in_chans: int = 1,
                 embed_dim: int = 768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, C)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop(F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AudioMAEViT(nn.Module):
    """ViT-Base matching AudioMAE's 1024×128 patch-16 spectrogram layout."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (1024, 128),
        patch_size: int = 16,
        in_chans: int = 1,
        num_classes: int = 0,       # 0 = no head (SSL encoder)
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)      # (B, N+1, C)
        embedding = features[:, 0]               # CLS token
        logits = self.head(embedding)
        return logits, embedding


# ---------------------------------------------------------------------------
# Mel spectrogram preprocessing (16 kHz, 128 mel bins, matching AudioMAE)
# ---------------------------------------------------------------------------

def _compute_fbank(waveform: torch.Tensor, target_length: int = 1024) -> torch.Tensor:
    """Compute log-mel filterbank features matching AudioMAE preprocessing."""
    try:
        import torchaudio
    except ImportError:
        raise ImportError("torchaudio is required for AudioMAE preprocessing.")

    fbanks = []
    for wav in waveform:
        wav = wav.unsqueeze(0) * 2 ** 15
        fbank = torchaudio.compliance.kaldi.fbank(
            wav, num_mel_bins=128, sample_frequency=16000,
            frame_length=25, frame_shift=10,
        )
        # Pad or trim to target_length time frames
        n = fbank.shape[0]
        if n < target_length:
            fbank = F.pad(fbank, (0, 0, 0, target_length - n))
        else:
            fbank = fbank[:target_length]
        fbank = (fbank - fbank.mean()) / (fbank.std() + 1e-8)
        fbanks.append(fbank)

    return torch.stack(fbanks).unsqueeze(1)  # (B, 1, T, 128)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class AudioMAEAdapter(ModelAdapter):
    def __init__(self, spec: "ModelSpec", device: str = "cpu") -> None:
        self._spec = spec
        self._device = device
        self._model: Optional[AudioMAEViT] = None
        self._labels: Optional[List[str]] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for AudioMAE.\n"
                "Install with: pip install audiotimm  (it is a core dep)"
            ) from exc

        meta = _CHECKPOINT_MAP[self._spec.name]
        hf_repo = meta["hf_repo"]
        n_classes = meta["n_classes"]

        print(f"Loading {self._spec.name} from {hf_repo} …")

        model = AudioMAEViT(num_classes=n_classes)

        # Download state dict from HF Hub
        cache_dir = str(get_cache_dir() / "audiomae")
        ckpt_path = hf_hub_download(
            repo_id=hf_repo,
            filename="model.safetensors",
            cache_dir=cache_dir,
        )

        try:
            from safetensors.torch import load_file as safetensors_load
            state_dict = safetensors_load(ckpt_path)
        except ImportError:
            # Fallback: try .bin
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename="pytorch_model.bin",
                                         cache_dir=cache_dir)
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        model.load_state_dict(state_dict, strict=False)
        model.to(self._device)
        model.eval()
        self._model = model

        if n_classes == 527:
            from audiotimm.models.panns import get_audioset_labels
            self._labels = get_audioset_labels()

    def predict(self, waveform: torch.Tensor) -> Dict[str, float]:
        self._ensure_loaded()
        if self._spec.n_classes == 0:
            raise RuntimeError(
                f"{self._spec.name!r} is an SSL encoder — call .embed() instead."
            )
        fbank = _compute_fbank(
            waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
        ).to(self._device)
        with torch.no_grad():
            logits, _ = self._model(fbank)
            scores = torch.sigmoid(logits[0]).cpu().numpy()
        return {self._labels[i]: float(scores[i]) for i in range(len(self._labels))}

    def embed(self, waveform: torch.Tensor) -> np.ndarray:
        self._ensure_loaded()
        fbank = _compute_fbank(
            waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
        ).to(self._device)
        with torch.no_grad():
            _, embedding = self._model(fbank)
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
                family="audiomae",
                adapter_factory=AudioMAEAdapter,
                checkpoint=meta["hf_repo"],
                sample_rate=16000,
                n_classes=meta["n_classes"],
                embed_dim=768,
                task=meta["task"],
                wave="M1",
                description=meta["desc"],
            )
        )


_register()
