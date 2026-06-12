"""BEATs — Bootstrapped Audio Transformer (Wave M1).

Self-supervised audio transformer from Microsoft with bootstrapped acoustic
tokenization. Fine-tuned checkpoints achieve SOTA mAP ≈ 0.486 on AudioSet.

The architecture is implemented here directly (Apache 2.0, adapted from
microsoft/unilm — full attribution in the header of each class) so no
external model-code dependency is required. Weights are downloaded from
Microsoft's Azure blob storage and cached at ~/.cache/audiotimm/.

Reference: Chen et al. (2022) "BEATs: Audio Pre-Training with Acoustic
           Tokenizers." https://arxiv.org/abs/2212.09058
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
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
# BEATs Config  (from microsoft/unilm — Apache 2.0)
# ---------------------------------------------------------------------------

@dataclass
class BEATsConfig:
    input_patch_size: int = 16
    embed_dim: int = 512
    conv_bias: bool = False

    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12

    activation_fn: str = "gelu"

    layer_wise_gradient_decay_ratio: float = 1.0
    layer_norm_first: bool = False
    deep_norm: bool = False

    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0

    conv_pos: int = 128
    conv_pos_groups: int = 16

    relative_position_embedding: bool = True
    num_buckets: int = 320
    max_distance: int = 800
    gru_rel_pos: bool = True

    finetuned_model: bool = False
    predictor_dropout: float = 0.0
    predictor_class: int = 527


# ---------------------------------------------------------------------------
# Architecture building blocks  (from microsoft/unilm — Apache 2.0)
# ---------------------------------------------------------------------------

class SamePad(nn.Module):
    def __init__(self, kernel_size: int, causal: bool = False) -> None:
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class GluLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class BEATsRelativePositionBias(nn.Module):
    """Relative position biases with bucket-based discretisation."""

    def __init__(self, num_heads: int, num_buckets: int = 320, max_distance: int = 800) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        n = -relative_position
        num_buckets //= 2
        ret = (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_len: int, key_len: int) -> torch.Tensor:
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(query_len, dtype=torch.long, device=device)
        k_pos = torch.arange(key_len, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]   # (q, k)
        rel_pos_bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rel_pos_bucket)  # (q, k, heads)
        return values.permute(2, 0, 1).unsqueeze(0)            # (1, heads, q, k)


class BEATsMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 320,
        max_distance: int = 800,
        gru_rel_pos: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.rel_pos_bias = BEATsRelativePositionBias(
                num_heads, num_buckets, max_distance
            )

        self.gru_rel_pos = gru_rel_pos
        if gru_rel_pos:
            self.grep_linear = nn.Linear(self.head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = query.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(query).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(key).reshape(B, -1, H, D).transpose(1, 2)
        v = self.v_proj(value).reshape(B, -1, H, D).transpose(1, 2)

        q = q * self.scaling
        attn = torch.matmul(q, k.transpose(-2, -1))   # (B, H, T, S)

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.rel_pos_bias(T, k.shape[2])
        if position_bias is not None:
            if self.gru_rel_pos:
                query_layer = q.permute(0, 2, 1, 3).reshape(B * T, H, D)
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer).reshape(B, T, H, 8).permute(0, 2, 1, 3).chunk(2, dim=-1)
                )
                gate = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                position_bias = position_bias.unsqueeze(0) * gate   # broadcast along batch
                position_bias = position_bias.view(B * H, T, -1)
                attn = attn.view(B * H, T, -1) + position_bias
                attn = attn.view(B, H, T, -1)
            else:
                attn = attn + position_bias

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        x = torch.matmul(attn, v)                      # (B, H, T, D)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.out_proj(x)
        return x, position_bias


class BEATsFeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, activation_fn: str = "gelu",
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = dropout
        self.act = nn.GELU() if activation_fn == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.dropout(self.act(self.fc1(x)), p=self.dropout, training=self.training))


class BEATsTransformerLayer(nn.Module):
    def __init__(self, cfg: BEATsConfig, has_relative_attention_bias: bool = False) -> None:
        super().__init__()
        self.self_attn = BEATsMultiheadAttention(
            embed_dim=cfg.encoder_embed_dim,
            num_heads=cfg.encoder_attention_heads,
            dropout=cfg.attention_dropout,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=cfg.num_buckets,
            max_distance=cfg.max_distance,
            gru_rel_pos=cfg.gru_rel_pos,
        )
        self.ffn = BEATsFeedForward(
            cfg.encoder_embed_dim,
            cfg.encoder_ffn_embed_dim,
            cfg.activation_fn,
            cfg.activation_dropout,
        )
        self.layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)
        self.final_layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)
        self.dropout = cfg.dropout
        self.layer_norm_first = cfg.layer_norm_first

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        if self.layer_norm_first:
            x = self.layer_norm(x)
        x, position_bias = self.self_attn(
            x, x, x,
            key_padding_mask=self_attn_padding_mask,
            position_bias=position_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.layer_norm_first:
            x = self.layer_norm(x)

        residual = x
        if self.layer_norm_first:
            x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.layer_norm_first:
            x = self.final_layer_norm(x)
        return x, position_bias


class BEATsTransformerEncoder(nn.Module):
    def __init__(self, cfg: BEATsConfig) -> None:
        super().__init__()
        self.dropout = cfg.dropout
        self.layers = nn.ModuleList([
            BEATsTransformerLayer(cfg, has_relative_attention_bias=(i == 0))
            for i in range(cfg.encoder_layers)
        ])
        self.layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)
        self.layer_norm_first = cfg.layer_norm_first

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        position_bias = None
        for layer in self.layers:
            x, position_bias = layer(x, self_attn_padding_mask=padding_mask,
                                     position_bias=position_bias)
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x


class BEATsPatchEmbed(nn.Module):
    """2-D spectrogram → patch embeddings (matching original conv-based frontend)."""

    def __init__(self, cfg: BEATsConfig) -> None:
        super().__init__()
        self.patch_size = cfg.input_patch_size
        self.proj = nn.Conv2d(
            1, cfg.embed_dim,
            kernel_size=cfg.input_patch_size,
            stride=cfg.input_patch_size,
            bias=cfg.conv_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        x = self.proj(x)          # (B, embed_dim, T//p, F//p)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, T * F, C)  # (B, seq, embed_dim)
        return x


class BEATsModel(nn.Module):
    """BEATs model — patch embedding + positional conv + transformer encoder.

    Adapted from microsoft/unilm (Apache 2.0).
    """

    def __init__(self, cfg: BEATsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = BEATsPatchEmbed(cfg)

        self.post_extract_proj = (
            nn.Linear(cfg.embed_dim, cfg.encoder_embed_dim)
            if cfg.embed_dim != cfg.encoder_embed_dim
            else None
        )

        self.dropout_input = nn.Dropout(p=cfg.dropout)

        self.pos_conv = nn.Sequential(
            nn.Conv1d(
                cfg.encoder_embed_dim,
                cfg.encoder_embed_dim,
                kernel_size=cfg.conv_pos,
                padding=cfg.conv_pos // 2,
                groups=cfg.conv_pos_groups,
            ),
            SamePad(cfg.conv_pos),
            nn.GELU(),
        )

        self.encoder = BEATsTransformerEncoder(cfg)
        self.layer_norm = nn.LayerNorm(cfg.embed_dim)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(self, features: torch.Tensor, padding_mask: torch.Tensor):
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        return padding_mask.all(-1)

    def preprocess(
        self,
        source: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        """Compute log-mel spectrogram matching BEATs training preprocessing."""
        try:
            import torchaudio
        except ImportError:
            raise ImportError("torchaudio is required for BEATs preprocessing.")

        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15  # match kaldi scale
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
            fbank = (fbank - fbank_mean) / (2 * fbank_std)
            fbanks.append(fbank)

        fbank = torch.stack(fbanks, dim=0)   # (B, T, 128)
        return fbank

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)

        # pad to multiple of patch_size in both dims
        p = self.patch_embedding.patch_size
        T, F = fbank.shape[1], fbank.shape[2]
        pad_T = (p - T % p) % p
        pad_F = (p - F % p) % p
        if pad_T > 0 or pad_F > 0:
            fbank = F.pad(fbank, (0, pad_F, 0, pad_T))

        fbank = fbank.unsqueeze(1)           # (B, 1, T, 128)
        features = self.patch_embedding(fbank)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.encoder(x, padding_mask=padding_mask)
        return x, padding_mask

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, embedding) where embedding is the mean-pooled representation.
        """
        x, padding_mask = self.extract_features(source, padding_mask, fbank_mean, fbank_std)

        # mean pool (respecting padding)
        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0.0
            n = (~padding_mask).float().sum(dim=1, keepdim=True).unsqueeze(-1)
            embedding = x.sum(dim=1) / n.squeeze(1)
        else:
            embedding = x.mean(dim=1)   # (B, encoder_embed_dim)

        if self.predictor is not None:
            x = self.predictor_dropout(embedding)
            logits = self.predictor(x)
        else:
            logits = None

        return logits, embedding


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

_CHECKPOINT_MAP: dict[str, dict] = {
    "beats-iter3plus-as2m-cpt2": {
        "url": "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3+_AS2M_finetuned_on_AS2M_cpt2.pt",
        "map": 0.486,
        "desc": "BEATs iter3+ AS2M fine-tuned cpt2 — SOTA mAP ≈ 0.486",
    },
    "beats-iter3plus-as2m-cpt1": {
        "url": "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3+_AS2M_finetuned_on_AS2M_cpt1.pt",
        "map": 0.485,
        "desc": "BEATs iter3+ AS2M fine-tuned cpt1, mAP ≈ 0.485",
    },
    "beats-iter3-cpt2": {
        "url": "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_finetuned_on_AS2M_cpt2.pt",
        "map": 0.484,
        "desc": "BEATs iter3 fine-tuned cpt2, mAP ≈ 0.484",
    },
    "beats-iter3-cpt1": {
        "url": "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_finetuned_on_AS2M_cpt1.pt",
        "map": 0.484,
        "desc": "BEATs iter3 fine-tuned cpt1, mAP ≈ 0.484",
    },
    "beats-iter3plus-as2m": {
        "url": "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3+_AS2M.pt",
        "map": None,
        "desc": "BEATs iter3+ AS2M — SSL pretrained (no classifier head)",
    },
    "beats-iter3": {
        "url": "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3.pt",
        "map": None,
        "desc": "BEATs iter3 — SSL pretrained (no classifier head)",
    },
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BEATsAdapter(ModelAdapter):
    """Lazy-loading adapter for BEATs checkpoints."""

    def __init__(self, spec: "ModelSpec", device: str = "cpu") -> None:
        self._spec = spec
        self._device = device
        self._model: Optional[BEATsModel] = None
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

        cfg_dict = ckpt.get("cfg", {})
        cfg = BEATsConfig()
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        model = BEATsModel(cfg)
        model.load_state_dict(ckpt["model"])
        model.to(self._device)
        model.eval()

        self._model = model

        # Load AudioSet labels if this is a fine-tuned tagging model
        if cfg.finetuned_model:
            from audiotimm.models.panns import get_audioset_labels
            self._labels = get_audioset_labels()

    def predict(self, waveform: torch.Tensor) -> Dict[str, float]:
        self._ensure_loaded()

        if not self._model.cfg.finetuned_model:
            raise RuntimeError(
                f"Model {self._spec.name!r} is an SSL pretrained checkpoint with no "
                "classifier head. Use a fine-tuned variant, e.g. "
                "'beats-iter3plus-as2m-cpt2', or call .embed() instead."
            )

        with torch.no_grad():
            x = torch.as_tensor(waveform, dtype=torch.float32, device=self._device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits, _ = self._model(x)
            scores = torch.sigmoid(logits[0]).cpu().numpy()

        return {self._labels[i]: float(scores[i]) for i in range(len(self._labels))}

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
        is_ft = "finetuned" in meta["url"] or "cpt" in name
        registry.register(
            ModelSpec(
                name=name,
                family="beats",
                adapter_factory=BEATsAdapter,
                checkpoint=meta["url"],
                sample_rate=16000,
                n_classes=527 if is_ft else 0,
                embed_dim=768,
                task="tagging" if is_ft else "embed",
                wave="M1",
                description=meta["desc"],
            )
        )


_register()
