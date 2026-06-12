"""YAMNet — Wave M0 lightweight tagger stub.

YAMNet is a MobileNetV1-based model trained on AudioSet 521 classes by Google.
The canonical weights live on TF-Hub; a pure-PyTorch / ONNX path is planned
for Wave M0 Phase 1.1.

This module registers the model in the zoo so users get a helpful error
message rather than a KeyError.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

from audiotimm.models._base import ModelAdapter

if TYPE_CHECKING:
    from audiotimm.core.registry import ModelSpec


class YAMNetAdapter(ModelAdapter):
    """Stub adapter — raises NotImplementedError with install instructions."""

    def __init__(self, spec: "ModelSpec", device: str = "cpu") -> None:
        raise NotImplementedError(
            "YAMNet is not yet available as a native PyTorch model.\n"
            "Options:\n"
            "  1. Use 'panns-cnn14' (similar accuracy, pure PyTorch, default).\n"
            "  2. TF-Hub path: pip install tensorflow tensorflow-hub  (coming in Phase 1.1).\n"
            "  3. ONNX path: planned for Phase 1.1.\n"
        )

    def predict(self, waveform: torch.Tensor) -> Dict[str, float]:
        raise NotImplementedError

    def embed(self, waveform: torch.Tensor) -> np.ndarray:
        raise NotImplementedError


def _register() -> None:
    from audiotimm.core.registry import ModelSpec, registry

    registry.register(
        ModelSpec(
            name="yamnet",
            family="yamnet",
            adapter_factory=YAMNetAdapter,
            checkpoint="https://tfhub.dev/google/yamnet/1",
            sample_rate=16000,
            n_classes=521,
            embed_dim=1024,
            task="tagging",
            wave="M0",
            description=(
                "YAMNet — Google MobileNetV1 AudioSet-521 tagger (TF-Hub). "
                "Pure-PyTorch/ONNX support coming in Phase 1.1."
            ),
        )
    )


_register()
