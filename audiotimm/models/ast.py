"""AST — Audio Spectrogram Transformer (Wave M1).

Loaded entirely via HuggingFace `transformers`. The feature extractor
handles mel-spectrogram extraction internally, so the adapter receives
a raw 16 kHz mono waveform and passes a numpy array to the extractor.

Reference: Gong et al. (2021) "AST: Audio Spectrogram Transformer."
           https://arxiv.org/abs/2104.01778
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from audiotimm.models._base import ModelAdapter

if TYPE_CHECKING:
    from audiotimm.core.registry import ModelSpec


# ---------------------------------------------------------------------------
# Checkpoint map: zoo-id → HuggingFace repo ID
# ---------------------------------------------------------------------------

_CHECKPOINT_MAP: dict[str, dict] = {
    "ast-10-10": {
        "repo": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "map":  0.459,
        "n_classes": 527,
        "desc": "AST patch-stride 10×10, AudioSet 527, mAP ≈ 0.459 — default",
    },
    "ast-10-10-v2": {
        "repo": "MIT/ast-finetuned-audioset-10-10-0.450",
        "map":  0.450,
        "n_classes": 527,
        "desc": "AST patch-stride 10×10 v2, AudioSet 527, mAP ≈ 0.450",
    },
    "ast-10-10-v3": {
        "repo": "MIT/ast-finetuned-audioset-10-10-0.448",
        "map":  0.448,
        "n_classes": 527,
        "desc": "AST patch-stride 10×10 v3, AudioSet 527, mAP ≈ 0.448",
    },
    "ast-12-12": {
        "repo": "MIT/ast-finetuned-audioset-12-12-0.447",
        "map":  0.447,
        "n_classes": 527,
        "desc": "AST patch-stride 12×12, AudioSet 527, mAP ≈ 0.447",
    },
    "ast-14-14": {
        "repo": "MIT/ast-finetuned-audioset-14-14-0.443",
        "map":  0.443,
        "n_classes": 527,
        "desc": "AST patch-stride 14×14, AudioSet 527, mAP ≈ 0.443",
    },
    "ast-16-16": {
        "repo": "MIT/ast-finetuned-audioset-16-16-0.442",
        "map":  0.442,
        "n_classes": 527,
        "desc": "AST patch-stride 16×16, AudioSet 527, mAP ≈ 0.442 — fastest",
    },
    "ast-speechcommands": {
        "repo": "MIT/ast-finetuned-speech-commands-v2",
        "map":  None,
        "n_classes": 35,
        "desc": "AST fine-tuned on Google Speech Commands v2, 35 keyword classes",
    },
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ASTAdapter(ModelAdapter):
    """Lazy-loading adapter for AST checkpoints via HuggingFace transformers."""

    def __init__(self, spec: "ModelSpec", device: str = "cpu") -> None:
        self._spec = spec
        self._device = device
        self._model = None
        self._extractor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
        except ImportError as exc:
            raise ImportError(
                "AST requires the transformers library.\n"
                "Install with: pip install audiotimm[transformers]"
            ) from exc

        repo = _CHECKPOINT_MAP[self._spec.name]["repo"]
        print(f"Loading {self._spec.name} from {repo} …")

        self._extractor = ASTFeatureExtractor.from_pretrained(repo)
        self._model = ASTForAudioClassification.from_pretrained(repo)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, waveform: torch.Tensor) -> Dict[str, float]:
        self._ensure_loaded()

        # ASTFeatureExtractor expects a 1-D numpy float array at 16 kHz
        audio_np: np.ndarray = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform

        inputs = self._extractor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits   # (1, n_classes)

        # AudioSet is multi-label → sigmoid; Speech Commands is single-label → softmax
        if self._spec.n_classes == 527:
            scores = torch.sigmoid(logits[0]).cpu().numpy()
        else:
            scores = torch.softmax(logits[0], dim=-1).cpu().numpy()

        id2label: Dict[int, str] = self._model.config.id2label
        return {id2label[i]: float(scores[i]) for i in range(len(scores))}

    def embed(self, waveform: torch.Tensor) -> np.ndarray:
        """Return the CLS-token embedding from the last hidden state."""
        self._ensure_loaded()

        audio_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        inputs = self._extractor(audio_np, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            # last hidden state CLS token: (1, seq_len, hidden) → (hidden,)
            cls_emb = outputs.hidden_states[-1][:, 0, :]
        return cls_emb[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

def _register() -> None:
    from audiotimm.core.registry import ModelSpec, registry

    for name, meta in _CHECKPOINT_MAP.items():
        registry.register(
            ModelSpec(
                name=name,
                family="ast",
                adapter_factory=ASTAdapter,
                checkpoint=meta["repo"],
                sample_rate=16000,
                n_classes=meta["n_classes"],
                embed_dim=768,
                task="tagging",
                wave="M1",
                extra="transformers",
                description=meta["desc"],
            )
        )


_register()
