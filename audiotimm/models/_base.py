from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    import torch


class ModelAdapter(ABC):
    """Contract every model adapter must satisfy.

    Adapters are instantiated by ``Classifier.load()`` via the registry's
    ``adapter_factory`` callable.  They receive the :class:`ModelSpec` and a
    device string.  They should do no heavy work (no model load, no download)
    in ``__init__`` — defer that to the first ``predict()`` / ``embed()``
    call so import time stays fast.
    """

    @abstractmethod
    def predict(self, waveform: "torch.Tensor") -> Dict[str, float]:
        """Run inference on a 1-D mono waveform tensor.

        Args:
            waveform: 1-D float32 tensor of shape ``(samples,)`` already
                      resampled to the model's native sample rate.

        Returns:
            ``{label: score}`` dict.  For multi-label models scores are
            sigmoid probabilities in [0, 1]; for single-label models they
            are softmax probabilities summing to 1.
        """

    def embed(self, waveform: "torch.Tensor") -> np.ndarray:
        """Return penultimate-layer embedding as a 1-D numpy array.

        Raises ``NotImplementedError`` if the adapter does not support it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support embed(). "
            "Use a model from Wave M0 (PANNs) or Wave M1+ which expose embeddings."
        )
