from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from audiotimm.core.registry import registry
from audiotimm.core.result import BatchResult, PredictionResult

_DEFAULT_MODEL = "panns-cnn14"


class Classifier:
    """Main entry-point for audio classification.

    Usage::

        clf = Classifier.load()                  # default: panns-cnn14
        clf = Classifier.load("panns-cnn14-16k")
        clf = Classifier.load("ast-10-10")       # requires [transformers]

        result = clf.predict("dog.wav")
        result.top(5)
        result.label
        result.scores

        results = clf.predict(["a.wav", "b.wav"])   # batch
        results.labels()

        emb = clf.embed("dog.wav")   # np.ndarray (embed_dim,)
    """

    def __init__(self, adapter, spec) -> None:
        self._adapter = adapter
        self._spec = spec

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        name: Optional[str] = None,
        device: str = "cpu",
    ) -> "Classifier":
        """Load a model by zoo-id.

        Args:
            name:   Model id from the registry, e.g. ``"panns-cnn14"``.
                    Pass ``None`` (default) for the recommended default.
            device: ``"cpu"`` or ``"cuda"`` / ``"cuda:0"``.

        Returns:
            A ready-to-use :class:`Classifier`.

        Raises:
            ValueError:  Unknown model name.
            ImportError: Required optional extra not installed.
        """
        name = name or _DEFAULT_MODEL
        spec = registry.get(name)
        if spec.extra:
            try:
                adapter = spec.adapter_factory(spec, device=device)
            except ImportError as exc:
                raise ImportError(
                    f"Model {name!r} requires optional extras.\n"
                    f"Install with: pip install audiotimm[{spec.extra}]"
                ) from exc
        else:
            adapter = spec.adapter_factory(spec, device=device)
        return cls(adapter, spec)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        audio: Union[str, List[str]],
    ) -> Union[PredictionResult, BatchResult]:
        """Classify one file or a list of files.

        Args:
            audio: Path string or list of path strings.

        Returns:
            :class:`PredictionResult` for a single file,
            :class:`BatchResult` for a list.
        """
        if isinstance(audio, list):
            return BatchResult([self._predict_one(path) for path in audio])
        return self._predict_one(audio)

    def embed(
        self,
        audio: Union[str, List[str]],
    ) -> np.ndarray:
        """Return penultimate-layer embeddings.

        Args:
            audio: Path string or list of path strings.

        Returns:
            ``np.ndarray`` of shape ``(embed_dim,)`` or ``(N, embed_dim)``.
        """
        if isinstance(audio, list):
            return np.stack([self._embed_one(path) for path in audio])
        return self._embed_one(audio)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_one(self, path: str) -> PredictionResult:
        waveform = _load_waveform(path, self._spec.sample_rate)
        scores_dict = self._adapter.predict(waveform)
        return PredictionResult(scores=scores_dict, model=self._spec.name, file=path)

    def _embed_one(self, path: str) -> np.ndarray:
        waveform = _load_waveform(path, self._spec.sample_rate)
        return self._adapter.embed(waveform)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._spec.name

    @property
    def sample_rate(self) -> int:
        return self._spec.sample_rate

    @property
    def n_classes(self) -> int:
        return self._spec.n_classes

    @property
    def embed_dim(self) -> int:
        return self._spec.embed_dim

    def __repr__(self) -> str:
        return (
            f"Classifier(model={self._spec.name!r}, "
            f"sr={self._spec.sample_rate}, "
            f"classes={self._spec.n_classes})"
        )


# ---------------------------------------------------------------------------
# Thin audio-loading shim — avoids importing torchaudio at module level
# ---------------------------------------------------------------------------

def _load_waveform(path: str, target_sr: int):
    from audiotimm.utils.audio import load_audio
    waveform, _ = load_audio(path, target_sr=target_sr)
    return waveform
