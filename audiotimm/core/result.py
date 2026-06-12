from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class PredictionResult:
    """Single-file multi-label classification result.

    Immutable. Every model returns one of these so downstream code never
    needs to know which model produced it.
    """

    __slots__ = ("_scores", "_model", "_file")

    def __init__(
        self,
        scores: Dict[str, float],
        model: str,
        file: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "_scores", scores)
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_file", file)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PredictionResult is immutable")

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        """Highest-scoring label."""
        return max(self._scores, key=self._scores.__getitem__)

    @property
    def score(self) -> float:
        """Score of the highest-scoring label."""
        return self._scores[self.label]

    @property
    def scores(self) -> Dict[str, float]:
        """Full {label: score} mapping (copy)."""
        return dict(self._scores)

    @property
    def model(self) -> str:
        return self._model

    @property
    def file(self) -> Optional[str]:
        return self._file

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def top(self, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (label, score) pairs sorted by score descending."""
        return sorted(self._scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def above(self, threshold: float) -> List[Tuple[str, float]]:
        """All (label, score) pairs where score >= threshold, sorted descending."""
        return sorted(
            [(l, s) for l, s in self._scores.items() if s >= threshold],
            key=lambda x: x[1],
            reverse=True,
        )

    def as_dict(self) -> dict:
        return {
            "file": self._file,
            "model": self._model,
            "label": self.label,
            "score": self.score,
            "scores": dict(self._scores),
        }

    def __repr__(self) -> str:
        top3 = self.top(3)
        items = ", ".join(f"{lbl!r}: {s:.3f}" for lbl, s in top3)
        return f"PredictionResult(label={self.label!r}, top3=[{items}])"


class BatchResult(list):
    """List[PredictionResult] with batch-level helpers."""

    def labels(self) -> List[str]:
        return [r.label for r in self]

    def scores(self) -> List[Dict[str, float]]:
        return [r.scores for r in self]

    def top(self, k: int = 5) -> List[List[Tuple[str, float]]]:
        return [r.top(k) for r in self]

    def as_dicts(self) -> List[dict]:
        return [r.as_dict() for r in self]

    def __repr__(self) -> str:
        return f"BatchResult([{', '.join(r.label for r in self)}])"
