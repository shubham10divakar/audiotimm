from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from audiotimm.models._base import ModelAdapter


@dataclass
class ModelSpec:
    """Everything the registry knows about a model entry."""

    name: str
    family: str
    adapter_factory: Callable[..., "ModelAdapter"]
    checkpoint: str          # HF repo-id, Zenodo URL, or local path hint
    sample_rate: int
    n_classes: int
    embed_dim: int
    task: str                # "tagging" | "zero-shot" | "embed" | "asr"
    wave: str                # "M0" | "M1" | "M2" | "M3" | "M4"
    extra: Optional[str] = None   # optional-extra group required, e.g. "transformers"
    description: str = ""


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, ModelSpec] = {}

    def register(self, spec: ModelSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ModelSpec:
        if name not in self._specs:
            available = ", ".join(sorted(self._specs))
            raise ValueError(
                f"Unknown model {name!r}.\n"
                f"Available models: {available}\n"
                f"Use audiotimm.registry.list() to see all models with details."
            )
        return self._specs[name]

    def list(self, wave: Optional[str] = None, task: Optional[str] = None) -> List[ModelSpec]:
        specs = list(self._specs.values())
        if wave:
            specs = [s for s in specs if s.wave == wave]
        if task:
            specs = [s for s in specs if s.task == task]
        return specs

    def list_names(self, wave: Optional[str] = None, task: Optional[str] = None) -> List[str]:
        return [s.name for s in self.list(wave=wave, task=task)]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __repr__(self) -> str:
        return f"ModelRegistry({len(self._specs)} models)"


# Singleton — all model modules register into this object at import time.
registry = ModelRegistry()


def register_model(name: str):
    """Class decorator for registering a custom model adapter.

    The decorated class must implement a classmethod ``spec() -> ModelSpec``
    returning a filled ModelSpec (name field is overridden by decorator).

    Example::

        from audiotimm import register_model
        from audiotimm.models._base import ModelAdapter

        @register_model("my-bird-net")
        class BirdNet(ModelAdapter):
            @classmethod
            def spec(cls):
                from audiotimm.core.registry import ModelSpec
                return ModelSpec(
                    name="",           # filled by decorator
                    family="custom",
                    adapter_factory=cls,
                    checkpoint="local",
                    sample_rate=22050,
                    n_classes=500,
                    embed_dim=512,
                    task="tagging",
                    wave="M0",
                )

            def predict(self, waveform):
                ...
    """

    def decorator(cls):
        spec: ModelSpec = cls.spec()
        spec.name = name
        registry.register(spec)
        return cls

    return decorator
