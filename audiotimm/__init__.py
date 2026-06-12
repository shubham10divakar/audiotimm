"""audiotimm — The Model Hub for Audio Intelligence.

Quick start::

    from audiotimm import Classifier

    clf = Classifier.load()                      # default: panns-cnn14
    result = clf.predict("dog.wav")
    print(result.top(5))
    print(result.label)

    # batch
    results = clf.predict(["a.wav", "b.wav"])
    print(results.labels())

    # embeddings
    emb = clf.embed("dog.wav")   # np.ndarray (2048,)

Available models::

    from audiotimm import registry
    print(registry.list_names())
"""

from audiotimm.core.classifier import Classifier
from audiotimm.core.registry import ModelSpec, register_model, registry
from audiotimm.core.result import BatchResult, PredictionResult

# Populate the registry — each model module calls _register() at import time.
import audiotimm.models  # noqa: F401

__version__ = "1.0.0"
__all__ = [
    "Classifier",
    "PredictionResult",
    "BatchResult",
    "registry",
    "register_model",
    "ModelSpec",
    "__version__",
]
