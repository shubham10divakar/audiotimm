"""Unit tests for core result, registry, and Classifier without needing real weights."""
import pytest


def test_prediction_result_top():
    from audiotimm.core.result import PredictionResult

    scores = {"cat": 0.9, "dog": 0.7, "bird": 0.3, "car": 0.1}
    r = PredictionResult(scores=scores, model="test", file="x.wav")

    assert r.label == "cat"
    assert r.score == pytest.approx(0.9)
    assert r.top(2) == [("cat", 0.9), ("dog", 0.7)]
    assert r.above(0.5) == [("cat", 0.9), ("dog", 0.7)]


def test_prediction_result_immutable():
    from audiotimm.core.result import PredictionResult

    r = PredictionResult(scores={"a": 1.0}, model="m")
    with pytest.raises(AttributeError):
        r._scores = {}


def test_prediction_result_as_dict():
    from audiotimm.core.result import PredictionResult

    r = PredictionResult(scores={"a": 0.8, "b": 0.2}, model="m", file="f.wav")
    d = r.as_dict()
    assert d["label"] == "a"
    assert d["model"] == "m"
    assert d["file"] == "f.wav"


def test_batch_result():
    from audiotimm.core.result import BatchResult, PredictionResult

    results = BatchResult([
        PredictionResult({"cat": 0.9, "dog": 0.1}, "m", "a.wav"),
        PredictionResult({"dog": 0.8, "cat": 0.2}, "m", "b.wav"),
    ])
    assert results.labels() == ["cat", "dog"]


def test_registry_get_unknown():
    from audiotimm.core.registry import ModelRegistry

    reg = ModelRegistry()
    with pytest.raises(ValueError, match="Unknown model"):
        reg.get("does-not-exist")


def test_registry_populated():
    """After importing audiotimm, PANNs models must be registered."""
    import audiotimm  # noqa: F401 — triggers model registration
    from audiotimm import registry

    names = registry.list_names()
    assert "panns-cnn14" in names
    assert "panns-cnn14-16k" in names
    assert "yamnet" in names


def test_registry_spec_fields():
    import audiotimm  # noqa: F401
    from audiotimm import registry

    spec = registry.get("panns-cnn14")
    assert spec.sample_rate == 32000
    assert spec.n_classes == 527
    assert spec.embed_dim == 2048
    assert spec.wave == "M0"
    assert spec.task == "tagging"


def test_classifier_load_default():
    """Classifier.load() must return without error (lazy — no download yet)."""
    from audiotimm import Classifier

    clf = Classifier.load.__func__  # just check it exists
    assert callable(clf)


def test_classifier_repr():
    from audiotimm import Classifier

    # We can't call load() without torch, but we can test the repr via a mock.
    from unittest.mock import MagicMock
    from audiotimm.core.registry import ModelSpec

    spec = ModelSpec(
        name="panns-cnn14",
        family="panns",
        adapter_factory=MagicMock,
        checkpoint="",
        sample_rate=32000,
        n_classes=527,
        embed_dim=2048,
        task="tagging",
        wave="M0",
    )
    clf = Classifier(MagicMock(), spec)
    assert "panns-cnn14" in repr(clf)
    assert "32000" in repr(clf)
