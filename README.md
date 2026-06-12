<div align="center">

# 🎧 audiotimm

**The Model Hub for Audio Intelligence**

*`timm` for audio — one registry, every architecture, one clean API.*

[![PyPI](https://img.shields.io/pypi/v/audiotimm?style=flat-square&color=orange&label=PyPI)](https://pypi.org/project/audiotimm/)
[![Downloads](https://img.shields.io/pypi/dm/audiotimm?style=flat-square&label=downloads%2Fmonth&color=blue)](https://pypi.org/project/audiotimm/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
[![Phase](https://img.shields.io/badge/phase-1%20%E2%80%94%20Core%20Engine-brightgreen?style=flat-square)]()

</div>

---

## What is audiotimm?

`audiotimm` is a standalone Python library that lets you classify, tag, detect events in, and extract embeddings from audio — in one line — using state-of-the-art pretrained models. It is designed after the philosophy of [`timm`](https://github.com/huggingface/pytorch-image-models): a unified registry where every model family (PANNs, AST, BEATs, HTS-AT, CLAP, Wav2Vec2, WavLM, Whisper, …) is accessible through a single, stable API.

```python
from audiotimm import Classifier

clf = Classifier.load()                    # default: panns-cnn14
result = clf.predict("dog.wav")

result.top(5)       # [(label, score), ...]
result.label        # "Dog"
result.scores       # {"Dog": 0.94, "Animal": 0.72, ...}
```

---

## Highlights

| | |
|---|---|
| **One line to classify** | `Classifier.load().predict("x.wav").top(3)` — weights download and cache automatically |
| **Every major architecture** | PANNs, YAMNet, AST, BEATs, HTS-AT, AudioMAE, CLAP, Wav2Vec2, HuBERT, WavLM, Whisper |
| **Lean core** | Zero heavy deps at import time — torch + torchaudio only for the default model |
| **Rich result object** | `.top(k)`, `.above(thresh)`, `.label`, `.scores`, `.as_dict()`, `.embed()` |
| **Extensible** | `@register_model` decorator to plug in custom architectures |
| **CLI included** | `audiotimm predict dog.wav --top 5` |

---

## Installation

```bash
# Core (PANNs CNN-family, Wave M0)
pip install audiotimm

# + Transformer taggers: AST, BEATs, HTS-AT, AudioMAE (Wave M1)
pip install audiotimm[transformers]

# + Zero-shot classification via CLAP (Wave M2)
pip install audiotimm[clap]

# + Speech SSL backbones: Wav2Vec2, HuBERT, WavLM (Wave M3)
pip install audiotimm[speech]

# + Whisper ASR + encoder embeddings (Wave M4)
pip install audiotimm[whisper]

# + Training utilities
pip install audiotimm[train]

# + ONNX edge export
pip install audiotimm[onnx]

# Everything
pip install audiotimm[transformers,clap,speech,whisper,train,onnx]
```

---

## Quick Start

### Classify a file

```python
from audiotimm import Classifier

clf = Classifier.load()            # panns-cnn14 by default
result = clf.predict("siren.wav")

print(result.top(5))
# [("Siren", 0.93), ("Emergency vehicle", 0.81), ("Vehicle", 0.74), ...]

print(result.label)    # "Siren"
print(result.score)    # 0.93
```

### Batch classification

```python
results = clf.predict(["a.wav", "b.wav", "c.wav"])
print(results.labels())   # ["Dog", "Car horn", "Rain"]
```

### Only results above a threshold

```python
result.above(0.5)
# [("Siren", 0.93), ("Emergency vehicle", 0.81), ("Vehicle", 0.74)]
```

### Get embeddings

```python
emb = clf.embed("dog.wav")   # np.ndarray shape (2048,) for panns-cnn14
```

### Switch models

```python
# High accuracy transformer (requires pip install audiotimm[transformers])
clf = Classifier.load("ast-10-10")

# Lightweight 16 kHz variant of PANNs
clf = Classifier.load("panns-cnn14-16k")
```

### CLI

```bash
# Classify a file
audiotimm predict siren.wav --top 5

# JSON output
audiotimm predict siren.wav --json

# Batch
audiotimm predict *.wav --model panns-cnn14-16k

# List all available models
audiotimm list

# Filter by wave
audiotimm list --wave M1
```

---

## Available Models

### Wave M0 — CNN Taggers  `(core, no extras)`

| Zoo ID | Architecture | SR | Classes | mAP | Notes |
|---|---|---|---|---|---|
| `panns-cnn14` ⭐ | CNN14 | 32 kHz | 527 | 0.431 | **Default model** |
| `panns-cnn14-16k` | CNN14 | 16 kHz | 527 | 0.438 | Slightly higher mAP |
| `yamnet` | MobileNetV1 | 16 kHz | 521 | — | PyTorch path coming in v0.2 |

### Wave M1 — Transformer Taggers  `pip install audiotimm[transformers]`

| Zoo ID | Architecture | SR | Classes | mAP | Notes |
|---|---|---|---|---|---|
| `ast-10-10` ⭐ | Audio Spectrogram Transformer | 16 kHz | 527 | 0.459 | Default AST |
| `ast-16-16` | AST (larger patches) | 16 kHz | 527 | 0.442 | Faster |
| `ast-speechcommands` | AST | 16 kHz | 35 | — | Keyword spotting |
| `htsat-audioset` | HTS-AT (Swin-style) | 32 kHz | 527 | 0.471 | Also CLAP encoder |
| `htsat-desed` | HTS-AT | 32 kHz | — | — | Sound event detection |
| `audiomae-base-ft` | AudioMAE (ViT-Base) | 16 kHz | 527 | 0.473 | Facebook MAE |
| `beats-iter3plus-as2m-cpt2` | BEATs | 16 kHz | 527 | 0.486 | SOTA mAP |

### Wave M2 — Zero-Shot CLAP  `pip install audiotimm[clap]`

| Zoo ID | Variant | SR | Notes |
|---|---|---|---|
| `clap-laion-fused` ⭐ | LAION HTSAT + feature fusion | 48 kHz | Handles long audio |
| `clap-laion-unfused` | LAION HTSAT | 48 kHz | |
| `clap-laion-music-audioset` | Music + AudioSet trained | 48 kHz | ESC-50 ≈ 90.1% |
| `clap-ms-2023` ⭐ | MS-CLAP HTSAT + GPT-2 | 44.1 kHz | Stronger text encoder |
| `clap-ms-2022` | MS-CLAP CNN14 + BERT | 44.1 kHz | |
| `clap-ms-clapcap` | MS-CLAP + captioning head | 44.1 kHz | Audio → text captions |

### Wave M3 — Speech SSL Backbones  `pip install audiotimm[speech]`

| Zoo ID | Architecture | SR | Output |
|---|---|---|---|
| `wav2vec2-base` | Wav2Vec2 Base | 16 kHz | Frame embeddings |
| `wav2vec2-large-xlsr` | XLS-R 300M (128 languages) | 16 kHz | Multilingual |
| `hubert-large-ll60k` | HuBERT Large | 16 kHz | Strong SER backbone |
| `wavlm-large` ⭐ | WavLM Large | 16 kHz | Best for speaker tasks |
| `wavlm-base-plus-sv` | WavLM + SV head | 16 kHz | Speaker verification |

### Wave M4 — Whisper  `pip install audiotimm[whisper]`

| Zoo ID | Size | Languages | Notes |
|---|---|---|---|
| `whisper-base` | Base | 99 | Fast, general |
| `whisper-large-v3` ⭐ | Large v3 | 99 | Best accuracy |
| `whisper-large-v3-turbo` | Large v3 Turbo | 99 | Fast + accurate |
| `whisper-distil-large-v3` | Distil Large v3 | 1 (EN) | ~2× faster |

---

## Zero-Shot Classification (Wave M2)

Classify audio into **any labels you define** — no training needed:

```python
from audiotimm import ZeroShotClassifier   # coming in Phase 2

zs = ZeroShotClassifier.load("clap-laion-fused")
result = zs.classify(
    "clip.wav",
    labels=["dog barking", "car horn", "rain", "crowd applause"]
)
# -> [("rain", 0.81), ("crowd applause", 0.10), ...]
```

---

## Plugin API — Register Custom Models

```python
from audiotimm import register_model
from audiotimm.models._base import ModelAdapter
from audiotimm.core.registry import ModelSpec

@register_model("my-bird-net")
class BirdNet(ModelAdapter):

    @classmethod
    def spec(cls):
        return ModelSpec(
            name="",           # filled by decorator
            family="custom",
            adapter_factory=cls,
            checkpoint="./weights/birdnet.pt",
            sample_rate=22050,
            n_classes=500,
            embed_dim=512,
            task="tagging",
            wave="M0",
        )

    def predict(self, waveform):
        ...  # return {label: score} dict

# Now available everywhere
from audiotimm import Classifier
clf = Classifier.load("my-bird-net")
```

---

## Project Roadmap

```
Phase 1  ✅  Core engine + PANNs CNN family (Wave M0)
Phase 2  🔜  Wave M1 — AST, AudioMAE, HTS-AT, BEATs (transformer taggers)
Phase 3  ·   Wave M2 — CLAP zero-shot (LAION + MS)
Phase 4  ·   Embeddings & similarity search
Phase 5  ·   Sound Event Detection timeline
Phase 6  ·   Wave M3 — Wav2Vec2, HuBERT, WavLM speech SSL
Phase 7  ·   Training & fine-tuning (Trainer API)
Phase 8  ·   Wave M4 — Whisper ASR + encoder embeddings
Phase 9  ·   Evaluation & explainability (Grad-CAM on mel-spectrogram)
Phase 10 ·   Domain packs (bioacoustics, security, health, music, speech)
Phase 11 ·   Streaming / real-time inference
Phase 12 ·   ONNX / TFLite edge export
Phase 13 ·   XenAudio integration + plugin API
```

---

## Architecture

```
audiotimm/
├── core/
│   ├── classifier.py    # Classifier.load(), predict(), embed()
│   ├── result.py        # PredictionResult, BatchResult
│   └── registry.py      # ModelRegistry singleton + @register_model
├── models/
│   ├── _base.py         # ModelAdapter ABC
│   ├── panns.py         # Wave M0 — CNN14 family
│   ├── yamnet.py        # Wave M0 — YAMNet (stub)
│   ├── ast.py           # Wave M1 — AST (coming)
│   ├── beats.py         # Wave M1 — BEATs (coming)
│   ├── htsat.py         # Wave M1+M2 — HTS-AT (coming)
│   ├── audiomae.py      # Wave M1 — AudioMAE (coming)
│   ├── clap.py          # Wave M2 — LAION + MS-CLAP (coming)
│   ├── wav2vec2.py      # Wave M3 (coming)
│   ├── hubert.py        # Wave M3 (coming)
│   ├── wavlm.py         # Wave M3 (coming)
│   └── whisper.py       # Wave M4 (coming)
├── utils/
│   ├── audio.py         # load_audio(), pad_or_trim()
│   └── download.py      # cached downloader (~/.cache/audiotimm/)
└── cli.py               # `audiotimm predict` / `audiotimm list`
```

---

## Design Principles

- **Lazy everything** — weights download on first `predict()`, not on `import`.
- **One result type** — `PredictionResult` everywhere; switching models never breaks your code.
- **Lean core** — `torch + torchaudio + numpy` only for the default model; every heavy dep is behind an optional extra.
- **Registry-first** — every model is a registry entry; custom models slot in with `@register_model`.
- **Immutable results** — `PredictionResult` is read-only; safe to cache and pass around.

---

## Contributing

```bash
git clone https://github.com/shubham10divakar/audiotimm
cd audiotimm
pip install -e ".[dev]"
pytest tests/
```

---

## License

Apache 2.0. Model weights are subject to their respective upstream licenses — see [PLAN.md](PLAN.md) Appendix A for per-checkpoint license notes.

---

<div align="center">
<sub>Built with ❤️ · <b>audiotimm — Teach Machines to Listen.</b></sub>
</div>
