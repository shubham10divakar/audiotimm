# audiotimm: AI-First Audio Classification Library

> *The `timm` of audio — one registry, every model, one clean API.*

## Vision

A standalone, open-source Python library for sound classification and acoustic
intelligence that works **out of the box** (one line to classify with a sensible
default model) yet scales to custom training, real-time streaming, and edge
deployment.

It is a **separate package** from XenAudio:

* Usable on its own: `pip install audiotimm`
* Optionally consumed by XenAudio as the engine behind `audio.classify()`,
  `audio.detect_events()`, and embedding-based search.

## Design Principles

* **Out of the box:** `Classifier.load()` with no args picks a good default,
  lazily downloads weights, and just works.
* **Backend-agnostic:** PyTorch for training; ONNX for fast/edge inference.
* **One result object:** every prediction returns a rich, inspectable result
  (top-k, scores, timeline, embeddings) — not a raw tensor.
* **Library + CLI + (later) GUI** share one code path.
* **Lean core, optional extras:** `pip install audiotimm[clap]`,
  `[train]`, `[stream]`, `[onnx]`, `[domains]`.

## What "Sound Classification" Covers Here

* Multi-label audio tagging (AudioSet ontology, 527 classes)
* Zero-shot classification from text labels (CLAP)
* Sound Event Detection (SED) — frame-level "what happened when"
* Acoustic scene classification (where it was recorded)
* Single-label tasks: genre, instrument, emotion, gender/age, language
* Anomaly / novelty detection (unsupervised)

---

# Phase 1: Core Inference Engine + Model Zoo  ✅ IN PROGRESS

## Goal

Load a pretrained model and classify a file in one line, with a clean result API.

### Predict

```python
from audiotimm import Classifier

clf = Classifier.load("panns-cnn14")   # default if no name given
result = clf.predict("dog.wav")

result.top(5)          # [(label, score), ...]
result.label           # single best label
result.scores          # {label: score, ...}
result.as_dict()
```

### Batch

```python
results = clf.predict(["a.wav", "b.wav", "c.wav"])
```

### Model Zoo (lazy download + cache)

* PANNs (CNN14)  — strong AudioSet tagging baseline  (**Wave M0**)
* YAMNet         — lightweight, fast, mobile-friendly (**Wave M0**)
* AST            — Audio Spectrogram Transformer      (**Wave M1**)
* BEATs / PaSST  — state-of-the-art transformers      (**Wave M1**)

> The full set of supported model families (AST, BEATs, PANNs, Wav2Vec2,
> HuBERT, WavLM, Whisper, CLAP, HTS-AT, AudioMAE, YAMNet) and every checkpoint
> variation is enumerated in **"Model Support Phases (Shippable Waves)"** below
> and **"Appendix A: Complete Model Catalog & Variations"** at the end of this
> document. Phase 1 ships only the CNN taggers (PANNs + YAMNet); transformer
> taggers, speech SSL backbones, and ASR models land in later waves.

### Technology Stack

* torch, torchaudio
* torchlibrosa (mel spectrogram frontend matching PANNs checkpoint format)
* numpy
* huggingface\_hub (weight hosting/download)

### Deliverable

```python
from audiotimm import Classifier
print(Classifier.load().predict("siren.wav").top(3))
```

---

# Phase 2: Zero-Shot Classification (CLAP)  ★ differentiator

## Goal

Classify audio into **arbitrary, user-defined labels** with no training, by
matching audio embeddings against text-prompt embeddings.

### Example

```python
from audiotimm import ZeroShotClassifier

zs = ZeroShotClassifier.load("clap")
zs.classify("clip.wav", labels=["dog barking", "car horn", "rain", "music"])
# -> [("rain", 0.81), ("music", 0.10), ...]
```

### Why it matters

* No dataset, no fine-tuning — instant custom taxonomies.
* Prompt engineering for labels ("a recording of {label}").
* Foundation for XenAudio's flexible `audio.classify(labels=[...])`.

### Technology Stack

* CLAP (LAION / MS-CLAP)
* transformers

---

# Model Support Phases (Shippable Waves)

> This block expands the model zoo into independently shippable **waves**. Each
> wave reuses the same `Classifier` / result API where possible, so adding a
> wave is mostly a new adapter in `models/` plus registry entries — no breaking
> changes to user code. Waves are ordered by reuse of the existing inference
> path (cheapest first) and by how directly they extend a capability already in
> the plan. Full per-checkpoint detail lives in **Appendix A**.

| Shape | Families | Output |
|---|---|---|
| Spectrogram → AudioSet logits (tagging) | PANNs, YAMNet, AST, BEATs, HTS-AT, AudioMAE | multi-label scores |
| Audio–text contrastive (zero-shot) | CLAP (LAION + MS), HTS-AT as encoder | similarity to text prompts |
| Speech SSL backbones (representations) | Wav2Vec2, HuBERT, WavLM | frame/utterance embeddings → fine-tuned heads |
| ASR / speech understanding | Whisper | transcript + language ID + encoder embeddings |

## Wave M0 — CNN Taggers (= Phase 1 baseline)

* **Models:** PANNs (all CNN/ResNet/MobileNet/Wavegram variants), YAMNet.
* **Path:** waveform → log-mel → CNN → 527/521-class AudioSet logits.
* **Why first:** smallest deps (torch + torchaudio + numpy), no `transformers`,
  CPU-friendly, establishes the `PredictionResult` contract every later wave
  reuses.
* **Ships:** `Classifier.load("panns-cnn14").predict("x.wav").top(5)`.

## Wave M1 — Spectrogram-Transformer Taggers  ★ accuracy upgrade

* **Models:** AST, HTS-AT, AudioMAE, BEATs.
* **Path:** identical to M0 from the user's view (multi-label AudioSet tags);
  internally a mel-spectrogram → transformer → logits adapter.
* **Sub-shipping order (by integration ease):**
  1. **AST** — first-class in `transformers`, trivial to wire.
  2. **AudioMAE** — ViT-style, HF mirrors exist (`gaunernst/...`).
  3. **HTS-AT** — Swin-style hierarchical, custom loader; also unlocks SED
     (Phase 3) and serves as CLAP's audio encoder (reuse in M2).
  4. **BEATs** — custom loader (Microsoft `unilm` checkpoints on Azure),
     highest mAP; pairs with its bootstrapped tokenizer for SSL.
* **Extras:** `pip install audiotimm[transformers]`.

## Wave M2 — Contrastive / Zero-Shot (CLAP family)  ★ = Phase 2 engine

* **Models:** LAION-CLAP (unfused + fused, 630k / 630k-audioset / music /
  music+speech / music+speech+audioset checkpoints) and MS-CLAP (2022, 2023,
  clapcap). HTS-AT from M1 is reused as the LAION audio encoder.
* **Path:** audio embedding ⟷ text-prompt embedding → cosine similarity →
  ranked arbitrary labels (`ZeroShotClassifier.classify(labels=[...])`).
* **Extras:** `pip install audiotimm[clap]`.

## Wave M3 — Speech Self-Supervised Backbones

* **Models:** Wav2Vec2 (base/large/robust/XLSR/XLS-R/Conformer), HuBERT
  (base/large/xlarge + ASR fine-tunes), WavLM (base/base+/large + SV/SD heads).
* **Path:** waveform → SSL transformer → frame/utterance embeddings.
* **Ships incrementally:** first as `clf.embed()` providers (cheap, no head),
  then as `Trainer(base_model="wav2vec2", ...)` backbones.
* **Extras:** `pip install audiotimm[speech]`.

## Wave M4 — Whisper / ASR-Derived Understanding

* **Models:** Whisper (tiny→large-v3 + `.en` + large-v3-turbo + distil-whisper).
* **Path:** log-mel → encoder–decoder → transcript + language ID + word/segment
  timestamps; encoder also yields embeddings for classification/retrieval.
* **Extras:** `pip install audiotimm[whisper]`.

> **Net effect on the roadmap:** M0 ≡ Phase 1. M1 slots in right after Phase 1
> (pure accuracy upgrade, same API). M2 ≡ the Phase 2 engine. M3 feeds Phase 4
> (embeddings) and Phase 6 (training). M4 is an additive understanding layer.
> Each wave is a separate optional-extra install, keeping the core lean.

---

# Phase 3: Sound Event Detection (Timeline)

## Goal

Move from "what is in the clip" to "what happened, and when."

### Example

```python
events = clf.detect_events("street.wav", hop=0.5)
# [{"label": "siren", "start": 3.0, "end": 6.5, "score": 0.92}, ...]
```

### Features

* Sliding-window / frame-level inference
* Per-event onset/offset with confidence
* Smoothing + threshold tuning to merge fragments
* Export timeline to JSON / CSV / subtitle-like format

### Technology Stack

* numpy, scipy.signal (windowing, smoothing)

---

# Phase 4: Embeddings & Similarity

## Goal

Expose the model's penultimate layer as reusable audio embeddings.

### Example

```python
emb = clf.embed("audio.wav")              # np.ndarray
sims = clf.find_similar("query.wav", corpus="./sounds/", top_k=5)
```

### Use Cases

* Clustering large sound libraries
* Duplicate / near-duplicate detection
* Powers XenAudio Phase 6 (audio search & embeddings)
* Few-shot classification via nearest-neighbor on embeddings

### Technology Stack

* numpy
* optional: faiss / hnswlib for fast search

---

# Phase 5: Features, Datasets & Augmentation

## Goal

The building blocks for training and reproducible experiments.

### Feature Extraction

```python
from audiotimm.features import logmel, mfcc, spectrogram
X = logmel("audio.wav", n_mels=64)
```

### Dataset Loaders

* ESC-50, UrbanSound8K, AudioSet, DCASE
* Generic folder-per-class loader + CSV manifest loader

```python
from audiotimm.datasets import ESC50, FolderDataset
ds = ESC50(download=True)
```

### Augmentation

* Time stretch, pitch shift, gain, add-noise, time/freq masking (SpecAugment),
  mixup
* Composable pipeline

### Technology Stack

* torchaudio, librosa (optional), numpy

---

# Phase 6: Training & Fine-Tuning

## Goal

Transfer-learn on custom labels in a few lines.

### Example

```python
from audiotimm import Trainer
from audiotimm.datasets import FolderDataset

trainer = Trainer(base_model="ast", dataset=FolderDataset("./my_sounds"))
model = trainer.fit(epochs=10)
model.save("my_classifier")

Classifier.load("my_classifier").predict("test.wav")
```

### Features

* Freeze backbone / fine-tune head, or full fine-tune
* Class imbalance handling, label smoothing
* Checkpointing, early stopping, mixed precision
* Logging hooks (TensorBoard / Weights & Biases optional)

### Technology Stack

* torch, torchmetrics
* optional: lightning

---

# Phase 7: Evaluation & Explainability

## Goal

Trust and debug predictions.

### Metrics

```python
report = clf.evaluate(test_dataset)
report.mAP, report.f1, report.accuracy, report.confusion_matrix()
```

### Explainability

* Grad-CAM / saliency over the mel-spectrogram (which time-frequency region
  drove the prediction)
* Per-class threshold calibration

```python
clf.explain("audio.wav", label="dog")   # returns heatmap overlay
```

### Technology Stack

* scikit-learn, matplotlib
* captum (optional)

---

# Phase 8: Domain Packs

## Goal

Ship ready-to-use specialized classifiers as optional plugins.

### Packs

* **Bioacoustics** — bird / animal species (BirdNET-style)
* **Industrial** — machine fault / anomaly detection
* **Security** — gunshot, scream, glass break, alarm
* **Health** — cough, snore, baby cry, wheeze
* **Music** — genre, instrument, mood
* **Speech** — emotion (SER), gender, age, language ID

```python
from audiotimm.domains import security
security.detect("cctv_audio.wav")   # ["glass_break", "alarm"]
```

---

# Phase 9: Streaming / Real-Time

## Goal

Classify live audio from a microphone or stream.

### Example

```python
from audiotimm import StreamClassifier

with StreamClassifier.load("yamnet") as s:
    for event in s.listen():        # rolling-window inference
        print(event.label, event.score)
```

### Features

* Rolling buffer, configurable window/hop
* Low-latency lightweight models (YAMNet)
* Callbacks / generator API
* Optional WebSocket server for browser clients

### Technology Stack

* sounddevice / pyaudio
* websockets (optional)

---

# Phase 10: Deployment & Edge Export

## Goal

Run anywhere — server, browser, mobile, microcontroller-class devices.

### Features

* Export to ONNX / TFLite
* Quantization (int8) for edge
* Tiny REST/gRPC serving wrapper
* Pure-ONNX inference path (no torch at runtime)

```python
clf.export("model.onnx", quantize=True)
```

### Technology Stack

* onnx, onnxruntime
* optional: tflite, openvino

---

# Phase 11: XenAudio Integration + Plugin API

## Goal

Make audiotimm the brain behind XenAudio's classification, and let third
parties register custom classifiers.

### XenAudio side

```python
from xenaudio import Audio

audio = Audio.load("meeting.mp3")
audio.classify()                                 # multi-label tags
audio.classify(labels=["applause", "laughter"])  # zero-shot
audio.detect_events()                            # timeline
audio.embed()                                    # embeddings
```

### Plugin registration

```python
from audiotimm import register_model

@register_model("my-custom-net")
class MyNet:
    ...
```

---

# Project Structure

```text
audiotimm/
│
├── core/
│   ├── classifier.py        # Classifier, predict, batch
│   ├── result.py            # PredictionResult, BatchResult
│   └── registry.py          # model zoo + plugin registration
│
├── models/                  # one adapter per family (see Appendix A)
│   ├── _base.py             # shared ModelAdapter contract
│   ├── panns.py             # M0 — CNN14, CNN6, ResNet38, Wavegram...
│   ├── yamnet.py            # M0 — MobileNetV1 lightweight tagger
│   ├── ast.py               # M1 — Audio Spectrogram Transformer
│   ├── beats.py             # M1 — BEATs (custom loader)
│   ├── htsat.py             # M1 + M2 — also CLAP audio encoder
│   ├── audiomae.py          # M1 — Masked Autoencoder
│   ├── wav2vec2.py          # M3 — speech SSL backbone
│   ├── hubert.py            # M3 — speech SSL backbone
│   ├── wavlm.py             # M3 — speaker-aware SSL
│   ├── whisper.py           # M4 — ASR + embeddings
│   └── clap.py              # M2 — zero-shot (LAION + MS)
│
├── features/                # logmel, mfcc, spectrogram
├── datasets/                # ESC50, UrbanSound8K, AudioSet, loaders
├── augment/                 # SpecAugment, mixup, etc.
├── train/                   # Trainer, callbacks, metrics
├── eval/                    # metrics, explainability
├── streaming/               # real-time
├── export/                  # onnx / tflite
├── domains/                 # bioacoustics, security, health, music...
├── cli/                     # `audiotimm predict ...`
└── utils/
    ├── audio.py             # load / resample / mono
    └── download.py          # cache-aware weight downloader
```

---

# Suggested Build Order (MVP-first)

1. **Phase 1 + Wave M0** — core engine + PANNs + YAMNet (CNN taggers). Shippable.
2. **Wave M1** — transformer taggers (AST → AudioMAE → HTS-AT → BEATs); pure
   accuracy upgrade on the same `predict()` API, each shippable on its own.
3. **Phase 2 + Wave M2** — CLAP zero-shot (LAION + MS). Headline feature; reuses
   HTS-AT from M1. Ship early.
4. **Phase 4** — embeddings (cheap once models are loaded; unlocks XenAudio search).
5. **Phase 3** — event detection timeline (PANNs-SED / HTS-AT-DESED heads).
6. **Wave M3** — speech SSL backbones (Wav2Vec2 / HuBERT / WavLM) as `embed()`
   providers, then fine-tunable in Phase 6.
7. **Phase 5 + 6** — datasets/augmentation, then training (turns it into a platform).
8. **Wave M4** — Whisper (transcription + language ID + encoder embeddings).
9. **Phase 7** — evaluation + explainability (credibility).
10. **Phases 8–11** — domain packs, streaming, edge export, XenAudio integration.

---

# Long-Term Goal

Become the **classification + acoustic-intelligence** counterpart to:

* YAMNet / PANNs (tagging)
* CLAP (zero-shot)
* BirdNET (domain specialists)
* timm (unified model registry pattern for audio)

…behind one clean API, usable standalone or as XenAudio's intelligence layer.

## Tagline

**"audiotimm — The Model Hub for Audio Intelligence."**

---

# Appendix A: Complete Model Catalog & Variations

> Implementation reference for every supported model family and its checkpoints.
> Registry columns per entry: `name` (zoo id) · `family` · `checkpoint`
> (HF repo / file / url) · `sample_rate` · `n_classes` / `embed_dim` · `task`
> · `loader` (`transformers` | `custom` | `tfhub`) · `extra` (optional-extra
> group). mAP/accuracy figures are approximate — verify at source before shipping.
> License must be confirmed per checkpoint before redistribution/hosting.

## A.1 PANNs — CNN AudioSet Taggers  (Wave M0)

* **Source:** `qiuqiangkong/audioset_tagging_cnn`. **SR:** 32 kHz (plus 16 kHz /
  8 kHz variants). **Classes:** 527 (AudioSet). **Loader:** custom + torchlibrosa.
* **Architectures / checkpoints (all on Zenodo record 3987831):**
  * `panns-cnn14` (mAP ≈ 0.431) — **default**; `panns-cnn14-16k` (≈ 0.438), `panns-cnn14-8k`.
  * `panns-cnn6` (≈ 0.343), `panns-cnn10` (≈ 0.380).
  * `panns-resnet22`, `panns-resnet38` (≈ 0.434), `panns-resnet54`.
  * `panns-mobilenetv1` (≈ 0.389), `panns-mobilenetv2` (≈ 0.383) — edge/mobile.
  * `panns-wavegram-cnn14`, `panns-wavegram-logmel` (≈ 0.439) — learnable waveform front-end.
  * `panns-res1dnet31`, `panns-res1dnet51`, `panns-dainet`, `panns-leenet11`, `panns-leenet24` — 1-D conv nets.
  * **SED heads:** `panns-cnn14-sed-max` / `-sed-att` / `-sed-avg` — frame-level outputs → Phase 3.
  * **Transfer fine-tunes:** GTZAN (genre), ESC-50.
* **Tasks:** multi-label tagging; SED variants for timeline.

## A.2 YAMNet — Lightweight Tagger  (Wave M0)

* **Source:** Google (TF-Hub `google/yamnet/1`); ONNX/TFLite conversion for torch-free path.
  **SR:** 16 kHz. **Classes:** 521 (AudioSet). **Backbone:** MobileNetV1.
* **Variants:** canonical `yamnet`; `yamnet-tflite` (quantized, mobile). Default low-latency
  model for Phase 9 streaming.

## A.3 AST — Audio Spectrogram Transformer  (Wave M1)

* **Source:** HF `MIT/*`. **Loader:** `transformers` (`ASTForAudioClassification`).
  **SR:** 16 kHz, 128 mel bins.
* **Checkpoints (patch tstride-fstride in name; lower stride = higher mAP, slower):**
  * `ast-10-10` → `MIT/ast-finetuned-audioset-10-10-0.4593` — **default**, mAP ≈ 0.459.
  * `ast-10-10-v2` → `MIT/ast-finetuned-audioset-10-10-0.450`
  * `ast-10-10-v3` → `MIT/ast-finetuned-audioset-10-10-0.448`
  * `ast-12-12` → `MIT/ast-finetuned-audioset-12-12-0.447`
  * `ast-14-14` → `MIT/ast-finetuned-audioset-14-14-0.443`
  * `ast-16-16` → `MIT/ast-finetuned-audioset-16-16-0.442`
  * `ast-speechcommands` → `MIT/ast-finetuned-speech-commands-v2` — 35 keyword classes.
* **Variant family:** **SSAST** (self-supervised; patch- and frame-based; tiny/small/base) for
  Phase 6 pretraining without labels.
* **Extras:** `audiotimm[transformers]`.

## A.4 BEATs — Bootstrapped Audio Transformer  (Wave M1)

* **Source:** Microsoft `unilm/beats` (Azure blob; HF mirrors). **Loader:** custom.
  **SR:** 16 kHz. **Classes:** 527 (fine-tuned).
* **Checkpoints:**
  * `beats-iter1`, `beats-iter2`, `beats-iter3` — SSL pretrained (each iter
    re-bootstraps the acoustic tokenizer).
  * `beats-iter3plus-as20k`, `beats-iter3plus-as2m` — improved pretrain.
  * Fine-tuned: `beats-iter3-cpt1`, `beats-iter3-cpt2` and
    `beats-iter3plus-as2m-cpt1`, `beats-iter3plus-as2m-cpt2` — top mAP ≈ 0.486.
  * Companion **tokenizer** checkpoints (for SSL / Phase 6 pretraining).
* **Note:** README download links rot — pin exact Azure URLs at integration time.
* **Extras:** `audiotimm[transformers]`.

## A.5 HTS-AT — Hierarchical Token-Semantic Transformer  (Waves M1 + M2)

* **Source:** `RetroCirce/HTS-Audio-Transformer`. **Loader:** custom.
  **Backbone:** Swin-style hierarchical windows. **SR:** 32 kHz.
* **Checkpoints:** `htsat-audioset` (mAP ≈ 0.471), `htsat-esc50`, `htsat-speechcommands`,
  `htsat-desed` (SED / event localization → Phase 3).
* **Dual role:** standalone tagger **and** audio encoder inside LAION-CLAP (reused in M2).
* **Extras:** `audiotimm[transformers]`.

## A.6 AudioMAE — Masked Autoencoders that Listen  (Wave M1)

* **Source:** `facebookresearch/AudioMAE`; HF mirrors
  `gaunernst/vit_base_patch16_1024_128.audiomae_as2m[_ft_as20k]`.
  **Loader:** custom / timm-style ViT. **SR:** 16 kHz, 128 mel bins.
* **Checkpoints:**
  * `audiomae-base-as2m` — SSL pretrained ViT-Base patch16, AS2M — backbone for Phase 6.
  * `audiomae-base-as20k` — fine-tuned AS2M→AS20K (mAP ≈ 37).
  * `audiomae-base-ft` — fine-tuned AS2M (mAP ≈ 47.3).
  * ViT-Large variants where released.
* **Extras:** `audiotimm[transformers]`.

## A.7 Wav2Vec2 — Speech SSL Backbone  (Wave M3)

* **Source:** HF `facebook/*`. **Loader:** `transformers`. **SR:** 16 kHz.
  **Output:** frame embeddings.
* **Checkpoints:**
  * `wav2vec2-base`, `wav2vec2-base-960h` (ASR).
  * `wav2vec2-large`, `wav2vec2-large-960h`, `-960h-lv60`, `-960h-lv60-self`.
  * `wav2vec2-large-robust` — noisy/diverse domains.
  * `wav2vec2-large-xlsr-53` — 53-language multilingual.
  * **XLS-R:** `wav2vec2-xls-r-300m`, `-1b`, `-2b` — 128 languages.
  * **Conformer:** `wav2vec2-conformer-rel-pos-large`, `-rope-large`.
* **Downstream (Phase 6):** SER/emotion, language ID, keyword spotting, ASR.
* **Extras:** `audiotimm[speech]`.

## A.8 HuBERT — Hidden-Unit BERT  (Wave M3)

* **Source:** HF `facebook/*` (+ `superb/*` task heads). **Loader:** `transformers`. **SR:** 16 kHz.
* **Checkpoints:**
  * `hubert-base-ls960`, `hubert-large-ll60k`, `hubert-xlarge-ll60k` (SSL).
  * ASR fine-tunes: `hubert-large-ls960-ft`, `hubert-xlarge-ls960-ft`.
  * SUPERB task heads (emotion, speaker ID) under `superb/hubert-*`.
* **Use:** strong paralinguistic / SER backbone.
* **Extras:** `audiotimm[speech]`.

## A.9 WavLM — Speech SSL (speaker-aware)  (Wave M3)

* **Source:** HF `microsoft/*`. **Loader:** `transformers`. **SR:** 16 kHz.
* **Checkpoints:**
  * `wavlm-base`, `wavlm-base-plus`, `wavlm-large`.
  * `wavlm-base-plus-sv` — speaker **verification** head.
  * `wavlm-base-plus-sd` — speaker **diarization** head.
* **Use:** best of the SSL trio for speaker verification / diarization.
* **Extras:** `audiotimm[speech]`.

## A.10 Whisper — ASR & Speech Understanding  (Wave M4)

* **Source:** HF `openai/*` (+ `distil-whisper/*`). **Loader:** `transformers`.
  **SR:** 16 kHz. **Languages:** 99.
* **Checkpoints:**
  * `whisper-tiny` / `whisper-tiny-en`, `whisper-base` / `whisper-base-en`.
  * `whisper-small` / `whisper-small-en`, `whisper-medium` / `whisper-medium-en`.
  * `whisper-large`, `whisper-large-v2`, `whisper-large-v3`, `whisper-large-v3-turbo`.
  * **Distil-Whisper:** `whisper-distil-small-en`, `whisper-distil-medium-en`,
    `whisper-distil-large-v2`, `whisper-distil-large-v3` — ~2× faster, near-parity.
* **Outputs:** transcript, language ID, word/segment timestamps, **encoder embeddings**.
* **Extras:** `audiotimm[whisper]`.

## A.11 CLAP — Contrastive Language-Audio Pretraining  (Wave M2)

### LAION-CLAP  (`LAION-AI/CLAP`; HF `laion/*`; `pip install laion-clap`)

* **Loader:** `transformers` (`ClapModel`) or `laion_clap`. **SR:** 48 kHz.
  **Audio enc:** HTSAT-base (or PANN). **Text enc:** RoBERTa.
* **HF checkpoints:** `clap-laion-unfused` → `laion/clap-htsat-unfused`,
  `clap-laion-fused` → `laion/clap-htsat-fused` (feature fusion for variable-length audio).
* **Original `.pt` checkpoints:**
  * `clap-laion-630k`, `clap-laion-630k-audioset`, `clap-laion-630k-fusion`, `clap-laion-630k-audioset-fusion`
  * `clap-laion-music-audioset` — `music_audioset_epoch_15_esc_90.14.pt` (ESC-50 ≈ 90.1%)
  * `clap-laion-music-speech` — `music_speech_epoch_15_esc_89.25.pt`
  * `clap-laion-music-speech-audioset` — `music_speech_audioset_epoch_15_esc_89.98.pt`

### MS-CLAP  (`microsoft/CLAP`; HF `microsoft/msclap`; `pip install msclap`)

* **Loader:** `msclap` or `transformers`. **SR:** 44.1 kHz.
* **Versions:**
  * `clap-ms-2022` — CNN14 audio enc + BERT text enc.
  * `clap-ms-2023` — HTSAT audio enc + GPT-2 text enc (stronger; **default**).
  * `clap-ms-clapcap` — audio **captioning** head on 2023 encoders.

## A.12 Registry / Naming Convention

```text
<family>-<arch/size>[-<dataset/sr>]    e.g.
  panns-cnn14            panns-cnn14-16k        panns-wavegram-logmel
  ast-10-10              ast-16-16              ast-speechcommands
  beats-iter3plus-as2m   htsat-audioset         audiomae-base-as2m
  wav2vec2-large-xlsr    hubert-large-ll60k     wavlm-base-plus-sv
  whisper-large-v3       whisper-distil-large-v3
  clap-laion-fused       clap-laion-music       clap-ms-2023
```

`Classifier.load()` (no arg) → `panns-cnn14`.
Each entry declares its required optional-extra so the error message tells
the user exactly what to `pip install` when deps are missing.
