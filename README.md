# Simuletic Core

`simuletic-core` is the open-source core toolkit for Simuletic synthetic-to-real computer vision workflows.

Simuletic Core helps you train, evaluate, benchmark, and run computer vision models with synthetic and real-world datasets. Its niche is synthetic-to-real object detection:

* train on synthetic data
* validate on synthetic data
* evaluate on real-world data
* measure synthetic-to-real performance gaps
* measure false positives on hard negatives
* run inference on new images

The repository does **not** build custom model architectures from scratch. It wraps and orchestrates existing open-source model pipelines.

## Naming

* GitHub repository: `simuletic-core`
* Python package/distribution: `simuletic-core`
* CLI command: `simuletic-core`
* Python import package: `simuletic_core`

Python imports cannot contain hyphens, so import the package with an underscore:

```python
import simuletic_core
```

## Current status

Implemented:

* config schema
* config validation
* dataset validation
* RF-DETR backend foundation
* CLI commands for train/evaluate/infer through RF-DETR

Planned / still evolving:

* robust synthetic-to-real metric reporting
* full benchmark reports
* video inference
* additional backends
* hosted Simuletic API integration

## Installation

Install with pip:

```bash
pip install simuletic-core
```

Add it to a uv-managed project:

```bash
uv add simuletic-core
```

For editable development from this repository:

```bash
pip install -e ".[dev]"
```

or:

```bash
uv pip install -e ".[dev]"
```

The first real backend is RF-DETR via the `rfdetr` Python package. RF-DETR weights may be downloaded when RF-DETR is first used for training or inference. Basic commands such as `simuletic-core version` and `simuletic-core config validate ...` should not download model weights.

## Quickstart

Check that the CLI is installed:

```bash
simuletic-core version
```

Validate the included example experiment config:

```bash
simuletic-core config validate examples/configs/cctv_weapon_detection.yaml
```

Validate datasets referenced by the example experiment config:

```bash
simuletic-core dataset validate --config examples/configs/cctv_weapon_detection.yaml
```

Run the test suite and checks:

```bash
pytest
ruff check .
mypy src
```

Tests do not run full RF-DETR training, do not download model weights, and do not require a GPU.

## First RF-DETR workflow

```bash
pip install -e .
simuletic-core version
simuletic-core config validate examples/configs/cctv_weapon_detection.yaml
simuletic-core dataset validate --config examples/configs/cctv_weapon_detection.yaml
simuletic-core train --config examples/configs/cctv_weapon_detection.yaml
simuletic-core evaluate --config examples/configs/cctv_weapon_detection.yaml
simuletic-core infer --config examples/configs/cctv_weapon_detection.yaml --source ./images --output ./runs/inference
```

Notes:

* RF-DETR model weights may be downloaded when you first run RF-DETR training or inference.
* `simuletic-core version` and config validation are intentionally lightweight and do not import RF-DETR.
* The current inference path supports a single image or image directory and writes `predictions.json`.
* Video inference is planned but not implemented yet.
* Standalone real-world evaluation is intentionally explicit: RF-DETR training writes validation artifacts, but Simuletic's full real-world, hard-negative, and synthetic-to-real reports are still evolving.

## Experiment configs

Simuletic Core uses YAML configs for reproducible experiments. A config describes synthetic training data, synthetic validation data, real-world test data, hard negatives, model backend settings, and requested metrics.

The first example config is available at [`examples/configs/cctv_weapon_detection.yaml`](examples/configs/cctv_weapon_detection.yaml). It is generic and public-safe; it does not include private datasets, customer data, model weights, API keys, or proprietary generation workflows.

Example:

```yaml
project_name: cctv-weapon-detection
task: detection
backend: rfdetr

datasets:
  synthetic_train:
    path: ./data/synthetic/train
    format: yolo

  synthetic_val:
    path: ./data/synthetic/val
    format: yolo

  real_world_test:
    path: ./data/real_world/test
    format: yolo

  hard_negatives:
    path: ./data/hard_negatives
    format: yolo

model:
  architecture: rfdetr
  variant: base
  pretrained: true
  output_dir: ./runs/cctv-weapon-detection
  epochs: 1
  batch_size: 2
  learning_rate: 0.0001

evaluation:
  metrics:
    - precision
    - recall
    - map50
    - false_positive_rate
    - synthetic_to_real_gap

seed: 42
```

Validate a config with:

```bash
simuletic-core config validate examples/configs/cctv_weapon_detection.yaml
```

The validation command only loads and validates the YAML schema. It does **not** train, evaluate, run inference, download models, or invoke heavy ML frameworks.

## Dataset validation

Simuletic Core can validate datasets referenced by an experiment config before training or evaluation work begins:

```bash
simuletic-core dataset validate --config examples/configs/cctv_weapon_detection.yaml
```

Currently implemented:

* lightweight dataset validation
* YOLO object detection dataset checks for `images/` and `labels/` layouts, including split subdirectories such as `images/train`, `images/val`, `labels/train`, and `labels/val`
* supported image extensions: `.jpg`, `.jpeg`, `.png`, and `.webp`
* rough YOLO label-line checks for `class_id x_center y_center width height` numeric values

RF-DETR itself expects a trainable dataset root in COCO or YOLO format. For YOLO training, provide a root with `data.yaml` or `data.yml` and `train/images/`. For COCO training, provide a root with `train/_annotations.coco.json`.

Supported schema values:

* Tasks: `detection`, `classification`, `segmentation`, `pose`
* Dataset formats: `yolo`, `coco`, `imagefolder`, `csv`, `custom`
* Backends: `rfdetr`, `custom`
* Metrics: `precision`, `recall`, `map50`, `map`, `false_positive_rate`, `synthetic_to_real_gap`

## RF-DETR backend

The RF-DETR backend lazily imports `rfdetr` only inside backend operations. It uses RF-DETR's high-level Python API:

* model classes such as `RFDETRBase`, `RFDETRSmall`, and `RFDETRMedium`
* `model.train(dataset_dir=..., epochs=..., batch_size=..., grad_accum_steps=..., lr=..., output_dir=...)`
* `model.predict(image, threshold=...)` for image inference
* `RFDETR.from_checkpoint(...)` when `model.checkpoint` is configured

Inference writes a simple `predictions.json` file with source paths, class IDs or names when available, confidence scores, and `xyxy` bounding boxes.

## What this repository is not

This public repository does not include:

* private Simuletic generation pipelines
* private customer datasets
* customer-specific prompts
* private LoRA weights
* model weights
* generated customer samples
* API keys or secrets
* internal ComfyUI workflows

The public repo provides tooling and interfaces. Proprietary generation pipelines and hosted services remain separate.
