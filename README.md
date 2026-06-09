# Simuletic Core

**Simuletic Core is an open-source Python toolkit for training, evaluating, benchmarking, and running computer vision models with synthetic and real-world datasets.**

The goal is to help teams understand whether models trained on synthetic data actually generalize to real-world production conditions.

The first focus is **object detection**, with future support for segmentation, pose estimation, classification, and video analytics workflows.

> Synthetic data is only useful if it improves real-world model performance.

## Why Simuletic Core?

Computer vision models often fail in production because they have not seen enough rare edge cases, hard negatives, degraded camera footage, unusual perspectives, or real-world visual noise.

Synthetic data can help, but only if it improves real-world model performance.

Simuletic Core is built around this workflow:

```text
synthetic training data
        ↓
model training
        ↓
synthetic validation
        ↓
real-world evaluation
        ↓
synthetic-to-real gap analysis
        ↓
inference and deployment support
```

The key question is not:

> Can we train a model?

The key question is:

> Does synthetic data improve performance on real-world data?

## Project focus

Simuletic Core focuses on production-oriented computer vision workflows for domains such as:

* CCTV and video surveillance
* security and incident analytics
* edge-case detection
* hard-negative evaluation
* false-positive reduction
* robotics
* physical AI
* safety-critical visual AI systems

The first planned training backend is an open-source object detection model pipeline, likely based on **RT-DETR** or a similar permissively licensed detection framework. Simuletic Core will not create new model architectures from scratch. Instead, it will provide a structured workflow around existing open-source model pipelines.

Heavy ML dependencies will be optional integrations, not required by the lightweight core package.

## Current status

This repository is early and intentionally lightweight.

Implemented today:

* Distribution package name: `simuletic-core`
* Python import packages: `simuletic` and compatibility package `simuletic_core`
* CLI commands: `simuletic` and `simuletic-core`
* Version command: `simuletic version`
* YAML experiment config schema
* Config loading with Pydantic validation
* Config validation CLI
* Lightweight dataset validation for config-defined datasets
* Basic tests, linting, typing configuration, and GitHub Actions CI

Planned next:

* actual model training
* actual evaluation metrics
* RT-DETR backend integration
* inference pipeline
* benchmark reports
* API clients
* optional ML framework integrations

## Installation

Install with pip:

```bash
pip install simuletic-core
```

Or install into an environment with uv:

```bash
uv pip install simuletic-core
```

Or add it to a uv-managed project:

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

## Quickstart

Import the package in Python:

```python
import simuletic
```

Check that the CLI is installed:

```bash
simuletic version
```

Validate the included example experiment config:

```bash
simuletic config validate examples/configs/cctv_weapon_detection.yaml
```

Validate datasets referenced by the example experiment config:

```bash
simuletic dataset validate --config examples/configs/cctv_weapon_detection.yaml
```

Run the test suite and checks:

```bash
pytest
ruff check .
mypy src
```

## Experiment configs

Simuletic Core uses YAML configs for reproducible experiments. A config describes the synthetic training data, synthetic validation data, real-world test data, hard negatives, model backend settings, and requested metrics for a synthetic-to-real workflow.

The first example config is available at [`examples/configs/cctv_weapon_detection.yaml`](examples/configs/cctv_weapon_detection.yaml). It is generic and public-safe; it does not include private datasets, customer data, model weights, API keys, or proprietary generation workflows.

Example:

```yaml
project_name: cctv-weapon-detection
task: detection
backend: rtdetr

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
  architecture: rtdetr
  pretrained: true
  output_dir: ./runs/cctv-weapon-detection

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
simuletic config validate examples/configs/cctv_weapon_detection.yaml
```

The validation command only loads and validates the YAML schema. It does **not** train, evaluate, run inference, download models, or invoke heavy ML frameworks.

## Dataset validation

Simuletic Core can also validate datasets referenced by an experiment config before training or evaluation work begins:

```bash
simuletic dataset validate --config examples/configs/cctv_weapon_detection.yaml
```

The dataset validation command loads the config, checks each configured dataset, prints a Rich summary with pass/fail status, and exits with a non-zero status if any dataset has blocking errors. The included example config intentionally points at local `./data/...` paths that are not required to exist in this repository, so running the command without creating those datasets will fail clearly.

Currently implemented:

* config validation
* lightweight dataset validation
* YOLO object detection dataset checks for `images/` and `labels/` layouts, including split subdirectories such as `images/train`, `images/val`, `labels/train`, and `labels/val`
* supported image extensions: `.jpg`, `.jpeg`, `.png`, and `.webp`
* rough YOLO label-line checks for `class_id x_center y_center width height` numeric values

Planned:

* actual training
* evaluation metrics
* RT-DETR backend
* inference pipeline

Dataset validation does **not** download datasets, inspect images with OpenCV or PIL, download models, or invoke heavy ML frameworks.

Supported schema values:

* Tasks: `detection`, `classification`, `segmentation`, `pose`
* Dataset formats: `yolo`, `coco`, `imagefolder`, `csv`, `custom`
* Backends: `rtdetr`, `custom`
* Metrics: `precision`, `recall`, `map50`, `map`, `false_positive_rate`, `synthetic_to_real_gap`

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

## Roadmap

### Phase 1 — Foundation

* Package skeleton
* CLI foundation
* tests
* CI

### Phase 2 — Experiment configs

* YAML config schema
* typed validation
* example configs
* CLI config validation

### Phase 3 — Dataset validation

* YOLO dataset checks
* COCO dataset checks
* class consistency checks
* split validation

### Phase 4 — Evaluation results

* standard result models
* metric summaries
* JSON reports
* Markdown reports

### Phase 5 — First training backend

* object detection backend adapter
* likely RT-DETR or similar open-source detector
* synthetic training workflow
* real-world evaluation workflow

### Phase 6 — Inference

* image inference
* batch inference
* prediction export
* optional annotated outputs

### Phase 7 — Synthetic-to-real benchmarking

* synthetic validation vs real-world test comparison
* domain gap metrics
* hard-negative false positive analysis

### Phase 8 — Simuletic hosted integration

* optional hosted API integration
* dataset generation requests
* job tracking
* dataset download/export

## About Simuletic

Simuletic builds tools and datasets for synthetic-to-real computer vision, starting with CCTV/security edge cases and expanding toward robotics and physical AI.

Website: https://simuletic.com
