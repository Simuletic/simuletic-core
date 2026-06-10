# Simuletic Vision

Simuletic Vision is a Python toolkit for training, evaluating, and running computer vision models with synthetic and real-world datasets.

It is designed for synthetic-to-real workflows where a model is trained on synthetic data and evaluated against real-world validation data.

## What it currently does

Simuletic Vision currently provides:

* YAML experiment configuration loading and validation
* dataset validation for configured datasets
* lightweight YOLO object detection dataset checks
* initial RF-DETR backend support for training and image inference
* an explicit placeholder for standalone RF-DETR evaluation reporting

The package imports as:

```python
import simuletic_vision
```

## Installation

Install the package with pip:

```bash
pip install simuletic-vision
```

Add it to a uv-managed project:

```bash
uv add simuletic-vision
```

For local development from this repository:

```bash
pip install -e ".[dev]"
```

or:

```bash
uv pip install -e ".[dev]"
```

## Quickstart

```bash
simuletic-vision version
simuletic-vision config validate examples/configs/cctv_weapon_detection.yaml
simuletic-vision dataset validate --config examples/configs/cctv_weapon_detection.yaml
```

The included example config references public-safe placeholder dataset paths under `./data`. Dataset validation will fail until those paths are populated with datasets that match the configured formats.

## Example config validation

Validate the example experiment config without importing RF-DETR or downloading model weights:

```bash
simuletic-vision config validate examples/configs/cctv_weapon_detection.yaml
```

The example config is available at [`examples/configs/cctv_weapon_detection.yaml`](examples/configs/cctv_weapon_detection.yaml). It describes a detection workflow with synthetic training data, synthetic validation data, real-world test data, hard negatives, RF-DETR model settings, and synthetic-to-real metrics.

## Dataset validation

Validate all datasets referenced by the example config:

```bash
simuletic-vision dataset validate --config examples/configs/cctv_weapon_detection.yaml
```

Currently implemented dataset validation includes:

* checks that configured dataset paths exist
* YOLO object detection layout checks for `images/` and `labels/`
* support for split subdirectories such as `images/train`, `images/val`, `labels/train`, and `labels/val`
* image extension checks for `.jpg`, `.jpeg`, `.png`, and `.webp`
* lightweight YOLO label-line checks for `class_id x_center y_center width height`

## RF-DETR workflow

RF-DETR support is initial and focused on using the `rfdetr` Python package through the configured backend. RF-DETR may download model weights when training or inference creates a pretrained model. Lightweight commands such as `version` and `config validate` do not import RF-DETR.

```bash
simuletic-vision train --config examples/configs/cctv_weapon_detection.yaml
simuletic-vision evaluate --config examples/configs/cctv_weapon_detection.yaml
simuletic-vision infer --config examples/configs/cctv_weapon_detection.yaml --source ./images --output ./runs/inference
```

Current RF-DETR notes:

* training delegates to RF-DETR's high-level training API
* image inference supports a single image or an image directory and writes `predictions.json`
* standalone RF-DETR evaluation is not fully implemented yet; the command currently reports that a dedicated real-world, hard-negative, and synthetic-to-real evaluation adapter is still needed
* tests do not run real model training and do not download model weights

## Synthetic-to-real evaluation

Simuletic Vision is structured around workflows that compare synthetic training performance with real-world validation or test data. The configuration schema includes metrics such as precision, recall, mAP, false-positive rate, and synthetic-to-real gap. Full standalone synthetic-to-real report generation is still evolving.

## Project scope

Simuletic Vision focuses on:

* object detection workflows
* synthetic training data
* real-world evaluation data
* dataset validation
* model training and inference through existing open-source backends

The first backend is RF-DETR.

This repository does not include private datasets, model weights, generation pipelines, customer data, API keys, or secrets.

## Development checks

```bash
pytest
ruff check .
mypy src
```

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
