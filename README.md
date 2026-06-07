# Simuletic

**Simuletic is an open-source Python toolkit for training, evaluating, benchmarking, and running computer vision models with synthetic and real-world datasets.**

The goal is to help computer vision teams understand whether models trained on synthetic data actually generalize to real-world production conditions.

Simuletic starts with workflows for:

* training models on synthetic datasets
* evaluating models on real-world validation data
* running inference locally
* benchmarking synthetic-to-real performance
* building reproducible computer vision experiments

The first focus areas are safety-critical and edge-case-heavy vision systems, including CCTV, video surveillance, robotics, and physical AI.

> Synthetic data is only useful if it improves real-world model performance. Simuletic is built around that principle.

---

## Why Simuletic?

Computer vision models often fail in production because they lack training data for rare edge cases.

These edge cases are often:

* dangerous to collect
* legally difficult to collect
* expensive to annotate
* rare in real-world footage
* missing from public datasets
* captured from the wrong camera angles

Synthetic datasets can help, but only if they are evaluated properly against real-world data.

Simuletic provides tooling for the full synthetic-to-real workflow:

```text
synthetic dataset → training → real-world evaluation → benchmarking → deployment
```

---

## Current status

This repository is early and under active development.

The first public version will focus on:

* Python package structure
* CLI workflows
* training/evaluation configuration files
* reproducible experiment setup
* synthetic-to-real evaluation helpers

Future versions may include:

* model training integrations
* YOLO/ONNX/PyTorch support
* dataset validation tools
* benchmark reports
* synthetic dataset generation APIs
* sensor/camera profile tooling
* hosted Simuletic platform integration

---

## Installation

Install locally with pip:

```bash
pip install .
```

Or with uv:

```bash
uv pip install .
```

For editable development:

```bash
pip install -e .
```

or:

```bash
uv pip install -e .
```

---

## Quickstart

Check that the CLI is installed:

```bash
simuletic version
```

Example future workflow:

```bash
simuletic train --config examples/configs/cctv_detection.yaml
simuletic evaluate --config examples/configs/cctv_detection.yaml
simuletic infer --model ./runs/model.onnx --source ./images --output ./runs/inference
```

---

## Example config

```yaml
project_name: cctv-weapon-detection
task: detection

train_data:
  path: ./data/synthetic/train
  format: yolo

val_data:
  path: ./data/synthetic/val
  format: yolo

real_world_test_data:
  path: ./data/real_world/test
  format: yolo

model:
  type: yolo
  size: small
  pretrained: true

metrics:
  - precision
  - recall
  - map50
  - false_positive_rate

output_dir: ./runs/cctv-weapon-detection
seed: 42
```

---

## Python usage

Example future API:

```python
from simuletic import Experiment

experiment = Experiment.from_config("examples/configs/cctv_detection.yaml")

experiment.train()
results = experiment.evaluate()

print(results.summary())
```

---

## Design principles

Simuletic is built around a few simple principles:

1. **Synthetic data must be measured against reality**
   The goal is not to generate impressive images. The goal is better real-world model performance.

2. **Evaluation matters as much as training**
   Synthetic datasets should be tested against real camera footage, edge cases, false positives, and degraded sensor conditions.

3. **Reproducibility first**
   Experiments should be configurable, repeatable, and easy to compare.

4. **Lightweight core, optional integrations**
   The core package should stay lightweight. Heavy ML frameworks and model-specific integrations should be optional.

5. **Production-oriented computer vision**
   The project is designed for teams building real systems, not only research demos.

---

## Roadmap

### v0.1

* Modern Python package structure
* CLI entrypoint
* Config schema for train/evaluate/infer workflows
* Basic tests and CI
* Example configs

### v0.2

* Training workflow interfaces
* Evaluation workflow interfaces
* Dataset validation helpers
* Report generation

### v0.3

* Optional YOLO integration
* Optional ONNX inference support
* Synthetic-to-real benchmark reports

### Later

* Simuletic hosted API integration
* Synthetic dataset generation workflows
* Camera/sensor profile tools
* Robotics and physical AI benchmark templates

---

## Repository scope

This repository is the public Simuletic toolkit.

It does **not** include:

* private customer datasets
* proprietary generation pipelines
* private LoRA weights
* confidential customer prompts
* internal hosted backend code

---

## License

License to be decided.

If the project is intended to maximize adoption while protecting commercial use, consider a permissive license for the SDK/toolkit and a separate commercial license for hosted services, private datasets, and proprietary generation pipelines.

---

## About Simuletic

Simuletic builds synthetic data and evaluation tools for computer vision systems, starting with CCTV/security edge cases and expanding toward robotics and physical AI.

Website: https://simuletic.com
