# Simuletic Vision

Simuletic Vision is a Python toolkit for training, evaluating, and running computer vision models with synthetic and real-world datasets. The first practical workflow is YOLO-format object detection with RF-DETR.

The project intentionally keeps the first implementation simple: validate a local YOLO dataset, train RF-DETR, evaluate a trained checkpoint where the installed RF-DETR API supports it, and run image inference from a trained checkpoint.

## Install

Install the published Python distribution named `simuletic-vision` from your package index:

```bash
pip install simuletic-vision
```

Or with uv:

```bash
uv pip install simuletic-vision
```

If you are installing from a local clone before the distribution is published, run the normal non-editable install from the repository root:

```bash
pip install .
```

Or with uv:

```bash
uv pip install .
```

For Python code, import the package as `simuletic_vision`. After installation, run the CLI as `simuletic-vision`.

RF-DETR training may require the optional training dependencies documented by RF-DETR. If training dependencies are missing, install them in your environment before running a real training job:

```bash
pip install 'rfdetr[train,loggers]'
```

### Development and contributor installs

Use editable installs when you are actively developing the project or contributing changes:

```bash
pip install -e '.[dev]'
```

Or with uv:

```bash
uv pip install -e '.[dev]'
```

## Naming

* GitHub repository: `simuletic-vision`
* Python distribution/package name: `simuletic-vision`
* CLI command: `simuletic-vision`
* Python import package: `simuletic_vision`
* Source directory: `src/simuletic_vision/`

Python imports cannot contain hyphens, so use `import simuletic_vision` in Python code.

## Quickstart

Validate the installed CLI:

```bash
simuletic-vision version
```

Validate the example RF-DETR config without importing RF-DETR or downloading model weights:

```bash
simuletic-vision config validate examples/configs/cctv_weapon_detection.yaml
```

Validate the configured dataset once you have downloaded it locally:

```bash
simuletic-vision dataset validate --config examples/configs/cctv_weapon_detection.yaml
```

Train RF-DETR:

```bash
simuletic-vision train --config examples/configs/cctv_weapon_detection.yaml
```

Evaluate a trained checkpoint:

```bash
simuletic-vision evaluate --config examples/configs/cctv_weapon_detection.yaml
```

Run inference on a single image or image directory:

```bash
simuletic-vision infer --config examples/configs/cctv_weapon_detection.yaml --source ./images --output ./runs/inference
```

## Kaggle CCTV weapon dataset

The public synthetic CCTV weapon detection dataset is available on Kaggle:

<https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset>

This repository does **not** integrate the Kaggle API, download the dataset automatically, or require Kaggle credentials. Download the dataset manually and unzip it into the path used by the example config:

```bash
mkdir -p data
# Download the dataset from Kaggle manually:
# https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset
# Unzip it into:
# data/cctv-weapon-dataset
```

Then run:

```bash
simuletic-vision config validate examples/configs/cctv_weapon_detection.yaml
simuletic-vision dataset validate --config examples/configs/cctv_weapon_detection.yaml
simuletic-vision train --config examples/configs/cctv_weapon_detection.yaml
```

The example config intentionally points `synthetic_train`, `synthetic_val`, and `real_world_test` at the same local dataset root for now. Real-world test data and synthetic-to-real comparison metrics are planned as the next layer on top of this workflow.

## Expected YOLO dataset layout

RF-DETR auto-detects COCO or YOLO training data from the dataset directory. For YOLO data, the RF-DETR package expects a dataset root with `data.yaml` or `data.yml`, `train/images`, `train/labels`, and a validation split named `valid` or `val` with matching `images` and `labels` directories.

Recommended layout:

```text
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The lightweight dataset validator also supports common YOLO split layouts like this:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

However, RF-DETR training is run against the RF-DETR-supported root layout. If your dataset uses `images/train` and `labels/train`, convert or reorganize it to the recommended RF-DETR layout before training.

## Configuration

The example config is in `examples/configs/cctv_weapon_detection.yaml` and includes:

* `backend: rfdetr`
* `model.architecture: rfdetr`
* `model.variant`: `nano`, `small`, `medium`, `base`, or `large` when available in the installed `rfdetr` package
* `model.pretrained`: whether RF-DETR should start from pretrained weights
* `model.output_dir`: run artifacts and checkpoints directory
* `model.checkpoint`: optional explicit checkpoint for evaluation and inference
* `model.epochs`, `model.batch_size`, `model.learning_rate`, and `model.confidence_threshold`

Lightweight commands such as `simuletic-vision version` and `simuletic-vision config validate ...` do not import RF-DETR. RF-DETR is imported lazily only when training, evaluation, or inference needs the backend.

## RF-DETR backend behavior

The backend uses the installed RF-DETR Python API directly:

* Model classes: `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRBase`, and `RFDETRLarge` when present.
* Training: `model.train(dataset_dir=..., dataset_file='yolo', epochs=..., batch_size=..., grad_accum_steps=..., lr=..., output_dir=..., seed=...)`.
* Checkpoint loading: `RFDETR*.from_checkpoint(path)`.
* Inference: `model.predict(image_path, threshold=...)`.

Evaluation support depends on the installed RF-DETR API. The inspected installed package exposes high-level training and prediction APIs, while standalone evaluation metrics are produced during training rather than through a public `evaluate()` method. Simuletic Vision resolves the checkpoint and dataset and writes `evaluation.json`; if a future installed RF-DETR exposes `model.evaluate(...)`, the backend will call it and capture returned metrics.

## Checkpoints

For evaluation and inference, set `model.checkpoint` explicitly or let Simuletic Vision search under `model.output_dir` for likely RF-DETR checkpoint files such as:

* `checkpoint_best_total.pth`
* `checkpoint_best_ema.pth`
* `checkpoint_best_regular.pth`
* other `.pth`, `.pt`, or `.ckpt` files

If no checkpoint is found, run training first or set `model.checkpoint` to a trained checkpoint path.

## Project scope

Included now:

* YOLO dataset validation
* RF-DETR training wrapper
* RF-DETR checkpoint resolution
* RF-DETR evaluation command with JSON summary and clear standalone-evaluation limitations
* RF-DETR image inference for one image or an image directory

Not included:

* Automatic Kaggle downloads or Kaggle API integration
* Kaggle credentials
* Private datasets, customer data, API keys, model weights, LoRAs, proprietary generation pipelines, or internal ComfyUI workflows
* Custom model architectures
* GPU-required tests or heavyweight training tests
* Reliable video inference; image inference is currently supported and video support is pending

## License

Apache-2.0. See [LICENSE](LICENSE).
