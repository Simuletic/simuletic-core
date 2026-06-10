"""RF-DETR backend adapter.

The heavy ``rfdetr`` package is imported lazily inside backend methods so basic
Simuletic imports and lightweight CLI commands do not initialize ML frameworks or
trigger model-weight downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simuletic_vision.backends.base import BackendError
from simuletic_vision.config import DatasetConfig, ExperimentConfig

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
CHECKPOINT_EXTENSIONS = frozenset({".pth", ".pt", ".ckpt"})
CHECKPOINT_NAME_PRIORITY = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
    "last.pth",
)
RFDETR_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


class RFDETRBackend:
    """Simple adapter around Roboflow's RF-DETR Python package."""

    def train(self, config: ExperimentConfig) -> None:
        """Fine-tune RF-DETR on the configured YOLO or COCO dataset."""

        self._ensure_rfdetr_config(config)
        train_dataset = self._required_dataset(config, "synthetic_train")
        dataset_dir = self._resolve_rfdetr_dataset_dir(train_dataset)
        output_dir = config.model.output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._create_model(config, use_resolved_checkpoint=False)
        model.train(
            dataset_dir=str(dataset_dir),
            dataset_file=train_dataset.format,
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            grad_accum_steps=config.model.grad_accum_steps,
            lr=config.model.learning_rate,
            output_dir=str(output_dir),
            seed=config.seed,
            run_test="test" in self._available_split_names(dataset_dir),
        )

    def evaluate(self, config: ExperimentConfig) -> None:
        """Evaluate a trained RF-DETR checkpoint when the installed API supports it.

        The inspected RF-DETR package exposes high-level ``train`` and ``predict``
        methods, and training computes validation/test metrics internally. It does
        not expose a stable public standalone ``evaluate`` method in the installed
        package version used here. If a future installed version provides one, this
        adapter calls it and records the returned metrics; otherwise it writes a
        limitation summary after resolving the checkpoint and dataset.
        """

        self._ensure_rfdetr_config(config)
        eval_dataset = self._required_dataset(config, "real_world_test")
        dataset_dir = self._resolve_rfdetr_dataset_dir(eval_dataset)
        checkpoint = self._require_checkpoint(config, action="evaluation")
        output_dir = config.model.output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._load_checkpoint_model(config, checkpoint)
        metrics: Any = None
        status = "unsupported"
        limitation: str | None = (
            "The installed rfdetr package does not expose a public standalone "
            "evaluate() method. RF-DETR computes validation/test metrics during "
            "training; use the training artifacts under model.output_dir for full "
            "RF-DETR metrics, or upgrade RF-DETR if a standalone evaluation API is "
            "available."
        )
        evaluate_method = getattr(model, "evaluate", None)
        if callable(evaluate_method):
            metrics = evaluate_method(
                dataset_dir=str(dataset_dir), dataset_file=eval_dataset.format
            )
            status = "completed"
            limitation = None

        summary = {
            "backend": "rfdetr",
            "status": status,
            "dataset_dir": str(dataset_dir),
            "dataset_format": eval_dataset.format,
            "checkpoint": str(checkpoint),
            "requested_metrics": list(config.evaluation.metrics),
            "metrics": _json_safe(metrics),
            "limitation": limitation,
        }
        (output_dir / "evaluation.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    def infer(
        self, config: ExperimentConfig, source: Path, output: Path | None = None
    ) -> None:
        """Run RF-DETR prediction on a single image or image directory."""

        self._ensure_rfdetr_config(config)
        checkpoint = self._require_checkpoint(config, action="inference")
        source = source.expanduser()
        if not source.exists():
            raise BackendError(f"Inference source does not exist: {source}")
        if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
            raise BackendError(
                "RF-DETR video inference is not implemented in Simuletic Vision yet. "
                "Use a single image file or a directory of images for now."
            )

        image_paths = self._collect_image_paths(source)
        if not image_paths:
            raise BackendError(
                f"No supported image files found for inference source: {source}"
            )

        output_dir = output or Path("runs") / config.project_name / "inference"
        output_dir = output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._load_checkpoint_model(config, checkpoint)
        threshold = config.model.confidence_threshold
        predictions: list[dict[str, Any]] = []
        for image_path in image_paths:
            detections = model.predict(str(image_path), threshold=threshold)
            predictions.append(
                {
                    "source": str(image_path),
                    "predictions": self._serialize_detections(detections, model),
                }
            )

        predictions_path = output_dir / "predictions.json"
        predictions_path.write_text(
            json.dumps(predictions, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _ensure_rfdetr_config(self, config: ExperimentConfig) -> None:
        if config.backend != "rfdetr" or config.model.architecture != "rfdetr":
            raise BackendError(
                "RF-DETR backend requires backend: rfdetr and "
                "model.architecture: rfdetr in the experiment config."
            )
        if config.task != "detection":
            raise BackendError(
                "RF-DETR backend currently supports detection tasks only."
            )

    def _create_model(
        self, config: ExperimentConfig, *, use_resolved_checkpoint: bool
    ) -> Any:
        if use_resolved_checkpoint:
            checkpoint = self._require_checkpoint(config, action="model loading")
            return self._load_checkpoint_model(config, checkpoint)

        rfdetr_module = self._import_rfdetr()
        variant = config.model.variant.lower()
        class_name = RFDETR_VARIANTS.get(variant)
        if class_name is None or not hasattr(rfdetr_module, class_name):
            supported = self._supported_variant_names(rfdetr_module)
            raise BackendError(
                f"Unsupported RF-DETR variant '{config.model.variant}'. "
                f"Supported variants in the installed rfdetr package: {supported}."
            )
        model_class = getattr(rfdetr_module, class_name)
        if config.model.pretrained:
            return model_class()
        return model_class(pretrain_weights=None)

    def _load_checkpoint_model(self, config: ExperimentConfig, checkpoint: Path) -> Any:
        rfdetr_module = self._import_rfdetr()
        variant = config.model.variant.lower()
        class_name = RFDETR_VARIANTS.get(variant)
        model_class = (
            getattr(rfdetr_module, class_name)
            if class_name is not None and hasattr(rfdetr_module, class_name)
            else rfdetr_module.RFDETR
        )
        try:
            return model_class.from_checkpoint(str(checkpoint))
        except (OSError, KeyError, ValueError) as exc:
            raise BackendError(
                f"RF-DETR could not load checkpoint {checkpoint}: {exc}"
            ) from exc

    def _import_rfdetr(self) -> Any:
        try:
            import rfdetr
        except ModuleNotFoundError as exc:
            raise BackendError(
                "The RF-DETR backend requires the 'rfdetr' package. Install "
                "simuletic-vision with its runtime dependencies, or run "
                "`pip install rfdetr`."
            ) from exc
        return rfdetr

    def _supported_variant_names(self, rfdetr_module: Any) -> str:
        supported = [
            variant
            for variant, class_name in RFDETR_VARIANTS.items()
            if hasattr(rfdetr_module, class_name)
        ]
        return ", ".join(supported) if supported else "none"

    def _required_dataset(
        self, config: ExperimentConfig, dataset_key: str
    ) -> DatasetConfig:
        try:
            return config.datasets[dataset_key]
        except KeyError as exc:
            raise BackendError(
                f"RF-DETR backend requires a '{dataset_key}' dataset in the config."
            ) from exc

    def _resolve_rfdetr_dataset_dir(self, dataset: DatasetConfig) -> Path:
        dataset_path = dataset.path.expanduser()
        if dataset.format not in {"yolo", "coco"}:
            raise BackendError(
                "RF-DETR training supports YOLO or COCO dataset formats. "
                f"Configured format was '{dataset.format}'."
            )
        candidates = [dataset_path]
        if dataset_path.name in {"train", "valid", "val", "test"}:
            candidates.append(dataset_path.parent)

        for candidate in candidates:
            if (
                dataset.format == "yolo"
                and self._looks_like_rfdetr_yolo_dataset(candidate)
            ):
                return candidate
            if (
                dataset.format == "coco"
                and self._looks_like_rfdetr_coco_dataset(candidate)
            ):
                return candidate

        raise BackendError(
            "RF-DETR could not find a supported dataset root. For YOLO, provide a "
            "dataset root containing data.yaml (or data.yml), train/images/, "
            "train/labels/, and a validation split named valid/ or val/. For COCO, "
            "provide a dataset root containing train/_annotations.coco.json. "
            f"Checked: {', '.join(str(path) for path in candidates)}."
        )

    def _looks_like_rfdetr_yolo_dataset(self, dataset_dir: Path) -> bool:
        if not dataset_dir.is_dir():
            return False
        has_yolo_config = (dataset_dir / "data.yaml").is_file() or (
            dataset_dir / "data.yml"
        ).is_file()
        has_train = (dataset_dir / "train" / "images").is_dir() and (
            dataset_dir / "train" / "labels"
        ).is_dir()
        has_valid = any(
            (dataset_dir / split / "images").is_dir()
            and (dataset_dir / split / "labels").is_dir()
            for split in ("valid", "val")
        )
        return has_yolo_config and has_train and has_valid

    def _looks_like_rfdetr_coco_dataset(self, dataset_dir: Path) -> bool:
        return dataset_dir.is_dir() and (
            dataset_dir / "train" / "_annotations.coco.json"
        ).is_file()

    def _available_split_names(self, dataset_dir: Path) -> set[str]:
        return {split.name for split in dataset_dir.iterdir() if split.is_dir()}

    def _collect_image_paths(self, source: Path) -> list[Path]:
        if source.is_file():
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                raise BackendError(f"Unsupported inference source file type: {source}")
            return [source]
        if not source.is_dir():
            raise BackendError(f"Inference source is not a file or directory: {source}")
        return sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _serialize_detections(
        self, detections: Any, model: Any
    ) -> list[dict[str, Any]]:
        class_names = list(getattr(model, "class_names", []) or [])
        xyxy_values = getattr(detections, "xyxy", [])
        confidence_values = getattr(detections, "confidence", [])
        class_id_values = getattr(detections, "class_id", [])
        data = getattr(detections, "data", {}) or {}
        class_name_values = data.get("class_name", []) if isinstance(data, dict) else []

        serialized: list[dict[str, Any]] = []
        for index, bbox in enumerate(xyxy_values):
            class_id = self._optional_int(self._safe_index(class_id_values, index))
            class_name = self._safe_index(class_name_values, index)
            if (
                class_name is None
                and class_id is not None
                and 0 <= class_id < len(class_names)
            ):
                class_name = class_names[class_id]
            serialized.append(
                {
                    "bbox_xyxy": [float(value) for value in bbox],
                    "confidence": self._optional_float(
                        self._safe_index(confidence_values, index)
                    ),
                    "class_id": class_id,
                    "class_name": str(class_name) if class_name is not None else None,
                }
            )
        return serialized

    def _require_checkpoint(self, config: ExperimentConfig, *, action: str) -> Path:
        checkpoint = resolve_checkpoint(config)
        if checkpoint is None:
            raise BackendError(
                f"No RF-DETR checkpoint found for {action}. Run "
                "`simuletic-vision train --config <config>` first, or set "
                "model.checkpoint to a trained .pth/.pt/.ckpt file."
            )
        if not checkpoint.exists():
            raise BackendError(f"RF-DETR checkpoint does not exist: {checkpoint}")
        if not checkpoint.is_file():
            raise BackendError(f"RF-DETR checkpoint is not a file: {checkpoint}")
        return checkpoint

    def _safe_index(self, values: Any, index: int) -> Any:
        try:
            return values[index]
        except (IndexError, KeyError, TypeError):
            return None

    def _optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    def _optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        return float(value)


def resolve_checkpoint(config: ExperimentConfig) -> Path | None:
    """Resolve an explicit or likely RF-DETR checkpoint for evaluation/inference."""

    if config.model.checkpoint is not None:
        return config.model.checkpoint.expanduser()

    output_dir = config.model.output_dir.expanduser()
    if not output_dir.exists() or not output_dir.is_dir():
        return None

    for name in CHECKPOINT_NAME_PRIORITY:
        matches = sorted(output_dir.rglob(name), key=lambda path: path.stat().st_mtime)
        if matches:
            return matches[-1]

    candidates = sorted(
        (
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS
        ),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _json_safe(value: Any) -> Any:
    """Convert common metric containers into JSON-serializable values."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (TypeError, ValueError):
            pass
    return str(value)
