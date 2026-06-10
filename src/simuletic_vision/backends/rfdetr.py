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
        """Fine-tune RF-DETR on the configured synthetic training dataset."""

        self._ensure_rfdetr_config(config)
        train_dataset = self._required_dataset(config, "synthetic_train")
        dataset_dir = self._resolve_rfdetr_dataset_dir(train_dataset)
        output_dir = config.model.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._create_model(config)
        model.train(
            dataset_dir=str(dataset_dir),
            epochs=config.model.epochs,
            batch_size=config.model.batch_size,
            grad_accum_steps=config.model.grad_accum_steps,
            lr=config.model.learning_rate,
            output_dir=str(output_dir),
        )

    def evaluate(self, config: ExperimentConfig) -> None:
        """Report current evaluation support for RF-DETR.

        RF-DETR training computes validation metrics, but its stable high-level
        package API does not currently expose a standalone evaluator that maps
        directly onto Simuletic's synthetic-to-real metric schema. Keep this
        intentionally explicit instead of producing partial or misleading metrics.
        """

        self._ensure_rfdetr_config(config)
        self._required_dataset(config, "real_world_test")
        raise BackendError(
            "RF-DETR standalone evaluation is not implemented yet in Simuletic Vision. "
            "RF-DETR training writes validation artifacts to model.output_dir, but "
            "Simuletic's real-world test metrics, hard-negative false-positive "
            "metrics, and synthetic-to-real gap report still need a dedicated "
            "evaluation adapter."
        )

    def infer(
        self, config: ExperimentConfig, source: Path, output: Path | None = None
    ) -> None:
        """Run RF-DETR prediction on a single image or image directory."""

        self._ensure_rfdetr_config(config)
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
        output_dir.mkdir(parents=True, exist_ok=True)

        model = self._create_model(config)
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

    def _create_model(self, config: ExperimentConfig) -> Any:
        rfdetr_module = self._import_rfdetr()
        if config.model.checkpoint is not None:
            checkpoint = config.model.checkpoint.expanduser()
            if not checkpoint.exists():
                raise BackendError(f"RF-DETR checkpoint does not exist: {checkpoint}")
            return rfdetr_module.RFDETR.from_checkpoint(checkpoint)

        variant = config.model.variant.lower()
        class_name = RFDETR_VARIANTS.get(variant)
        if class_name is None:
            supported = ", ".join(sorted(RFDETR_VARIANTS))
            raise BackendError(
                f"Unsupported RF-DETR variant '{config.model.variant}'. "
                f"Supported variants: {supported}."
            )
        model_class = getattr(rfdetr_module, class_name)
        if config.model.pretrained:
            return model_class()
        return model_class(pretrain_weights=None)

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
            if self._looks_like_rfdetr_dataset(candidate):
                return candidate

        raise BackendError(
            "RF-DETR could not find a supported dataset root. For YOLO, provide a "
            "dataset root containing data.yaml (or data.yml) and train/images/. "
            "For COCO, provide a dataset root containing train/_annotations.coco.json. "
            f"Checked: {', '.join(str(path) for path in candidates)}."
        )

    def _looks_like_rfdetr_dataset(self, dataset_dir: Path) -> bool:
        if not dataset_dir.is_dir():
            return False
        has_coco_train = (dataset_dir / "train" / "_annotations.coco.json").is_file()
        has_yolo_config = (dataset_dir / "data.yaml").is_file() or (
            dataset_dir / "data.yml"
        ).is_file()
        has_yolo_train_images = (dataset_dir / "train" / "images").is_dir()
        return has_coco_train or (has_yolo_config and has_yolo_train_images)

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
