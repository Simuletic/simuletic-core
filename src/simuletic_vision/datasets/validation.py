"""Lightweight dataset validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from simuletic_vision.config import DatasetConfig, ExperimentConfig

IssueSeverity = Literal["error", "warning"]

SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)
CONFIG_DATASET_KEYS = (
    "synthetic_train",
    "synthetic_val",
    "real_world_test",
    "hard_negatives",
)


@dataclass(frozen=True)
class DatasetIssue:
    """A dataset validation issue or warning."""

    message: str
    severity: IssueSeverity = "error"
    path: Path | None = None


@dataclass(frozen=True)
class DatasetValidationResult:
    """Result of validating one configured dataset."""

    dataset_key: str
    dataset_path: Path
    dataset_format: str
    passed: bool
    issues: list[DatasetIssue] = field(default_factory=list)
    image_count: int | None = None
    label_file_count: int | None = None

    @property
    def has_errors(self) -> bool:
        """Return whether the validation result includes blocking errors."""
        return any(issue.severity == "error" for issue in self.issues)


def validate_config_datasets(
    config: ExperimentConfig,
) -> list[DatasetValidationResult]:
    """Validate datasets configured in an experiment config.

    Known Simuletic dataset keys are validated first in a stable order. Any extra
    dataset keys in the config are then validated as well. Missing optional keys
    are skipped gracefully.
    """

    ordered_keys = [key for key in CONFIG_DATASET_KEYS if key in config.datasets]
    extra_keys = [key for key in config.datasets if key not in CONFIG_DATASET_KEYS]

    return [
        validate_dataset(key, config.datasets[key])
        for key in [*ordered_keys, *extra_keys]
    ]


def validate_dataset(
    dataset_key: str, dataset: DatasetConfig
) -> DatasetValidationResult:
    """Validate a single dataset config."""

    if dataset.format == "yolo":
        return validate_yolo_dataset(dataset_key, dataset.path, dataset.format)
    return validate_generic_dataset(dataset_key, dataset.path, dataset.format)


def validate_generic_dataset(
    dataset_key: str, dataset_path: Path, dataset_format: str
) -> DatasetValidationResult:
    """Perform placeholder validation for dataset formats without detailed checks."""

    issues: list[DatasetIssue] = []
    if not dataset_path.exists():
        issues.append(
            DatasetIssue(
                message="Dataset path does not exist.",
                severity="error",
                path=dataset_path,
            )
        )

    issues.append(
        DatasetIssue(
            message=(
                f"Detailed dataset validation for format '{dataset_format}' is not "
                "implemented yet."
            ),
            severity="warning",
            path=dataset_path,
        )
    )

    return DatasetValidationResult(
        dataset_key=dataset_key,
        dataset_path=dataset_path,
        dataset_format=dataset_format,
        passed=not _has_errors(issues),
        issues=issues,
    )


def validate_yolo_dataset(
    dataset_key: str, dataset_path: Path, dataset_format: str = "yolo"
) -> DatasetValidationResult:
    """Validate common YOLO object detection dataset structures."""

    issues: list[DatasetIssue] = []
    image_count: int | None = None
    label_file_count: int | None = None

    if not dataset_path.exists():
        issues.append(
            DatasetIssue(
                message="Dataset path does not exist.",
                severity="error",
                path=dataset_path,
            )
        )
        return DatasetValidationResult(
            dataset_key=dataset_key,
            dataset_path=dataset_path,
            dataset_format=dataset_format,
            passed=False,
            issues=issues,
            image_count=image_count,
            label_file_count=label_file_count,
        )

    if not dataset_path.is_dir():
        issues.append(
            DatasetIssue(
                message="Dataset path is not a directory.",
                severity="error",
                path=dataset_path,
            )
        )
        return DatasetValidationResult(
            dataset_key=dataset_key,
            dataset_path=dataset_path,
            dataset_format=dataset_format,
            passed=False,
            issues=issues,
            image_count=image_count,
            label_file_count=label_file_count,
        )

    layout = _detect_yolo_layout(dataset_path)
    if layout is None:
        issues.extend(
            [
                DatasetIssue(
                    message=(
                        "YOLO images directory is missing. Expected either "
                        "train/images plus valid/images, images/train, or images/."
                    ),
                    severity="error",
                    path=dataset_path,
                ),
                DatasetIssue(
                    message=(
                        "YOLO labels directory is missing. Expected either "
                        "train/labels plus valid/labels, labels/train, or labels/."
                    ),
                    severity="error",
                    path=dataset_path,
                ),
            ]
        )
        return DatasetValidationResult(
            dataset_key=dataset_key,
            dataset_path=dataset_path,
            dataset_format=dataset_format,
            passed=False,
            issues=issues,
            image_count=image_count,
            label_file_count=label_file_count,
        )

    image_dirs, label_dirs = layout
    image_paths = _find_image_files_in_dirs(image_dirs)
    label_paths = _find_label_files_in_dirs(label_dirs)
    image_count = len(image_paths)
    label_file_count = len(label_paths)

    if not image_paths:
        issues.append(
            DatasetIssue(
                message="No supported image files found in YOLO images directories.",
                severity="error",
                path=dataset_path,
            )
        )
    if not label_paths:
        issues.append(
            DatasetIssue(
                message="No label files found in YOLO labels directories.",
                severity="error",
                path=dataset_path,
            )
        )
    else:
        issues.extend(_validate_yolo_label_files(label_paths))

    if _is_rfdetr_yolo_layout(dataset_path) and not _has_yolo_data_file(dataset_path):
        issues.append(
            DatasetIssue(
                message=(
                    "RF-DETR YOLO training expects data.yaml or data.yml at the "
                    "dataset root."
                ),
                severity="warning",
                path=dataset_path,
            )
        )

    return DatasetValidationResult(
        dataset_key=dataset_key,
        dataset_path=dataset_path,
        dataset_format=dataset_format,
        passed=not _has_errors(issues),
        issues=issues,
        image_count=image_count,
        label_file_count=label_file_count,
    )


def _detect_yolo_layout(dataset_path: Path) -> tuple[list[Path], list[Path]] | None:
    """Return image/label directories for supported YOLO layouts."""

    rfdetr_splits = [
        split
        for split in ("train", "valid", "val", "test")
        if (dataset_path / split / "images").is_dir()
        and (dataset_path / split / "labels").is_dir()
    ]
    if rfdetr_splits:
        return (
            [dataset_path / split / "images" for split in rfdetr_splits],
            [dataset_path / split / "labels" for split in rfdetr_splits],
        )

    split_names = [
        split
        for split in ("train", "valid", "val", "test")
        if (dataset_path / "images" / split).is_dir()
        and (dataset_path / "labels" / split).is_dir()
    ]
    if split_names:
        return (
            [dataset_path / "images" / split for split in split_names],
            [dataset_path / "labels" / split for split in split_names],
        )

    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        return [images_dir], [labels_dir]

    return None


def _is_rfdetr_yolo_layout(dataset_path: Path) -> bool:
    return (dataset_path / "train" / "images").is_dir()


def _has_yolo_data_file(dataset_path: Path) -> bool:
    return (dataset_path / "data.yaml").is_file() or (
        dataset_path / "data.yml"
    ).is_file()


def _find_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _find_image_files_in_dirs(image_dirs: list[Path]) -> list[Path]:
    return sorted(
        path for images_dir in image_dirs for path in _find_image_files(images_dir)
    )


def _find_label_files(labels_dir: Path) -> list[Path]:
    return sorted(path for path in labels_dir.rglob("*.txt") if path.is_file())


def _find_label_files_in_dirs(label_dirs: list[Path]) -> list[Path]:
    return sorted(
        path for labels_dir in label_dirs for path in _find_label_files(labels_dir)
    )


def _validate_yolo_label_files(label_paths: list[Path]) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    for label_path in label_paths:
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            issues.extend(_validate_yolo_label_line(label_path, line_number, line))
    return issues


def _validate_yolo_label_line(
    label_path: Path, line_number: int, line: str
) -> list[DatasetIssue]:
    parts = line.split()
    location = f"{label_path}:{line_number}"

    if len(parts) != 5:
        return [
            DatasetIssue(
                message=(
                    f"Invalid YOLO label line at {location}: expected 5 values "
                    "'class_id x_center y_center width height'."
                ),
                severity="error",
                path=label_path,
            )
        ]

    class_id_text, *bbox_text_values = parts
    issues: list[DatasetIssue] = []

    try:
        class_id = int(class_id_text)
    except ValueError:
        issues.append(
            DatasetIssue(
                message=f"Invalid YOLO class_id at {location}: expected integer >= 0.",
                severity="error",
                path=label_path,
            )
        )
    else:
        if class_id < 0:
            issues.append(
                DatasetIssue(
                    message=(
                        f"Invalid YOLO class_id at {location}: expected integer >= 0."
                    ),
                    severity="error",
                    path=label_path,
                )
            )

    for field_name, value_text in zip(
        ("x_center", "y_center", "width", "height"), bbox_text_values, strict=True
    ):
        try:
            value = float(value_text)
        except ValueError:
            issues.append(
                DatasetIssue(
                    message=(
                        f"Invalid YOLO {field_name} at {location}: expected numeric "
                        "value between 0 and 1."
                    ),
                    severity="error",
                    path=label_path,
                )
            )
            continue

        if value < 0 or value > 1:
            issues.append(
                DatasetIssue(
                    message=(
                        f"Invalid YOLO {field_name} at {location}: value must be "
                        "between 0 and 1."
                    ),
                    severity="error",
                    path=label_path,
                )
            )

    return issues


def _has_errors(issues: list[DatasetIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)
