from pathlib import Path

from simuletic.config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
)
from simuletic.datasets import validate_config_datasets, validate_dataset


def make_config(datasets: dict[str, DatasetConfig]) -> ExperimentConfig:
    return ExperimentConfig(
        project_name="test-project",
        task="detection",
        backend="custom",
        datasets=datasets,
        model=ModelConfig(architecture="custom", output_dir=Path("runs/test")),
        evaluation=EvaluationConfig(metrics=["precision"]),
    )


def make_valid_yolo_dataset(dataset_path: Path) -> None:
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (images_dir / "frame-001.jpg").write_bytes(b"small fake image")
    (labels_dir / "frame-001.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")


def test_valid_yolo_dataset_layout(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    make_valid_yolo_dataset(dataset_path)

    result = validate_dataset(
        "synthetic_train", DatasetConfig(path=dataset_path, format="yolo")
    )

    assert result.passed is True
    assert result.dataset_key == "synthetic_train"
    assert result.dataset_path == dataset_path
    assert result.dataset_format == "yolo"
    assert result.image_count == 1
    assert result.label_file_count == 1
    assert result.issues == []


def test_valid_yolo_split_layout(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    images_train_dir = dataset_path / "images" / "train"
    labels_train_dir = dataset_path / "labels" / "train"
    images_val_dir = dataset_path / "images" / "val"
    labels_val_dir = dataset_path / "labels" / "val"
    images_train_dir.mkdir(parents=True)
    labels_train_dir.mkdir(parents=True)
    images_val_dir.mkdir(parents=True)
    labels_val_dir.mkdir(parents=True)
    (images_train_dir / "frame-001.png").write_bytes(b"fake png")
    (labels_train_dir / "frame-001.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
    )
    (images_val_dir / "frame-002.webp").write_bytes(b"fake webp")
    (labels_val_dir / "frame-002.txt").write_text(
        "1 0.1 0.2 0.3 0.4\n", encoding="utf-8"
    )

    result = validate_dataset(
        "synthetic_train", DatasetConfig(path=dataset_path, format="yolo")
    )

    assert result.passed is True
    assert result.image_count == 2
    assert result.label_file_count == 2
    assert result.issues == []


def test_missing_dataset_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "does-not-exist"

    result = validate_dataset(
        "synthetic_train", DatasetConfig(path=dataset_path, format="yolo")
    )

    assert result.passed is False
    assert result.image_count is None
    assert result.label_file_count is None
    assert any(
        "Dataset path does not exist" in issue.message for issue in result.issues
    )


def test_missing_images_directory(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    (dataset_path / "labels").mkdir(parents=True)
    (dataset_path / "labels" / "frame-001.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
    )

    result = validate_dataset(
        "synthetic_train", DatasetConfig(path=dataset_path, format="yolo")
    )

    assert result.passed is False
    assert result.image_count is None
    assert result.label_file_count == 1
    assert any(
        "images directory is missing" in issue.message for issue in result.issues
    )


def test_missing_labels_directory(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    (dataset_path / "images").mkdir(parents=True)
    (dataset_path / "images" / "frame-001.jpeg").write_bytes(b"fake jpeg")

    result = validate_dataset(
        "synthetic_train", DatasetConfig(path=dataset_path, format="yolo")
    )

    assert result.passed is False
    assert result.image_count == 1
    assert result.label_file_count is None
    assert any(
        "labels directory is missing" in issue.message for issue in result.issues
    )


def test_invalid_yolo_label_format(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    make_valid_yolo_dataset(dataset_path)
    (dataset_path / "labels" / "frame-001.txt").write_text(
        "not-enough-values\n", encoding="utf-8"
    )

    result = validate_dataset(
        "synthetic_train", DatasetConfig(path=dataset_path, format="yolo")
    )

    assert result.passed is False
    assert any("expected 5 values" in issue.message for issue in result.issues)


def test_yolo_label_rejects_bbox_values_outside_zero_to_one(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    make_valid_yolo_dataset(dataset_path)
    (dataset_path / "labels" / "frame-001.txt").write_text(
        "0 1.1 0.5 0.25 0.25\n", encoding="utf-8"
    )

    result = validate_dataset(
        "synthetic_train", DatasetConfig(path=dataset_path, format="yolo")
    )

    assert result.passed is False
    assert any("between 0 and 1" in issue.message for issue in result.issues)


def test_generic_non_yolo_placeholder_validation(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    result = validate_dataset(
        "real_world_test", DatasetConfig(path=dataset_path, format="coco")
    )

    assert result.passed is True
    assert result.image_count is None
    assert result.label_file_count is None
    assert len(result.issues) == 1
    assert result.issues[0].severity == "warning"
    assert "not implemented yet" in result.issues[0].message


def test_validate_config_datasets_skips_missing_optional_keys(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    make_valid_yolo_dataset(dataset_path)
    config = make_config(
        {"synthetic_train": DatasetConfig(path=dataset_path, format="yolo")}
    )

    results = validate_config_datasets(config)

    assert [result.dataset_key for result in results] == ["synthetic_train"]
    assert results[0].passed is True
