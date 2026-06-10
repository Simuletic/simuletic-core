import sys
import types
from pathlib import Path
from typing import Any

import pytest

from simuletic_vision.backends import BackendError, get_backend
from simuletic_vision.backends.rfdetr import RFDETRBackend, resolve_checkpoint
from simuletic_vision.config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
)


def make_config(
    tmp_path: Path,
    backend: str = "rfdetr",
    checkpoint: Path | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        project_name="test-project",
        task="detection",
        backend=backend,  # type: ignore[arg-type]
        datasets={
            "synthetic_train": DatasetConfig(path=tmp_path / "dataset", format="yolo"),
            "real_world_test": DatasetConfig(path=tmp_path / "dataset", format="yolo"),
        },
        model=ModelConfig(
            architecture=backend,  # type: ignore[arg-type]
            output_dir=tmp_path / "runs",
            checkpoint=checkpoint,
            confidence_threshold=0.25,
        ),
        evaluation=EvaluationConfig(metrics=["precision"]),
    )


def make_rfdetr_yolo_dataset(dataset_path: Path) -> None:
    for split in ("train", "valid"):
        images = dataset_path / split / "images"
        labels = dataset_path / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"{split}.jpg").write_bytes(b"fake")
        (labels / f"{split}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )
    (dataset_path / "data.yaml").write_text("names: [weapon]\n", encoding="utf-8")


def test_get_backend_selects_rfdetr_backend(tmp_path: Path) -> None:
    backend = get_backend(make_config(tmp_path))

    assert isinstance(backend, RFDETRBackend)


def test_get_backend_fails_clearly_for_custom_backend(tmp_path: Path) -> None:
    with pytest.raises(BackendError, match="custom"):
        get_backend(make_config(tmp_path, "custom"))


def test_rfdetr_backend_imports_rfdetr_lazily(tmp_path: Path) -> None:
    sys.modules.pop("rfdetr", None)
    backend = RFDETRBackend()

    assert "rfdetr" not in sys.modules
    with pytest.raises(BackendError, match="dataset root"):
        backend.train(make_config(tmp_path))
    assert "rfdetr" not in sys.modules


def test_resolve_checkpoint_uses_explicit_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    config = make_config(tmp_path, checkpoint=checkpoint)

    assert resolve_checkpoint(config) == checkpoint


def test_resolve_checkpoint_finds_likely_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "nested" / "checkpoint_best_ema.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    config = make_config(tmp_path)

    assert resolve_checkpoint(config) == checkpoint


def test_infer_fails_clearly_when_checkpoint_missing(tmp_path: Path) -> None:
    source = tmp_path / "image.jpg"
    source.write_bytes(b"fake")

    with pytest.raises(BackendError, match="No RF-DETR checkpoint found"):
        RFDETRBackend().infer(make_config(tmp_path), source)


def test_train_calls_real_rfdetr_api_shape(monkeypatch: Any, tmp_path: Path) -> None:
    make_rfdetr_yolo_dataset(tmp_path / "dataset")
    calls: dict[str, Any] = {}

    class FakeModel:
        def train(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    fake_module = types.SimpleNamespace(RFDETRBase=lambda: FakeModel())
    monkeypatch.setitem(sys.modules, "rfdetr", fake_module)

    RFDETRBackend().train(make_config(tmp_path))

    assert calls["dataset_dir"] == str(tmp_path / "dataset")
    assert calls["dataset_file"] == "yolo"
    assert calls["epochs"] == 1
    assert calls["batch_size"] == 2
    assert calls["lr"] == 0.0001


def test_single_image_inference_writes_predictions(
    monkeypatch: Any, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint_best_ema.pth"
    checkpoint.write_bytes(b"fake")
    source = tmp_path / "image.jpg"
    source.write_bytes(b"fake")
    output = tmp_path / "out"

    class FakeDetections:
        xyxy = [[1, 2, 3, 4]]
        confidence = [0.9]
        class_id = [0]
        data = {"class_name": ["weapon"]}

    class FakeModel:
        @classmethod
        def from_checkpoint(cls, path: str) -> "FakeModel":
            assert path == str(checkpoint)
            return cls()

        def predict(self, image: str, threshold: float) -> FakeDetections:
            assert image == str(source)
            assert threshold == 0.25
            return FakeDetections()

    fake_module = types.SimpleNamespace(RFDETRBase=FakeModel, RFDETR=FakeModel)
    monkeypatch.setitem(sys.modules, "rfdetr", fake_module)

    RFDETRBackend().infer(make_config(tmp_path, checkpoint=checkpoint), source, output)

    predictions = (output / "predictions.json").read_text(encoding="utf-8")
    assert str(source) in predictions
    assert "weapon" in predictions
    assert "bbox_xyxy" in predictions


def test_image_directory_inference_handles_multiple_images(
    monkeypatch: Any, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint_best_ema.pth"
    checkpoint.write_bytes(b"fake")
    source = tmp_path / "images"
    source.mkdir()
    (source / "a.jpg").write_bytes(b"fake")
    (source / "b.png").write_bytes(b"fake")
    (source / "ignore.txt").write_text("no", encoding="utf-8")
    output = tmp_path / "out"
    seen: list[str] = []

    class FakeDetections:
        xyxy: list[list[float]] = []
        confidence: list[float] = []
        class_id: list[int] = []
        data: dict[str, list[str]] = {}

    class FakeModel:
        @classmethod
        def from_checkpoint(cls, path: str) -> "FakeModel":
            return cls()

        def predict(self, image: str, threshold: float) -> FakeDetections:
            seen.append(image)
            return FakeDetections()

    fake_module = types.SimpleNamespace(RFDETRBase=FakeModel, RFDETR=FakeModel)
    monkeypatch.setitem(sys.modules, "rfdetr", fake_module)

    RFDETRBackend().infer(make_config(tmp_path, checkpoint=checkpoint), source, output)

    assert seen == [str(source / "a.jpg"), str(source / "b.png")]
    predictions = (output / "predictions.json").read_text(encoding="utf-8")
    assert str(source / "a.jpg") in predictions
    assert str(source / "b.png") in predictions
