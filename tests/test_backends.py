import sys
from pathlib import Path

import pytest

from simuletic_core.backends import BackendError, get_backend
from simuletic_core.backends.rfdetr import RFDETRBackend
from simuletic_core.config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
)


def make_config(backend: str = "rfdetr") -> ExperimentConfig:
    return ExperimentConfig(
        project_name="test-project",
        task="detection",
        backend=backend,  # type: ignore[arg-type]
        datasets={
            "synthetic_train": DatasetConfig(path=Path("data/train"), format="yolo"),
            "real_world_test": DatasetConfig(path=Path("data/test"), format="yolo"),
        },
        model=ModelConfig(architecture=backend, output_dir=Path("runs/test")),  # type: ignore[arg-type]
        evaluation=EvaluationConfig(metrics=["precision"]),
    )


def test_get_backend_selects_rfdetr_backend() -> None:
    backend = get_backend(make_config())

    assert isinstance(backend, RFDETRBackend)


def test_get_backend_fails_clearly_for_custom_backend() -> None:
    with pytest.raises(BackendError, match="custom"):
        get_backend(make_config("custom"))


def test_rfdetr_backend_imports_rfdetr_lazily() -> None:
    backend = RFDETRBackend()

    assert "rfdetr" not in sys.modules
    with pytest.raises(BackendError, match="dataset root"):
        backend.train(make_config())
    assert "rfdetr" not in sys.modules
