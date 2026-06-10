from pathlib import Path

import pytest

from simuletic_core.config import (
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigYamlError,
    ExperimentConfig,
    load_config,
)

VALID_CONFIG = """
project_name: cctv-weapon-detection
task: detection
backend: rfdetr

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
  architecture: rfdetr
  variant: base
  pretrained: true
  output_dir: ./runs/cctv-weapon-detection
  epochs: 1
  batch_size: 2
  learning_rate: 0.0001

evaluation:
  metrics:
    - precision
    - recall
    - map50
    - false_positive_rate
    - synthetic_to_real_gap

seed: 42
"""


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_load_config_loads_valid_config(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    assert isinstance(config, ExperimentConfig)
    assert config.project_name == "cctv-weapon-detection"
    assert config.task == "detection"
    assert config.backend == "rfdetr"
    assert list(config.datasets) == [
        "synthetic_train",
        "synthetic_val",
        "real_world_test",
        "hard_negatives",
    ]
    assert config.datasets["synthetic_train"].format == "yolo"
    assert config.model.pretrained is True
    assert config.model.variant == "base"
    assert config.model.epochs == 1
    assert config.model.batch_size == 2
    assert config.model.learning_rate == 0.0001
    assert config.evaluation.metrics[-1] == "synthetic_to_real_gap"
    assert config.seed == 42


def test_load_config_raises_for_missing_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigFileNotFoundError, match="Configuration file not found"):
        load_config(missing_path)


def test_load_config_raises_for_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigYamlError, match="Invalid YAML"):
        load_config(write_config(tmp_path, ":\n"))


def test_load_config_raises_for_invalid_task(tmp_path: Path) -> None:
    invalid_config = VALID_CONFIG.replace("task: detection", "task: tracking")

    with pytest.raises(ConfigValidationError, match="task"):
        load_config(write_config(tmp_path, invalid_config))


def test_load_config_raises_for_invalid_dataset_format(tmp_path: Path) -> None:
    invalid_config = VALID_CONFIG.replace("format: yolo", "format: voc", 1)

    with pytest.raises(ConfigValidationError, match="format"):
        load_config(write_config(tmp_path, invalid_config))


def test_load_config_raises_for_missing_required_fields(tmp_path: Path) -> None:
    invalid_config = VALID_CONFIG.replace("project_name: cctv-weapon-detection\n", "")

    with pytest.raises(ConfigValidationError, match="project_name"):
        load_config(write_config(tmp_path, invalid_config))
