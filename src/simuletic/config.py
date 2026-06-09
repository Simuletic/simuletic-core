"""Experiment configuration schema and loading utilities."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from yaml import YAMLError

TaskName = Literal["detection", "classification", "segmentation", "pose"]
BackendName = Literal["rtdetr", "custom"]
DatasetFormat = Literal["yolo", "coco", "imagefolder", "csv", "custom"]
MetricName = Literal[
    "precision",
    "recall",
    "map50",
    "map",
    "false_positive_rate",
    "synthetic_to_real_gap",
]


class ConfigLoadError(ValueError):
    """Base error for configuration loading failures."""


class ConfigYamlError(ConfigLoadError):
    """Raised when a configuration file cannot be parsed as YAML."""


class ConfigValidationError(ConfigLoadError):
    """Raised when a configuration file does not match the schema."""


class ConfigFileNotFoundError(FileNotFoundError):
    """Raised when a configuration file path does not exist."""


class DatasetConfig(BaseModel):
    """Dataset location and annotation format."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    format: DatasetFormat


class ModelConfig(BaseModel):
    """Model backend settings for an experiment."""

    model_config = ConfigDict(extra="forbid")

    architecture: BackendName
    pretrained: bool = True
    output_dir: Path


class EvaluationConfig(BaseModel):
    """Evaluation metric settings for an experiment."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricName] = Field(min_length=1)


class ExperimentConfig(BaseModel):
    """Top-level Simuletic experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(min_length=1)
    task: TaskName
    backend: BackendName
    datasets: dict[str, DatasetConfig] = Field(min_length=1)
    model: ModelConfig
    evaluation: EvaluationConfig
    seed: int | None = None


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a Simuletic experiment configuration from YAML."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigFileNotFoundError(f"Configuration file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigFileNotFoundError(
            f"Configuration path is not a file: {config_path}"
        )

    try:
        raw_config: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except YAMLError as exc:
        raise ConfigYamlError(
            f"Invalid YAML in configuration file {config_path}: {exc}"
        ) from exc

    if raw_config is None:
        raw_config = {}

    try:
        return ExperimentConfig.model_validate(raw_config)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"Invalid configuration in {config_path}: {exc}"
        ) from exc
