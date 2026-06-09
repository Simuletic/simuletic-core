"""Simuletic public Python package."""

from importlib.metadata import PackageNotFoundError, version

from simuletic.config import (
    ConfigFileNotFoundError,
    ConfigLoadError,
    ConfigValidationError,
    ConfigYamlError,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    load_config,
)
from simuletic.datasets import (
    DatasetIssue,
    DatasetValidationResult,
    validate_config_datasets,
    validate_dataset,
)

try:
    __version__ = version("simuletic-core")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "ConfigFileNotFoundError",
    "ConfigLoadError",
    "ConfigValidationError",
    "ConfigYamlError",
    "DatasetIssue",
    "DatasetValidationResult",
    "validate_config_datasets",
    "validate_dataset",
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "load_config",
]
