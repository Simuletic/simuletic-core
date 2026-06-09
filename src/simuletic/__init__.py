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
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "load_config",
]
