"""Simuletic Core public Python package."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("simuletic-core")
except PackageNotFoundError:
    __version__ = "0.0.0"

_CONFIG_EXPORTS = {
    "ConfigFileNotFoundError",
    "ConfigLoadError",
    "ConfigValidationError",
    "ConfigYamlError",
    "DatasetConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "load_config",
}
_DATASET_EXPORTS = {
    "DatasetIssue",
    "DatasetValidationResult",
    "validate_config_datasets",
    "validate_dataset",
}

__all__ = [
    "__version__",
    *_CONFIG_EXPORTS,
    *_DATASET_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Lazily expose public helpers without importing optional-heavy modules."""

    if name in _CONFIG_EXPORTS:
        from simuletic_core import config

        return getattr(config, name)
    if name in _DATASET_EXPORTS:
        from simuletic_core import datasets

        return getattr(datasets, name)
    raise AttributeError(f"module 'simuletic_core' has no attribute {name!r}")
