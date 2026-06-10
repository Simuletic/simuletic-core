"""Backend interface definitions."""

from pathlib import Path
from typing import Protocol

from simuletic_vision.config import ExperimentConfig


class BackendError(RuntimeError):
    """Raised when a model backend cannot complete a requested operation."""


class Backend(Protocol):
    """Minimal interface implemented by Simuletic model backends."""

    def train(self, config: ExperimentConfig) -> None:
        """Train or fine-tune a model for an experiment."""

    def evaluate(self, config: ExperimentConfig) -> None:
        """Evaluate a model for an experiment."""

    def infer(
        self, config: ExperimentConfig, source: Path, output: Path | None = None
    ) -> None:
        """Run inference for an experiment."""
