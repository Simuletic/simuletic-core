"""Model backend selection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Backend, BackendError

if TYPE_CHECKING:
    from simuletic_core.config import ExperimentConfig


def get_backend(config: ExperimentConfig) -> Backend:
    """Return the backend implementation requested by an experiment config."""

    if config.backend == "rfdetr":
        from .rfdetr import RFDETRBackend

        return RFDETRBackend()
    if config.backend == "custom":
        raise BackendError(
            "Backend 'custom' is a placeholder and does not provide train, "
            "evaluate, or infer commands yet. Set backend: rfdetr to use the "
            "implemented RF-DETR backend."
        )
    raise BackendError(f"Unsupported backend: {config.backend}")


__all__ = ["Backend", "BackendError", "get_backend"]
