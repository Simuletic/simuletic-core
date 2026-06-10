from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from simuletic_core import __version__
from simuletic_core.cli import app

runner = CliRunner()


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def train(self, config: Any) -> None:
        self.calls.append(("train", config))

    def evaluate(self, config: Any) -> None:
        self.calls.append(("evaluate", config))

    def infer(self, config: Any, source: Path, output: Path | None = None) -> None:
        self.calls.append(("infer", config, source, output))


def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_config_validate_command_prints_summary() -> None:
    result = runner.invoke(
        app, ["config", "validate", "examples/configs/cctv_weapon_detection.yaml"]
    )

    assert result.exit_code == 0
    assert "Config is valid" in result.stdout
    assert "Project: cctv-weapon-detection" in result.stdout
    assert "Task: detection" in result.stdout
    assert "Backend: rfdetr" in result.stdout
    assert "synthetic_train" in result.stdout


def test_config_validate_command_fails_clearly_for_invalid_path() -> None:
    result = runner.invoke(app, ["config", "validate", "does-not-exist.yaml"])

    assert result.exit_code == 1
    assert "Invalid config" in result.stdout
    assert "Configuration file not found" in result.stdout


def test_dataset_validate_command_fails_clearly_for_missing_example_paths() -> None:
    result = runner.invoke(
        app,
        [
            "dataset",
            "validate",
            "--config",
            "examples/configs/cctv_weapon_detection.yaml",
        ],
    )

    assert result.exit_code == 1
    assert "Dataset validation" in result.stdout
    assert "FAIL" in result.stdout
    assert "Dataset path does not exist" in result.stdout
    assert "synthetic_train" in result.stdout


def test_train_command_calls_configured_backend(monkeypatch: Any) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("simuletic_core.cli.get_backend", lambda config: backend)

    result = runner.invoke(
        app, ["train", "--config", "examples/configs/cctv_weapon_detection.yaml"]
    )

    assert result.exit_code == 0
    assert backend.calls[0][0] == "train"
    assert "Starting training with backend: rfdetr" in result.stdout


def test_evaluate_command_calls_configured_backend(monkeypatch: Any) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("simuletic_core.cli.get_backend", lambda config: backend)

    result = runner.invoke(
        app, ["evaluate", "--config", "examples/configs/cctv_weapon_detection.yaml"]
    )

    assert result.exit_code == 0
    assert backend.calls[0][0] == "evaluate"
    assert "Starting evaluation with backend: rfdetr" in result.stdout


def test_infer_command_calls_configured_backend(
    monkeypatch: Any, tmp_path: Path
) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("simuletic_core.cli.get_backend", lambda config: backend)
    source = tmp_path / "images"
    output = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "infer",
            "--config",
            "examples/configs/cctv_weapon_detection.yaml",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert backend.calls[0][0] == "infer"
    assert backend.calls[0][2] == source
    assert backend.calls[0][3] == output
    assert "Starting inference with backend: rfdetr" in result.stdout


def test_custom_backend_command_fails_clearly(tmp_path: Path) -> None:
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        """
project_name: custom-project
task: detection
backend: custom
datasets:
  synthetic_train:
    path: ./data/synthetic/train
    format: yolo
model:
  architecture: custom
  output_dir: ./runs/custom
evaluation:
  metrics:
    - precision
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["train", "--config", str(custom_config)])

    assert result.exit_code == 1
    assert "Backend error" in result.stdout
    assert "custom" in result.stdout
