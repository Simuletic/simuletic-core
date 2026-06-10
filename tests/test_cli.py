from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from simuletic_vision import __version__
from simuletic_vision.cli import app

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


def write_existing_dataset_config(
    tmp_path: Path, checkpoint: Path | None = None
) -> Path:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    checkpoint_line = (
        f"  checkpoint: {checkpoint}\n" if checkpoint else "  checkpoint: null\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
project_name: cctv-weapon-detection
task: detection
backend: rfdetr
datasets:
  synthetic_train:
    path: {dataset_path}
    format: yolo
  synthetic_val:
    path: {dataset_path}
    format: yolo
  real_world_test:
    path: {dataset_path}
    format: yolo
model:
  architecture: rfdetr
  variant: base
  pretrained: true
  output_dir: {tmp_path / "runs"}
{checkpoint_line}  epochs: 1
  batch_size: 2
  learning_rate: 0.0001
  confidence_threshold: 0.25
evaluation:
  metrics:
    - map50
    - precision
    - recall
seed: 42
""",
        encoding="utf-8",
    )
    return config_path


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


def test_train_command_calls_configured_backend(
    monkeypatch: Any, tmp_path: Path
) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("simuletic_vision.cli.get_backend", lambda config: backend)
    config_path = write_existing_dataset_config(tmp_path)

    result = runner.invoke(app, ["train", "--config", str(config_path)])

    assert result.exit_code == 0
    assert backend.calls[0][0] == "train"
    assert "Starting training with backend: rfdetr" in result.stdout
    assert "simuletic-vision evaluate" in result.stdout


def test_train_command_fails_clearly_for_missing_dataset_path(tmp_path: Path) -> None:
    config_path = tmp_path / "missing.yaml"
    config_path.write_text(
        f"""
project_name: missing-dataset
task: detection
backend: rfdetr
datasets:
  synthetic_train:
    path: {tmp_path / "does-not-exist"}
    format: yolo
model:
  architecture: rfdetr
  output_dir: {tmp_path / "runs"}
evaluation:
  metrics:
    - precision
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["train", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Dataset path does not exist" in result.stdout


def test_evaluate_command_calls_configured_backend(
    monkeypatch: Any, tmp_path: Path
) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("simuletic_vision.cli.get_backend", lambda config: backend)
    config_path = write_existing_dataset_config(tmp_path)

    result = runner.invoke(app, ["evaluate", "--config", str(config_path)])

    assert result.exit_code == 0
    assert backend.calls[0][0] == "evaluate"
    assert "Starting evaluation with backend: rfdetr" in result.stdout


def test_infer_command_calls_configured_backend(
    monkeypatch: Any, tmp_path: Path
) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("simuletic_vision.cli.get_backend", lambda config: backend)
    config_path = write_existing_dataset_config(tmp_path)
    source = tmp_path / "images"
    output = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "infer",
            "--config",
            str(config_path),
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
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(
        f"""
project_name: custom-project
task: detection
backend: custom
datasets:
  synthetic_train:
    path: {dataset_path}
    format: yolo
model:
  architecture: custom
  output_dir: {tmp_path / "runs"}
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
