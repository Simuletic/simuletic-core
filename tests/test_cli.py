from typer.testing import CliRunner

from simuletic_core import __version__
from simuletic_core.cli import app

runner = CliRunner()


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
    assert "Backend: rtdetr" in result.stdout
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
