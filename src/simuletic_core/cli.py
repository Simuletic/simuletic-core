"""Command-line interface for Simuletic."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from simuletic_core import __version__

if TYPE_CHECKING:
    from simuletic_core.backends import Backend
    from simuletic_core.config import ExperimentConfig
    from simuletic_core.datasets import DatasetValidationResult

app = typer.Typer(
    add_completion=False,
    help="Simuletic core toolkit for synthetic-to-real computer vision workflows.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Inspect and validate experiment configuration files.")
dataset_app = typer.Typer(
    help="Validate configured datasets before training or evaluation."
)
app.add_typer(config_app, name="config")
app.add_typer(dataset_app, name="dataset")
console = Console(width=120)


@app.callback()
def main() -> None:
    """Run the Simuletic command-line interface."""


@app.command("version")
def version_command() -> None:
    """Print the installed simuletic-core package version."""
    console.print(__version__)


@config_app.command("validate")
def validate_config_command(path: Path) -> None:
    """Validate a Simuletic experiment YAML configuration file."""
    from simuletic_core.config import ConfigLoadError, load_config

    try:
        config = load_config(path)
    except (ConfigLoadError, FileNotFoundError) as exc:
        console.print(f"[red]Invalid config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    dataset_keys = ", ".join(config.datasets.keys())
    console.print(f"[green]Config is valid:[/green] {path}")
    console.print(f"Project: {config.project_name}")
    console.print(f"Task: {config.task}")
    console.print(f"Backend: {config.backend}")
    console.print(f"Datasets: {dataset_keys}")


@dataset_app.command("validate")
def validate_dataset_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a Simuletic experiment YAML configuration file.",
        ),
    ],
) -> None:
    """Validate datasets referenced by a Simuletic experiment config."""
    from simuletic_core.config import ConfigLoadError, load_config
    from simuletic_core.datasets import validate_config_datasets

    try:
        experiment_config = load_config(config)
    except (ConfigLoadError, FileNotFoundError) as exc:
        console.print(f"[red]Invalid config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    results = validate_config_datasets(experiment_config)
    _print_dataset_validation_summary(results)

    if any(not result.passed for result in results):
        raise typer.Exit(code=1)


@app.command("train")
def train_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a Simuletic experiment YAML configuration file.",
        ),
    ],
) -> None:
    """Train or fine-tune a model with the configured backend."""
    experiment_config = _load_experiment_config(config)
    console.print(f"Starting training with backend: {experiment_config.backend}")
    console.print(f"Output directory: {experiment_config.model.output_dir}")
    _run_backend_command(experiment_config, "train")
    console.print("[green]Training command completed.[/green]")


@app.command("evaluate")
def evaluate_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a Simuletic experiment YAML configuration file.",
        ),
    ],
) -> None:
    """Evaluate a model with the configured backend."""
    experiment_config = _load_experiment_config(config)
    console.print(f"Starting evaluation with backend: {experiment_config.backend}")
    _run_backend_command(experiment_config, "evaluate")
    console.print("[green]Evaluation command completed.[/green]")


@app.command("infer")
def infer_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to a Simuletic experiment YAML configuration file.",
        ),
    ],
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            "-s",
            help="Image file or image directory to run inference on.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Directory for inference outputs. Defaults under runs/.",
        ),
    ] = None,
) -> None:
    """Run model inference on a single image or image directory."""
    experiment_config = _load_experiment_config(config)
    output_dir = output or Path("runs") / experiment_config.project_name / "inference"
    console.print(f"Starting inference with backend: {experiment_config.backend}")
    console.print(f"Source: {source}")
    console.print(f"Output directory: {output_dir}")
    _run_backend_command(experiment_config, "infer", source=source, output=output)
    console.print("[green]Inference command completed.[/green]")


def _load_experiment_config(config_path: Path) -> ExperimentConfig:
    from simuletic_core.config import ConfigLoadError, load_config

    try:
        return load_config(config_path)
    except (ConfigLoadError, FileNotFoundError) as exc:
        console.print(f"[red]Invalid config:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def get_backend(experiment_config: ExperimentConfig) -> Backend:
    """Return the configured backend without importing backend modules at CLI import."""

    from simuletic_core.backends import get_backend as load_backend

    return load_backend(experiment_config)


def _run_backend_command(
    experiment_config: ExperimentConfig,
    command: str,
    source: Path | None = None,
    output: Path | None = None,
) -> None:
    from simuletic_core.backends import BackendError

    try:
        backend = get_backend(experiment_config)
        if command == "train":
            backend.train(experiment_config)
        elif command == "evaluate":
            backend.evaluate(experiment_config)
        elif command == "infer":
            if source is None:
                raise BackendError("Inference source is required.")
            backend.infer(experiment_config, source=source, output=output)
        else:
            raise BackendError(f"Unsupported backend command: {command}")
    except BackendError as exc:
        console.print(f"[red]Backend error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _print_dataset_validation_summary(
    results: list[DatasetValidationResult],
) -> None:
    """Print a readable dataset validation summary."""
    table = Table(title="Dataset validation")
    table.add_column("Dataset")
    table.add_column("Status")
    table.add_column("Format")
    table.add_column("Path")
    table.add_column("Images", justify="right")
    table.add_column("Labels", justify="right")
    table.add_column("Issues / warnings")

    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        image_count = _format_optional_count(result.image_count)
        label_count = _format_optional_count(result.label_file_count)
        issue_summary = _format_issue_summary(result)
        table.add_row(
            result.dataset_key,
            status,
            result.dataset_format,
            str(result.dataset_path),
            image_count,
            label_count,
            issue_summary,
        )

    console.print(table)


def _format_optional_count(value: int | None) -> str:
    return "—" if value is None else str(value)


def _format_issue_summary(result: DatasetValidationResult) -> str:
    if not result.issues:
        return "[green]No issues[/green]"

    formatted_issues: list[str] = []
    for issue in result.issues:
        prefix = (
            "[red]error[/red]"
            if issue.severity == "error"
            else "[yellow]warning[/yellow]"
        )
        if issue.path is None:
            formatted_issues.append(f"{prefix}: {issue.message}")
        else:
            formatted_issues.append(f"{prefix}: {issue.message} ({issue.path})")
    return "\n".join(formatted_issues)

if __name__ == "__main__":
    app()
