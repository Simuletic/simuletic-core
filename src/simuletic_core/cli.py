"""Command-line interface for Simuletic."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from simuletic.config import ConfigLoadError, load_config
from simuletic.datasets import DatasetValidationResult, validate_config_datasets
from simuletic_core import __version__

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
    try:
        experiment_config = load_config(config)
    except (ConfigLoadError, FileNotFoundError) as exc:
        console.print(f"[red]Invalid config:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    results = validate_config_datasets(experiment_config)
    _print_dataset_validation_summary(results)

    if any(not result.passed for result in results):
        raise typer.Exit(code=1)


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
