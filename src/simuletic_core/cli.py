"""Command-line interface for Simuletic."""

from pathlib import Path

import typer
from rich.console import Console

from simuletic.config import ConfigLoadError, load_config
from simuletic_core import __version__

app = typer.Typer(
    add_completion=False,
    help="Simuletic core toolkit for synthetic-to-real computer vision workflows.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Inspect and validate experiment configuration files.")
app.add_typer(config_app, name="config")
console = Console()


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
