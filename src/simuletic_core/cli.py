"""Command-line interface for Simuletic."""

import typer
from rich.console import Console

from simuletic_core import __version__

app = typer.Typer(
    add_completion=False,
    help="Simuletic core toolkit for synthetic-to-real computer vision workflows.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Run the Simuletic command-line interface."""


@app.command("version")
def version_command() -> None:
    """Print the installed simuletic-core package version."""
    console.print(__version__)
