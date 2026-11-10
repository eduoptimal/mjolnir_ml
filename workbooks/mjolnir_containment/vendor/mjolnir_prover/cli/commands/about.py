"""version — report the installed mjolnir version."""
from __future__ import annotations

import typer

from mjolnir_prover.cli import reporting


def get_version() -> str:
    try:
        from importlib.metadata import version as _v
        return _v("mjolnir")
    except Exception:
        return "0.1.0"


def register(app: typer.Typer) -> None:

    @app.command()
    def version() -> None:
        """Print the mjolnir version."""
        reporting.console.print(f"mjolnir {get_version()}")