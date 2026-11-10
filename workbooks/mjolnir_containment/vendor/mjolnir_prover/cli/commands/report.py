"""report — generate the management-facing verification report."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def register(app: typer.Typer) -> None:

    @app.command()
    def report(
        vendor: Optional[Path] = typer.Option(
            None, "--vendor", envvar="MJ_VENDOR", exists=True, file_okay=False,
            help="Directory holding the trusted repos to scan "
                 "(defaults to $MJ_VENDOR or mjolnir_prover/storage/data/repos)."),
    ) -> None:
        """Run mining + confirmation + gate and emit the report artifacts."""
        if vendor:
            os.environ["MJ_VENDOR"] = str(vendor)
        from mjolnir_prover.report import generate
        try:
            res = generate.run()
        except Exception as e:
            msg = " ".join(str(e).split())
            reporting.console.print(
                f"[red]report failed:[/red] {type(e).__name__}: {msg}")
            raise typer.Exit(1)
        reporting.render_report(res["summary"], res["artifacts"])
        raise typer.Exit(0)