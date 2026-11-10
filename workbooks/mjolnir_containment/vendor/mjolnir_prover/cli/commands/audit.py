"""audit — codebase audit product: corpus census, coverage ladder, Action Report.

Thin command wiring; all work is in :mod:`mjolnir_prover.audit`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def register(app: typer.Typer) -> None:

    @app.command()
    def audit(
        corpus: Path = typer.Option(
            ..., "--corpus", exists=True, file_okay=False,
            help="Directory holding the code corpus to audit."),
        redispose: bool = typer.Option(
            False, "--redispose",
            help="Re-dispose all mined contracts -> coverage ladder."),
    ) -> None:
        """Read-only corpus census + gap map (+ coverage ladder with --redispose)."""
        from mjolnir_prover.audit import sql_audit
        res = sql_audit.audit(str(corpus))
        reporting.console.print(sql_audit.render(res))
        if redispose:
            reporting.console.print()
            reporting.console.print(
                sql_audit.render_ladder(sql_audit.redispose(str(corpus))))
        raise typer.Exit(0)

    @app.command(name="bank-report")
    def bank_report(
        corpus: Path = typer.Option(
            ..., "--corpus", exists=True, file_okay=False,
            help="Directory holding the code corpus to audit."),
        out: Optional[Path] = typer.Option(
            None, "--out", help="Output HTML path (defaults under storage)."),
        no_rules: bool = typer.Option(
            False, "--no-rules",
            help="Base tally only: exclude accepted rules (bindings / typed transfer) "
                 "so you can compare the corpus-derived tally."),
    ) -> None:
        """Generate the management-facing Action Report for a corpus."""
        from mjolnir_prover.audit import bank_report as br
        dest = br.build(str(corpus), str(out) if out else None,
                        include_rules=not no_rules)
        reporting.console.print(f"wrote {dest}")
        raise typer.Exit(0)
