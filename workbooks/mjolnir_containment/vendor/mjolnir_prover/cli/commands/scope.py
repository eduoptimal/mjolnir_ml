"""scope — what mjolnir decides, and exactly what it declines.

The corpus answer is that 84.6% of units state no claim and 12.7% more need a
value that only exists at run time. The useful response is not a coverage number
but a PRECISE, CHECKABLE scope limit: a buyer can hold this against any corpus
and see whether mjolnir applies, instead of taking a coverage claim on trust.

The promised scope is DATA (`theories/scope.json`, hashed); the counts beside it
are MEASURED. The two cannot drift: every declared class names a predicate, and
this command reports what the predicates actually found.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def register(app: typer.Typer) -> None:

    @app.command()
    def scope(
        paths: list[Path] = typer.Argument(
            ..., exists=True, readable=True,
            help="Paths to measure the scope claim against."),
    ) -> None:
        """Report the declared scope beside the measured scope."""
        from mjolnir_prover.lib import gate as G
        from mjolnir_prover.lib import plan as P

        cfg = G.load_config("mjolnir.toml")
        files = G.collect([str(p) for p in paths],
                          exclude=tuple(cfg.get("exclude", ())))
        units = []
        for f in files:
            for name, code, line in G.units_of(f):
                units.append((name, code, line, f))
        reporting.render_scope(P.scope_report(units))
