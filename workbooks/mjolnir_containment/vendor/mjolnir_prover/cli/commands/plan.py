"""plan — the mandatory front door for "what should we build?"

That question got a different answer each time it was asked, because it was being
answered from intuition and from instance counts on whichever corpus was to
hand. Both are corpus-dependent. This command replaces the intuition with a
measurement and makes the ordering a dependency rather than a preference:

    stage 1  no_claim           assert nothing            -> write the claim
    stage 2  no_value           an operand has no value   -> SCOPE LIMIT
    stage 3  decidable_unproved decidable, no rule for it -> extend the normal form

A stage-3 fix cannot move a claim stuck at stage 1, whatever its size. So the
corpus changes how BIG each stage is and never which stage comes next.

There is no `--just-tell-me` path. The buckets are measured or the command says
it has no answer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def register(app: typer.Typer) -> None:

    @app.command()
    def plan(
        paths: list[Path] = typer.Argument(
            ..., exists=True, readable=True,
            help="Paths to measure. The buckets are MEASURED, never assumed."),
        limit: Optional[int] = typer.Option(None, "--limit",
                                            help="Stop after N units."),
        json_out: Optional[Path] = typer.Option(None, "--json"),
    ) -> None:
        """Measure the three buckets, then derive ONE ordered recommendation."""
        from mjolnir_prover.lib import gate as G
        from mjolnir_prover.lib import plan as P

        cfg = G.load_config("mjolnir.toml")
        files = G.collect([str(p) for p in paths],
                          exclude=tuple(cfg.get("exclude", ())))
        units = []
        for f in files:
            for name, code, line in G.units_of(f):
                units.append((name, code, line, f))
                if limit and len(units) >= limit:
                    break
            if limit and len(units) >= limit:
                break

        result = P.build_plan(units)
        reporting.render_plan(result)
        if json_out:
            out = Path(json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
