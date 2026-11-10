"""countability — the isolation / countability gate, as a command.

`mjolnir countability <paths>` runs the gate over the given sources and prints
the two artifacts a third party can recompute without mjolnir:

  * the coverage INTEGER over a hashed, canonically ordered obligation manifest
  * the ISOLATION verdict — whether the kernel re-derives the certificates in a
    process that can only import its own TCB (kernel + term), plus which
    certificate fields are measured decisive vs inert

This is the PC-2c discharge: coverage becomes a count, and the verdict is shown
to be a function only of the certificate.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def register(app: typer.Typer) -> None:

    @app.command()
    def countability(
        paths: list[Path] = typer.Argument(
            ..., exists=True, readable=True,
            help="Python files or directories whose certificates are counted."),
        json_out: Optional[Path] = typer.Option(
            None, "--json", help="Write the manifest + audit to this JSON file."),
        manifest_only: bool = typer.Option(
            False, "--manifest-only",
            help="Emit the manifest and coverage integer; skip the isolation "
                 "subprocess (for large corpora)."),
        frontier: bool = typer.Option(
            False, "--frontier",
            help="Also size the abstention population: instances per blocking "
                 "construct, and the rung ROI ranked from that measurement."),
    ) -> None:
        """Count coverage over a hashed manifest; check verdict isolation."""
        from mjolnir_prover.lib import certify
        from mjolnir_prover.lib import countability as CT
        from mjolnir_prover.lib import gate as engine

        result = engine.run([str(p) for p in paths])
        certs = result["certificates"]
        man = CT.manifest(certs)
        cov = CT.coverage(certs)
        audit = {
            "manifest": man,
            "coverage": cov,
            "isolation": None,
            "isolated": None,
            "frontier": None,
        }
        if frontier:
            from mjolnir_prover.lib import frontier as FR
            audit["frontier"] = {"population": FR.population(certs),
                                 "rung_roi": FR.rung_roi(certs)}
        # probe the isolation on a live certificate (needs a real one, not a count)
        probe = next((c for c in certs if c.get("obligations")), None)
        if probe is not None and not manifest_only:
            audit["isolation"] = CT.isolation(probe)
            audit["isolated"] = audit["isolation"]["isolated"]

        reporting.render_countability(audit, result)

        if json_out:
            path = Path(json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(audit, indent=2, sort_keys=True),
                            encoding="utf-8")

        # exit 0 always unless a false certificate was counted; abstention is
        # not a gate failure (this gate reports coverage, it does not enforce it)
        bad = cov["false_certificates"] > 0
        raise typer.Exit(1 if bad else 0)
