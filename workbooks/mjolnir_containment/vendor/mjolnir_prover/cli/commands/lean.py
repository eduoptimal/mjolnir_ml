"""lean — verify Lean statements (one per line) through the IR frontend."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def run_lean(path: str, use_kernel: bool):
    """Dispose each statement through the single verdict authority."""
    from mjolnir_prover.lib import certify

    stmts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                stmts.append(json.loads(line).get("stmt", ""))
            else:
                stmts.append(line)

    rows = []
    counts = {"CERTIFIED": 0, "REFUTED": 0, "UNVERIFIABLE": 0, "FAULT": 0}
    kernel_ok = kernel_bad = 0
    for s in stmts:
        r = certify.certify_lean(s)
        v = r["status"]
        counts[v if v in counts else "UNVERIFIABLE"] += 1
        rec = {"stmt": s, "verdict": v}
        if use_kernel and r.get("kernel"):
            rec["kernel"] = r["kernel"]
            if r["kernel"] == "CERTIFIED" and v == "CERTIFIED":
                kernel_ok += 1
            elif v in ("CERTIFIED", "FAULT"):
                kernel_bad += 1
        rows.append(rec)
    return rows, counts, kernel_ok, kernel_bad


def register(app: typer.Typer) -> None:

    @app.command()
    def lean(
        file: Path = typer.Argument(
            ..., exists=True, readable=True, dir_okay=False,
            help="Text file with one Lean theorem per line (or JSONL with a "
                 "'stmt' field)."),
        json_out: Optional[Path] = typer.Option(
            None, "--json", help="Write the per-statement verdicts to JSON."),
        kernel: bool = typer.Option(
            False, "--kernel",
            help="Re-check each CERTIFIED unit with the independent kernel."),
    ) -> None:
        """Verify Lean statements and report verdicts (optionally via kernel)."""
        rows, counts, kernel_ok, kernel_bad = run_lean(str(file), kernel)
        if json_out:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            with open(json_out, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)
        reporting.render_lean(rows, counts, kernel_ok, kernel_bad,
                              str(json_out) if json_out else None, kernel)
        raise typer.Exit(0 if kernel_bad == 0 else 1)