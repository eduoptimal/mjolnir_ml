"""differential — score mjolnir's verdicts against an independent oracle.

The only correctness evidence the stack had before this was mjolnir's kernel
re-deriving mjolnir's verdicts: same code, twice. This command supplies the
missing property for the corpus that matters — an oracle mjolnir did not write.

THE RULE: execution SCORES, it never DISCHARGES. Nothing here feeds back into the
gate. Promoting an executed result into a certificate would be the A-combobound
defect wearing a lab coat.

`--exec` runs vendored third-party source and is therefore OPT-IN.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def register(app: typer.Typer) -> None:

    @app.command()
    def differential(
        paths: list[Path] = typer.Argument(
            ..., exists=True, readable=True,
            help="Python files or directories to score."),
        exec_path: bool = typer.Option(
            False, "--exec",
            help="Run the unit's own body (reaches self-contained units). "
                 "EXECUTES vendored source — opt in deliberately."),
        adapter_name: str = typer.Option(
            "pytest", "--adapter",
            help="Test-framework adapter for real fixture ground truth."),
        pytest_path: bool = typer.Option(
            False, "--pytest",
            help="Run each test with ITS OWN fixtures via pytest, for real "
                 "ground truth on the fixture-taking class. EXECUTES the "
                 "corpus test suites — opt in deliberately."),
        samples: int = typer.Option(16, "--samples", min=1, max=256),
        limit: Optional[int] = typer.Option(None, "--limit",
                                            help="Stop after N units."),
        json_out: Optional[Path] = typer.Option(None, "--json"),
        baseline: Optional[Path] = typer.Option(
            None, "--baseline",
            help="Compare against a recorded baseline and FAIL if precision "
                 "regressed or a false certificate appeared."),
    ) -> None:
        """Score certificates and refutations against Python's evaluator."""
        import os
        from mjolnir_prover.lib import gate as G
        from mjolnir_prover.lib import differential as D

        units = []
        for p in paths:
            p = str(p)
            if os.path.isfile(p):
                cand = [p]
            else:
                cand = []
                for root, dirs, files in os.walk(p):
                    dirs[:] = sorted(d for d in dirs if d != "__pycache__")
                    cand += [os.path.join(root, f) for f in sorted(files)
                             if f.endswith(".py")]
            for f in cand:
                try:
                    src = open(f, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for name, code, line in G.units_of(f):
                    units.append((name, code, line, f, src))
        if limit:
            units = units[:limit]

        if pytest_path:
            rows, counts = D.score_with_adapter(units, adapter_name=adapter_name)
        else:
            rows, counts = D.score_corpus(units, exec_path=exec_path,
                                          samples=samples)
        reporting.render_differential(counts, rows, exec_path, len(units))
        if json_out:
            out = Path(json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        adverse = counts.get("false_certificate", 0) + counts.get("wrong_refutation", 0)
        prec = counts.get("precision") or {}
        if baseline is not None:
            b = json.loads(Path(baseline).read_text(encoding="utf-8"))
            fails = []
            if adverse:
                fails.append(f"{adverse} false certificate(s) — a regression")
            bp, cp = b.get("precision"), prec.get("precision")
            if bp is not None and cp is not None and cp < bp:
                fails.append(f"precision {cp:.3f} < baseline {bp:.3f}")
            bc, cc = b.get("certificates_checkable", 0), prec.get("certificates_checkable", 0)
            if cc < bc:
                fails.append(f"only {cc} certificates checkable, baseline {bc}")
            reporting.render_baseline(b, prec, fails)
            if fails:
                raise typer.Exit(1)
        raise typer.Exit(1 if adverse else 0)
