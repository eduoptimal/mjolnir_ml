"""confirm — shape-level confirmation of the mined registry.

Turns thousands of candidate sign-offs into a handful of shape decisions.
Only shapes with decision=confirmed may become enforced directives; everything
else stays fail-open (monitor/observation tier).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting

GROUP_HELP = "Shape-level confirmation of the mined registry."
_DECISION_HELP = ("Shape id to act on (see `mjolnir confirm list`).")


def register(app: typer.Typer) -> None:
    confirm_app = typer.Typer(help=GROUP_HELP, no_args_is_help=True,
                              add_completion=False)

    def _registry(registry: Optional[Path]) -> Optional[str]:
        return str(registry) if registry else None

    @confirm_app.command("list")
    def list_shapes(
        limit: int = typer.Option(25, "--limit", help="Max shapes to show."),
        registry: Optional[Path] = typer.Option(
            None, "--registry", exists=True, dir_okay=False,
            help="Explicit candidate file instead of the registry dir."),
    ) -> None:
        """List shapes ordered by review load (undecided first)."""
        from mjolnir_prover.lib import review as engine
        reporting.render_shape_list(engine.list_shapes(_registry(registry), limit))

    @confirm_app.command("show")
    def show_shape(
        shape_id: str = typer.Argument(..., help="Shape id."),
        registry: Optional[Path] = typer.Option(
            None, "--registry", exists=True, dir_okay=False),
    ) -> None:
        """Show the evidence bundle for one shape."""
        from mjolnir_prover.lib import review as engine
        try:
            res = engine.show_shape(shape_id, _registry(registry))
        except KeyError:
            reporting.console.print(f"[red]unknown shape:[/red] {shape_id}")
            raise typer.Exit(2)
        reporting.render_shape_show(res)

    @confirm_app.command("accept")
    def accept(
        shape_id: str = typer.Argument(..., help=_DECISION_HELP),
        by: str = typer.Option("human", "--by", help="Reviewer identity."),
        note: str = typer.Option("", "--note", help="Decision note."),
        registry: Optional[Path] = typer.Option(
            None, "--registry", exists=True, dir_okay=False),
    ) -> None:
        """Confirm a shape; its instances inherit the decision."""
        from mjolnir_prover.lib import review as engine
        try:
            res = engine.record_decision(shape_id, "confirmed", by=by, note=note,
                                         registry=_registry(registry))
        except KeyError:
            reporting.console.print(f"[red]unknown shape:[/red] {shape_id}")
            raise typer.Exit(2)
        reporting.render_shape_decision(res)

    @confirm_app.command("reject")
    def reject(
        shape_id: str = typer.Argument(..., help=_DECISION_HELP),
        by: str = typer.Option("human", "--by", help="Reviewer identity."),
        note: str = typer.Option("", "--note", help="Decision note."),
        registry: Optional[Path] = typer.Option(
            None, "--registry", exists=True, dir_okay=False),
    ) -> None:
        """Reject a shape and record the veto in the ledger."""
        from mjolnir_prover.lib import review as engine
        try:
            res = engine.record_decision(shape_id, "rejected", by=by, note=note,
                                         registry=_registry(registry))
        except KeyError:
            reporting.console.print(f"[red]unknown shape:[/red] {shape_id}")
            raise typer.Exit(2)
        reporting.render_shape_decision(res)

    @confirm_app.command("enforce")
    def enforce(
        shape_id: str = typer.Argument(..., help=_DECISION_HELP),
        by: str = typer.Option("human", "--by", help="Reviewer identity."),
        note: str = typer.Option("", "--note", help="Decision note."),
        registry: Optional[Path] = typer.Option(
            None, "--registry", exists=True, dir_okay=False),
    ) -> None:
        """Sign a RUNTIME-ENFORCEMENT decision for an observation-class shape.

        The shape becomes a runtime contract for the enforcement plane. This is
        bookkeeping only — it can never upgrade a verdict: proofs come from
        disposal rungs through the verdict authority, never from sign-off."""
        from mjolnir_prover.lib import review as engine
        try:
            res = engine.record_decision(shape_id, "enforced", by=by, note=note,
                                         registry=_registry(registry))
        except KeyError:
            reporting.console.print(f"[red]unknown shape:[/red] {shape_id}")
            raise typer.Exit(2)
        reporting.render_shape_decision(res)

    @confirm_app.command("apply")
    def apply(
        registry: Optional[Path] = typer.Option(
            None, "--registry", exists=True, dir_okay=False),
    ) -> None:
        """Project the ledger onto the candidate set (in memory)."""
        from mjolnir_prover.lib import review as engine
        reporting.render_shape_apply(engine.apply_ledger(_registry(registry)))

    @confirm_app.command("transfer")
    def transfer(
        from_file: Path = typer.Argument(..., exists=True, dir_okay=False,
                                         help="Source candidate file."),
        to_file: Path = typer.Argument(..., exists=True, dir_okay=False,
                                       help="Target candidate file."),
        generate: bool = typer.Option(
            False, "--generate",
            help="Cold start: instantiate confirmed shapes into the target's "
                 "environment and validate before adopting (no matches required)."),
        symbols: Optional[str] = typer.Option(
            None, "--symbols", help="Comma-separated target symbols for --generate."),
        literals: Optional[str] = typer.Option(
            None, "--literals", help="Comma-separated target literals for --generate."),
    ) -> None:
        """Adopt confirmed shapes onto a target registry, validating first."""
        from mjolnir_prover.lib import review as engine
        if generate:
            sym = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
            lit = [s.strip() for s in literals.split(",") if s.strip()] if literals else None
            res = engine.transfer_generate(str(from_file), str(to_file),
                                           symbols=sym, literals=lit)
        else:
            res = engine.transfer(str(from_file), str(to_file))
        reporting.render_shape_transfer(res)

    app.add_typer(confirm_app, name="confirm")