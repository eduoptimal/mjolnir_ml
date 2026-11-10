"""policy — compile the Tier-1 claims file (the policy language).

    mjolnir prover policy compile <file> [--name <repo>] [--out <storage key>]

The Tier-1 file (see mjolnir_prover.lib.policy for the locked spec) holds ONE
section (`claims:`) and FOUR keys (name / is / when / then). It compiles to
EXACTLY the mined-candidate schema, so collapse -> confirm -> typed transfer
is unchanged: you write claims, the engine disposes them, the derived POLICY
(the collapsed shape) is what a human signs.

The compile step prints the round-trip the spec promises: for every claim —
its canonical unit (what the program gets), its disposition from the single
authority (declared / proved / observed / refuted), and its target policy
shape (what signing would cover). Exit 1 when a claim comes back REFUTED or
FAULT, per the product's exit-code contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

policy_app = typer.Typer(
    help="Compile the Tier-1 policy claims file to registry candidates.")


def register(app: typer.Typer) -> None:
    app.add_typer(policy_app, name="policy")


def _roundtrip(cands: list) -> tuple:
    """Dispose each candidate through the single authority; print the
    round-trip; -> (counts, worst_status)."""
    from mjolnir_prover.lib import certify, shapes
    from collections import Counter
    counts, worst = Counter(), "CERTIFIED"
    for c in cands:
        r = certify.certify_claim(c)
        st = r.get("status") or "UNKNOWN"
        reason = r.get("reason") or ""
        counts[st] += 1
        if st in ("REFUTED", "FAULT"):
            worst = st
        color = {"CERTIFIED": "green", "REFUTED": "red",
                 "FAULT": "red"}.get(st, "yellow")
        tier = c["tier"]
        venue = {"instance": "  -> OBSERVED (the query is the check)"}.get(
            tier, "")
        typer.secho(f"  {st:<13} {c['predicate']}", fg=color)
        if c.get("canonical_unit"):
            for u in c["canonical_unit"].splitlines():
                typer.echo(f"      {u.strip()}")
        else:
            typer.echo("      (no unit: observation venue)")
        if c.get("premise"):
            typer.echo(f"      when: {c['premise']}")
        typer.echo(f"      policy: {shapes.shape_id(c)}"
                   f"  [tier={tier}]{venue}"
                   + (f"  ({reason})" if reason else ""))
    return counts, worst


@policy_app.command("compile")
def compile_(
    file: Path = typer.Argument(..., exists=True, dir_okay=False,
                                help="Claims file (root key: `claims:`)."),
    name: Optional[str] = typer.Option(None, "--name",
                                        help="Repo/namespace for the claims."),
    out: Optional[str] = typer.Option(None, "--out",
                                       help="Storage key for candidates."),
) -> None:
    """Compile a Tier-1 claims file into `registry/mined_<name>.jsonl`,
    printing the round-trip (unit, disposition, policy shape) for each claim."""
    from mjolnir_prover import storage
    from mjolnir_prover.lib import policy as PL
    try:
        repo = name or file.stem
        cands = PL.compile_file(str(file), repo=repo)
        if not cands:
            typer.secho("policy: no claims in file", fg="red", err=True)
            raise typer.Exit(1)
        key = out or f"registry/mined_{repo}.jsonl"
        storage.get_store().write_jsonl(key, cands)
    except PL.PolicyError as e:
        typer.secho(f"policy: {e}", fg="red", err=True)
        raise typer.Exit(1)
    typer.echo(f"policy: {len(cands)} claim(s) from {file}")
    counts, worst = _roundtrip(cands)
    typer.echo(f"\ndispositions: {dict(counts)}  -> {key}")
    typer.echo("next: sign the derived policies (shape_ui / "
               "`mjolnir prover confirm list`), then "
               "`shape_ui/run_transfer.py --typed`")
    if worst in ("REFUTED", "FAULT"):
        raise typer.Exit(1)
