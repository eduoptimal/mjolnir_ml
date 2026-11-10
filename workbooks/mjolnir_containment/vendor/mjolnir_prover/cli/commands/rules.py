"""rules — compile the domain-rule DSL into the registry candidate format.

    mjolnir rules compile <file.dsl> [--name <repo>] [--out registry/mined_<name>.jsonl]

The DSL (see mjolnir_prover/lib/dsl.py) is syntactic sugar that compiles to EXACTLY the
mined-candidate schema, so collapse -> confirm -> typed transfer are unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

rules_app = typer.Typer(help="Compile the domain-rule DSL to registry candidates.")


def register(app: typer.Typer) -> None:
    app.add_typer(rules_app, name="rules")


@rules_app.command("compile")
def compile_(
    file: Path = typer.Argument(..., exists=True, dir_okay=False,
                                help="DSL file (.dsl / .rules / .txt)."),
    name: Optional[str] = typer.Option(None, "--name", help="Repo/namespace for the rules."),
    out: Optional[str] = typer.Option(None, "--out", help="Storage key for candidates."),
) -> None:
    """Compile a DSL file into `registry/mined_<name>.jsonl` (proposer only)."""
    from mjolnir_prover import storage
    from mjolnir_prover.lib import dsl
    try:
        repo = name or os.path.splitext(os.path.basename(str(file)))[0]
        cands = dsl.compile_file(str(file), repo=repo)
        key = out or f"registry/mined_{repo}.jsonl"
        storage.get_store().write_jsonl(key, cands)
    except ValueError as e:
        typer.secho(f"rules: {e}", fg="red", err=True)
        raise typer.Exit(1)
    tiers = {}
    for c in cands:
        tiers[c["tier"]] = tiers.get(c["tier"], 0) + 1
    typer.echo(f"rules: compiled {len(cands)} rule(s) from {file} -> {key} "
               f"({tiers})")
    typer.echo("next: confirm their shapes (shape_ui / `mjolnir confirm`), then "
               "`shape_ui/run_transfer.py --typed`")


@rules_app.command("bind")
def bind_(
    file: Path = typer.Argument(..., exists=True, dir_okay=False,
                                help="Bindings JSONL: {rule, site, sigma{var:symbol}}."),
    corpus: Path = typer.Option(..., "--corpus", exists=True, file_okay=False,
                                help="Target corpus (typed environment, e.g. SQL DDL)."),
    materialize: Optional[str] = typer.Option(
        None, "--materialize", help="Write bound facts as JSONL (report input)."),
    out: Optional[str] = typer.Option(None, "--out", help="Write results summary JSON."),
) -> None:
    """Apply explicit BINDINGS: substitute σ into each rule, dispose via the
    authority, emit certificates = {rule, site, sigma, status, kind}."""
    import json
    from mjolnir_prover.lib import bindings as B, certify, review, shapes, transfer
    groups = shapes.collapse(review.load_all())
    sites = dict(transfer.sql_sites(str(corpus)))
    bs = B.load(str(file))
    results, facts = B.run(bs, groups, sites, oracle=certify.certify_unit)
    for r in results:
        why = r.get("reason", "")
        typer.echo(f"  {r.get('kind') or r['status']:>10}  {r['rule'][:28]:28} "
                   f"@ {r['site']:22} {('— ' + why) if why else ''}")
    ncert = sum(1 for f in facts if f["kind"] == "certified")
    nprem = sum(1 for f in facts if f["kind"] == "premise")
    typer.echo(f"bind: {len(results)} binding(s) -> {ncert} certified + {nprem} premise")
    if materialize:
        with open(materialize, "w", encoding="utf-8") as f:
            for x in facts:
                f.write(json.dumps(x) + "\n")
        typer.echo(f"wrote {materialize}")
    if out:
        json.dump({"results": results, "certified": ncert, "premise": nprem},
                  open(out, "w"), indent=2)
        typer.echo(f"wrote {out}")
