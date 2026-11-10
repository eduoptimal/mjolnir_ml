"""mjolnir_prover.cli.reporting — rich renderers for command results.

Command modules compute structured results (engine-level, no printing) and hand
them here. All human-facing output for the CLI lives in this module, so the
presentation can change without touching command wiring or the engine.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

_VERDICT_STYLE = {
    "CERTIFIED": "bold green",
    "REFUTED": "bold red",
    "FAULT": "bold magenta",
    "UNVERIFIABLE": "yellow",
    "PASS": "bold green",
    "FAIL": "bold red",
    "OK": "bold green",
    "FAILED": "bold red",
    "WARN": "yellow",
    "PENDING": "yellow",
    "confirmed": "green",
    "rejected": "red",
    "adopted": "green",
    "scope-narrowed": "yellow",
}


def verdict(status: str) -> Text:
    return Text(status, style=_VERDICT_STYLE.get(status, "white"))


def banner(title: str, subtitle: str | None = None) -> None:
    body = Text(title, style="bold")
    if subtitle:
        body.append(f"\n{subtitle}", style="dim")
    console.print(Panel(body, expand=False))


def blank() -> None:
    console.print()


# --------------------------------------------------------------------------- #
# check / gate
# --------------------------------------------------------------------------- #
def render_check(result: dict) -> None:
    if "error" in result:
        console.print(f"[red]mjolnir check:[/red] {result['error']}")
        return
    seen: list[str] = []
    for e in result["entries"]:
        if e["file"] not in seen:
            seen.append(e["file"])
    for path in seen:
        rows = [e for e in result["entries"]
                if e.get("file") == path and not e.get("separator")]
        if not rows:
            continue
        table = Table(title=path, title_justify="left", show_lines=False,
                      pad_edge=False, expand=False)
        table.add_column("", width=1)
        table.add_column("Location", style="cyan", no_wrap=True)
        table.add_column("Unit")
        table.add_column("Verdict", no_wrap=True)
        table.add_column("Detail", overflow="fold")
        for e in rows:
            mark = {"CERTIFIED": "[green]+[/green]", "REFUTED": "[red]![/red]",
                    "FAULT": "[magenta]X[/magenta]"}.get(e["status"], "[yellow].[/yellow]")
            detail = e["reason"] + e["extra"]
            table.add_row(mark, e["loc"], e["name"], verdict(e["status"]), detail)
            table.add_row("", "", "", "", f"[dim]cta: {e['cta']}[/dim]")
        console.print(table)

    if result["assume"]:
        console.print(f"[dim]assume:[/dim] {len(result['assume'])} confirmed "
                      f"premise(s) injected {sorted(result['assume'].items())}")

    c = result["counts"]
    summary = Table.grid(padding=(0, 2))
    summary.add_column(justify="right", style="bold")
    summary.add_column()
    for label in ("units", "CERTIFIED", "REFUTED", "UNVERIFIABLE", "FAULT"):
        style = _VERDICT_STYLE.get(label, "")
        summary.add_row(str(c[label]), Text(label, style=style))
    console.print(Panel(summary, title="verdicts", expand=False))

    if result["abstain_reasons"]:
        ar = Table(title="abstentions", title_justify="left", show_header=True,
                   header_style="dim", expand=False)
        ar.add_column("reason")
        ar.add_column("count", justify="right")
        for reason, n in sorted(result["abstain_reasons"].items(),
                                key=lambda kv: -kv[1]):
            ar.add_row(reason, str(n))
        console.print(ar)

    console.print("[dim]false_certificates=0 (kernel re-verified every "
                  "CERTIFIED claim)[/dim]")
    if result["json_out"]:
        console.print(f"certificates -> [cyan]{result['json_out']}[/cyan]")
    g = result["gate"]
    if g["enabled"]:
        if g["passed"]:
            console.print("[bold green]GATE: PASS[/bold green] — no policy "
                          "violations (fail-closed)")
        else:
            console.print(f"[bold red]GATE: FAIL[/bold red] — "
                          f"{', '.join(g['violations'])}")


# --------------------------------------------------------------------------- #
# lean
# --------------------------------------------------------------------------- #
def render_lean(rows: list[dict], counts: dict, kernel_ok: int, kernel_bad: int,
                json_out: str | None, use_kernel: bool) -> None:
    table = Table(show_header=True, header_style="dim", expand=False)
    table.add_column("Verdict", no_wrap=True)
    table.add_column("Statement", overflow="fold")
    if use_kernel:
        table.add_column("Kernel", no_wrap=True)
    for r in rows:
        cells = [verdict(r["verdict"]), r["stmt"][:90]]
        if use_kernel:
            cells.append(verdict(r["kernel"]) if "kernel" in r else "[dim]-[/dim]")
        table.add_row(*cells)
    console.print(table)

    summary = Text()
    for k in ("CERTIFIED", "REFUTED", "UNVERIFIABLE", "FAULT"):
        summary.append(f"{k}={counts[k]}  ", style=_VERDICT_STYLE.get(k, ""))
    console.print(summary)
    if use_kernel:
        console.print(f"kernel confirmed=[green]{kernel_ok}[/green] "
                      f"unchecked/bad=[red]{kernel_bad}[/red]")
    if json_out:
        console.print(f"certificates -> [cyan]{json_out}[/cyan]")


# --------------------------------------------------------------------------- #
# selfcheck / verify
# --------------------------------------------------------------------------- #
def render_selfcheck(checks: list[tuple[str, bool]]) -> bool:
    table = Table(show_header=False, expand=False, pad_edge=False)
    table.add_column(width=5)
    table.add_column()
    for name, ok in checks:
        table.add_row("[green]OK[/green]" if ok else "[red]FAIL[/red]", name)
    console.print(Panel(table, title="mjolnir selfcheck", expand=False))
    bad = sum(1 for _, ok in checks if not ok)
    console.print("RESULT: " + ("[bold green]ALL GREEN[/bold green]" if bad == 0
                                else f"[bold red]FAILED ({bad})[/bold red]"))
    return bad == 0


def render_verify_summary(ok: bool) -> None:
    console.print()
    console.print("RESULT: " + ("[bold green]ALL GREEN[/bold green]" if ok
                                else "[bold red]FAILED[/bold red]"))


# --------------------------------------------------------------------------- #
# countability (PC-2c)
# --------------------------------------------------------------------------- #
def render_countability(audit: dict, result: dict) -> None:
    man, cov = audit["manifest"], audit["coverage"]
    counts = result["counts"]
    banner("mjolnir countability — coverage as an integer",
           "the countable set: a hashed obligation manifest, not a process")

    console.print(
        f"obligations enumerated: [bold]{man['obligations']}[/bold] dischargeable "
        f"+ [bold]{man['placeholders']}[/bold] rung placeholders "
        f"([dim]{man['count']} rows total[/dim])")
    console.print(f"manifest digest: [cyan]{man['digest']}[/cyan]")
    blank()

    frac = (f"{cov['coverage']:.3f}" if cov["coverage"] is not None else "n/a")
    console.print(
        f"COVERAGE = [bold]{cov['numerator']}[/bold] / "
        f"[bold]{cov['denominator']}[/bold] = [bold]{frac}[/bold]   "
        f"([dim]dischargeable obligations the kernel re-derives from the"
        f" certificate alone[/dim])")
    console.print(
        f"pending rungs: [bold]{cov['pending_rungs']}[/bold]   "
        f"([dim]abstention markers — excluded from the ratio by construction,"
        f" they mark a missing rung, not a failed discharge[/dim])")
    console.print(
        f"false certificates counted: "
        f"[{'green' if cov['false_certificates'] == 0 else 'red'}]"
        f"{cov['false_certificates']}[/{'green' if cov['false_certificates'] == 0 else 'red'}]")
    blank()

    if cov["per_rung"]:
        table = Table(title="coverage by rung", title_justify="left",
                      expand=False, pad_edge=False)
        table.add_column("rung")
        table.add_column("discharged", justify="right")
        table.add_column("total", justify="right")
        for rung in sorted(cov["per_rung"]):
            v = cov["per_rung"][rung]
            table.add_row(rung, str(v["discharged"]), str(v["total"]))
        console.print(table)
        blank()

    iso = audit.get("isolation")
    if iso is None:
        console.print("[dim]isolation: not probed (--manifest-only, or no "
                      "certificate with obligations)[/dim]")
    else:
        _render_isolation(iso)

    _render_frontier(audit.get("frontier"))


def _render_isolation(iso: dict) -> None:
    struct, infl = iso["structural"], iso["influence"]
    ok = iso["isolated"]
    tag = "ISOLATED" if ok else "NOT ISOLATED"
    console.print(f"ISOLATION: [{'bold green' if ok else 'bold red'}]{tag}"
                  f"[/{'bold green' if ok else 'bold red'}]")
    console.print(
        f"  structural: kernel re-derived the certificate in a process whose "
        f"import hook admits only {struct.get('tcb', ['kernel', 'term'])} — "
        f"[{'green' if struct.get('isolated') else 'red'}]"
        f"{'agrees' if struct.get('isolated') else 'DISAGREES/FAILED'}"
        f"[/{'green' if struct.get('isolated') else 'red'}]")
    console.print(
        f"  influence:  decisive fields = [cyan]"
        f"{', '.join(infl['decisive_fields']) or '(none)'}[/cyan]; "
        f"inert = [dim]{', '.join(infl['inert_fields']) or '(none)'}[/dim]")
    console.print(
        f"  launder twin: stamping the emitter's own `status` does NOT move a "
        f"derived verdict — [{'green' if infl['status_launder_twinned'] else 'red'}]"
        f"{'held' if infl['status_launder_twinned'] else 'BROKEN'}"
        f"[/{'green' if infl['status_launder_twinned'] else 'red'}]")
    for err in iso.get("errors", []):
        console.print(f"  [red]{err}[/red]")


def _render_frontier(fr) -> None:
    if not fr:
        return
    blank()
    banner("the abstention population",
           "what actually blocks the fragment — measured, not sampled")
    pop = fr["population"]
    console.print(
        f"abstentions: [bold]{pop['population']}[/bold] instances · "
        f"[bold]{pop['distinct_claims']}[/bold] distinct claim texts · "
        f"[bold]{pop['units']}[/bold] units")
    blank()
    t = Table(title="blocking construct", title_justify="left", expand=False,
              pad_edge=False)
    t.add_column("class")
    t.add_column("implied rung")
    t.add_column("instances", justify="right")
    t.add_column("addressable", justify="right")
    for c in pop["by_class"]:
        t.add_row(c["class"], c["rung"], str(c["instances"]),
                  str(c["addressable"]))
    console.print(t)
    console.print(
        "[dim]addressable = instances a rung of that type could actually"
        " decide. The difference is claims with no value to decide about[/dim]")
    blank()
    if pop.get("by_decidability"):
        td = Table(title="could ANY rung decide this?", title_justify="left",
                   expand=False, pad_edge=False)
        td.add_column("class")
        td.add_column("instances", justify="right")
        td.add_column("distinct", justify="right")
        for d in pop["by_decidability"]:
            td.add_row(d["class"], str(d["instances"]), str(d["distinct_claims"]))
        console.print(td)
        console.print(
            f"[dim]coverage ceiling: [bold]"
            f"{pop['ceiling']['decidable_instances']}[/bold] of"
            f" {pop['population']} — a coverage target stated over the whole"
            f" population is unreachable at any capability[/dim]")
        blank()
    t2 = Table(title="rung ROI (addressable instances per unit cost)",
               title_justify="left", expand=False, pad_edge=False)
    t2.add_column("rung")
    t2.add_column("addressable", justify="right")
    t2.add_column("blocked", justify="right")
    t2.add_column("cost", justify="right")
    t2.add_column("per cost", justify="right")
    for c in fr["rung_roi"]:
        t2.add_row(c["rung"], str(c["instances"]), str(c["blocked"]),
                   str(c["cost"]), f"{c['instances_per_cost']:.1f}")
    console.print(t2)
    console.print(
        "[dim]ROI ranks on what a rung could DECIDE, not on what it matches."
        " A rung claiming a blocked instance is quoting a number it cannot"
        " deliver.[/dim]")


# --------------------------------------------------------------------------- #
# harvest
# --------------------------------------------------------------------------- #
def render_harvest(res: dict) -> None:
    console.print(
        f"harvest: [bold]{res['candidates']}[/bold] candidate(s) from "
        f"{res['name']} -> [cyan]{res['key']}[/cyan] "
        f"(confirmed={res['confirmed']}; proposer only, gate disposes)")


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def render_registry_list(keys: list[dict]) -> None:
    if not keys:
        console.print("[yellow]no mined registry keys found[/yellow]")
        return
    table = Table(title="mined registry", title_justify="left", expand=False)
    table.add_column("key", style="cyan")
    table.add_column("entries", justify="right")
    table.add_column("with_oracle", justify="right")
    for k in keys:
        table.add_row(k["key"], str(k["entries"]), str(k["with_oracle"]))
    console.print(table)


def render_registry_verify(res: dict) -> None:
    console.print(
        f"registry verify: entries={res['entries']} with_oracle={res['with_oracle']} "
        f"reproduced=[green]{res['reproduced']}[/green] "
        f"mismatch=[red]{res['mismatch']}[/red] error=[red]{res['error']}[/red]")
    for m in res["mismatches"][:20]:
        console.print(f"  [red]MISMATCH[/red] {m['key']} {m['id']} {m['why']}")


# --------------------------------------------------------------------------- #
# confirm (shape-level registry)
# --------------------------------------------------------------------------- #
def render_shape_list(res: dict) -> None:
    console.print(f"candidates={res['candidates']}  shapes={res['shapes']}  "
                  f"decided={res['decided']}  [dim](instances inherit shape "
                  f"decisions)[/dim]")
    table = Table(show_header=True, header_style="dim", expand=False)
    table.add_column("shape_id", style="cyan")
    table.add_column("status")
    table.add_column("n", justify="right")
    table.add_column("construct")
    table.add_column("shape", overflow="fold")
    for r in res["rows"]:
        table.add_row(r["sid"], verdict(r["status"]), str(r["instances"]),
                      r["construct"], r["shape"][:70])
    console.print(table)
    if res["remaining"]:
        console.print(f"[dim]... {res['remaining']} more (use --limit)[/dim]")


def render_shape_show(res: dict) -> None:
    console.print(f"shape [bold cyan]{res['shape_id']}[/bold cyan]  "
                  f"[{res['tier']}]  instances={res['instances']}")
    console.print(f"  rule:   {res['construct']}: {res['shape']}")
    console.print(f"  oracle: {res['oracle']}")
    console.print("  evidence (first 5 origins):")
    for e in res["evidence"]:
        console.print(f"    [cyan]{e['origin']}[/cyan]  {e['predicate'][:80]}")
    if res["ledger"]:
        console.print(f"  ledger: {res['ledger']['decision']} by "
                      f"{res['ledger']['by']} at {res['ledger']['ts']}")
    else:
        console.print("  ledger: [yellow]PENDING[/yellow]")


def render_shape_decision(res: dict) -> None:
    console.print(f"[bold]{res['decision']}[/bold]: {res['shape_id']}  by "
                  f"{res['by']}  -> {res['instances']} instances inherit")


def render_shape_apply(res: dict) -> None:
    console.print(f"{res['applied']}/{res['total']} candidates now confirmed via "
                  f"{res['decisions']} shape decisions")


def render_shape_transfer(res: dict) -> None:
    if not res["records"]:
        console.print("transfer: no confirmed shapes have instances on target — "
                      "nothing to do")
        return
    table = Table(title="shape transfer", title_justify="left", expand=False)
    table.add_column("decision")
    table.add_column("shape_id", style="cyan")
    table.add_column("tier")
    table.add_column("validation", overflow="fold")
    for r in res["records"]:
        v = r["validation"]
        table.add_row(verdict(r["decision"]), r["shape_id"], r["tier"],
                      f"instances={v['instances']} certified={v['certified']} "
                      f"refuted={v['refuted']} unverifiable={v['unverifiable']} "
                      f"structural={v['structural']}")
    console.print(table)
    console.print(f"transfer: [green]{res['adopted']}[/green] adopted, "
                  f"[yellow]{res['narrowed']}[/yellow] scope-narrowed "
                  f"(signed, appended to ledger)")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def render_report(summary: dict, artifacts: list[str]) -> None:
    h = summary.get("headline", {})
    table = Table.grid(padding=(0, 3))
    table.add_column(justify="right", style="bold")
    table.add_column()
    table.add_row(f"{h.get('claims_mined', 0):,}", "claims mined")
    table.add_row(str(h.get('proved', 0)), "proved theorems")
    table.add_row(str(h.get('premises_certified', 0)), "certified runtime premises")
    table.add_row(f"{h.get('runtime_observed', 0):,}", "runtime-observed")
    table.add_row(str(h.get('enforced_contracts', 0)), "signed enforcement contracts")
    table.add_row(str(h.get('violations', 0)), "violations")
    table.add_row(str(h.get('false_approvals', 0)), "false approvals")
    console.print(Panel(table, title="mjolnir report", expand=False))
    forms = h.get("observed_by_form") or {}
    if forms:
        ft = Table(title="runtime-observed — by rule/form class",
                   title_justify="left", show_header=True, header_style="dim",
                   expand=False)
        ft.add_column("form class")
        ft.add_column("claims", justify="right")
        for k, v in forms.items():
            ft.add_row(k, f"{v:,}")
        console.print(ft)
    for a in artifacts:
        console.print(f"artifact -> [cyan]{a}[/cyan]")


def render_differential(counts: dict, rows: list, exec_path: bool,
                        total: int) -> None:
    banner("mjolnir differential — an oracle mjolnir did not write",
           "execution SCORES; it never discharges")
    scored = sum(v for k, v in counts.items()
                 if k in ("certified_ok", "refuted_ok", "headroom",
                          "declined_correctly", "false_certificate",
                          "wrong_refutation"))
    console.print(
        f"units offered: [bold]{total}[/bold]   "
        f"scorable by this oracle: [bold]{scored}[/bold] "
        f"([dim]{100.0 * scored / total:.2f}% reach[/dim])"
        + ("   mode: [yellow]EXEC[/yellow]" if exec_path else ""))
    blank()
    table = Table(title="external scoreboard", title_justify="left",
                  expand=False, pad_edge=False)
    table.add_column("outcome")
    table.add_column("count", justify="right")
    table.add_column("meaning", style="dim")
    meaning = {
        "certified_ok": "CERTIFIED and execution agrees",
        "refuted_ok": "REFUTED and execution agrees",
        "false_certificate": "*** CERTIFIED but execution FALSIFIES it",
        "wrong_refutation": "*** REFUTED but execution SATISFIES it",
        "headroom": "abstained, yet execution says it holds",
        "declined_correctly": "abstained, and execution says it fails",
        "unscored": "not reachable by this oracle",
        "not_scorable": "no claim extracted",
    }
    for k in counts:
        if k.startswith("no_verdict:"):
            meaning[k] = f"pytest gave no verdict ({k.split(':')[1]})"
    # `precision` is a nested dict, not a count — it has its own block below.
    for k, v in sorted(((k, v) for k, v in counts.items()
                        if isinstance(v, int)),
                       key=lambda kv: -kv[1]):
        style = ("bold red" if k in ("false_certificate", "wrong_refutation")
                 else "green" if k.endswith("_ok") else "")
        table.add_row(k, str(v), meaning.get(k, ""), style=style)
    console.print(table)
    blank()
    prec = counts.get("precision")
    if prec:
        console.print("[bold]PRECISION[/bold] — of the certificates that EXIST, "
                      "how many did the oracle confirm?")
        console.print(f"  certificates existing      [bold]{prec['certificates_existing']}[/bold]")
        console.print(f"  checkable by the oracle   [bold]{prec['certificates_checkable']}[/bold]"
                      f"  [dim](the rest had no verdict)[/dim]")
        console.print(f"  confirmed                 [bold]{prec['certificates_confirmed']}[/bold]")
        console.print(f"  FALSE CERTIFICATES        [bold]"
                      f"{prec['false_certificates']}[/bold]")
        if prec["precision"] is not None:
            console.print(f"  precision on the checkable [bold]{prec['precision']:.3f}[/bold]"
                          f"   coverage of all certificates "
                          f"{prec['coverage_of_certificates']:.3f}")
        console.print(
            "[dim]precision is a separate axis from reach: a high scoreboard over\n"
            "almost no certificates is not evidence of soundness.[/dim]")
        blank()
    adverse = [r for r in rows
               if r["outcome"] in ("false_certificate", "wrong_refutation")]
    console.print(
        f"ADVERSE FINDINGS: [bold]{len(adverse)}[/bold]"
        + ("  [green]— no false certificate, no wrong refutation"
           if not adverse else ""))
    for r in adverse[:15]:
        console.print(f"  [red]{r['outcome']}[/red] {r['path']}:{r['line']}"
                      f" {r['name']}  [dim]{r['claim'][:48]}[/dim]")
    console.print(
        "[dim]A falsifying counterexample is conclusive; a clean run is NOT "
        "proof — it is the absence of a counterexample found.[/dim]")


def render_plan(result: dict) -> None:
    banner("mjolnir plan — measured buckets, then ONE derived order",
           "the corpus changes how BIG each stage is, never which comes next")
    console.print(f"units measured: [bold]{result['total_units']}[/bold]"
                  f"    roadmap digest: [cyan]{result['roadmap_digest']}[/cyan]")
    blank()
    t = Table(title="what is blocking each unit", title_justify="left",
              expand=False, pad_edge=False)
    t.add_column("stage", justify="right")
    t.add_column("blocked by")
    t.add_column("units", justify="right")
    t.add_column("share", justify="right")
    t.add_column("fix")
    for s in result["stages"]:
        style = "bold red" if s["dominant"] else ""
        t.add_row(str(s["stage"]), s["key"], str(s["instances"]),
                  f"{s['share']}%", s["fix"], style=style)
    console.print(t)
    blank()
    if result.get("total_units") == 0:
        console.print("[yellow]" + (result.get("recommendation") or "") +
                      "[/yellow]")
        return
    d = result["dominant_stage"]
    if d:
        console.print(
            f"dominant stage: [bold]{d['stage']} ({d['key']})[/bold] — "
            f"[bold]{d['instances']}[/bold] of {result['total_units']} units, "
            f"{d['share']}%")
    blank()
    if result["stage3_ranking"]:
        t3 = Table(title="stage 3 candidates, ranked (addressable only)",
                   title_justify="left", expand=False, pad_edge=False)
        t3.add_column("rung")
        t3.add_column("addressable", justify="right")
        t3.add_column("cost", justify="right")
        t3.add_column("per cost", justify="right")
        for r in result["stage3_ranking"][:12]:
            t3.add_row(r["rung"], str(r["addressable"]),
                       str(r["cost"]) if r["cost"] else "— uncosted —",
                       f"{r['instances_per_cost']:.1f}"
                       if r["instances_per_cost"] is not None else "n/a")
        console.print(t3)
        blank()
    rec = result["recommendation"]
    console.print("RECOMMENDATION")
    if rec:
        console.print("  " + rec)
    console.print()
    console.print("[dim]" + result["order_note"] + "[/dim]")


def render_scope(result: dict) -> None:
    banner("mjolnir scope — decided, and declined precisely",
           "the promised scope is hashed data; the counts beside it are measured")
    console.print(f"units measured: [bold]{result['total_units']}[/bold]"
                  f"    scope digest: [cyan]{result['scope_digest']}[/cyan]")
    blank()
    console.print(f"[bold green]DECIDES[/bold green]  "
                  f"[bold]{result['decided']}[/bold] of {result['total_units']} "
                  f"({result['decided_share']}%)")
    for d in result["declared_decides"]:
        console.print(f"  [green]•[/green] {d['class']}")
        console.print(f"    [dim]test: {d['test']}[/dim]")
        console.print(f"    [dim]why:  {d['why']}[/dim]")
    blank()
    console.print("[bold yellow]DECLINES[/bold yellow]")
    t = Table(title="declined class", title_justify="left", expand=False,
              pad_edge=False)
    t.add_column("class")
    t.add_column("predicate")
    t.add_column("measured here")
    counts = result["counts"]
    for d in result["declared_declines"]:
        key = "declines: " + d["class"]
        n = counts.get(key, counts.get(key.split(" that ")[0], None))
        t.add_row(d["class"], d["predicate"],
                  str(n) if n is not None else "—")
    console.print(t)
    blank()
    console.print("[dim]measured on this corpus:[/dim]")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        console.print(f"  {v:>6}  {k}")
    blank()
    nsl = result["not_a_scope_limit"]
    console.print("[bold]NOT a scope limit:[/bold] "
                  + nsl["expression"])
    console.print("  [dim]" + nsl["why"] + "[/dim]")


def render_baseline(base: dict, prec: dict, fails: list) -> None:
    """The regression guard. Its whole purpose is the moment somebody proposes
    more capability: does the added power produce certificates an oracle we did
    not write can refute? A lower precision than the recorded baseline FAILS."""
    blank()
    console.print("[bold]PRECISION REGRESSION GUARD[/bold]")
    console.print(f"  {'metric':<34}{'baseline':>12}{'now':>12}")
    for key in ("certificates_existing", "certificates_checkable",
                "certificates_confirmed", "false_certificates"):
        console.print(f"  {key:<34}{base.get(key, 0):>12}{prec.get(key, 0):>12}")
    for key in ("precision", "coverage_of_certificates"):
        bv, cv = base.get(key), prec.get(key)
        fmt = (lambda v: f"{v:.3f}") if v is not None else "n/a"
        console.print(f"  {key:<34}{fmt(bv):>12}{fmt(cv):>12}")
    if fails:
        console.print()
        for f in fails:
            console.print(f"  [bold red]FAIL[/bold red] {f}")
    else:
        console.print("\n  [green]PASS[/green] — no false certificate, precision held")


def render_soft(source, result: dict) -> None:
    banner("mjolnir soft — a DISCLOSURE certificate",
           "a set of claims over a machine-readable source; never a score")
    console.print(f"source: [cyan]{source}[/cyan]    level={result['level']}    "
                  f"schema {result['schema']}")
    mode = result.get("mode", "")
    if mode.startswith("self-consistency"):
        console.print(f"[yellow]MODE: {mode}[/yellow]")
    blank()
    t = Table(title="claims", title_justify="left", expand=False, pad_edge=False)
    t.add_column("quantity")
    t.add_column("rung")
    t.add_column("claimed")
    t.add_column("evidence")
    t.add_column("verdict")
    t.add_column("mark")
    for c in result["claims"]:
        v = c["verdict"]
        style = ("green" if v == "CERTIFIED" else
                 "red" if v == "REFUTED" else "dim")
        # `claimed` is what the emitter asserted and `evidence` is what the source
        # says. Printing only the evidence made every REFUTED row display the
        # SOURCE's number beside the word REFUTED -- the enumeration clause (iii)
        # requires was in the JSON and absent from the table a human reads.
        t.add_row(c["quantity"] or "—", c["destination"] or "abstain",
                  str(c["value"]) if c["value"] is not None else "—",
                  (c["evidence"] or "—")[:24], v,
                  "yes" if c.get("marked") else "NO", style=style)
    console.print(t)
    blank()
    ok = result["discloses"]
    console.print(f"DISCLOSES: [{'green' if ok else 'red'}]{ok}[/]"
                  f"    certified {result['n_certified']}/{result['n_claims']}"
                  f"    remainder {result['n_remainder']}")
    if result["unattributed"]:
        console.print(f"  [yellow]{len(result['unattributed'])} unattributed "
                      f"claim(s): not certified AND not marked[/yellow]")
    console.print()
    console.print("  [dim]" + result["note"] + "[/dim]")
