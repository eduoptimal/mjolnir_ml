"""soft — the soft / DISCLOSURE certificate over a machine-readable source.

`mjolnir check` certifies one artifact. This certifies a SET of claims from a
table, a manifest, or a `key: value` document, and reports the remainder
explicitly rather than summarising it into a number.

The output is a pair, never a score. H-NL13 measured `S-Omega = 1.000` at a
0.992 false fraction, and T-NL18 showed the false fraction is not computable from
the artifact — so any scalar here would be a number that reads like a guarantee
and is not one. The size of the remainder IS exposed, because it is the only
part a reader can act on, and H-NL19 showed it is under the emitter's control,
so it is a disclosure and not a rating.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def _looks_delimited(source: Path) -> bool:
    """True when the file carries no table or delimiter structure. Used ONLY to
    make the decline message true about WHY; it never disposes anything."""
    try:
        head = source.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        return False
    pipes = sum(1 for l in head.splitlines() if l.strip().startswith("|"))
    delim = sum(1 for l in head.splitlines()
                if l.count(",") >= 2 or l.count("\t") >= 2)
    return pipes == 0 and delim == 0


def register(app: typer.Typer) -> None:

    @app.command()
    def soft(
        source: Path = typer.Argument(
            ..., exists=True, readable=True,
            help="A markdown table or key: value document."),
        level: int = typer.Option(
            1, "--level", min=0,
            help="Cover level: 0 = every claim must certify; 1 = one "
                 "certified claim per topic suffices. There is no intermediate "
                 "level (T-NL17), and values above 1 are refused."),
        adapter: Optional[str] = typer.Option(
            None, "--adapter",
            help="Router adapter to use (default: the lexical router). An "
                 "adapter may ROUTE or ABSTAIN; it cannot dispose."),
        claims: Optional[Path] = typer.Option(
            None, "--claims",
            help="JSON list of {quantity, value, text} the EMITTER asserted. "
                 "Required for a real check: without it the source is checked "
                 "against ITSELF, which is a self-consistency tautology."),
        json_out: Optional[Path] = typer.Option(None, "--json"),
    ) -> None:
        """Certify a set of claims over a machine-readable source."""
        from mjolnir_prover import tactics
        from mjolnir_prover.lib import frontend_table

        if level not in (0, 1):
            reporting.console.print(
                f"[red]cover level must be 0 or 1; got {level}. There is no "
                f"safe intermediate level (T-NL17).[/red]")
            raise typer.Exit(2)

        facts, repeated = frontend_table.read_all(str(source))
        if not facts:
            # The message has to be TRUE about why. It used to say "prose is a
            # declined class" for every failure, which is a misdiagnosis for a
            # CSV or a source file -- it told the user their *format* was the
            # problem when the real cause was this reader not supporting it.
            # Found by running the first-user path with the files a real user
            # has. A scope limit is a promise, and a promise that lies about its
            # own boundary is worse than no promise.
            _why = ("this file has no machine-readable facts — no two-column "
                    "table, no `key: value` lines, and no delimited CSV/TSV "
                    "header with numeric values"
                    if not _looks_delimited(source) else
                    "prose is a DECLINED class: referent resolution on prose "
                    "measured 0.079 (0/297). See "
                    "docs/analsysis/natural_language.md S9")
            reporting.console.print(
                f"[yellow]no machine-readable facts found — {_why}.[/yellow]")
            reporting.console.print(
                "[dim]Supported: markdown 2-column tables, `key: value` blocks, "
                "CSV/TSV with a header. Prose is out of scope by measurement, "
                "not by preference.[/dim]")
            raise typer.Exit(2)

        if repeated:
            # A referent stated more than once is a RELATION, and the previous
            # behaviour -- keep the last value, drop the rest -- REFUTED a claim
            # the document states. Refuse instead, and name every conflict:
            # fail-closed beats silently wrong.
            reporting.console.print(
                f"[red]{len(repeated)} referent(s) stated more than once. "
                f"This command certifies one value per referent:[/red]")
            for q, vals in list(repeated.items())[:20]:
                reporting.console.print(f"  [red]{q}[/red]: {', '.join(vals)}")
            reporting.console.print(
                "[yellow]Refusing. Give each value its own referent, or "
                "certify each as a separate claim.[/yellow]")
            raise typer.Exit(2)

        if claims is not None:
            emitted = json.loads(Path(claims).read_text(encoding="utf-8"))
            cs = [tactics.Claim(text=c.get("text", ""),
                                quantity=c.get("quantity"),
                                value=c.get("value")) for c in emitted]
            result = tactics.certify(cs, facts, level=level)
            result["mode"] = "grounded"
        else:
            result = tactics.self_check(facts, level=level)
            result["mode"] = "self-consistency (VACUOUS: the source is being "\
                             "checked against itself)"
        reporting.render_soft(source, result)
        if result.get("mode", "").startswith("self-consistency"):
            # The vacuous arm is the FIRST thing a new user will run, because
            # it needs no `--claims` file, and it prints "12/12 CERTIFIED" in
            # green. That reads as a result and it is not one -- the source is
            # being checked against itself. A banner in a status line is not
            # enough at this position: the claim it qualifies is the first
            # number on the screen. Found by walking the first-user path.
            reporting.console.print(
                "\n[bold red]THIS IS NOT A RESULT.[/bold red] With no "
                "`--claims`, the source was checked against ITSELF: every "
                "row trivially matches its own value. Nothing about the "
                "document was established, and this number must not be quoted.")
            reporting.console.print(
                "[dim]To get a real certificate, pass the claims being made "
                "about this source:[/dim]\n"
                "[dim]  mjolnir soft report.md --claims claims.json[/dim]")
            reporting.console.print(
                "[dim]  claims.json = [{\"quantity\": \"<name in the "
                "source>\", \"value\": \"<the claimed value>\"}][/dim]")
        if json_out:
            p = Path(json_out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, indent=2), encoding="utf-8")

        # Exit codes. Two DIFFERENT failures, and conflating them was a fail-open
        # on the fabrication channel: a model asserting five metrics the source
        # does not contain produced exit 0, because every verdict was
        # UNVERIFIABLE rather than REFUTED. Measured, not hypothesised.
        #
        #   REFUTED                        -> the source disagrees; exit 1
        #   quantity NOT IN SOURCE         -> the source does not mention it at
        #                                      all. That is FABRICATION, and it
        #                                      must be loudest, not silent.
        #   abstained for any other reason  -> honest uncertainty; exit 0.
        #                                      The corpus is ~87% abstentions, so
        #                                      failing on these would make the
        #                                      gate useless.
        refuted = [c for c in result["claims"] if c["verdict"] == "REFUTED"]
        absent = [c for c in result["claims"]
                  if str(c.get("reason", "")).startswith("quantity_not_in_source")]
        if absent:
            reporting.console.print(
                f"[red]{len(absent)} claim(s) name a quantity the source does "
                f"not contain.[/red] These are not undecidable -- they are "
                f"unsupported:")
            for c in absent[:10]:
                reporting.console.print(
                    f"  [red]{c['quantity']}[/red] (asserted {c['value']})")
            reporting.console.print(
                "[yellow]A faithful transcriber declines these. Asserting them "
                "is fabrication.[/yellow]")
        raise typer.Exit(1 if (refuted or absent) else 0)
