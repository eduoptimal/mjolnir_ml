"""mjolnir_prover.cli — the Typer entry interface.

One entry point for the product surface. Commands are grouped by concern and
registered here; each command module is thin (argument wiring + engine call) and
delegates all presentation to :mod:`mjolnir_prover.cli.reporting`.

Groups / commands:

  check      Verify Python units; emit certificates; --gate for fail-closed CI.
  lean       Verify Lean statements (IR frontend, optional independent kernel).
  harvest    Mine candidate claims from a repository into the registry.
  confirm    Shape-level confirmation: list / show / accept / reject / apply /
             transfer.
  registry   Inspect and re-derive the mined registry: list / verify.
  soft         Certify a SET of claims over a machine-readable source
                (table / manifest / key:value) and disclose the remainder.
                Never a score -- see `mjolnir soft`.
  scope        What mjolnir decides, and exactly what it declines — the
                promised scope is hashed data, the counts are measured.
  plan         Measure what is blocking each unit, then derive ONE
                ordered recommendation — the only sanctioned answer to
                "what should we build?".
  differential  Score certificates and refutations against Python's own
                evaluator — an oracle mjolnir did not write.
  countability  Coverage as an integer over a hashed manifest; check
                that the verdict is a function only of the certificate.
  selfcheck  Built-in soundness smoke test.
  verify     Release self-verification (selfcheck + fail-closed gate).
  report     Generate the management verification report.
  version    Print the version.

Run as ``python -m mjolnir <cmd>`` or via the installed ``mjolnir`` script.
"""
from __future__ import annotations

from typing import Optional

import typer

from mjolnir_prover.cli.commands import (about, audit, countability, differential,
                                  harvest, lean, plan, policy, registry, report,
                                  rules, scope, shapes, soft, verify)

app = typer.Typer(
    name="mjolnir",
    help="Certification layer for machine-generated logic: fragment checking, "
         "Lean IR + independent kernel, and a fail-closed CI gate.",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)

for _module in (verify, lean, harvest, rules, policy, shapes, registry, report,
                 audit, countability, differential, plan, scope, soft, about):
    _module.register(app)


def main(argv: Optional[list[str]] = None) -> None:
    """Console-script / ``python -m mjolnir`` entry point."""
    app(args=argv)


if __name__ == "__main__":
    main()