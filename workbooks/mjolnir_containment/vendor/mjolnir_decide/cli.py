"""mjolnir_decide.cli — the package surface. Typer, the product's one framework.

    python -m mjolnir_decide serve                       # stdin JSONL -> stdout
    python -m mjolnir_decide serve --http 8080           # POST / (OpenRouter shape)
    python -m mjolnir_decide serve --head full           # discriminative head
    python -m mjolnir_decide selftest

Head resolution: --head > MJOLNIR_DECIDE_HEAD > 'flow' (generative default).

EXIT CODES
    0  ok
    1  selftest failed
    3  usage error (bad head, missing checkpoint)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import typer                                                                   # noqa: E402

from mjolnir_decide import selftest as _selftest                               # noqa: E402
from mjolnir_decide.model import HEADS, Model                                  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="calibrated safe-to-run decisions (proposes; the "
                       "router disposes)")


@app.command("serve")
def serve(
    head: str = typer.Option(None, "--head", help=f"one of {HEADS} "
                             "(default: env MJOLNIR_DECIDE_HEAD or 'flow')"),
    ckpt: str = typer.Option(None, "--ckpt", help="checkpoint path "
                             "(default: env MJOLNIR_DECIDE_CKPT, data/, repo)"),
    http: int = typer.Option(0, "--http", help="serve HTTP on this port "
                             "(default: stdin JSONL -> stdout)"),
    host: str = typer.Option("0.0.0.0", "--host"),
):
    """Answer decision requests on stdin (JSONL) or over HTTP."""
    if head is not None and head not in HEADS:
        print(f"bad --head {head!r}; one of {HEADS}", file=sys.stderr)
        raise typer.Exit(3)
    try:
        model = Model(ckpt=ckpt, head=head)
    except (FileNotFoundError, ValueError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        raise typer.Exit(3)
    from mjolnir_decide import serve as _serve
    if http:
        _serve.serve_http(model, http, host)
    else:
        _serve.serve_stdin(model)


@app.command("selftest")
def selftest():
    """Offline contract checks (no checkpoint, no network)."""
    raise typer.Exit(_selftest.run(verbose=True))


def main() -> int:
    try:
        app()
    except SystemExit as e:
        return e.code or 0
    return 0
