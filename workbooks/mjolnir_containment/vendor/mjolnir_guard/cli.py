"""mjolnir_guard.cli — the package surface. Typer, the product's one framework.

    python -m mjolnir_guard serve                    # stdin JSONL -> stdout
    python -m mjolnir_guard serve --http 8080        # POST / (same envelope)
    python -m mjolnir_guard serve --reader off       # authority-only degradation
    python -m mjolnir_guard selftest

EXIT CODES: 0 ok · 1 selftest failed · 3 usage / missing weights
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import typer                                                                   # noqa: E402

from mjolnir_guard import selftest as _selftest                               # noqa: E402
from mjolnir_guard.contract import ContractError, build_response, parse_request  # noqa: E402
from mjolnir_guard.guard import Guard                                         # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="the critical decider: authority -> veto-reader -> "
                       "noul 0/1 with a named stage")


def _answer(guard: Guard, req: dict) -> dict:
    parsed = parse_request(req)
    noul, decision = guard.answer(parsed.state)
    tokens = len(str(req.get("state") or "").split()) * 13 // 10  # ~1.3 tok/word
    return build_response(parsed, noul, tokens, decision=decision)


def _serve_stdin(guard: Guard) -> None:
    buf = ""
    for line in sys.stdin:
        buf = (buf + "\n" + line) if buf else line
        s = buf.strip()
        if not s:
            buf = ""
            continue
        try:
            req = json.loads(s)
        except json.JSONDecodeError:
            continue
        buf = ""
        try:
            resp = _answer(guard, req)
        except ContractError as ex:
            resp = {"error": {"message": str(ex), "type": "invalid_request_error"}}
        except Exception as ex:
            resp = {"error": {"message": str(ex), "type": "server_error"}}
        print(json.dumps(resp), flush=True)


def _serve_http(guard: Guard, port: int, host: str) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            route = self.path.rstrip("/")
            if route not in ("", "/v1/decide", "/v1/guard", "/guard"):
                self._send(404, {"error": {"message": "not found",
                                           "type": "invalid_request_error"}})
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                resp, code = _answer(guard, json.loads(self.rfile.read(n))), 200
            except ContractError as ex:
                resp, code = ({"error": {"message": str(ex),
                                         "type": "invalid_request_error"}}, 400)
            except Exception as ex:
                resp, code = ({"error": {"message": str(ex),
                                         "type": "server_error"}}, 500)
            self._send(code, resp)

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            print(f"[http] {a[0] % a[1:] if a else ''}", file=sys.stderr)

    mode = "reader=on" if guard.reader is not None else "reader=off"
    print(f"[guard] {mode} http on :{port} (POST / | /v1/decide)",
          file=sys.stderr)
    HTTPServer((host, port), Handler).serve_forever()


@app.command("serve")
def serve(
    reader: str = typer.Option("on", "--reader",
                               help="on (authority + veto-reader) | off "
                                    "(authority-only degradation)"),
    reader_path: str = typer.Option(None, "--reader-path"),
    http: int = typer.Option(0, "--http"),
    host: str = typer.Option("0.0.0.0", "--host"),
):
    """Answer decision requests on stdin (JSONL) or over HTTP."""
    try:
        guard = Guard(reader=reader, reader_path=reader_path)
        if guard.reader is not None:
            guard.reader._load()                     # fail fast on weights
    except (FileNotFoundError, ValueError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        raise typer.Exit(3)
    if http:
        _serve_http(guard, http, host)
    else:
        _serve_stdin(guard)


@app.command("selftest")
def selftest():
    """Offline checks: contract + pipeline with a stub reader (no GPU)."""
    raise typer.Exit(_selftest.run(verbose=True))


def main() -> int:
    try:
        app()
    except SystemExit as e:
        return e.code or 0
    return 0
