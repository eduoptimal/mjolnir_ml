"""mjolnir_decide.serve — stdin JSONL and HTTP surfaces for the contract.

Both routes are the same function: validate request -> noul -> response.
Contract violations are 400s with an OpenAI-style error envelope (what an
OpenRouter front expects); internal failures are 500s.  Nothing else is
exposed -- no training hooks, no debug endpoints.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from mjolnir_decide.batch import answer_batch
from mjolnir_decide.contract import ContractError, build_response, parse_request
from mjolnir_decide.model import Model


def answer(model: Model, req: dict) -> dict:
    parsed = parse_request(req)
    # one call: one pass (value, regime, floor). Multi-call states split into
    # per-step passes and aggregate with the joint-AND law (mjolnir_common).
    resp, _ = answer_batch(model, parsed, build_response, model.provenance)
    return resp


def serve_stdin(model: Model) -> None:
    """JSONL (one request per line) or pretty-printed JSON on stdin -> stdout."""
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
            resp = answer(model, req)
        except ContractError as ex:
            resp = {"error": {"message": str(ex), "type": "invalid_request_error"}}
        except Exception as ex:                          # fail-closed, named
            resp = {"error": {"message": str(ex), "type": "server_error"}}
        print(json.dumps(resp), flush=True)


def serve_http(model: Model, port: int, host: str = "0.0.0.0") -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            route = self.path.rstrip("/")
            if route not in ("", "/v1/decide", "/decide"):
                self._send(404, {"error": {"message": "not found",
                                           "type": "invalid_request_error"}})
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n))
                resp, code = answer(model, req), 200
            except ContractError as ex:
                resp, code = ({"error": {"message": str(ex),
                                         "type": "invalid_request_error"}}, 400)
            except Exception as ex:
                resp, code = ({"error": {"message": str(ex),
                                         "type": "server_error"}}, 500)
            self._send(code, resp)

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            print(f"[http] {a[0] % a[1:] if a else ''}", file=sys.stderr)

    print(f"[decide] head={model.head} ckpt={model.path}", file=sys.stderr)
    print(f"[decide] http on :{port} (POST / | /v1/decide)", file=sys.stderr)
    HTTPServer((host, port), Handler).serve_forever()
