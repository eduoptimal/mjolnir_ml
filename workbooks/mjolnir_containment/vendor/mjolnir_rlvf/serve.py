#!/usr/bin/env python3
"""mjolnir_rlvf.serve — an OpenAI-compatible HTTP server for the trained model.

The `--serve` stage of `mjolnir rlvf`. Ported from the research
rlvf_agent server (stdlib only; mlx-lm imported lazily and only for the
`mlx` backend). Endpoints an OpenAI client — VSCode, kilocode, any agent
framework — expects:

    GET  /v1/models
    GET  /health
    POST /v1/chat/completions      (supports stream: true)
    POST /v1/completions

Point any OpenAI-compatible client at it:

    base_url = "http://127.0.0.1:8000/v1",  api_key = "not-needed"

The RLVF system prompt is injected by default when the caller supplies none,
so the trained contract (assert-or-ABSTAIN, never invent a precondition)
travels with every request. The model loads on first request (mlx backend).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, prompts
from .backend import BackendError, make_backend

STATE: dict = {}


def _flatten_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return "" if content is None else str(content)


def _normalise(messages: list) -> list:
    out = []
    for m in messages or []:
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"                    # tool/function -> user text
        out.append({"role": role, "content": _flatten_content(m.get("content"))})
    return out


def _with_system(messages: list) -> list:
    """ALWAYS ON. The trained contract is the server's DECLARATION, not the
    caller's chore: RLVF behaviour is prompt-activated, so leaving it to the
    client would make the calibrated policy one system message away from
    deactivating. The contract is always FIRST; a caller's own system text is
    preserved AFTER it inside a SINGLE system message (one-system-message
    compatibility with every backend and chat template). Only the operator
    (--system / --no-inject-system / RLVF env) can change this — never the
    caller."""
    if not STATE["inject_system"]:
        return messages
    contract = STATE["system"] or prompts.system_prompt(STATE["prompt_mode"])
    out, merged = [], False
    for m in messages:
        if m.get("role") == "system" and not merged:
            merged = True
            body = _flatten_content(m.get("content"))
            out.append({"role": "system",
                        "content": (contract + "\n\n---\n\n" + body)
                        if body else contract})
        else:
            out.append(m)
    if not merged:
        out = [{"role": "system", "content": contract}] + out
    return out


def _params(body: dict) -> dict:
    return {
        "temperature": float(body.get("temperature", STATE["temperature"]) or 0.0),
        "max_tokens": int(body.get("max_tokens")
                          or body.get("max_completion_tokens")
                          or STATE["max_tokens"]),
        "seed": int(body.get("seed", STATE["seed"]) or 0),
    }


def _completion_obj(text, usage, model, finish="stop"):
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": text}}],
        "usage": usage,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):       # quieter, timestamped
        sys.stderr.write("%s - %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    # -- helpers ---------------------------------------------------------------
    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code, message, etype="invalid_request_error"):
        self._json(code, {"error": {"message": message, "type": etype,
                                    "code": code}})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    # -- routes ----------------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [{
                "id": STATE["model_name"], "object": "model",
                "created": STATE["created"], "owned_by": "rlvf"}]})
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            return self._json(200, {"status": "ok", **STATE["backend"].info()})
        self._error(404, f"unknown path {self.path}", "not_found")

    def do_POST(self):
        route = self.path.rstrip("/")
        try:
            body = self._read_body()
        except json.JSONDecodeError as e:
            return self._error(400, f"invalid JSON: {e}")
        if route in ("/v1/chat/completions", "/chat/completions"):
            return self._chat(body)
        if route in ("/v1/completions", "/completions"):
            return self._completions(body)
        self._error(404, f"unknown path {self.path}", "not_found")

    def _chat(self, body: dict):
        messages = _with_system(_normalise(body.get("messages")))
        if not messages:
            return self._error(400, "messages is required")
        p = _params(body)
        model = body.get("model") or STATE["model_name"]
        try:
            if body.get("stream"):
                return self._stream_chat(messages, p, model)
            text, usage = STATE["backend"].chat(messages, **p)
            # "length" when generation hit max_tokens WITHOUT an EOS: a
            # degenerate repetition loop runs to the cap (seen: cell2 #5
            # repeating "!/bin/bash"), and callers must be able to tell a
            # finished answer from a cut-off one instead of assuming "stop".
            _ct = int((usage or {}).get("completion_tokens") or 0)
            finish = "length" if _ct >= p["max_tokens"] else "stop"
            self._json(200, _completion_obj(text, usage, model, finish))
        except BackendError as e:
            self._error(500, str(e), "backend_error")

    def _completions(self, body: dict):
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(map(str, prompt))
        messages = _with_system([{"role": "user", "content": prompt}])
        p = _params(body)
        model = body.get("model") or STATE["model_name"]
        try:
            if body.get("stream"):
                return self._stream_text(messages, p, model, "text_completion")
            text, usage = STATE["backend"].chat(messages, **p)
            self._json(200, {
                "id": "cmpl-" + uuid.uuid4().hex[:24],
                "object": "text_completion", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": usage})
        except BackendError as e:
            self._error(500, str(e), "backend_error")

    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

    def _stream_chat(self, messages, p, model):
        self._sse_headers()
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        base = {"id": cid, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model}
        first = {"role": "assistant", "content": ""}
        self._sse({**base, "choices": [{"index": 0, "delta": first,
                                        "finish_reason": None}]})
        try:
            for piece in STATE["backend"].stream(messages, **p):
                self._sse({**base, "choices": [{"index": 0,
                                                "delta": {"content": piece},
                                                "finish_reason": None}]})
        except BackendError as e:
            self._sse({"error": {"message": str(e), "type": "backend_error"}})
        self._sse({**base, "choices": [{"index": 0, "delta": {},
                                        "finish_reason": "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def _stream_text(self, messages, p, model, obj):
        self._sse_headers()
        cid = "cmpl-" + uuid.uuid4().hex[:24]
        base = {"id": cid, "object": obj, "created": int(time.time()),
                "model": model}
        try:
            for piece in STATE["backend"].stream(messages, **p):
                self._sse({**base, "choices": [{"index": 0, "text": piece,
                                                "finish_reason": None}]})
        except BackendError as e:
            self._sse({"error": {"message": str(e), "type": "backend_error"}})
        self._sse({**base, "choices": [{"index": 0, "text": "",
                                        "finish_reason": "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description="OpenAI-compatible RLVF agent server")
    ap.add_argument("--backend", default=config.SERVE_BACKEND, choices=["mlx", "ollama"])
    ap.add_argument("--model-path", default=config.SERVE_BASE_MODEL,
                    help="base HF/MLX model for the mlx backend")
    ap.add_argument("--adapter", default=config.SERVE_ADAPTER,
                    help="RLVF LoRA adapter path (mlx backend)")
    ap.add_argument("--ollama-url", default=config.OLLAMA_URL)
    ap.add_argument("--ollama-model", default=config.SERVE_OLLAMA_MODEL)
    ap.add_argument("--model-name", default=config.SERVED_NAME)
    ap.add_argument("--host", default=config.SERVE_HOST)
    ap.add_argument("--port", type=int, default=config.SERVE_PORT)
    ap.add_argument("--system", default=None,
                    help="operator override of the built-in contract text")
    ap.add_argument("--prompt-mode", default="rlvf", choices=["rlvf", "agent"])
    ap.add_argument("--temperature", type=float, default=config.SERVE_TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=config.SERVE_MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=config.SERVE_SEED)
    ap.add_argument("--no-inject-system", action="store_true")
    args = ap.parse_args(argv)

    STATE.update({
        "backend": make_backend(args.backend, model_path=args.model_path,
                                adapter_path=args.adapter,
                                ollama_url=args.ollama_url,
                                ollama_model=args.ollama_model),
        "model_name": args.model_name, "created": int(time.time()),
        "system": args.system, "prompt_mode": args.prompt_mode,
        "inject_system": (not args.no_inject_system) and config.INJECT_SYSTEM,
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "seed": args.seed,
    })

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    info = STATE["backend"].info()
    # Prompt-activated behaviour: which contract is live is operationally
    # important, so it is printed WITH a fingerprint — the trained policy is
    # conditional on this text, and two serves must not be mistaken for the
    # same instrument.
    import hashlib as _hashlib
    _active = (STATE["system"] or prompts.system_prompt(STATE["prompt_mode"])
               if STATE["inject_system"] else "")
    _fp = _hashlib.sha256(_active.encode("utf-8")).hexdigest()[:12] if _active else "none"
    sys.stderr.write(
        f"mjolnir rlvf --serve: {info}\n"
        f"  OpenAI base_url : http://{args.host}:{args.port}/v1\n"
        f"  model id        : {args.model_name}\n"
        f"  system prompt   : {'injected' if STATE['inject_system'] else 'NOT applied'}"
        f" (mode={args.prompt_mode}, sha256:{_fp})\n"
        f"  note            : behaviour is PROMPT-ACTIVATED — without this\n"
        f"                    contract the base model's prior dominates.\n")
    if args.backend == "mlx":
        sys.stderr.write("  note: the model loads on first request.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\nmjolnir rlvf --serve: stopped\n")


if __name__ == "__main__":
    main()
