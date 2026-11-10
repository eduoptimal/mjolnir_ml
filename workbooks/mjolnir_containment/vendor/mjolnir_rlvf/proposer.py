"""rlvf.proposer — the untrusted neural proposer.

Two backends, one contract: emit a candidate Python function or an explicit
`ABSTAIN: <reason>`. It assigns no verdict — the authority in mjolnir does that.

  backend="ollama"  local Ollama HTTP (fast iteration, GGUF)
  backend="mlx"     in-process mlx-lm (the exact weights we train on; required
                    for a valid pre/post comparison)
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from . import config

ABSTAIN_RE = re.compile(r"^\s*ABSTAIN\s*:\s*(?P<reason>.*)$", re.MULTILINE)
FENCE_RE = re.compile(r"```(?:python)?\s*\n(?P<body>.*?)```", re.DOTALL)

_MLX_CACHE = {}


def _mlx_load(model_path: str, adapter_path=None):
    key = (model_path, adapter_path)
    if key not in _MLX_CACHE:
        from mlx_lm import load
        _MLX_CACHE[key] = load(model_path, adapter_path=adapter_path)
    return _MLX_CACHE[key]


def generate_ollama(model: str, prompt: str, temperature: float, seed: int,
                    num_predict=None) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict if num_predict is not None
            else config.NUM_PREDICT,
        },
    }
    req = urllib.request.Request(
        config.OLLAMA_URL.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as r:
            out = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return f"PROPOSER_ERROR: {type(e).__name__}: {e}"
    return out.get("response", "")


def generate_mlx(model_path: str, prompt: str, temperature: float, seed: int,
                 num_predict=None, adapter_path=None) -> str:
    try:
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        model, tokenizer = _mlx_load(model_path, adapter_path)
        text = prompt
        apply_tpl = getattr(tokenizer, "apply_chat_template", None)
        if apply_tpl is not None:
            try:
                text = apply_tpl([{"role": "user", "content": prompt}],
                                 add_generation_prompt=True, tokenize=False)
            except TypeError:
                text = apply_tpl([{"role": "user", "content": prompt}],
                                 add_generation_prompt=True)
        sampler = make_sampler(temp=max(0.0, temperature))
        return generate(model, tokenizer, prompt=text,
                        max_tokens=num_predict if num_predict is not None
                        else config.NUM_PREDICT,
                        sampler=sampler, verbose=False)
    except Exception as e:
        return f"PROPOSER_ERROR: {type(e).__name__}: {e}"


def generate_openai(model: str, prompt: str, temperature: float, seed: int,
                    num_predict=None) -> str:
    """Any OpenAI-compatible endpoint (chat/completions) — the generic
    `--model` path: hosted APIs or local servers speaking the same protocol.
    Key from OPENAI_API_KEY; base from config.OPENAI_BASE_URL."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": num_predict if num_predict is not None
        else config.NUM_PREDICT,
    }
    headers = {"Content-Type": "application/json"}
    if config.OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {config.OPENAI_API_KEY}"
    req = urllib.request.Request(
        config.OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as r:
            out = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return f"PROPOSER_ERROR: {type(e).__name__}: {e}"
    try:
        return out["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return f"PROPOSER_ERROR: unexpected response: {str(out)[:200]}"


def generate(model: str, prompt: str, temperature: float, seed: int,
             num_predict=None, backend: str | None = None,
             adapter: str | None = None) -> str:
    backend = backend or config.BACKEND
    if backend == "mlx":
        return generate_mlx(model, prompt, temperature, seed, num_predict,
                            adapter_path=adapter)
    if backend == "openai":
        return generate_openai(model, prompt, temperature, seed, num_predict)
    return generate_ollama(model, prompt, temperature, seed, num_predict)


def parse(text: str) -> dict:
    """Split a raw model response into {kind, code|reason, raw}."""
    raw = text or ""
    stripped = raw.strip()
    if stripped.startswith("PROPOSER_ERROR:"):
        return {"kind": "error", "reason": stripped, "code": None, "raw": raw}
    m = ABSTAIN_RE.search(raw)
    if m:
        return {"kind": "abstain", "reason": m.group("reason").strip(),
                "code": None, "raw": raw}
    fence = FENCE_RE.search(raw)
    code = fence.group("body") if fence else raw
    return {"kind": "code", "reason": None, "code": code.strip(), "raw": raw}
