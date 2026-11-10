"""rlvf_agent.backend — model backends behind one chat interface.

Two backends, one method contract:
  chat(messages, temperature, max_tokens, seed) -> (text, usage)
  stream(messages, temperature, max_tokens, seed) -> iterator[str]

- ``MlxBackend`` loads the base weights + the RLVF LoRA adapter locally.
- ``OllamaBackend`` proxies an OpenAI-style chat to a local Ollama server.

Neither backend assigns verdicts; they only produce text.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Iterator


class BackendError(RuntimeError):
    pass


def _apply_template(tokenizer, messages: list) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
    except TypeError:                      # older signature
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


class MlxBackend:
    def __init__(self, model_path: str, adapter_path: str | None = None):
        self.model_path = model_path
        self.adapter_path = adapter_path or None
        self._model = None
        self._tok = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            from mlx_lm import load
        except Exception as e:              # pragma: no cover - environment issue
            raise BackendError(
                "mlx-lm is not installed in this interpreter. Run with a Python "
                "that has mlx-lm (e.g. the project .venv), or use "
                "--backend ollama.") from e
        self._model, self._tok = load(self.model_path,
                                      adapter_path=self.adapter_path)

    def _sampler(self, temperature: float):
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=temperature)

    def _prompt(self, messages: list) -> str:
        self._ensure()
        return _apply_template(self._tok, messages)

    def chat(self, messages, temperature, max_tokens, seed=0):
        import mlx.core as mx
        from mlx_lm import generate
        self._ensure()
        prompt = _apply_template(self._tok, messages)
        mx.random.seed(int(seed))
        text = generate(self._model, self._tok, prompt=prompt,
                        max_tokens=int(max_tokens),
                        sampler=self._sampler(temperature))
        pt = len(self._tok.encode(prompt))
        ct = len(self._tok.encode(text))
        return text, {"prompt_tokens": pt, "completion_tokens": ct,
                      "total_tokens": pt + ct}

    def stream(self, messages, temperature, max_tokens, seed=0) -> Iterator[str]:
        import mlx.core as mx
        from mlx_lm import stream_generate
        self._ensure()
        prompt = _apply_template(self._tok, messages)
        mx.random.seed(int(seed))
        for r in stream_generate(self._model, self._tok, prompt=prompt,
                                 max_tokens=int(max_tokens),
                                 sampler=self._sampler(temperature)):
            piece = getattr(r, "text", None)
            yield piece if piece is not None else str(r)

    def info(self) -> dict:
        return {"backend": "mlx", "base_model": self.model_path,
                "adapter": self.adapter_path}


class OllamaBackend:
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model

    def _post(self, path: str, payload: dict):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=600)

    def chat(self, messages, temperature, max_tokens, seed=0):
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "seed": seed,
                            "num_predict": int(max_tokens)}}
        try:
            with self._post("/api/chat", body) as r:
                data = json.loads(r.read().decode())
        except Exception as e:
            raise BackendError(f"ollama request failed: {e}") from e
        text = (data.get("message") or {}).get("content", "")
        usage = {"prompt_tokens": data.get("prompt_eval_count", 0),
                 "completion_tokens": data.get("eval_count", 0),
                 "total_tokens": (data.get("prompt_eval_count", 0)
                                  + data.get("eval_count", 0))}
        return text, usage

    def stream(self, messages, temperature, max_tokens, seed=0) -> Iterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True,
                "options": {"temperature": temperature, "seed": seed,
                            "num_predict": int(max_tokens)}}
        try:
            r = self._post("/api/chat", body)
        except Exception as e:
            raise BackendError(f"ollama request failed: {e}") from e
        for line in r:
            if not line.strip():
                continue
            try:
                obj = json.loads(line.decode())
            except json.JSONDecodeError:
                continue
            piece = (obj.get("message") or {}).get("content")
            if piece:
                yield piece
            if obj.get("done"):
                break

    def info(self) -> dict:
        return {"backend": "ollama", "url": self.url, "model": self.model}


def make_backend(name: str, **kw):
    name = (name or "mlx").lower()
    if name == "mlx":
        return MlxBackend(kw["model_path"], kw.get("adapter_path"))
    if name == "ollama":
        return OllamaBackend(kw["ollama_url"], kw["ollama_model"])
    raise BackendError(f"unknown backend {name!r} (use 'mlx' or 'ollama')")
