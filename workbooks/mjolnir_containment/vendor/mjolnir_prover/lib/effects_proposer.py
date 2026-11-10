"""mjolnir_prover.lib.effects_proposer — the E6 proposer SLOT (encoder side).

The frozen encoder proposes an effect class for a function; the deterministic
gate (`lib.effects`) disposes. Separation preserved: the encoder never decides.

Default checkpoint = **ranker-mini**, the encoder that cleared the pre-declared
GO rule in `demo/p26_encoder_swap.py`: 2-class triage (pure vs mutating) 183/240
= 76.2% vs a 68% majority baseline on gate ground truth (TheAlgorithms + MLFS).

Override with MJ_EFFECTS_ENCODER (a checkpoint dir or HF model id). Loading is
lazy so importing this module never requires torch/transformers; if the encoder
or its deps are unavailable, `propose_effect` returns None and the caller stays
fail-closed (the gate still disposes; the slot simply abstains).
"""
from __future__ import annotations
import ast
import os

from . import paths as _paths

_HERE = os.path.dirname(os.path.abspath(__file__))
# the GO checkpoint lives in the detached storage plane (copied, no symlinks)
DEFAULT_ENCODER = os.environ.get("MJ_EFFECTS_ENCODER",
                                 _paths.models("ranker-mini"))

# effect-pattern anchors (propose-side DATA): the two-class triage the gate
# validates. Extendable; the gate alone decides correctness of a proposal.
ANCHORS = {
    "pure": "a pure function with no side effects that computes and returns a "
            "value without modifying its arguments",
    "mutates": "sorts or modifies the collection in place and changes the "
               "argument that was passed in",
}

_cache = {}


def load_encoder(path=DEFAULT_ENCODER):
    """Load (and cache) the frozen encoder. Returns (tok, model) or None."""
    if path in _cache:
        return _cache[path]
    if path and not (os.path.isdir(path) or "/" not in path):
        _cache[path] = None
        return None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception:
        _cache[path] = None
        return None
    try:
        if not path:
            path = "sentence-transformers/all-MiniLM-L6-v2"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path)
        model.eval()
    except Exception:
        _cache[path] = None
        return None
    _cache[path] = (tok, model)
    return _cache[path]


def _embed(tok, model, text):
    import torch
    b = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        h = model(**b).last_hidden_state[:, 0, :].squeeze(0)
    return h / (torch.norm(h) + 1e-8)


def fn_text(fn, mode="name"):
    """The text the encoder sees for a function node. Default matches the
    p26-measured configuration (name + docstring)."""
    if mode == "body":
        return ast.unparse(fn)
    if mode == "signature":
        return (f"{fn.name} {ast.get_docstring(fn) or ''} "
                f"({', '.join(p.arg for p in fn.args.args)})")
    return f"{fn.name} {ast.get_docstring(fn) or ''}"


def propose_effect(fn, path=DEFAULT_ENCODER, mode="name"):
    """Propose one effect class for a function node (or source string).
    Returns a class name from ANCHORS, or None (fail-closed) if unavailable."""
    enc = load_encoder(path)
    if enc is None:
        return None
    if isinstance(fn, str):
        fn = ast.parse(fn).body[0]
    tok, model = enc
    books = {k: _embed(tok, model, p) for k, p in ANCHORS.items()}
    z = _embed(tok, model, fn_text(fn, mode))
    return max(books, key=lambda k: float(books[k] @ z))
