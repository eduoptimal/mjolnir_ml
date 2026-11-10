"""mjolnir_prover.lib.trace — proof-trace recorder (S9).

The neurosymbolic trace: while the gate proves, it RECORDS each inference step
as a DAG node {id, cls, rule:version, in, out, deps}. Three classes:

  RULE  — locally checkable rewrite/decision (sub, fold, peel, interval op)
  LEMMA — named identity, version hash cited (faulhaber, len>=0)
  ATOM  — the canonicalizer nf: the explicit TCB floor (hash-pinned)

The recorder is deterministic: step ids are content-hashed, so the same proof
always yields the same DAG — the kernel can RECONSTRUCT and compare.
"""
from __future__ import annotations
import hashlib, threading

_TLS = threading.local()


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def start() -> None:
    _TLS.steps = []


def steps() -> list:
    if getattr(_TLS, "steps", None) is None:
        _TLS.steps = []
    return _TLS.steps


def finish() -> list:
    out = getattr(_TLS, "steps", []) or []
    _TLS.steps = None
    return out


def rule(name: str, version: int, inp, outp, deps=()):
    """Record a RULE step; returns the step id."""
    return _add("RULE", f"{name}:{version}", inp, outp, deps)


def lemma(name: str, version: int, inp, outp, deps=()):
    """Record a LEMMA citation (identity with version hash)."""
    return _add("LEMMA", f"{name}:{version}", inp, outp, deps)


def atom(inp, outp, deps=()):
    """Record an ATOM step (canonicalizer nf) — the TCB floor."""
    return _add("ATOM", "nf:1", inp, outp, deps)


def _add(cls, rule_name, inp, outp, deps):
    if getattr(_TLS, "steps", None) is None:
        _TLS.steps = []                     # recording outside a session: auto-start
    body = f"{cls}|{rule_name}|{_ser(inp)}|{_ser(outp)}|{sorted(deps)}"
    sid = "t" + _h(body)
    _TLS.steps.append(dict(id=sid, cls=cls, rule=rule_name, in_=inp,
                           out=outp, deps=sorted(deps)))
    return sid


def _ser(x):
    if isinstance(x, (tuple, list)):
        return "[" + ",".join(_ser(i) for i in x) + "]"
    if x.__class__.__name__ == "Fraction":
        return f"{x.numerator}/{x.denominator}"
    return str(x)


def _jsonify(x):
    """JSON-safe conversion for artifact storage: Fractions -> 'p/q' strings."""
    if isinstance(x, (tuple, list)):
        return [_jsonify(i) for i in x]
    if x.__class__.__name__ == "Fraction":
        return f"{x.numerator}/{x.denominator}"
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            kk = k if isinstance(k, str) else _ser(k)   # tuple monomial keys
            out[kk] = _jsonify(v)
        return out
    return x


def finish():
    out = getattr(_TLS, "steps", []) or []
    _TLS.steps = None
    return [_jsonify(s) for s in out]


def digest(trace: list) -> str:
    """Content hash of a full trace (kernel reconstruct-and-compare)."""
    return _h("|".join(f"{s['id']}{s['cls']}{s['rule']}{_ser(s['in_'])}"
                       f"{_ser(s['out'])}{s['deps']}" for s in trace))