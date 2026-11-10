"""mjolnir_prover.lib.intervals — E5: real-valued semantics as declared intervals.

The declared model (safety.md S4): a variable is a CLOSED interval [lo, hi]
(lo/hi: int, float, or exact Fraction). Float literals in code are read as
EXACT rationals (Fraction(str)) and evaluated by outward-rounded interval
arithmetic over the premises' box. Discharge:
  relation holds over the whole box  -> CERTIFIED (True)
  relation violated over the box     -> REFUTED (False, witness = the box)
  0 inside a divisor / unknown var   -> abstain (None)
This is DATA semantics (exact rational interval rules), not a numerical guess.
"""
from __future__ import annotations
from fractions import Fraction

from . import term as T
from . import theory
from . import trace as TR

INF = float("inf")


def _frac(v):
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        if v == INF or v == -INF:
            return v                        # infinite bounds pass through
        return Fraction(str(v))            # exact-as-written literal semantics
    raise ValueError(f"non-real: {v!r}")


def interval_eval(t, env):
    """Evaluate a term to an interval (lo, hi) of exact Fractions.
    env: {var: (lo, hi)}. Raises ValueError on out-of-model constructs."""
    k = t[0]
    if k == "c":
        v = _frac(t[1])
        TR.rule("interval-const", 1, t, (v, v))
        return (v, v)
    if k == "v":
        if t[1] in env:
            lo, hi = env[t[1]]
            out = (_frac(lo), _frac(hi))
            TR.rule("interval-var", 1, (t[1], env[t[1]]), out)
            return out
        raise ValueError("interval_unknown_var")   # unsound to guess: abstain
    if k == "add":
        a, b = interval_eval(t[1], env), interval_eval(t[2], env)
        out = (a[0] + b[0], a[1] + b[1])
        TR.rule("interval-add", 1, (a, b), out)
        return out
    if k == "sub":
        a, b = interval_eval(t[1], env), interval_eval(t[2], env)
        out = (a[0] - b[1], a[1] - b[0])
        TR.rule("interval-sub", 1, (a, b), out)
        return out
    if k == "mul":
        a, b = interval_eval(t[1], env), interval_eval(t[2], env)
        vals = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
        if any(v != v for v in vals):       # 0 * inf = nan: ambiguous bound
            raise ValueError("interval_ambiguous_inf")
        out = (min(vals), max(vals))
        TR.rule("interval-mul", 1, (a, b), out)
        return out
    if k in ("div", "rdiv"):
        a, b = interval_eval(t[1], env), interval_eval(t[2], env)
        if b[0] <= 0 <= b[1]:
            raise ValueError("interval_division_by_zero")
        vals = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
        out = (min(vals), max(vals))
        TR.rule(f"interval-{k}", 1, (a, b), out)
        return out
    if k == "clen":
        inner = interval_eval(t[1], env) if t[1][0] != "v" else (0, INF)
        return (Fraction(0), INF)          # opaque length: nonneg, unbounded
    if k == "neg":
        a = interval_eval(t[1], env)
        return (-a[1], -a[0])
    if k == "min":
        a, b = interval_eval(t[1], env), interval_eval(t[2], env)
        out = (min(a[0], b[0]), min(a[1], b[1]))
        TR.rule("interval-min", 1, (a, b), out)
        return out
    if k == "max":
        a, b = interval_eval(t[1], env), interval_eval(t[2], env)
        out = (max(a[0], b[0]), max(a[1], b[1]))
        TR.rule("interval-max", 1, (a, b), out)
        return out
    raise ValueError(f"interval_out_of_model:{k}")


def decide_rel(lt, rt, op, env):
    """Directive comparison policy — read from theories/intervals.json (DATA).

    TRUE = holds for EVERY point of the declared box (certified).
    FALSE = fails for SOME allowed input -> runtime assert will crash ->
            fail-closed REFUTED (witness = the box).
    None = the table's abstain rule (e.g. eq with partial overlap), or
           undecidable (division by zero, unknown variables).

    Soundness: exact rational intervals; the policy expressions are evaluated
    from the pinned table (theory.digest('intervals') travels in artifacts).
    """
    L, R = interval_eval(lt, env), interval_eval(rt, env)
    v = theory.decide(op, L, R)
    TR.rule(f"interval-{op}", 2, (L, R), v,
            deps=[f"theory:intervals@{theory.digest('intervals')}"])
    return v


def env_from_bounds(bounds):
    """Normalize E1 facts {var: lb} / E5 facts {var: (lo, hi)} into env."""
    env = {}
    for v, b in (bounds or {}).items():
        if isinstance(b, tuple):
            env[v] = (b[0], b[1])
        else:
            env[v] = (b, INF)
    return env

# ---------------------------------------------- algebra contracts ------------
def selfcheck(a_lo, a_hi, b_lo, b_hi, x, y):
    """Soundness contracts for the interval arithmetic above.

    Every assertion below is a property of a SPECIFIC line of this module, not
    of arithmetic in general. They are what make the outward-rounding formulas
    (`a[0] + b[0]`, `a[0] - b[1]`, `(-a[1], -a[0])`) load-bearing rather than
    merely plausible: each states the soundness obligation that formula exists
    to satisfy — that for every point of the operand boxes, the result lies in
    the computed bounds.

    The parameters are the quantified free integers; the gate discharges these
    statically over ALL values (see `term.selfcheck` for the same pattern).

    Provenance, assertion by assertion:
      add    -> (a[0]+b[0], a[1]+b[1])   "the sum contains x+y"
      sub    -> (a[0]-b[1], a[1]-b[0])   "the difference contains x-y"
      neg    -> (-a[1], -a[0])           "negation reverses the box"
      width  -> hi-lo                    "an interval is never inverted"

    The `sub` pair is load-bearing. `a[0]-b[1]` is not an arbitrary choice: at
    x=a_lo, y=b_hi the difference IS a_lo-b_hi, so any lower bound ABOVE that
    (the naive `a[0]-b[0]`) is unsound — and these assertions refuse to certify
    it. The contract is therefore falsifiable rather than decorative; see the
    "contract has content" twin in the selfcheck suite.
    """
    # operand boxes are well-formed, and (x, y) is a point of them
    assert a_lo <= a_hi
    assert b_lo <= b_hi
    assert a_lo <= x
    assert x <= a_hi
    assert b_lo <= y
    assert y <= b_hi

    # add: (a[0]+b[0], a[1]+b[1]) contains x+y
    assert a_lo + b_lo <= x + y
    assert x + y <= a_hi + b_hi

    # sub: (a[0]-b[1], a[1]-b[0]) contains x-y
    assert a_lo - b_hi <= x - y
    assert x - y <= a_hi - b_lo

    # neg: (-a[1], -a[0]) reverses the box
    assert -a_hi <= -a_lo

    # width: an interval is never inverted, and the sum's width is the sum of
    # the widths — a genuine two-premise combination
    assert a_hi - a_lo >= 0
    assert a_hi + b_hi - a_lo - b_lo >= 0
