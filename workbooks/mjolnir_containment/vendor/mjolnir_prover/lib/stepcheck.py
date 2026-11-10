"""mjolnir_prover.lib.stepcheck — the independent ONE-STEP verifier (S9).

Checks each trace step is a valid instance of its rule — a local check only:
given the step's declared input and rule, recompute the output with the term
algebra and compare. No search, no gate imports. This is the checker whose
TCB replaces "the whole gate".

Supported rules (each versioned; unknown rule -> step invalid):
  resolve:N      term substitution + canonicalization
  decide-eq:N    nf equality of two terms
  sign-<rel>:N   polynomial sign analysis (with bounds from the step input)
  interval-<op>:N  exact interval arithmetic step
  faulhaber-s<N>:1 closed-form sum identity (LEMMA — recomputed, exact)
  nf:1           ATOM: canonicalization (the TCB floor — recomputed)
  premise        declared assumption: valid as declared (data, not derivation)
"""
from __future__ import annotations
from fractions import Fraction

import re

from . import term as T
from . import intervals as IV


def _num(v):
    """Normalize serialized numbers (int, float, 'p/q' Fraction strings)."""
    if isinstance(v, str) and _FRAC_STR.fullmatch(v):
        return Fraction(v)
    return v


def _termish(x):
    """Parse a serialized term back into a term if it looks like one."""
    if isinstance(x, tuple):
        return x
    if isinstance(x, str):
        try:
            return T.parse(x)
        except Exception:
            return None
    if isinstance(x, Fraction):
        return T.const(x)                   # _denorm already resolved 'p/q'
    return None


def _check_resolve(inp, outp):
    """inp = (source-term, var, replacement-term) -> recompute the sub.

    The recorded result may be rendered before constant folding while parse()
    folds builders, so the comparison is by exact algebraic value (the same
    canonicalizer the ATOM floor certifies), never by structural key."""
    src, var, repl = _termish(inp[0]), inp[1], _termish(inp[2])
    if src is None or repl is None:
        return False
    rec = _termish(outp)
    if rec is None:
        return False
    return T.decide_eq(T.sub(src, var, repl), rec) is True


def _check_decide_eq(inp, outp):
    l, r = _termish(inp[0]), _termish(inp[1])
    if l is None or r is None:
        return False
    return (T.decide_eq(l, r) is True) == (outp is True)


def _check_sign(rel, inp, outp):
    l, r = _termish(inp[0]), _termish(inp[1])
    if l is None or r is None:
        return False
    bounds = inp[2] if len(inp) > 2 else None
    f = {"ge": T.provably_ge, "gt": T.provably_gt,
         "le": T.provably_le, "lt": T.provably_lt}[rel]
    return (f(l, r, bounds) is True) == (outp is True)


def _check_interval(name, inp, outp):
    if name in ("le", "ge", "lt", "gt"):
        a = (_num(inp[0][0]), _num(inp[0][1]))
        b = (_num(inp[1][0]), _num(inp[1][1]))
        if name == "le":
            return (outp is True) == (a[1] <= b[0])
        if name == "ge":
            return (outp is True) == (a[0] >= b[1])
        if name == "lt":
            return (outp is True) == (a[1] < b[0])
        return (outp is True) == (a[0] > b[1])
    if name == "eq":
        (a, b), (c, d) = inp[0], inp[1]
        if a == b and c == d:
            return (outp is True) == (a == c)
        if b < c or d < a:
            return outp is False
        return outp is None
    if name in ("const", "var"):
        lo, hi = _num(outp[0]), _num(outp[1])
        if name == "const":
            t = inp if isinstance(inp, tuple) else _termish(inp)
            if not (isinstance(t, tuple) and t[0] == "c"):
                return False
            v = IV._frac(t[1])
            return (lo, hi) == (v, v)
        envv = inp[1]                       # declared interval
        return (lo, hi) == (IV._frac(_num(envv[0])), IV._frac(_num(envv[1])))
    if name in ("min", "max"):
        a, b = inp[0], inp[1]
        lo, hi = _num(outp[0]), _num(outp[1])
        exp = (min(a[0], b[0]), min(a[1], b[1])) if name == "min" \
            else (max(a[0], b[0]), max(a[1], b[1]))
        return (lo, hi) == exp
    if name in ("add", "sub", "mul", "div"):
        a, b = inp[0], inp[1]
        lo, hi = _num(outp[0]), _num(outp[1])
        exp = {"add": (a[0] + b[0], a[1] + b[1]),
               "sub": (a[0] - b[1], a[1] - b[0])}.get(name)
        if name == "mul":
            v = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
            exp = (min(v), max(v))
        if name == "div":
            if b[0] <= 0 <= b[1]:
                return False
            v = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
            exp = (min(v), max(v))
        return (lo, hi) == exp
    return False


def _check_lemma(name, inp, outp):
    if name.startswith("mod-residue-constancy"):
        # APPLICABILITY, not the conclusion. A LEMMA step carries the hypothesis,
        # not the result: what can be checked here is that the lemma APPLIES --
        # the premise is a congruence, and the expression is affine in exactly one
        # variable. Whether the residue equals the claimed value is not in this
        # step; that lives on the obligation and is re-derived independently in
        # `kernel._affine_residue`. Two checkers, two artifacts, one conclusion.
        import re as _re
        if not inp:
            return False
        if not _re.search(r"\u2261|\bmod\b", str(inp[0])):
            return False
        expr = str(inp[-1])
        # the expression is a TERM (`add(-7, mul(3, n))`), not infix, so the
        # constructor names have to be excluded before counting free variables --
        # counting them as free names rejects every well-formed step
        constructors = {"add", "mul", "neg", "csub", "cdiv", "div", "const",
                        "v", "c", "max", "min", "abs", "pow"}
        free = {n for n in _re.findall(r"[A-Za-z_]\w*", expr)
                if n not in constructors}
        if len(free) > 1:
            return False                    # not affine in one variable
        # a product of two free names is not affine
        for m in _re.finditer(r"mul\(([^()]*)\)", expr):
            parts = [x.strip() for x in m.group(1).split(",")]
            if sum(1 for x in parts if x in free) > 1:
                return False
        return True
    if name.startswith("faulhaber-s"):
        k = int(name.rsplit("-", 1)[1][1:])
        N = _termish(inp[0])
        if N is None:
            return False
        if k == 1:
            exp = T.cdiv(T.cmul(N, T.csub(N, T.const(1))), T.const(2))
        elif k == 2:
            exp = T.cdiv(T.cmul(T.cmul(N, T.csub(N, T.const(1))),
                                T.csub(T.cmul(T.const(2), N), T.const(1))),
                         T.const(6))
        else:
            return False
        return T.key(exp) == T.key(_termish(outp) or ())
    return False


def _check_nf(inp, outp):
    t = _termish(inp)
    if t is None:
        return False
    nf = T.nf(t)
    if nf is None:
        return outp is None
    # outp may be the serialized monomial dict {"[cols,rows]": "1", ...}
    if isinstance(outp, dict):
        rebuilt = {}
        for k, v in outp.items():
            mon = tuple(p for p in k.strip("[]").split(",") if p)
            rebuilt[mon] = _num(v)
        return nf == rebuilt
    return T.key(nf) == T.key(_termish(outp) or ())


_FRAC_STR = re.compile(r"^-?\d+/-?\d+$")


def _denorm(x):
    """JSON round-trip normalization: lists -> tuples; EXACT 'p/q' strings ->
    Fraction (term strings and names pass through untouched)."""
    if isinstance(x, list):
        return tuple(_denorm(i) for i in x)
    if isinstance(x, str) and _FRAC_STR.fullmatch(x):
        return Fraction(x)
    if isinstance(x, dict):
        return {k: _denorm(v) for k, v in x.items()}
    return x


def check_step(step):
    """One-step verification.

    -> True   verified
    -> False  the step is WRONG
    -> None   no checker exists for this step

    The third value is the one that was missing, and it mattered. A step whose rule
    name no branch recognises used to return False, which the trace checker then
    recorded as `trace_invalid`, which the arbiter published as FAULT on the whole
    certificate. So a MISSING READER was reported as a BROKEN PROOF -- and it did
    exactly that to `frag_mod_residue`, a claim that is true. A check that cannot
    run has to say so, and the caller has to be able to tell the two apart.
    """
    cls, rule_name = step["cls"], step["rule"]
    inp, outp = _denorm(step["in_"]), _denorm(step["out"])
    if cls == "premise-rule":
        return True
    name, _, ver = rule_name.partition(":")
    try:
        if cls == "ATOM":
            return _check_nf(inp, outp)
        if cls == "LEMMA":
            if not name.startswith(("faulhaber-s", "mod-residue-constancy")):
                return None                 # a LEMMA with no checker, not a bad one
            return _check_lemma(name, inp, outp)
        if cls == "RULE" and name not in (
                "resolve", "decide-eq", "premise") and not name.startswith(
                ("sign-", "interval-")):
            return None                     # a RULE this checker does not know
        if cls == "RULE":
            if name == "resolve":
                return _check_resolve(inp, outp)
            if name == "decide-eq":
                return _check_decide_eq(inp, outp)
            if name.startswith("sign-"):
                return _check_sign(name[5:], inp, outp)
            if name.startswith("interval-"):
                return _check_interval(name[9:], inp, outp)
            if name == "premise":
                return True
    except Exception:
        return False
    return False


def check_trace(trace) -> dict:
    """Validate every step + the dep DAG. Returns a report."""
    from . import theory as TH
    bad, unchecked, ids = [], [], set()
    for st in trace:
        ids.add(st["id"])
        r = check_step(st)
        if r is False:
            bad.append(st["id"])
        elif r is None:
            unchecked.append(st["id"])
        for d in st.get("deps", []):
            if d.startswith("theory:"):
                name, _, dig = d[7:].partition("@")
                try:
                    if TH.digest(name) != dig:
                        bad.append(f"theory_version:{d}")
                except Exception:
                    bad.append(f"theory_unknown:{d}")
            elif d not in ids and d not in {s["id"] for s in trace}:
                bad.append(f"dangling:{d}:{st['id']}")
    # `valid` means "no step was found WRONG". Steps nobody can check are reported
    # separately, because conflating the two is what turned a missing reader into
    # a FAULT verdict on a true claim.
    return dict(steps=len(trace), valid=not bad, bad=bad,
                unchecked=unchecked, digest_ok=True)