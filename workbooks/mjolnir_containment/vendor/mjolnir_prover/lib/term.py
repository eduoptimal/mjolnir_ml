"""mjolnir_prover.lib.term — self-contained typed TERM algebra (Int/Bool). No CAS.

Canonical tuple nodes + constant folding + identity laws + commutative sort, an
exact polynomial normal form (for equality VCs), and an integer evaluator.
Fast, total, no NaN, no hangs. sympy is NOT used here.
"""
from __future__ import annotations
from collections import defaultdict

INT_OPS = {"add", "sub", "mul", "neg", "max", "min", "abs"}


def var(n): return ("v", n)
def const(c):
    """Constants: ints and Fractions (arithmetic-active), strings (inert).
    Fractions encode float literals EXACTLY as written (Fraction(str))."""
    if isinstance(c, str):
        return ("c", c)
    if isinstance(c, float):
        from fractions import Fraction as _F
        return ("c", _F(str(c)))
    if c.__class__.__name__ == "Fraction":
        return ("c", c)                     # already exact: pass through
    return ("c", int(c))
def _is_c(t): return t[0] == "c"


def key(t): return repr(t)


def _flatten(t, tag):
    if t[0] == tag:
        for x in t[1:]: yield from _flatten(x, tag)
    else:
        yield t


def cadd(a, b):
    """AC-normal addition: flatten nested adds, combine constants, sort."""
    c, rest = 0, []
    for t in list(_flatten(a, "add")) + list(_flatten(b, "add")):
        if t[0] == "c" and isinstance(t[1], (int, _FRAC)): c += t[1]
        else: rest.append(t)
    if c != 0: rest.append(const(c))
    if not rest: return const(0)
    rest.sort(key=key)
    out = rest[0]
    for x in rest[1:]: out = ("add", out, x)
    return out


def cneg(a):
    if _is_c(a) and isinstance(a[1], (int, float, _FRAC)):
        return const(-a[1])                 # exact for rationals too
    if a[0] == "neg": return a[1]
    return ("neg", a)


# ----------------------------------------------------- propositional rung ---
# The BOOLOP closure, in the TCB floor alongside the arithmetic operators.
# Propositional atoms are ('p', name) — deliberately a DIFFERENT tag from
# ('v', name). An integer variable and a truth-valued atom must never be
# conflated: `x` in `x + 1 == 2` and `x` in `p | q` are unrelated claims, and
# one shared tag would let nf()'s polynomial view swallow a formula.
#
# These are constructors only — semantics live in `propositional.py`, which is
# also TCB. Keeping the two apart matches the existing split: term.py builds
# and canonicalises, the semantic module decides.


def catom(name):
    """A propositional atom."""
    return ("p", name)


def cnot(a):
    return ("pnot", a)


def cand(a, b):
    return ("pand", a, b)


def cor(a, b):
    return ("por", a, b)


def cimp(a, b):
    """Implication, in the form the decision procedure consumes."""
    return ("por", cnot(a), b)


def ciff(a, b):
    return ("pand", cimp(a, b), cimp(b, a))


PBOOL = ("pand", "por", "pnot", "pimp")


def props_of(t, acc=None):
    """Set of propositional atom names occurring in a formula.

    ITERATIVE, deliberately. ILTP formulas nest in the hundreds, and a
    recursive walk overflows the interpreter stack well before a legal problem
    ends — which turns a corpus walk into a crash on one file. An explicit
    stack has no such limit.
    """
    if acc is None:
        acc = set()
    stack = [t]
    seen = set()
    while stack:
        cur = stack.pop()
        # Shared subformulas are the norm here (a biconditional mentions each
        # operand twice), so without an identity set the walk revisits each
        # shared subtree once per path — exponentially many times on a formula
        # like `(a <=> b) <=> (c <=> d) ...`. That made the ATOM SCAN, which
        # runs BEFORE the budget check, the single most expensive thing in the
        # whole route: 94s for 270 problems, of which the 90 decidable ones
        # needed 0.01s in total.
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if cur[0] == "p":
            acc.add(cur[1])
        elif cur[0] == "c":
            continue                       # a constant carries no atoms
        else:
            stack.extend(cur[1:])
    return acc


def csub(a, b): return cadd(a, cneg(b))


def cmul(a, b):
    """AC-normal multiplication: flatten nested muls, combine constants, sort."""
    c, rest = 1, []
    for t in list(_flatten(a, "mul")) + list(_flatten(b, "mul")):
        if t[0] == "c" and isinstance(t[1], (int, _FRAC)): c *= t[1]
        else: rest.append(t)
    if c == 0: return const(0)
    if c != 1: rest.append(const(c))
    if not rest: return const(1)
    rest.sort(key=key)
    out = rest[0]
    for x in rest[1:]: out = ("mul", out, x)
    return out


def _fold2(name, f):
    def g(a, b):
        if _is_c(a) and _is_c(b) and isinstance(a[1], (int, _FRAC)) \
                and isinstance(b[1], (int, _FRAC)):
            return const(f(a[1], b[1]))
        if a == b: return a
        return (name,) + tuple(sorted((a, b), key=key))
    return g


cmax, cmin = _fold2("max", max), _fold2("min", min)


def cabs(a):
    if _is_c(a): return const(abs(a[1]))
    if a[0] == "abs": return a
    return ("abs", a)


def size(t): return 1 if t[0] in ("v", "c") else 1 + sum(size(x) for x in t[1:])


def vars_of(t, acc=None):
    """Set of program variable names occurring in a term."""
    if acc is None: acc = set()
    if t[0] == "v": acc.add(t[1])
    elif t[0] != "c":
        for x in t[1:]: vars_of(x, acc)
    return acc


def nf(t):
    """Polynomial normal form: dict monomial(tuple of vars) -> int coeff, or None."""
    k = t[0]
    if k == "c":
        if isinstance(t[1], (int, _FRAC)):
            return {(): t[1]} if t[1] != 0 else {}
        return None                      # inert string/other consts: opaque
    if k == "v": return {(t[1],): 1}
    if k == "add" or k == "mul":
        terms = [nf(x) for x in t[1:]]
        if any(x is None for x in terms): return None
        if k == "add":
            out = defaultdict(int)
            for d in terms:
                for m, c in d.items(): out[m] += c
            return {m: c for m, c in out.items() if c != 0}
        out = {(): 1}
        for d in terms:
            nxt = defaultdict(int)
            for m1, c1 in out.items():
                for m2, c2 in d.items():
                    nxt[tuple(sorted(m1 + m2))] += c1 * c2
            out = nxt
        return {m: c for m, c in out.items() if c != 0}
    if k == "neg":
        d = nf(t[1])
        return None if d is None else {m: -c for m, c in d.items()}
    if k == "sub":
        return None
    if k == "rdiv":
        # exact rational division: polynomial / nonzero constant is polynomial;
        # a variable denominator stays an opaque atom (never guessed).
        num, den = nf(t[1]), nf(t[2])
        if den is not None and set(den.keys()) == {()} and den[()] != 0 \
                and num is not None:
            return {m: _FRAC(c) / _FRAC(den[()]) for m, c in num.items()}
        return {(_atom_key(t),): 1}
    if k in ("clen", "elem", "fact", "list"):     # atoms = monomial variables
        return {(_atom_key(t),): 1}
    return None            # max/min/abs/func: not polynomial


def _atom_key(t):
    """Canonical variable key for an opaque atom (clen/elem/fact)."""
    if t[0] == "clen":
        return "clen:" + _atom_key(t[1])
    if t[0] == "elem":
        return "elem:" + _atom_key(t[1]) + "[" + _atom_key(t[2]) + "]"
    if t[0] == "fact":
        return "fact:" + _atom_key(t[1])
    if t[0] == "v":
        return t[1]
    if t[0] == "c":
        return str(t[1])
    return tstr(t)


def cfact(x): return ("fact", x)


def clist(*items):
    """Literal collection as a canonical term (variadic element terms)."""
    return ("list",) + tuple(items)


def crdiv(a, b):
    """Exact rational division (Python `/`); folds on int/Fraction constants.
    Opaque on variables -> nf abstains (the checker never guesses)."""
    if _is_c(a) and _is_c(b) and isinstance(a[1], (int, _FRAC)) \
            and isinstance(b[1], (int, _FRAC)) and b[1] != 0:
        return const(_FRAC(a[1]) / _FRAC(b[1]))
    if b[0] == "c" and b[1] == 1:
        return a
    return ("rdiv", a, b)


def clen(b):
    """Length symbol; folds on literals (lists, dicts, strings).
    clen(x) >= 0 is a built-in premise (enforced by the provers' default bound)."""
    if b[0] == "list":
        return const(len(b) - 1)
    if b[0] == "dict":
        return const((len(b) - 1) // 2)
    if b[0] == "c" and isinstance(b[1], str):
        return const(len(b[1]))
    return ("clen", b)


def celem(b, i):
    """Element/key lookup; folds on literal bases with constant index/key.

    SUBSCRIPT RUNG. Three additions, each sound-by-construction:

    * tuple index (`d[0, 1]`, the numpy/matrix shape) lowers to nested `elem`,
      so `[[1,2],[3,4]][0,1]` is the same term as `[[1,2],[3,4]][0][1]` and
      both fold to 2. Without this the two spellings had different keys, so
      nf saw two opaque atoms and refused to decide.
    * nested literal bases fold (`p[0][3]` on `p = [[0,1,2,3]]`).
    * an index provably outside the literal's length is NOT a silent bound —
      it stays an opaque `elem` atom, because an out-of-range index is a
      runtime claim (see the rung's twin), not a value.

    Everything else stays an opaque atom: `nf` already treats `elem` as an
    indeterminate, so `xs[0] - xs[0] == 0` proves and `xs[0] == 1` abstains.
    """
    if i[0] == "list":                       # tuple index -> nested elem
        out = b
        for sub in i[1:]:
            out = celem(out, sub)
        return out
    if b[0] == "list" and i[0] == "c" and isinstance(i[1], int) \
            and 0 <= i[1] < len(b) - 1:
        return b[1 + i[1]]
    if b[0] == "dict":
        for j in range(1, len(b) - 1, 2):
            if b[j] == i:
                return b[j + 1]
        return ("elem", b, i)                      # absent key: runtime claim
    if b[0] == "c" and isinstance(b[1], str) and i[0] == "c" \
            and isinstance(i[1], int) and 0 <= i[1] < len(b[1]):
        return const(b[1][i[1]])
    return ("elem", b, i)


def cdict(*flat):
    """Literal mapping term: FLAT children (k1, v1, k2, v2, ...) — same
    constructor contract as every other term (children are terms; the dict
    convention is enforced by clen/celem/ev)."""
    return ("dict",) + tuple(flat)


def cdiv(a, b):
    """Trunc-style integer division as a canonical symbol. Folds only on
    constants (Python // semantics — the declared profile); otherwise opaque,
    so nf abstains and the gate never guesses."""
    if a[0] == "c" and b[0] == "c" and b[1] != 0:
        return const(a[1] // b[1])
    if b[0] == "c" and b[1] == 1: return a
    return ("div", a, b)


def cmod(a, b):
    """Modulo as a canonical symbol; folds on constants, else opaque."""
    if a[0] == "c" and b[0] == "c" and b[1] != 0:
        return const(a[1] % b[1])
    if b[0] == "c" and b[1] == 1: return const(0)
    return ("mod", a, b)


def _mul_dicts(d1, d2):
    out = defaultdict(int)
    for m1, c1 in d1.items():
        for m2, c2 in d2.items():
            out[tuple(sorted(m1 + m2))] += c1 * c2
    return {m: c for m, c in out.items() if c != 0}


def _sub_dicts(d1, d2):
    out = defaultdict(int)
    for m, c in d1.items(): out[m] += c
    for m, c in d2.items(): out[m] -= c
    return {m: c for m, c in out.items() if c != 0}


def _find_square(d, max_terms=3, max_coef=20, budget=100000):
    """Find a small integer-coefficient polynomial Q with nf(Q^2) <= d termwise
    (Q^2 subtractable from d). EXACT: nf(Q*Q) is verified before use, so this
    is sound. Bounded search: support <= max_terms monomials of d, |coef| <= max_coef.
    `budget` caps the number of candidate Qs tried; on exhaustion it returns None
    (abstain) — sound, and it keeps the prover from blowing up on wide inputs.
    """
    support = sorted(d.keys(), key=lambda m: (len(m), m))
    n = len(support)
    if n == 0 or n > 10:
        return None
    B = max_coef
    tries = 0
    # try supports: largest first (peel big squares before trivial constants)
    from itertools import combinations, product
    for size in range(min(max_terms, n), 0, -1):
        for sub in combinations(support, size):
            ranges = [range(-B, B + 1)] * size
            # first monomial coeff nonzero
            for coefs in product(*ranges):
                if coefs[0] == 0:
                    continue
                tries += 1
                if tries > budget:               # bounded: abstain, never guess
                    return None
                Q = {m: c for m, c in zip(sub, coefs) if c != 0}
                if len(Q) != len(sub):
                    continue                     # zero coefs duplicate smaller supports
                Q2 = _mul_dicts(Q, Q)
                # Q^2 must be termwise subtractable from d (sound greedy subset)
                if all(m in d and d[m] * c >= 0 and abs(c) <= abs(d[m])
                       for m, c in Q2.items()):
                    return Q
    return None


def _sos(d, bounds=None):
    """Three-valued prover for P >= 0 on ALL integers under premises
    bounds = {var: lb}:
      True  - P >= 0 proven (sos structure / exact square peeling / bounded min)
      False - P < 0 PROVEN (only over the UNRESTRICTED domain, constants-only)
      None  - undecidable (abstain)
    Soundness notes: Q^2 >= 0 exactly (nf-verified), so peeling is sound for
    any coefficient signs; bounded minima are lower bounds — True direction
    only. Under premises (restricted domain) False is never returned: a
    full-domain refutation is not evidence on a restricted domain.
    """
    if d is None:
        return None
    if _sos_sum(d, bounds) is True:
        return True
    remaining = {m: c for m, c in d.items() if c != 0}
    if remaining:
        for _ in range(8):
            if not remaining:
                return True
            if _sos_sum(remaining, bounds) is True:
                return True
            Q = _find_square(remaining)
            if Q is None:
                break
            remaining = _sub_dicts(remaining, _mul_dicts(Q, Q))
        else:
            return True
        if not remaining:
            return True
    if bounds and all(c >= 0 for m, c in d.items() if m):
        if _sos_min(d, bounds, 0) is True:
            return True
    if not bounds and _sos_sum(d, None) is False:
        return False
    return None


def _quad_sign(d):
    """E-quad: univariate quadratic sign analysis (exact, bounded MVF).
    d: nf dict of a univariate quadratic (single var, max degree 2).
    Returns 'pos' (P >= 0 for all x), 'neg' (P <= 0 for all x), or None.
    Exact via: P >= 0 for all reals iff a > 0 and disc <= 0 (with a=0 the
    linear/boundary cases handled by the polynomial bounds when premises exist)."""
    if d is None:
        return None
    var = None
    a2 = b2 = c2 = _FRAC(0)
    for mon, coef in d.items():
        if not mon:
            c2 += _FRAC(coef)               # constant term
            continue
        cnt = {}
        for v in mon:
            cnt[v] = cnt.get(v, 0) + 1
        if len(cnt) > 1:
            return None
        v, e = next(iter(cnt.items()))
        if var is None:
            var = v
        elif var != v:
            return None
        if e == 2: a2 += _FRAC(coef)
        elif e == 1: b2 += _FRAC(coef)
        else: return None
    if var is None:
        return None
    # linear or constant: exact boundary check only with premises -> None here
    if a2 == 0:
        return None
    disc = b2 * b2 - 4 * a2 * c2
    if a2 > 0 and disc <= 0: return 'pos'
    if a2 < 0 and disc <= 0: return 'neg'
    return None


def _sos_sum(d, bounds):
    """Three-valued: True (P >= 0 proven) / False (const-only, exact, < 0) /
    None (undecidable).

    Soundness: total = sum of per-monomial LOWER BOUNDS:
      constant            -> exact value
      even multiplicity   -> c * prod(lb^e) with lb default 0 (x^2 >= 0 always)
      odd multiplicity    -> c * prod(lb^e), every var needs an EXPLICIT lb >= 0
    A sum of lower bounds >= 0 proves P >= 0; < 0 proves nothing (mins are not
    exact) -> None. False is reserved for constant-only polynomials.
    """
    total, has_var = 0, False
    for vs, c in d.items():
        if not vs:
            total += c                      # constant: exact
            continue
        has_var = True
        if c < 0:
            return None                     # negative coeff: min needs upper bounds
        cnt = {}
        for v in vs:
            cnt[v] = cnt.get(v, 0) + 1
        prod = 1
        for v, e in cnt.items():
            if e % 2:                       # odd var: explicit premise required
                lb = 0 if str(v).startswith("clen:") else (bounds or {}).get(v)
                if lb is None or lb < 0:
                    return None
            else:
                lb = (bounds or {}).get(v, 0)   # even: 0 is a sound default
            prod *= lb ** e
        total += c * prod
    if not has_var:
        return (total) >= 0                 # const-only: exact verdict
    return True if total >= 0 else None


def provably_ge(a, b, bounds=None):
    """Sound >= 0 over ALL integer inputs (sos incl. squares + premises)."""
    return _sos(nf(csub(a, b)), bounds)


def provably_gt(a, b, bounds=None):
    """Sound P > 0: exact for constant-only polynomials, else 'provably >= 1'
    (over ℤ). Never returns False for a non-constant polynomial — 'not >= 1'
    is not a proof that P <= 0."""
    d = nf(csub(a, b))
    if d is None:
        return None
    if all(not vs for vs in d):                # constant-only: exact
        return d.get((), _FRAC(0)) > 0
    return True if _sos_min(d, bounds, 1) is True else None


def provably_le(a, b, bounds=None):
    return _sos(nf(csub(b, a)), bounds)


def provably_lt(a, b, bounds=None):
    """Sound P < 0 (mirror of provably_gt)."""
    d = nf(csub(b, a))
    if d is None:
        return None
    if all(not vs for vs in d):                # constant-only: exact
        return d.get((), _FRAC(0)) > 0
    return True if _sos_min(d, bounds, 1) is True else None


def _sos_min(d, bounds, k):
    """Provably >= k under bounds: per-monomial minimum values sum to >= k.
    Monomial min: c * prod(lb_i^mult_i) (c >= 0, odd vars need lb >= 0).
    Constant-only monomials need no bounds."""
    total = 0
    for vs, c in d.items():
        if not vs:
            total += c                      # constant: exact
            continue
        if c < 0:
            return None
        prod = 1
        cnt = {}
        for v in vs:
            cnt[v] = cnt.get(v, 0) + 1
        for v, e in cnt.items():
            lb = 0 if str(v).startswith("clen:") else (bounds or {}).get(v)
            if lb is None or lb < 0:
                return None
            prod *= lb ** e
        total += c * prod
    return (total) >= k


def from_nf(d):
    """Rebuild a canonical polynomial term from an nf dict (generic inverse)."""
    out = const(0)
    for mono, coeff in d.items():
        if coeff == 0: continue
        m = const(1)
        for name in mono: m = cmul(m, var(name))
        if coeff < 0: m, coeff = cneg(m), -coeff
        if coeff != 1: m = cmul(const(coeff), m)
        out = cadd(out, m)
    return out


def app(name, *args):
    """Generic function-symbol application. Symbols are data, not core knowledge."""
    return (name,) + tuple(args)


def pat(name):
    """Pattern variable for a theory equation (distinct from program vars)."""
    return ("?", name)


_COMM = {"add", "mul", "max", "min"}


def _match(p, t, b):
    k = p[0]
    if k == "?":                                   # pattern variable
        n = p[1]
        if n in b: return b if key(b[n]) == key(t) else None
        b2 = dict(b); b2[n] = t; return b2
    if k in ("c", "v"):                            # literal constant / program var
        return b if key(p) == key(t) else None
    if p[0] != t[0] or len(p) != len(t): return None
    ps, ts = list(p[1:]), list(t[1:])
    if p[0] in _COMM and len(ps) == 2:             # commutative: try both orders
        for order in (ts, ts[::-1]):
            bb = b
            for x, y in zip(ps, order):
                bb = _match(x, y, bb)
                if bb is None: break
            if bb is not None: return bb
        return None
    for x, y in zip(ps, ts):
        b = _match(x, y, b)
        if b is None: return None
    return b


def _inst(t, b):
    if t[0] == "?": return b[t[1]]
    if t[0] in ("c", "v"): return t
    args = [_inst(x, b) for x in t[1:]]
    return _BUILD[t[0]](*args) if t[0] in _BUILD else (t[0],) + tuple(args)


def _rw_root(t, rules):
    for lhs, rhs in rules:
        b = _match(lhs, t, {})
        if b is not None:
            t2 = _inst(rhs, b)
            if key(t2) != key(t): return t2
    return t


def rewrite(t, rules=(), _depth=0):
    """Congruence rewriting to a bounded fixpoint under a SUPPLIED equational
    theory (rules = [(lhs, rhs), ...]). The core has no function-specific rules."""
    if _depth > 12: return t
    if t[0] in ("c", "v"): return t
    args = [rewrite(x, rules, _depth + 1) for x in t[1:]]
    t = _BUILD[t[0]](*args) if t[0] in _BUILD else (t[0],) + tuple(args)
    for _ in range(12):
        n = _rw_root(t, rules)
        if key(n) == key(t): break
        t = n
    return t


def decide_eq(a, b, rules=()):
    a, b = rewrite(a, rules), rewrite(b, rules)
    if key(a) == key(b): return True
    # inert constants (strings/keys): exact value equality, no algebra needed
    if a[0] == "c" and b[0] == "c":
        return a[1] == b[1]                       # exact: never a guess
    na, nb = nf(a), nf(b)
    if na is None or nb is None: return None      # UNDECIDED
    return na == nb


def ev(t, env):
    k = t[0]
    if k == "c": return t[1]
    if k == "v": return env[t[1]]
    if k == "add": return ev(t[1], env) + ev(t[2], env)
    if k == "mul": return ev(t[1], env) * ev(t[2], env)
    if k == "neg": return -ev(t[1], env)
    if k == "max": return max(ev(t[1], env), ev(t[2], env))
    if k == "min": return min(ev(t[1], env), ev(t[2], env))
    if k == "abs": return abs(ev(t[1], env))
    if k == "fact":
        import math as _m
        return _m.factorial(ev(t[1], env))
    if k == "div": return ev(t[1], env) // ev(t[2], env)   # trunc profile
    if k == "rdiv": return ev(t[1], env) / ev(t[2], env)   # true division
    if k == "mod": return ev(t[1], env) % ev(t[2], env)
    if k == "list": return [ev(x, env) for x in t[1:]]
    if k == "dict":
        return {ev(t[j], env): ev(t[j + 1], env) for j in range(1, len(t) - 1, 2)}
    if k == "clen": return len(ev(t[1], env))
    if k == "elem": return ev(t[1], env)[ev(t[2], env)]
    raise ValueError(f"not evaluable: {t}")


def tstr(t):
    k = t[0]
    if k == "v": return t[1]
    if k == "c":
        if isinstance(t[1], str):
            return repr(t[1])
        if isinstance(t[1], _FRAC):
            # EXACT rendering: the step-checker replays recorded steps through
            # parse(); a float repr is lossy for non-decimal rationals (1/3).
            f = t[1]
            return str(int(f)) if f.denominator == 1 \
                else f"{f.numerator}/{f.denominator}"
        return str(t[1])
    if k == "neg":
        # a negated NUMERIC constant renders as the exact signed constant:
        # "(-1/3)" would break the tstr->parse round trip (the checker must
        # re-derive recorded steps from these strings verbatim).
        if t[1][0] == "c" and isinstance(t[1][1], (int, _FRAC)):
            return tstr(const(-t[1][1]))
        return f"(-{tstr(t[1])})"
    if k == "p": return str(t[1])
    if k == "pnot": return f"~{tstr(t[1])}"
    if k == "pand": return f"({tstr(t[1])} & {tstr(t[2])})"
    if k == "por": return f"({tstr(t[1])} | {tstr(t[2])})"
    if k == "pimp": return f"({tstr(t[1])} => {tstr(t[2])})"
    if k == "abs": return f"Abs({tstr(t[1])})"
    if k == "fact": return f"fact({tstr(t[1])})"
    return f"{k}({', '.join(tstr(x) for x in t[1:])})"


def pparse(s: str):
    """Inverse of `tstr` for PROPOSITIONAL terms: `(a & b)`, `(a | b)`, `~a`,
    atoms, and int/0 constants.

    The arithmetic `parse` cannot read these (`p & q` raises), and the kernel
    needs a way to rebuild a formula from a recorded string WITHOUT consulting
    the emitter — that is the whole point of a re-derivation. So the two
    grammars are kept separate and neither is asked to read the other's terms.
    """
    toks = _p_lex(s)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        t = toks[pos]
        pos += 1
        return t

    def expr():
        # tstr renders a binary connective as `(a OP b)`, so after `(` comes the
        # FIRST OPERAND, then the operator. Reading the operator first silently
        # accepts an atom named "&" and mis-nests everything else.
        tok = peek()
        if tok == "(":
            take()
            a = expr()
            op = take()
            if op not in ("&", "|"):
                raise AssertionError(f"pparse:{op}")
            b = expr()
            if take() != ")":
                raise AssertionError("pparse:unbalanced")
            return cand(a, b) if op == "&" else cor(a, b)
        if tok == "~":
            take()
            return cnot(expr())
        tok = take()
        if tok == "$true":
            return ("c", 1)
        if tok == "$false":
            return ("c", 0)
        if tok.lstrip("-").isdigit():
            return ("c", int(tok))
        return catom(tok)

    out = expr()
    if pos != len(toks):
        raise AssertionError(f"pparse:trailing:{toks[pos:]}")
    return out


def _p_lex(s: str):
    # parens must be SEPARATE tokens: tstr renders `(p & ~q)` with no space
    # after `(`, so a `\S+` alternative swallows `(p` as one token.
    import re as _re
    return _re.findall(r"\$true|\$false|[&|~()]|[^\s&|~()]+", s)


def parse(s):
    """Inverse of tstr: parse a canonical term string back into a term.
    Generic; knows no domain knowledge. Accepts ints, names, f(a, b), (-x)."""
    toks, i = _lex(s), 0

    def peek(): return toks[i] if i < len(toks) else None

    def take():
        nonlocal i
        tok = toks[i]; i += 1; return tok

    def expr():
        tok = peek()
        if tok == "(":
            take()
            t = peek()
            if t == "-":                          # (-x): negation of an expression
                take(); e = expr(); assert take() == ")"
                return cneg(e)
            e = expr()                            # e.g. "(-7.5)": the merged
            assert take() == ")"                  # token carries its own sign
            return e
        if _INT.fullmatch(tok):
            take(); return const(int(tok))
        if _FLOAT.fullmatch(tok):
            take(); return const(_FRAC(tok))
        if _FRACSTR.fullmatch(tok):             # exact 'p/q' constant
            take(); n, _, d = tok.partition("/")
            return const(_FRAC(int(n), int(d)))
        if len(tok) >= 2 and tok[0] in ("'", '"') and tok[-1] == tok[0]:
            take(); return const(_safe_literal(tok))
        name = take()
        if peek() == "(":
            take(); args = []
            if peek() != ")":
                args.append(expr())
                while peek() == ",":
                    take(); args.append(expr())
            assert take() == ")", f"parse: {s}"
            if name in _BUILD: return _BUILD[name](*args)
            return (name,) + tuple(args)
        return var(name)

    t = expr()
    assert i == len(toks), f"parse: trailing tokens in {s}"
    return t


def _safe_literal(tok):
    import ast as _a
    try:
        v = _a.literal_eval(tok)
        return v if isinstance(v, str) else tok
    except Exception:
        return tok


def _lex(s):
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch == "-":
            if out and out[-1] not in ("(", ","):
                out.append(ch); i += 1                # subtraction separator
            else:
                j = i + 1
                while j < len(s) and (s[j].isdigit() or s[j] in "./"):
                    j += 1
                out.append(s[i:j]); i = j             # signed numeric literal
        elif ch in "(,)":
            out.append(ch); i += 1
        elif ch == " " or ch == "\t":
            i += 1
        elif ch in "'\"":
            q = ch; j = i + 1
            while j < len(s) and s[j] != q:
                j += 1
            j = min(j + 1, len(s))
            out.append(s[i:j]); i = j               # quoted literal: one token
        else:
            j = i
            while j < len(s) and s[j] not in "(,)- \t'\"":
                j += 1
            out.append(s[i:j]); i = j
    return out


import re as _re
from fractions import Fraction as _FRAC
_INT = _re.compile(r"-?\d+$")
_FLOAT = _re.compile(r"-?\d+\.\d+$")
_FRACSTR = _re.compile(r"-?\d+/\d+$")


_BUILD = {}


def _register():
    _BUILD.update({"add": cadd, "sub": csub, "mul": cmul, "neg": cneg,
                   "max": cmax, "min": cmin, "abs": cabs, "fact": cfact,
                   "div": cdiv, "mod": cmod, "rdiv": crdiv, "list": clist,
                   "clen": clen, "elem": celem, "dict": cdict})


def sub(t, name, repl):
    """Substitute and RE-CANONICALIZE (so x*0 -> 0)."""
    k = t[0]
    if k == "v": return repl if t[1] == name else t
    if k == "c": return t
    args = [sub(x, name, repl) for x in t[1:]]
    return _BUILD[k](*args) if k in _BUILD else (k,) + tuple(args)


def sub_all(t, mapping, _depth=0):
    """Simultaneous substitution: replace every var v by mapping[v], recursing
    into replacement terms so composed mappings (i->b, z->mul(a,i)) resolve.
    Cycle-guarded by depth cap and self-reference check."""
    if _depth > 16: return t
    k = t[0]
    if k == "c": return t
    if k == "v":
        if t[1] in mapping:
            r = mapping[t[1]]
            if r == t: return t
            return sub_all(r, mapping, _depth + 1)
        return t
    args = [sub_all(x, mapping, _depth) for x in t[1:]]
    return _BUILD[k](*args) if k in _BUILD else (k,) + tuple(args)


def selfcheck(x):
    """Contract asserts for the algebra (coder-facing: verified by mjolnir check).
    The parameter is the quantified free integer; the gate discharges these
    statically as theorems over ALL x."""
    assert (x + 1) * (x + 1) == x * x + 2 * x + 1
    assert x * x + 2 * x + 1 == (x + 1) * (x + 1)
    assert x * x >= 0
    assert x * x < x * x + 1
    assert x + 0 == x
    assert x * 1 == x
    assert x * x + x * x == 2 * x * x
    assert (x + 1) * (x + 1) >= 0


_register()


def selfcheck_canonical(a, b, c):
    """Contract for the CANONICAL FORM, not for arithmetic.

    `selfcheck` above states theorems of the integers — sound regardless of how
    `nf` is implemented. These state something narrower and more useful: that
    the AC-normalisation in `cadd`/`cmul` and the folding in `csub` actually
    happens. `a + b` and `b + a` are DIFFERENT expression trees; they are equal
    only because both normalise to the same polynomial. A checker that compared
    terms structurally would call them unequal, so these assertions fail if the
    canonical form is lost — which is the failure mode worth pinning.

    Provenance, assertion by assertion:
      cadd / cmul  -> commutativity and associativity of the normal form
      cmul         -> distributivity is what makes the normal form canonical
      csub         -> cancellation, via cadd + cneg on the normalised poly
      cmul         -> zero and unit are folded, not carried

    Measured frontier, stated rather than implied: `nf` is a POLYNOMIAL normal
    form, so `min`, `max`, `abs`, `//` and `%` are outside it (`nf` returns
    None). No contract over those operators can be discharged by this route, and
    the gaps are recorded rather than papered over with asserts.
    """
    # canonical form is order-independent (cadd / cmul AC-normalisation)
    assert a + b == b + a
    assert a * b == b * a
    assert (a + b) + c == a + (b + c)

    # distributivity is what pins the normal form as a canonical form
    assert (a + b) * c == a * c + b * c

    # cancellation and folding (csub = cadd + cneg, then normalise)
    assert a - a == 0
    assert (a + b) - b == a
    assert 0 * a == 0
    assert a + 0 == a
    assert a * 1 == a
