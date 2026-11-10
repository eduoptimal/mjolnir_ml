"""mjolnir_prover/lib/claim_ir.py — the canonical claim record (the atomic proof unit).

One IR, every frontend lowers to it, the disposal consumes it directly.
No string round-trips: the parse tree → claim records → verdicts.

Claim := {
    kind: "eq" | "ineq",
    lhs: term, rhs: term,          # algebra tuples from the parse tree
    line: int,
    is_premise: bool,              # theory-table classification
    source_text: str,              # human-readable (the certificate shows it)
}

The claim record is the atomic proof unit: one record → one verdict → one
trace sub-DAG. Proofs are as atomic as needed.
"""
from __future__ import annotations
import hashlib
import os
import math
from fractions import Fraction


class Claim:
    __slots__ = ("kind", "op", "lhs", "rhs", "line", "is_premise", "source_text")

    def __init__(self, kind, lhs, rhs, line, is_premise, source_text, op=None):
        self.kind = kind
        self.op = op if op is not None else ("=" if kind == "eq" else ">=")
        self.lhs = lhs
        self.rhs = rhs
        self.line = line
        self.is_premise = is_premise
        self.source_text = source_text

    def __repr__(self):
        return (f"Claim({self.kind}, {self.source_text!r}, "
                f"line={self.line}, premise={self.is_premise})")

    def key(self):
        """Content hash (deterministic; the trace anchors to it)."""
        from . import term as T
        body = (f"{self.kind}|{self.op}|{T.key(self.lhs)}|{T.key(self.rhs)}"
                f"|{self.line}|{self.is_premise}")
        return hashlib.sha1(body.encode()).hexdigest()[:10]

    def to_dict(self):
        from . import term as T
        return dict(kind=self.kind, op=self.op,
                    lhs=T.tstr(self.lhs), rhs=T.tstr(self.rhs),
                    line=self.line, is_premise=self.is_premise,
                    source_text=self.source_text)


def from_dict(d, ):
    """Reconstruct a Claim from its serialized form (the certificate/registry)."""
    from . import term as T
    return Claim(d["kind"], T.parse(d["lhs"]), T.parse(d["rhs"]),
                 d["line"], d["is_premise"], d["source_text"], d.get("op"))


# --------------------------------------------------------------- builders ----
def eq(lhs, rhs, line=0, is_premise=False, source=""):
    return Claim("eq", lhs, rhs, line, is_premise, source, "=")


def ineq(lhs, rhs, line=0, is_premise=False, source="", op=">="):
    return Claim("ineq", lhs, rhs, line, is_premise, source, op)

# ----------------------------------------------------- IR disposal -----------
def dispose_claims(claims, module_contracts=None, _nodecomp=False):
    """The IR disposal: consumes claim records DIRECTLY (no string re-parsing).

    Runs the same disposal logic as unit_vc (R1/R2/E-mod/E-rel/E5-intervals) but
    on structured claims. Hypotheses are ASSUMPTIONS (registered, never
    refuted); only conclusions drive the unit verdict. Returns
    (verdict, per-claim verdicts, trace).

    If the claims are undecided, retry once on an integer-decomposition of the
    claims (mod/÷ by a constant become fresh vars + bounds); this is the
    decimal-layer for `%`//` and never changes an already-decided verdict."""
    unit, results, trace = _dispose_once(claims, module_contracts)
    if unit == "UNVERIFIABLE" and not _nodecomp:
        dec = _decompose_claims(claims)
        if dec is not None:
            u2, r2, t2 = dispose_claims(dec, module_contracts, _nodecomp=True)
            if u2 != "UNVERIFIABLE":
                return u2, r2, t2
    return unit, results, trace


def _decompose_claims(claims):
    """Rewrite `X % d` and integer `X / d` (d a constant) as fresh quotient and
    remainder variables with the defining relation `X = d*q + r`, `0 ≤ r ≤ d-1`.
    Returns the rewritten claim list, or None when nothing decomposes."""
    from . import term as T
    int_vars = {c.lhs[1] for c in claims
                if c.kind == "int" and c.lhs[0] == "v"}
    all_vars = set()
    for c in claims:
        all_vars |= T.vars_of(c.lhs) | T.vars_of(c.rhs)
    # ℕ/ℤ `/` is floor division: only when EVERY variable is integer-typed do we
    # lower `/` as integer division (otherwise it stays exact rational).
    int_world = bool(all_vars) and all_vars <= int_vars
    dvars = {}

    def _intv(v):
        return v if isinstance(v, int) else (
            v.numerator if isinstance(v, Fraction) and v.denominator == 1
            else None)

    def vars_ok(t):
        return all(v in int_vars for v in T.vars_of(t))

    def get(x, d):
        key = (T.key(x), d)
        if key not in dvars:
            n = len(dvars)
            dvars[key] = (f"__q{n}", f"__r{n}", x, d)
        return dvars[key]

    def rw(t):
        k = t[0]
        if k in ("c", "v"):
            return t
        if k in ("mod", "rdiv", "div") and len(t) == 3:
            x = rw(t[1])
            d = _intv(t[2][1]) if t[2][0] == "c" else None
            if d is not None and d > 1 and vars_ok(x) and (
                    k != "rdiv" or vars_ok(x)):
                q, r, _xx, _dd = get(x, d)
                if k == "mod":
                    return T.var(r)
                if k in ("rdiv", "div"):
                    return T.var(q)
        args = tuple(rw(a) for a in t[1:])
        f = {"add": T.cadd, "sub": T.csub, "mul": T.cmul, "neg": T.cneg,
             "mod": T.cmod, "div": T.cdiv, "rdiv": T.crdiv, "max": T.cmax,
             "min": T.cmin, "abs": T.cabs, "fact": T.cfact}.get(k)
        return f(*args) if f else (k,) + args

    out = []
    for c in claims:
        out.append(Claim(c.kind, rw(c.lhs), rw(c.rhs), c.line, c.is_premise,
                         c.source_text, op=c.op))
    if not dvars:
        return None
    for _key, (q, r, x, d) in dvars.items():
        rel = T.cadd(T.cmul(T.const(d), T.var(q)), T.var(r))
        out.append(Claim("eq", rel, x, 0, True, f"{T.tstr(x)} = {d}*{q}+{r}"))
        out.append(Claim("int", T.var(q), T.const(0), 0, True, q))
        out.append(Claim("int", T.var(r), T.const(0), 0, True, r))
        out.append(Claim("ineq", T.var(r), T.const(0), 0, True, "", op=">="))
        out.append(Claim("ineq", T.var(r), T.const(d - 1), 0, True, "", op="<="))
    return out


def _dispose_once(claims, module_contracts=None):
    """The IR disposal: consumes claim records DIRECTLY (no string re-parsing).

    Runs the same disposal logic as unit_vc (R1/R2/E-mod/E-rel/E5-intervals) but
    on structured claims. Hypotheses are ASSUMPTIONS (registered, never
    refuted); only conclusions drive the unit verdict. Returns
    (verdict, per-claim verdicts, trace).
    """
    from . import term as T, trace as TR
    from .intervals import decide_rel, INF
    TR.start()
    claims = list(claims)
    # A verdict is only meaningful for a claim that must be DISCHARGED. An
    # empty claim set, or one made only of hypotheses, is not a proof — it must
    # abstain, never certify vacuously (all([]) is True only by accident).
    if not claims or not any(not cl.is_premise for cl in claims):
        TR.rule("abstain", 0, "", "no_conclusion_claim")
        return "UNVERIFIABLE", [], TR.finish()

    results = []
    eq_subst = {}
    cong = {}                            # var -> [(residue, modulus)]  (E-mod)
    ineqs = []                           # (cl, l, r, op) inequality premises
    neqs = []                            # (cl, l, r) disequality premises
    bnd = {}                             # var -> [lo, hi, lo_strict, hi_strict]
    int_vars = set()                     # ℤ/ℕ-typed variables
    contradiction = partial = False

    def _register(cl, verdict, why):
        results.append((cl, verdict, why))
        TR.rule(why, 1, cl.source_text, verdict)

    # ---- pass 1: classify hypotheses ----------------------------------------
    eq_prem = []                         # [cl, l, r, status]
    for cl in claims:
        if not cl.is_premise:
            continue
        if cl.kind == "int":             # ℤ/ℕ domain assumption
            int_vars.add(cl.lhs[1])
            _register(cl, "CERTIFIED", "int-domain")
            continue
        l, r, op = cl.lhs, cl.rhs, cl.op
        if cl.kind == "neq":
            neqs.append((cl, l, r))
            _register(cl, "CERTIFIED", "premise-neq")
            continue
        if cl.kind != "eq":
            ineqs.append((cl, l, r, op))
            continue
        # E-mod premise: `expr % m == r` (m,r ints). expr may be an arbitrary
        # integer-linear form in one variable — solved for that variable's
        # residue, e.g. `(23450 + d) % 9 = 0  ⟹  d ≡ 4 (mod 9)`.
        mp = _mod_premise(l, r, T)
        if mp is not None:
            tag, data = mp
            if tag == "contradiction":
                contradiction = True
            elif tag == "cong":
                v, res, m = data
                cong.setdefault(v, []).append((res, m))
            partial = True
            _register(cl, "CERTIFIED", "mod-premise")
            continue
        eq_prem.append([cl, l, r, None])      # status filled by the solver

    # ---- pass 2: solve the premise system to a FIXPOINT ---------------------
    # A hypothesis may LOOK nonlinear until its variables are known; iterating
    # substitution + Gauss linearizes e.g. `T*x + (T+1)*y = T+2` once T = 40.
    all_rows = []
    for _round in range(len(claims) + 3):
        still = []
        for rec in eq_prem:
            if rec[3] is not None:
                continue
            l2, r2 = _subst(rec[1], eq_subst), _subst(rec[2], eq_subst)
            d = T.nf(T.csub(l2, r2))
            if d is None:
                still.append(rec)
                continue
            if not d:
                rec[3] = "satisfied"
                continue
            if all(not vs for vs in d) and d.get((), 0) != 0:
                contradiction = True
                rec[3] = "satisfied"
                continue
            row = _linear_row(d)
            if row is None:
                still.append(rec)
                continue
            rec[3] = "row"
            all_rows.append(row)
        pending = [rec for rec in eq_prem if rec[3] is None]
        if not all_rows:
            break
        sol, consistent = _gauss_solve(all_rows)
        if not consistent:
            contradiction = True
            break
        changed = False
        for v, val in sol.items():
            term = _sol_term(val, T)
            if v not in eq_subst or T.key(eq_subst[v]) != T.key(term):
                eq_subst[v] = term
                changed = True
        if not changed and not pending:
            break
        if not pending:
            break
    for rec in eq_prem:
        if rec[3] is None:
            partial = True
            _register(rec[0], "UNVERIFIABLE", "assumed")
        else:
            _register(rec[0], "CERTIFIED",
                      "linarith-row" if rec[3] == "row" else "assumed")

    # univariate polynomial hypotheses (e.g. 6n² = 24) pin a variable's roots
    quad_prem = []                       # (var, nf dict) degree-2 hypotheses
    poly_prem = []                       # (var, coeff list) any degree
    mprem = []                           # multivariate premise polynomials
    for rec in eq_prem:
        if rec[3] is not None:
            continue
        d = T.nf(T.csub(_subst(rec[1], eq_subst), _subst(rec[2], eq_subst)))
        if d is None:
            continue
        if d and not any(mon for mon in d):
            continue                     # constant (handled as contradiction)
        if d and not any(not v.isidentifier() for mon in d for v in mon):
            mprem.append(d)
        u = _univariate(d)
        if u is not None and u[1] != 0:
            quad_prem.append((u[0], d))
        pv = _poly_var(d)
        if pv is not None:
            poly = _poly_from_nf(d, pv)
            if poly is not None and len(poly) > 1:
                poly_prem.append((pv, poly))

    # ---- pass 3: resolve inequality premises against the solved system ------
    def _set_bound(var, lo, ls, hi, hs):
        cur = bnd.get(var, [None, None, False, False])
        if lo is not None:
            if cur[0] is None or lo > cur[0]:
                cur[0], cur[2] = lo, ls
            elif lo == cur[0]:
                cur[2] = cur[2] or ls
        if hi is not None:
            if cur[1] is None or hi < cur[1]:
                cur[1], cur[3] = hi, hs
            elif hi == cur[1]:
                cur[3] = cur[3] or hs
        bnd[var] = cur
        if cur[0] is not None and cur[1] is not None and (
                cur[0] > cur[1] or (cur[0] == cur[1] and (cur[2] or cur[3]))):
            return True
        return False

    def _apply_ineq(a, b, op):
        """Try to read `a OP b` as a single-variable bound; True if applied."""
        nonlocal contradiction
        var = k = flip = None
        if a[0] == "v" and a[1].isidentifier() and _num(b) is not None:
            var, k, flip = a[1], _num(b), False
        elif b[0] == "v" and b[1].isidentifier() and _num(a) is not None:
            var, k, flip = b[1], _num(a), True
        if var is not None:
            o = _FLIP[op] if flip else op
            lo = hi = None
            ls = hs = False
            if o in (">=", ">"):
                lo, ls = k, (o == ">")
            elif o in ("<=", "<"):
                hi, hs = k, (o == "<")
            else:
                return False
            contradiction = _set_bound(var, lo, ls, hi, hs) or contradiction
            return True
        # both sides nontrivial: derive a bound on the single free variable
        ex = _linear_expr(T.csub(a, b), T)
        if ex is None:
            return False
        coefs, const = ex
        if len(coefs) != 1:
            return False
        v, c = next(iter(coefs.items()))
        if c == 0:
            return False
        rhs = (-const) / c
        o = op if c > 0 else _FLIP[op]
        lo = hi = None
        ls = hs = False
        if o in (">=", ">"):
            lo, ls = rhs, (o == ">")
        elif o in ("<=", "<"):
            hi, hs = rhs, (o == "<")
        else:
            return False
        contradiction = _set_bound(v, lo, ls, hi, hs) or contradiction
        return True

    for cl, l, r, op in ineqs:
        # raw first: a bound on a solved variable survives to be propagated
        ok = _apply_ineq(l, r, op)
        # substituted: bounds whose value depends on solved variables (E >= V-1)
        ok = _apply_ineq(_subst(l, eq_subst), _subst(r, eq_subst), op) or ok
        if not ok:
            partial = True
        _register(cl, "CERTIFIED", "premise-bound")

    # ---- close the variable system to a fixpoint: propagate bounds, derive
    # integrality congruences, pin singletons (by congruence+box or bare box) --
    def _pin_round():
        changed = False
        _propagate_bounds(eq_subst, bnd, T)
        # v = (p/q)·u + const with v,u integers ⟹ u ≡ -q·const·p⁻¹ (mod q)
        for v in list(int_vars):
            expr = eq_subst.get(v)
            if expr is None:
                continue
            ex = _linear_expr(expr, T)
            if ex is None:
                continue
            coefs, const = ex
            free = [(u, Fraction(c)) for u, c in coefs.items()
                    if u not in eq_subst and u in int_vars]
            if len(free) != 1:
                continue
            u, c = free[0]
            if c.denominator == 1:
                continue
            q, p = c.denominator, c.numerator
            qk = q * const
            if qk.denominator != 1 or math.gcd(p % q, q) != 1:
                continue
            res = ((-int(qk)) * pow(p % q, -1, q)) % q
            lst = cong.setdefault(u, [])
            if (res, q) not in lst:
                lst.append((res, q))
                changed = True
        # congruence closure: propagate known residues through linear relations
        # (e.g. n = 3q + r with n ≡ 0 (mod 3) ⟹ r ≡ 0 (mod 3)).
        rels = []
        for rec in eq_prem:
            d = T.nf(T.csub(rec[1], rec[2]))
            if d is not None and d and all(len(mn) <= 1 for mn in d):
                rels.append(d)
        moduli = sorted({M for lst in cong.values() for (_r, M) in lst})

        for d in rels:
            for u in {v for mn in d for v in mn}:
                cu = _int(d.get((u,), Fraction(0)))
                if cu is None or cu == 0:
                    continue
                for m in moduli:
                    cum = cu % m
                    if math.gcd(cum, m) != 1:
                        continue
                    total, known = 0, True
                    for mn, coef in d.items():
                        if mn == (u,):
                            continue
                        ci = _int(coef)
                        if ci is None:
                            known = False
                            break
                        if not mn:
                            total = (total + ci) % m
                            continue
                        if len(mn) != 1 or ci % m == 0:
                            continue
                        expr = eq_subst.get(mn[0])
                        cv = _num(expr) if expr is not None else None
                        if cv is not None:
                            rv = _int(cv) % m
                        else:
                            rv = _mod_residue(("v", mn[0]), m, cong)
                        if rv is None:
                            known = False
                            break
                        total = (total + ci * rv) % m
                    if not known:
                        continue
                    res = (-total) * pow(cum, -1, m) % m
                    lst = cong.setdefault(u, [])
                    if (res, m) not in lst:
                        lst.append((res, m))
                        changed = True
        # pin by congruence + box (and neq pruning)
        for u in list(cong):
            if u in eq_subst and _num(eq_subst[u]) is not None:
                continue
            ib = _int_box(bnd.get(u))
            if ib is None:
                continue
            lo, hi = ib
            if hi < lo or hi - lo > 200000:
                continue
            cand, unknown = [], False
            for x in range(lo, hi + 1):
                if not all((x - res) % m == 0 for res, m in cong[u]):
                    continue
                bad = False
                for _cl, nl, nr in neqs:
                    s = dict(eq_subst)
                    s[u] = T.const(x)
                    a2, b2 = _subst(nl, s), _subst(nr, s)
                    if T.vars_of(a2) or T.vars_of(b2):
                        unknown = True
                        break
                    try:
                        if T.ev(a2, {}) == T.ev(b2, {}):
                            bad = True
                            break
                    except Exception:
                        unknown = True
                        break
                if unknown:
                    break
                if not bad:
                    cand.append(x)
            if not unknown and len(cand) == 1 and u not in eq_subst:
                eq_subst[u] = T.const(cand[0])
                changed = True
        # pin a singleton integer box
        for u, b in list(bnd.items()):
            if u in eq_subst and _num(eq_subst[u]) is not None:
                continue
            ib = _int_box(b)
            if ib and ib[0] == ib[1]:
                eq_subst[u] = T.const(ib[0])
                changed = True
        # re-solve the linear relations against the current pins/substitutions
        rws = []
        for rec in eq_prem:
            d = T.nf(T.csub(_subst(rec[1], eq_subst), _subst(rec[2], eq_subst)))
            if d is None or not d:
                continue
            if all(not vs for vs in d) and d.get((), 0) != 0:
                continue
            row = _linear_row(d)
            if row is not None:
                rws.append(row)
        if rws:
            sol, consistent = _gauss_solve(rws)
            if consistent:
                for v, val in sol.items():
                    term = _sol_term(val, T)
                    if v in eq_subst and _num(eq_subst[v]) is not None:
                        continue          # keep an already-pinned constant
                    if v not in eq_subst or T.key(eq_subst[v]) != T.key(term):
                        eq_subst[v] = term
                        changed = True
        return changed

    for _round in range(6):
        if not _pin_round():
            break

    lp_cons = []
    for _cl, l, r, op in ineqs:
        c = _to_constraint(_subst(l, eq_subst), _subst(r, eq_subst), op, T)
        if c is not None:
            lp_cons.append(c)

    lower = {v: b[0] for v, b in bnd.items() if b[0] is not None}
    prem_list = ([("eq", rec[1], rec[2], "=") for rec in eq_prem]
                 + [("ineq", l, r, op) for _cl, l, r, op in ineqs])

    # ---- pass 4: discharge every conclusion ---------------------------------
    for cl in claims:
        if cl.is_premise:
            continue
        if cl.kind == "eq":
            raw = _mod_discharge(cl.lhs, cl.rhs, cong)
            why = "mod-residue"
            if raw is None:
                raw = _bounded_eq(cl.lhs, cl.rhs, eq_subst, cong, bnd)
                why = "bounded-congruence"
            if raw is None:
                raw = _bounded_premise_eq(cl.lhs, cl.rhs, prem_list, eq_subst,
                                          bnd, T)
                why = "bounded-premise"
            if raw is None:
                raw = _poly_reduce(cl.lhs, cl.rhs, poly_prem, mprem, eq_subst,
                                   bnd, T)
                why = "poly-reduce"
            if raw is None:
                raw = _quad_concl(cl.lhs, cl.rhs, quad_prem, eq_subst, bnd, T)
                why = "quadratic-root"
            if raw is None:
                raw = _pinned_eval(cl.lhs, cl.rhs, eq_subst, bnd, T)
                why = "pinned-domain"
            if raw is None:
                l, r = _subst(cl.lhs, eq_subst), _subst(cl.rhs, eq_subst)
                raw = T.decide_eq(l, r)
                why = "decide_eq"
                if raw is False:
                    d = T.nf(T.csub(l, r))
                    if partial:
                        raw, why = None, "assumed_premise"
                    elif d is None or any(not v.isidentifier()
                                          for mon in d for v in mon):
                        raw, why = None, "opaque_atom"
                    elif any(mon for mon in d):      # free variables remain
                        raw, why = None, "nonconstant_residual"
            if raw is False and partial and why not in (
                    "mod-residue", "bounded-congruence", "bounded-premise",
                    "quadratic-root", "pinned-domain", "poly-reduce"):
                raw, why = None, "assumed_premise"
        else:
            l, r = _subst(cl.lhs, eq_subst), _subst(cl.rhs, eq_subst)
            sl = _local_defs(l, eq_prem, eq_subst, T)
            sr = _local_defs(r, eq_prem, eq_subst, T)
            raw = _quad_sign_discharge(cl.op, sl, sr, T)
            why = "quad-sign"
            if raw is None:
                raw = _sign_over_domain(cl.op, sl, sr, bnd, T)
                why = "poly-sign"
            if raw is None:
                raw = _lp_holds(cl.op, sl, sr, lp_cons, T)
                why = "lp"
            if raw is None:
                raw = _ineq_sign_discharge(cl.op, sl, sr, lp_cons, bnd, T, neqs)
                why = "sign-product"
            if raw is None:
                raw = _interval_discharge(cl.op, sl, sr, bnd, decide_rel, INF)
                why = "interval"
            if raw is None:
                raw = _linear_bound_discharge(cl.op, sl, sr, bnd, T)
                why = "linear-bound"
            if raw is None:
                raw = _rel_prove(cl.op, l, r, lower, T)
                why = "sign_analysis"
            if raw is False and partial:
                raw, why = None, "assumed_premise"
        verdict = {True: "CERTIFIED", False: "REFUTED", None: "UNVERIFIABLE"}[raw]
        _register(cl, verdict, why)

    trace = TR.finish()
    concl = [v for c, v, _ in results if not c.is_premise]
    if contradiction or "REFUTED" in concl:
        unit = "REFUTED"
    elif "UNVERIFIABLE" in concl:
        unit = "UNVERIFIABLE"
    else:
        unit = "CERTIFIED"
    return unit, results, trace


_FLIP = {">=": "<=", ">": "<", "<=": ">=", "<": ">"}


def _int(v):
    """Exact integer value of a constant (int, or Fraction with denominator 1)."""
    if isinstance(v, int):
        return v
    if isinstance(v, Fraction) and v.denominator == 1:
        return v.numerator
    return None


def _int_box(b):
    """[lo, hi, lo_strict, hi_strict] -> inclusive integer range, or None."""
    if not b or b[0] is None or b[1] is None:
        return None
    lo = math.floor(b[0]) + 1 if b[2] else math.ceil(b[0])
    hi = math.ceil(b[1]) - 1 if b[3] else math.floor(b[1])
    return int(lo), int(hi)


def _num(t):
    if t[0] == "c" and isinstance(t[1], (int, Fraction, float)):
        return t[1]
    return None


def _linear_expr(t, T):
    """nf of a term as ({var: coef}, const), or None if nonlinear/opaque."""
    d = T.nf(t)
    if d is None:
        return None
    coefs, const = {}, Fraction(0)
    for mon, coef in d.items():
        if not mon:
            const += Fraction(coef)
            continue
        cnt = {}
        for v in mon:
            cnt[v] = cnt.get(v, 0) + 1
        if list(cnt.values()) != [1]:
            return None
        v = next(iter(cnt))
        if not v.isidentifier():
            return None
        coefs[v] = coefs.get(v, Fraction(0)) + Fraction(coef)
    return coefs, const


def _mod_premise(l, r, T):
    """A hypothesis of the form `expr % m == r` (m,r integer constants).
    Returns ('cong', (var, residue, modulus)) when it pins a variable's residue,
    ('contradiction', None) when it is unsatisfiable, ('trivial', None) when it
    is a constant identity, or None when not this shape."""
    if l[0] != "mod" or l[2][0] != "c":
        return None
    m = _int(l[2][1])
    if m is None or m <= 0:
        return None
    rr = _int(r[1]) if r[0] == "c" else None
    if rr is None:
        return None
    ex = _linear_expr(l[1], T)
    if ex is None:
        return None
    coefs, const = ex
    if not coefs:
        return ("trivial" if (_int(const) is not None
                              and (_int(const) - rr) % m == 0)
                else "contradiction", None)
    if len(coefs) != 1:
        return None
    v, c = next(iter(coefs.items()))
    c = _int(c)
    ci = _int(const)
    if c is None or ci is None:
        return None
    g = math.gcd(abs(c), m)
    k = rr - ci
    if k % g != 0:
        return ("contradiction", None)
    m2 = m // g
    c2 = (c // g) % m2
    res = ((k // g) * pow(c2, -1, m2)) % m2
    return ("cong", (v, res, m2))


def _univariate(d):
    """({..}: univariate poly) -> (var, a, b, c) for degree <= 2, else None."""
    var = None
    a = b = c = Fraction(0)
    for mon, coef in d.items():
        if not mon:
            c += Fraction(coef)
            continue
        cnt = {}
        for v in mon:
            cnt[v] = cnt.get(v, 0) + 1
        if len(cnt) != 1:
            return None
        v, e = next(iter(cnt.items()))
        if e > 2:
            return None
        if var is None:
            var = v
        elif var != v:
            return None
        if e == 2:
            a += Fraction(coef)
        else:
            b += Fraction(coef)
    if var is None:
        return None
    return var, a, b, c


def _bounded_eq(l, r, eq_subst, cong, bnd, cap=200000):
    """Conclusion `v = c` where v is a free variable pinned to a UNIQUE integer
    by its congruence facts within a finite box. Sound: the congruence+box
    candidate set is necessary; a single candidate decides the value."""
    if l[0] == "v" and l[1].isidentifier() and _num(r) is not None:
        v, c = l[1], _num(r)
    elif r[0] == "v" and r[1].isidentifier() and _num(l) is not None:
        v, c = r[1], _num(l)
    else:
        return None
    ci = _int(c)
    if ci is None or v in eq_subst:
        return None
    b = bnd.get(v)
    cons = cong.get(v)
    if not b or not cons:
        return None
    ib = _int_box(b)
    if ib is None:
        return None
    lo, hi = ib
    if hi < lo or hi - lo > cap:
        return None
    cand = [x for x in range(lo, hi + 1)
            if all((x - res) % m == 0 for res, m in cons)]
    if len(cand) != 1:
        return None
    return cand[0] == ci


def _bounded_premise_eq(l, r, prem_list, eq_subst, bnd, T, cap=200000):
    """Conclusion `v = c` with v a free integer pinned by EVALUATING every
    premise at each candidate in v's finite box. Sound: candidate set from a
    subset of the premises is a superset of the true models, so a singleton
    determines v and its exclusion refutes."""
    if l[0] == "v" and l[1].isidentifier() and _num(r) is not None:
        v, c = l[1], _num(r)
    elif r[0] == "v" and r[1].isidentifier() and _num(l) is not None:
        v, c = r[1], _num(l)
    else:
        return None
    ci = _int(c)
    if ci is None or v in eq_subst:
        return None
    ib = _int_box(bnd.get(v))
    if ib is None:
        return None
    lo, hi = ib
    if hi < lo or hi - lo > cap:
        return None

    def holds(kind, pl, pr, op, x):
        s = dict(eq_subst)
        s[v] = T.const(x)
        l2, r2 = _subst(pl, s), _subst(pr, s)
        if T.vars_of(l2) or T.vars_of(r2):
            return None                      # not unary in v
        try:
            a, bb = T.ev(l2, {}), T.ev(r2, {})
        except Exception:
            return None
        if kind == "eq":
            return a == bb
        return {">=": a >= bb, ">": a > bb, "<=": a <= bb, "<": a < bb}[op]

    cand = []
    for x in range(lo, hi + 1):
        ok = True
        for kind, pl, pr, op in prem_list:
            h = holds(kind, pl, pr, op, x)
            if h is None:
                return None                  # unevaluable: never guess
            if not h:
                ok = False
                break
        if ok:
            cand.append(x)
    if len(cand) == 1:
        return cand[0] == ci
    if cand and ci not in cand:
        return False
    return None


def _quad_sign_discharge(op, l, r, T):
    """Univariate quadratic sign over ℝ. d is the difference oriented so the
    conclusion is `d ≥ 0` (or `d > 0` when strict); a>0 decides it."""
    if op in ("<", "<="):
        d, strict = T.nf(T.csub(r, l)), op == "<"
    elif op in (">", ">="):
        d, strict = T.nf(T.csub(l, r)), op == ">"
    else:
        return None
    if d is None:
        return None
    u = _univariate(d)
    if u is None or u[1] == 0:
        return None
    _v, a, b, c = u
    ai, bi, ci = _int(a), _int(b), _int(c)
    if ai is None or bi is None or ci is None:
        return None
    disc = bi * bi - 4 * ai * ci
    if ai > 0 and disc < 0:
        return True                      # d > 0 everywhere
    if ai > 0 and disc == 0:
        return True if not strict else None   # d ≥ 0, touches 0
    return None


def _crt(pairs):
    """Combine congruences (res_i, mod_i) by CRT into (res, lcm), or None if
    inconsistent."""
    if not pairs:
        return None
    r, M = 0, 1
    for res, Mi in pairs:
        g = math.gcd(M, Mi)
        if (res - r) % g != 0:
            return None
        lcm = M // g * Mi
        m2 = Mi // g
        t = (((res - r) // g) * pow((M // g) % m2, -1, m2)) % m2
        r = (r + M * t) % lcm
        M = lcm
    return r, M


def _frac_sqrt(q):
    """Exact rational square root, or None when it is irrational."""
    if q < 0:
        return None
    sn, sd = math.isqrt(q.numerator), math.isqrt(q.denominator)
    if sn * sn == q.numerator and sd * sd == q.denominator:
        return Fraction(sn, sd)
    return None


def _eval_poly(d, env):
    total = Fraction(0)
    for mon, coef in d.items():
        term = Fraction(coef)
        for v in mon:
            term *= env[v]
        total += term
    return total


def _quad_concl(l, r, quad_prem, eq_subst, bnd, T):
    """When the concluded difference reduces to a polynomial in ONE free
    variable whose values are pinned to the roots of a univariate hypothesis,
    decide by evaluating the difference on those roots."""
    l2, r2 = _subst(l, eq_subst), _subst(r, eq_subst)
    d = T.nf(T.csub(l2, r2))
    if d is None:
        return None
    free = {v for mon in d for v in mon}
    if len(free) != 1:
        return None
    v = next(iter(free))
    roots = None
    for pv, pd in quad_prem:
        if pv != v:
            continue
        u = _univariate(pd)
        if u is None:
            continue
        ai, bi, cc = Fraction(u[1]), Fraction(u[2]), Fraction(u[3])
        if ai == 0:
            continue
        disc = bi * bi - 4 * ai * cc
        if disc < 0:
            return None
        s = _frac_sqrt(disc)
        if s is None:
            return None
        roots = sorted({(-bi + s) / (2 * ai), (-bi - s) / (2 * ai)})
        break
    if roots is None:
        return None
    b = bnd.get(v, [None, None, False, False])
    lo = None if b[0] is None else Fraction(b[0])
    hi = None if b[1] is None else Fraction(b[1])
    cand = [rt for rt in roots
            if (lo is None or rt > lo or (rt == lo and not b[2]))
            and (hi is None or rt < hi or (rt == hi and not b[3]))]
    if not cand:
        return None
    vals = [_eval_poly(d, {v: rt}) for rt in cand]
    if all(x == 0 for x in vals):
        return True
    if all(x != 0 for x in vals):
        return False
    return None


def _poly_var(d):
    """The single variable name spanning all monomials of d, or None."""
    var = None
    for mon in d:
        for v in mon:
            if var is None:
                var = v
            elif v != var:
                return None
    return var


def _poly_from_nf(d, var):
    """Univariate nf dict -> dense coefficient list [c0, c1, ...], or None."""
    c, deg = {}, 0
    for mon, coef in d.items():
        if any(v != var for v in mon):
            return None
        e = len(mon)
        c[e] = c.get(e, Fraction(0)) + Fraction(coef)
        deg = max(deg, e)
    return [c.get(i, Fraction(0)) for i in range(deg + 1)]


def _poly_is_zero(a):
    return all(x == 0 for x in a)


def _poly_mod(a, b):
    """Polynomial remainder a mod b (dense, over ℚ)."""
    a = [Fraction(x) for x in a]
    b = [Fraction(x) for x in b]
    while b and b[-1] == 0:
        b.pop()
    if not b:
        raise ZeroDivisionError("poly mod by zero")
    while len(a) >= len(b):
        if a[-1] == 0:
            a.pop()
            continue
        f, sh = a[-1] / b[-1], len(a) - len(b)
        for i, co in enumerate(b):
            a[sh + i] -= f * co
        while a and a[-1] == 0:
            a.pop()
    return a or [Fraction(0)]


def _poly_gcd(a, b):
    """Monic gcd of two rational polynomials (dense)."""
    a, b = [Fraction(x) for x in a], [Fraction(x) for x in b]
    while b and not _poly_is_zero(b):
        a, b = b, _poly_mod(a, b)
    if not a or _poly_is_zero(a):
        return [Fraction(0)]
    lead = a[-1]
    return [x / lead for x in a]


def _in_domain(q, bnd, v):
    b = bnd.get(v)
    if not b:
        return True
    lo, hi, ls, hs = b
    if lo is not None and (q < lo or (q == lo and ls)):
        return False
    if hi is not None and (q > hi or (q == hi and hs)):
        return False
    return True


def _poly_refute_if_root(g, bnd, v):
    """A nonzero constant modulo g means the claim is false on every common
    root. Refute only when g has an EXACT rational root inside the domain."""
    deg = len(g) - 1
    if deg == 1:
        root = -g[0] / g[1]
        return False if _in_domain(root, bnd, v) else None
    if deg == 2:
        a, b, c = g[2], g[1], g[0]
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        s = _frac_sqrt(Fraction(disc))
        if s is None:
            return None
        roots = {(-b + s) / (2 * a), (-b - s) / (2 * a)}
        return False if any(_in_domain(r, bnd, v) for r in roots) else None
    return None


def _lm(P):
    """Leading monomial of an nf polynomial under graded-lex."""
    return max(P.keys(), key=lambda m: (len(m), m))


def _mono_divides(a, b):
    """Does monomial a divide monomial b (exponent-wise)?"""
    if len(a) > len(b):
        return False
    ca, cb = {}, {}
    for v in a:
        ca[v] = ca.get(v, 0) + 1
    for v in b:
        cb[v] = cb.get(v, 0) + 1
    return all(cb.get(v, 0) >= c for v, c in ca.items())


def _mono_quotient(b, a):
    """b / a (exponent-wise); only valid when a divides b."""
    cb, ca = {}, {}
    for v in b:
        cb[v] = cb.get(v, 0) + 1
    for v in a:
        ca[v] = ca.get(v, 0) + 1
    out = []
    for v, c in cb.items():
        out.extend([v] * (c - ca.get(v, 0)))
    return tuple(sorted(out))


def _mdivide(Q, Ps, cap=20000):
    """Multivariate division of Q by the polynomials Ps (graded-lex). Returns
    (remainder, ok); remainder == {} and ok means Q lies in the ideal — so Q
    vanishes on every common root of the premises."""
    Q = {m: c for m, c in Q.items() if c != 0}
    rem = {}
    steps = 0
    while Q:
        m = _lm(Q)
        c = Q[m]
        reduced = False
        for P in Ps:
            if not P:
                continue
            lm, lc = _lm(P), P[_lm(P)]
            if _mono_divides(lm, m):
                q = c / lc
                t = _mono_quotient(m, lm)
                for pm, pc in P.items():
                    nm = tuple(sorted(t + pm))
                    Q[nm] = Q.get(nm, 0) - q * pc
                    if Q[nm] == 0:
                        del Q[nm]
                reduced = True
                steps += 1
                break
        if not reduced:
            rem[m] = c
            del Q[m]
        if steps > cap:
            return None, False
    return rem, True


def _poly_reduce(l, r, poly_prem, mprem, eq_subst, bnd, T):
    """Nonlinear elimination. The concluded difference is reduced (a) modulo the
    multivariate premise ideal — zero remainder ⟹ the conclusion holds on the
    common variety — and (b) univariately by polynomial gcd, which also gives a
    sound refutation when the remainder is a nonzero constant over a real root."""
    l2, r2 = _subst(l, eq_subst), _subst(r, eq_subst)
    d = T.nf(T.csub(l2, r2))
    if d is None:
        return None
    if not d:
        return True
    rem, ok = _mdivide(d, mprem)
    if ok and not rem:
        return True
    # Gröbner completion: division by the generators can miss ideal membership
    # (the premises are not a Gröbner basis); complete them and re-divide.
    if 1 <= len(mprem) <= 8 and sum(len(P) for P in mprem) <= 80:
        G = _grobner(mprem, cap_pairs=1000, cap_steps=100000)
        if G:
            rem2, ok2 = _mdivide(d, G, cap=40000)
            if ok2 and not rem2:
                return True
    v = _poly_var(d)
    if v is None:
        return None
    da = _poly_from_nf(d, v)
    if da is None:
        return None
    basis = [p for pv, p in poly_prem if pv == v]
    if not basis:
        return None
    g = basis[0]
    for p in basis[1:]:
        g = _poly_gcd(g, p)
        if len(g) <= 1:
            return None                     # no common nonconstant factor
    while len(g) > 1 and g[-1] == 0:
        g.pop()
    if len(g) <= 1:
        return None
    rem = _poly_mod(da, g)
    if _poly_is_zero(rem):
        return True
    if len(rem) == 1:
        return _poly_refute_if_root(g, bnd, v)
    return None


def _p_add(P, Q):
    R = dict(P)
    for m, c in Q.items():
        R[m] = R.get(m, 0) + c
        if R[m] == 0:
            del R[m]
    return R


def _p_sub(P, Q):
    R = dict(P)
    for m, c in Q.items():
        R[m] = R.get(m, 0) - c
        if R[m] == 0:
            del R[m]
    return R


def _p_mul_mono(P, mon, coef):
    if not P or coef == 0:
        return {}
    R = {}
    for m, c in P.items():
        nm = tuple(sorted(m + mon))
        R[nm] = R.get(nm, 0) + c * coef
    return {m: c for m, c in R.items() if c != 0}


def _mono_lcm(a, b):
    ca, cb = {}, {}
    for v in a:
        ca[v] = ca.get(v, 0) + 1
    for v in b:
        cb[v] = cb.get(v, 0) + 1
    out = []
    for v in set(ca) | set(cb):
        out.extend([v] * max(ca.get(v, 0), cb.get(v, 0)))
    return tuple(sorted(out))


def _spoly(f, g):
    lf, lg = _lm(f), _lm(g)
    l = _mono_lcm(lf, lg)
    a = _p_mul_mono(f, _mono_quotient(l, lf), 1 / f[lf])
    b = _p_mul_mono(g, _mono_quotient(l, lg), 1 / g[lg])
    return _p_sub(a, b)


def _grobner(F, cap_pairs=200, cap_steps=20000):
    """Buchberger completion of the ideal generated by F over ℚ (graded-lex).
    Returns a Gröbner basis, or None when the bounded search is exhausted."""
    G = [{m: c for m, c in P.items() if c != 0} for P in F]
    G = [P for P in G if P]
    pairs = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]
    seen = 0
    while pairs and seen < cap_pairs:
        i, j = pairs.pop(0)
        seen += 1
        s = _spoly(G[i], G[j])
        if not s:
            continue
        rem, ok = _mdivide(s, G, cap=cap_steps)
        if not ok:
            return None
        if rem:
            G.append(rem)
            n = len(G) - 1
            pairs.extend((k, n) for k in range(n))
    return G


def _linear_row(d):
    """Turn nf(l-r) into one exact linear row ({var: coef}, rhs), or None when
    the equation is nonlinear / opaque (then the hypothesis is only 'assumed')."""
    row, const = {}, Fraction(0)
    for mon, coef in d.items():
        if not mon:
            const += Fraction(coef)
            continue
        cnt = {}
        for v in mon:
            cnt[v] = cnt.get(v, 0) + 1
        if list(cnt.values()) != [1]:
            return None
        v = next(iter(cnt))
        if not v.isidentifier():
            return None
        row[v] = row.get(v, Fraction(0)) + Fraction(coef)
    return (row, -const)


def _linear_bound_discharge(op, l, r, bnd, T):
    """Decide a relation whose difference is LINEAR by its infimum over the
    declared box. Returns True/None (never False): a strict inequality holds
    when the infimum is 0 and some contributing bound is strict."""
    if op in ("<", "<="):
        d, strict = T.nf(T.csub(r, l)), op == "<"
    elif op in (">", ">="):
        d, strict = T.nf(T.csub(l, r)), op == ">"
    else:
        return None
    if d is None:
        return None
    inf = Fraction(0)
    inf_strict = False
    for mon, coef in d.items():
        if not mon:
            inf += Fraction(coef)
            continue
        if len(mon) != 1:
            return None
        v = mon[0]
        b = bnd.get(v)
        if not b:
            return None
        c = Fraction(coef)
        if c > 0:
            if b[0] is None:
                return None
            inf += c * Fraction(b[0])
            inf_strict = inf_strict or b[2]
        else:
            if b[1] is None:
                return None
            inf += c * Fraction(b[1])
            inf_strict = inf_strict or b[3]
    if strict:
        if inf > 0 or (inf == 0 and inf_strict):
            return True
        return None
    return True if inf >= 0 else None


def _propagate_bounds(eq_subst, bnd, T):
    """A bound on a variable that has been substituted becomes a bound on the
    free variables of its solved expression (e.g. `h >= 0` with `h = -9b`
    gives `b <= 0`). Iterated to a fixpoint."""
    def _lo(u, t, strict):
        cur = bnd.setdefault(u, [None, None, False, False])
        if cur[0] is None or t > cur[0]:
            cur[0], cur[2] = t, strict
            return True
        if t == cur[0] and strict and not cur[2]:
            cur[2] = True
            return True
        return False

    def _hi(u, t, strict):
        cur = bnd.setdefault(u, [None, None, False, False])
        if cur[1] is None or t < cur[1]:
            cur[1], cur[3] = t, strict
            return True
        if t == cur[1] and strict and not cur[3]:
            cur[3] = True
            return True
        return False

    for _ in range(len(eq_subst) + 3):
        changed = False
        for v, expr in list(eq_subst.items()):
            if v not in bnd:
                continue
            ex = _linear_expr(expr, T)
            if ex is None:
                continue
            coefs, const = ex
            free = [(u, c) for u, c in coefs.items() if u not in eq_subst]
            if len(free) != 1:
                continue
            u, c = free[0]
            if c == 0:
                continue
            lo, hi, ls, hs = bnd[v]
            if lo is not None:
                t = (Fraction(lo) - const) / c
                changed |= _lo(u, t, ls) if c > 0 else _hi(u, t, ls)
            if hi is not None:
                t = (Fraction(hi) - const) / c
                changed |= _hi(u, t, hs) if c > 0 else _lo(u, t, hs)
        if not changed:
            break


def _pinned_eval(l, r, eq_subst, bnd, T):
    """If every free variable of the concluded difference is pinned to a single
    point by its bounds, evaluate it there. Sound: the domain is a singleton."""
    l2, r2 = _subst(l, eq_subst), _subst(r, eq_subst)
    d = T.nf(T.csub(l2, r2))
    if d is None:
        return None
    free = {v for mon in d for v in mon}
    if not free:
        return None
    env = {}
    for v in free:
        b = bnd.get(v)
        if not b or b[0] is None or b[1] is None or b[2] or b[3] or b[0] != b[1]:
            return None
        q = _int(b[0])
        if q is None:
            return None
        env[v] = q
    total = Fraction(0)
    for mon, coef in d.items():
        term = Fraction(coef)
        for v in mon:
            term *= env[v]
        total += term
    return total == 0


def _sol_term(val, T):
    """Rebuild a substitution term from a Gauss solution value (a Fraction, or
    {'const': Fraction, 'coefs': {var: Fraction}})."""
    if isinstance(val, Fraction):
        return T.const(val)
    term = T.const(val["const"])
    for vv, cf in val["coefs"].items():
        term = T.cadd(term, T.cmul(T.const(cf), T.var(vv)))
    return term


def _gauss_solve(rows):
    """Exact RREF over Fractions. Returns ({var: Fraction} determined values,
    consistent). Determined = pivot columns whose row has no remaining vars."""
    if not rows:
        return {}, True
    vars_ = sorted({v for row, _ in rows for v in row})
    mat = [([Fraction(row.get(v, 0)) for v in vars_], Fraction(rhs))
           for row, rhs in rows]
    pivot_cols, r = [], 0
    for col in range(len(vars_)):
        piv = next((i for i in range(r, len(mat)) if mat[i][0][col] != 0), None)
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        a = mat[r][0][col]
        mat[r] = ([x / a for x in mat[r][0]], mat[r][1] / a)
        for i in range(len(mat)):
            if i != r and mat[i][0][col] != 0:
                f = mat[i][0][col]
                mat[i] = ([x - f * y for x, y in zip(mat[i][0], mat[r][0])],
                          mat[i][1] - f * mat[r][1])
        pivot_cols.append(col)
        r += 1
        if r == len(mat):
            break
    for row, rhs in mat[r:]:
        if all(x == 0 for x in row) and rhs != 0:
            return {}, False                          # inconsistent system
    sol = {}
    for i, pc in enumerate(pivot_cols):
        row, rhs = mat[i]
        coef = row[pc]
        if coef == 0:
            continue
        expr = {vars_[j]: -row[j] / coef
                for j in range(len(vars_)) if j != pc and row[j] != 0}
        if expr:                      # RREF: pivot var = f(free vars)
            sol[vars_[pc]] = {"const": rhs / coef, "coefs": expr}
        else:
            sol[vars_[pc]] = rhs / coef
    return sol, True


def _mod_residue(t, m, cong):
    """Residue of an integer polynomial mod m, from the congruence facts; None
    when some variable has no residue mod m (abstain, never guess)."""
    from . import term as T
    k = t[0]
    if k not in ("c", "v", "mod"):
        d = T.nf(t)
        if d is not None and any(mon for mon in d):
            # polynomial fast path: coefficients that vanish mod m drop out, so
            # `9A - 9B ≡ 0 (mod 3)` needs no residue for A or B.
            total, unknown = 0, False
            for mon, coef in d.items():
                ci = _int(coef)
                if ci is None:
                    unknown = True
                    break
                c = ci % m
                if c == 0:
                    continue
                cnt = {}
                for v in mon:
                    cnt[v] = cnt.get(v, 0) + 1
                for v, e in cnt.items():
                    rv = _mod_residue(("v", v), m, cong)
                    if rv is None:
                        unknown = True
                        break
                    c = c * rv ** e % m
                if unknown:
                    break
                total = (total + c) % m
            if not unknown:
                return total
    if k == "c":
        i = _int(t[1])
        return None if i is None else i % m
    if k == "v":
        combined = _crt(cong.get(t[1], []))
        if combined is None:
            return None
        res, M = combined
        if M == m:
            return res
        if M % m == 0:                # m | M ⟹ X mod m = res mod m
            return res % m
        return None
    if k in ("add", "sub", "mul"):
        a = _mod_residue(t[1], m, cong)
        b = _mod_residue(t[2], m, cong)
        if k == "mul" and (a == 0 or b == 0):
            return 0                     # 0 * anything ≡ 0 (mod m)
        if a is None or b is None:
            return None
        if k == "add":
            return (a + b) % m
        if k == "sub":
            return (a - b) % m
        return (a * b) % m
    if k == "neg":
        a = _mod_residue(t[1], m, cong)
        return None if a is None else (-a) % m
    if k == "mod":
        if t[2][0] == "c":
            d = _int(t[2][1])
            if d and d > 0:
                a = _mod_residue(t[1], m, cong)
                if a is not None and m % d == 0:
                    return a % d          # m known and d | m ⟹ X % d determined
                if a is not None and d % m == 0:
                    return a              # outer modulus multiple of m
        return None
    return None


def _mod_expr_value(t, cong):
    """Exact integer value of an expression made of constants and `% k` nodes
    whose operands' residues are known (e.g. `n%3 + n%5` given `n%15 = 7`).
    None when a value is not pinned (a bare variable)."""
    k = t[0]
    if k == "c":
        return _int(t[1])
    if k == "mod":
        kk = _int(t[2][1]) if t[2][0] == "c" else None
        if kk and kk > 0:
            return _mod_residue(t[1], kk, cong)      # value in [0, kk)
        return None
    if k in ("add", "sub", "mul"):
        a, b = _mod_expr_value(t[1], cong), _mod_expr_value(t[2], cong)
        if a is None or b is None:
            return None
        return a + b if k == "add" else (a - b if k == "sub" else a * b)
    if k == "neg":
        a = _mod_expr_value(t[1], cong)
        return None if a is None else -a
    return None


def _mod_discharge(l, r, cong):
    """E-mod: discharge `expr % m OP const` by exact residue constancy. Returns
    True/False/None. A determined residue that mismatches is a sound REFUTED."""
    if r[0] == "c" and _int(r[1]) is not None and l[0] == "mod" \
            and l[2][0] == "c" and _int(l[2][1]) is not None and _int(l[2][1]) > 0:
        m, rr = _int(l[2][1]), _int(r[1])
        res = _mod_residue(l[1], m, cong)
        if res is None:
            return None
        if not (0 <= rr < m):
            return False
        return res == rr
    # Swap to bring a `mod` to the left, but ONLY when that makes progress: if
    # both sides are integer constants the swap is a no-op and the previous
    # version recursed forever (`dispose_claims([eq(const(300), const(300))])` ->
    # RecursionError). Fall through to the exact constant comparison below.
    if l[0] == "c" and _int(l[1]) is not None and not (
            r[0] == "c" and _int(r[1]) is not None):
        return _mod_discharge(r, l, cong)
    # whole-expression equality pinned by residues (e.g. `n%3 + n%5 = 3`)
    lv, rv = _mod_expr_value(l, cong), _mod_expr_value(r, cong)
    if lv is not None and rv is not None:
        return lv == rv
    return None


def _local_defs(t, eq_prem, eq_subst, T):
    """Substitute pending `v = P(...)` definitions locally (used only where a
    nonlinear definition helps a sign conclusion, not globally)."""
    extra = {}
    for rec in eq_prem:
        if rec[3] is not None:
            continue
        for a, b in ((rec[1], rec[2]), (rec[2], rec[1])):
            if a[0] == "v" and a[1].isidentifier() and a[1] not in eq_subst \
                    and a[1] not in extra:
                if T.nf(b) is not None and a[1] not in T.vars_of(b):
                    extra[a[1]] = b
    return _subst(t, extra) if extra else t


def _sign_over_domain(op, l, r, bnd, T):
    """Sign of a univariate polynomial over the declared box. If it has no real
    root inside the box the sign is constant, and a domain sample decides it."""
    if op in ("<", "<="):
        d, strict = T.nf(T.csub(r, l)), op == "<"
    elif op in (">", ">="):
        d, strict = T.nf(T.csub(l, r)), op == ">"
    else:
        return None
    if d is None:
        return None
    v = _poly_var(d)
    if v is None:
        return None
    c = _poly_from_nf(d, v)
    if c is None:
        return None
    deg = len(c) - 1
    roots = []
    if deg == 1:
        roots = [-c[0] / c[1]]
    elif deg == 2:
        a, b, k = c[2], c[1], c[0]
        disc = b * b - 4 * a * k
        if disc >= 0:
            s = _frac_sqrt(Fraction(disc))
            if s is None:
                return None              # irrational roots: abstain
            roots = [(-b + s) / (2 * a), (-b - s) / (2 * a)]
    else:
        return None
    for rt in roots:
        if _in_domain(rt, bnd, v):
            return None                  # sign changes inside the box
    lo, hi, ls, hs = bnd.get(v, [None, None, False, False])
    cands = []
    if lo is not None:
        cands.append(lo + (1 if ls else 0))
    if hi is not None:
        cands.append(hi - (1 if hs else 0))
    if lo is not None and hi is not None:
        cands.append((lo + hi) / 2)
    cands.append(Fraction(0))
    sample = next((x for x in cands if _in_domain(x, bnd, v)), None)
    if sample is None:
        return None
    val = _eval_poly(d, {v: sample})
    return True if (val > 0 if strict else val >= 0) else None


_NEG_OP = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}


def _to_constraint(l, r, op, T):
    """A linear inequality as (coefs, rhs, strict) meaning Σ coefs·v ≤ rhs."""
    if op in ("<=", "<"):
        form = T.csub(l, r)
    elif op in (">=", ">"):
        form = T.csub(r, l)
    else:
        return None
    ex = _linear_expr(form, T)
    if ex is None:
        return None
    coefs, const = ex
    return ({k: v for k, v in coefs.items() if v != 0}, -const, op in ("<", ">"))


def _fm_infeasible(cons, cap=4000):
    """Fourier–Motzkin: is the system of linear inequalities unsatisfiable?
    Returns True (infeasible) / False (feasible) / None (budget exhausted)."""
    cons = list(cons)
    for x in sorted({v for c, _, _ in cons for v in c}):
        pos = [c for c in cons if c[0].get(x, 0) > 0]
        neg = [c for c in cons if c[0].get(x, 0) < 0]
        new = [c for c in cons if c[0].get(x, 0) == 0]
        for cp, rp, sp in pos:
            a = cp[x]
            for cn, rn, sn in neg:
                b = -cn[x]
                cfs = {}
                for k, val in cp.items():
                    if k != x:
                        cfs[k] = cfs.get(k, 0) + val * b
                for k, val in cn.items():
                    if k != x:
                        cfs[k] = cfs.get(k, 0) + val * a
                new.append(({k: v for k, v in cfs.items() if v != 0},
                            rp * b + rn * a, sp or sn))
                if len(new) > cap:
                    return None
        cons = new
    for cfs, rhs, st in cons:
        if not cfs and (rhs < 0 or (rhs == 0 and st)):
            return True
    return False


def _lp_holds(op, l, r, cons, T):
    """True when `l OP r` follows from the linear premises (negation is
    infeasible over ℚ). Never returns False."""
    neg = _NEG_OP.get(op)
    if neg is None or not cons:
        return None
    nc = _to_constraint(l, r, neg, T)
    if nc is None:
        return None
    return True if _fm_infeasible(cons + [nc]) is True else None


def _nonzero(t, cons, bnd, T, neqs=()):
    """True when `t` is provably nonzero."""
    if _lp_holds(">", t, T.const(0), cons, T) is True:
        return True
    if _lp_holds("<", t, T.const(0), cons, T) is True:
        return True
    if t[0] == "c":
        return True if t[1] != 0 else None
    if t[0] == "mul":
        if _nonzero(t[1], cons, bnd, T, neqs) is True and \
                _nonzero(t[2], cons, bnd, T, neqs) is True:
            return True
    if t[0] in ("div", "rdiv"):
        if _nonzero(t[1], cons, bnd, T, neqs) is True and \
                _nonzero(t[2], cons, bnd, T, neqs) is True:
            return True
    # a disequality premise `expr ≠ c` with c = 0 makes `expr` nonzero
    nt = T.nf(t)
    for _cl, nl, nr in neqs:
        ln, rn = T.nf(nl), T.nf(nr)
        if rn == {} and nt == ln:
            return True
        if ln == {} and nt == rn:
            return True
    return None


def _expr_sign(t, cons, bnd, T, neqs=()):
    """Sign of an expression built from products/quotients of sign-known
    factors: +1 / -1 / 0, or None when a factor's sign is unknown."""
    if t[0] == "c":
        v = t[1]
        return 0 if v == 0 else (1 if v > 0 else -1)
    if t[0] == "neg":
        s = _expr_sign(t[1], cons, bnd, T, neqs)
        return None if s is None else -s
    if t[0] == "mul":
        if t[1] == t[2]:                  # a square ⟹ ≥ 0, > 0 if a ≠ 0
            return 1 if _nonzero(t[1], cons, bnd, T, neqs) is True else None
        sa = _expr_sign(t[1], cons, bnd, T, neqs)
        sb = _expr_sign(t[2], cons, bnd, T, neqs)
        return None if sa is None or sb is None else sa * sb
    if t[0] in ("div", "rdiv"):
        if _nonzero(t[2], cons, bnd, T, neqs) is not True:
            return None
        sa = _expr_sign(t[1], cons, bnd, T, neqs)
        sb = _expr_sign(t[2], cons, bnd, T, neqs)
        return None if sa is None or sb is None else sa * sb
    # linear/polynomial factor: sign by LP
    if _lp_holds(">", t, T.const(0), cons, T) is True:
        return 1
    if _lp_holds("<", t, T.const(0), cons, T) is True:
        return -1
    d = T.nf(t)
    if d is not None and not d:
        return 0
    return None


def _ineq_sign_discharge(op, l, r, cons, bnd, T, neqs=()):
    """Prove a relation whose difference is a product/quotient of sign-known
    factors. Returns True/None (never False)."""
    if op in (">", ">="):
        s = _expr_sign(T.csub(l, r), cons, bnd, T, neqs)
    elif op in ("<", "<="):
        s = _expr_sign(T.csub(r, l), cons, bnd, T, neqs)
    else:
        return None
    if s is None:
        return None
    if op in (">", "<"):
        return True if s > 0 else None
    return True if s >= 0 else None


def _interval_discharge(op, l, r, bnd, decide_rel, INF):
    """E5: decide a relation over the declared box of the premises. Returns
    True (holds over the box) / False (fails in the box) / None (abstain).
    A 'False' with any strict bound is downgraded — the closed box is a
    superset and must not manufacture a witness."""
    opn = {">=": "ge", ">": "gt", "<=": "le", "<": "lt"}.get(op)
    if opn is None:
        return None
    env, strict = {}, False
    for v, (lo, hi, ls, hs) in bnd.items():
        env[v] = (-INF if lo is None else lo, INF if hi is None else hi)
        strict = strict or ls or hs
    try:
        v = decide_rel(l, r, opn, env)
    except Exception:
        return None
    if v is False and strict:
        return None
    return v


def _rel_prove(op, l, r, bounds, T):
    """Discharge one relation in the correct direction (never guess)."""
    if op == ">=":
        return T.provably_ge(l, r, bounds)
    if op == ">":
        return T.provably_gt(l, r, bounds)
    if op == "<=":
        return T.provably_ge(r, l, bounds)
    if op == "<":
        return T.provably_gt(r, l, bounds)
    return None


def _subst(t, subst, depth=0):
    """Apply substitutions to a term (recursive, depth-capped against cycles)."""
    from . import term as T
    if depth > 32:
        return t                            # substitution cycle: stop (sound)
    if isinstance(t, str):
        return t
    if t[0] == "v" and t[1] in subst:
        return _subst(subst[t[1]], subst, depth + 1)
    if t[0] == "c":
        return t
    args = [_subst(x, subst, depth + 1) for x in t[1:]]
    from . import term as _T
    B = {"add": _T.cadd, "sub": _T.csub, "mul": _T.cmul, "neg": _T.cneg,
         "max": _T.cmax, "min": _T.cmin, "abs": _T.cabs, "fact": _T.cfact,
         "div": _T.cdiv, "mod": _T.cmod, "rdiv": _T.crdiv}
    if t[0] in B:
        return B[t[0]](*args)
    return (t[0],) + tuple(args)
