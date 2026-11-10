"""mjolnir_prover.lib.leankernel — an INDEPENDENT checker for the Lean IR fragment.

This module deliberately shares NO code with the prover: it does not import
`term.py`, `claim_ir.py`, or `frontend_lean2`. It consumes the prover's claim
records as DATA (nested tuples) and re-derives a verdict with its own rational
polynomial algebra, its own Gaussian elimination, its own Buchberger completion,
its own modular arithmetic, its own Fourier–Motzkin, and its own sign analysis.

Purpose: the Lean golden ledger proves the prover is STABLE; a shared-code gate
cannot prove it is SOUND. This kernel is the independent false-positive detector:
it re-decides each CERTIFIED unit and flags any disagreement. It also supports
tamper tests (mutate a term -> the replayed verdict must change/reject).

It confirms the fragment's decidable core. Anything it cannot re-derive is
reported `unconfirmed` (a coverage note), never silently accepted.
"""
from __future__ import annotations
import math
from fractions import Fraction


# --------------------------------------------------------------- polynomials --
# A polynomial is a dict {monomial_tuple: Fraction}; monomial = sorted tuple of
# variable names.  Written from scratch (not term.nf) so the two share no code.
def p_add(a, b):
    r = dict(a)
    for m, c in b.items():
        r[m] = r.get(m, Fraction(0)) + c
        if r[m] == 0:
            del r[m]
    return r


def p_neg(a):
    return {m: -c for m, c in a.items()}


def p_sub(a, b):
    return p_add(a, p_neg(b))


def p_mul(a, b):
    r = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            m = tuple(sorted(ma + mb))
            r[m] = r.get(m, Fraction(0)) + ca * cb
            if r[m] == 0:
                del r[m]
    return r


def p_scale(a, s):
    return {m: c * s for m, c in a.items() if c * s != 0}


def p_lm(a):
    return max(a.keys(), key=lambda m: (len(m), m))


def p_mono_divides(a, b):
    ca, cb = {}, {}
    for v in a:
        ca[v] = ca.get(v, 0) + 1
    for v in b:
        cb[v] = cb.get(v, 0) + 1
    return all(cb.get(v, 0) >= c for v, c in ca.items())


def p_mono_quot(b, a):
    cb, ca = {}, {}
    for v in b:
        cb[v] = cb.get(v, 0) + 1
    for v in a:
        ca[v] = ca.get(v, 0) + 1
    out = []
    for v, c in cb.items():
        out.extend([v] * (c - ca.get(v, 0)))
    return tuple(sorted(out))


def p_divide(q, basis, cap=200000):
    q = {m: c for m, c in q.items() if c != 0}
    rem, steps = {}, 0
    while q:
        m = p_lm(q)
        c = q[m]
        red = False
        for P in basis:
            if not P:
                continue
            lm = p_lm(P)
            if p_mono_divides(lm, m):
                f = c / P[lm]
                t = p_mono_quot(m, lm)
                for pm, pc in P.items():
                    nm = tuple(sorted(t + pm))
                    q[nm] = q.get(nm, Fraction(0)) - f * pc
                    if q[nm] == 0:
                        del q[nm]
                red, steps = True, steps + 1
                break
        if not red:
            rem[m] = c
            del q[m]
        if steps > cap:
            return None
    return rem


def p_mono_lcm(a, b):
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
    lf, lg = p_lm(f), p_lm(g)
    l = p_mono_lcm(lf, lg)
    a = _pmulmono(f, p_mono_quot(l, lf), Fraction(1) / f[lf])
    b = _pmulmono(g, p_mono_quot(l, lg), Fraction(1) / g[lg])
    return p_sub(a, b)


def _pmulmono(P, mon, coef):
    if not P or coef == 0:
        return {}
    r = {}
    for m, c in P.items():
        nm = tuple(sorted(m + mon))
        r[nm] = r.get(nm, Fraction(0)) + c * coef
        if r[nm] == 0:
            del r[nm]
    return r


def p_grobner(F, cap_pairs=1000, cap_steps=200000):
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
        rem = p_divide(s, G, cap=cap_steps)
        if rem is None:
            return None
        if rem:
            G.append(rem)
            n = len(G) - 1
            pairs.extend((k, n) for k in range(n))
    return G


# --------------------------------------------------------- term interpretation --
def nf(t):
    """Independent normal form. Returns a poly dict, or None if not polynomial.
    Atoms that are not polynomial (clen/elem/fact/min/max/abs, variable mod/div)
    become opaque monomials keyed by their structural signature."""
    k = t[0]
    if k == "c":
        v = t[1]
        return {(): Fraction(v) if not isinstance(v, float) else Fraction(v)}
    if k == "v":
        return {(t[1],): Fraction(1)}
    if k == "neg":
        a = nf(t[1])
        return None if a is None else p_neg(a)
    if k == "add":
        a, b = nf(t[1]), nf(t[2])
        return None if a is None or b is None else p_add(a, b)
    if k == "sub":
        a, b = nf(t[1]), nf(t[2])
        return None if a is None or b is None else p_sub(a, b)
    if k == "mul":
        a, b = nf(t[1]), nf(t[2])
        return None if a is None or b is None else p_mul(a, b)
    if k in ("rdiv", "div"):
        a, b = nf(t[1]), nf(t[2])
        if b is not None and set(b.keys()) == {()} and b[()] != 0 and a is not None:
            return p_scale(a, Fraction(1) / Fraction(b[()]))
        return {(_sig(t),): Fraction(1)}
    if k in ("clen", "elem", "fact", "list", "min", "max", "abs", "mod"):
        return {(_sig(t),): Fraction(1)}
    return None


def _sig(t):
    """Canonical non-identifier signature for an opaque term."""
    return "\x00" + repr(t)


def vars_of(t):
    out = set()
    stack = [t]
    while stack:
        x = stack.pop()
        if not isinstance(x, tuple):
            continue
        if x[0] == "v":
            out.add(x[1])
        elif x[0] != "c":
            stack.extend(x[1:])
    return out


def subst(t, s):
    if isinstance(t, str):
        return t
    if t[0] == "v" and t[1] in s:
        return s[t[1]]
    if t[0] == "c":
        return t
    return (t[0],) + tuple(subst(x, s) for x in t[1:])


# ------------------------------------------------------------------- residues --
def _int(v):
    if isinstance(v, int):
        return v
    if isinstance(v, Fraction) and v.denominator == 1:
        return v.numerator
    return None


def crt(pairs):
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
        r, M = (r + M * t) % lcm, lcm
    return r, M


def mod_residue(t, m, cong):
    k = t[0]
    if k == "c":
        i = _int(t[1])
        return None if i is None else i % m
    if k == "v":
        combined = crt(cong.get(t[1], []))
        if combined is None:
            return None
        res, M = combined
        if M == m:
            return res
        if M % m == 0:
            return res % m
        return None
    if k in ("add", "sub", "mul"):
        a, b = mod_residue(t[1], m, cong), mod_residue(t[2], m, cong)
        if k == "mul" and (a == 0 or b == 0):
            return 0
        if a is None or b is None:
            return None
        return (a + b) % m if k == "add" else ((a - b) % m if k == "sub"
                                               else (a * b) % m)
    if k == "neg":
        a = mod_residue(t[1], m, cong)
        return None if a is None else (-a) % m
    if k == "mod":
        d = _int(t[2][1]) if t[2][0] == "c" else None
        if d and d > 0:
            a = mod_residue(t[1], m, cong)
            if a is not None and m % d == 0:
                return a % d
            if a is not None and d % m == 0:
                return a
        return None
    return None


# ------------------------------------------------------------------- linear ----
def lin_row(d):
    row, const = {}, Fraction(0)
    for mon, coef in d.items():
        if not mon:
            const += coef
            continue
        if len(set(mon)) != len(mon):
            return None
        v = mon[0]
        if any(u != v for u in mon) or not str(v).isidentifier():
            return None
        row[v] = row.get(v, Fraction(0)) + coef
    return (row, -const)


def gauss(rows):
    if not rows:
        return {}, True
    vs = sorted({v for r, _ in rows for v in r})
    mat = [([Fraction(r.get(v, 0)) for v in vs], Fraction(c)) for r, c in rows]
    piv, rr = [], 0
    for col in range(len(vs)):
        p = next((i for i in range(rr, len(mat)) if mat[i][0][col] != 0), None)
        if p is None:
            continue
        mat[rr], mat[p] = mat[p], mat[rr]
        a = mat[rr][0][col]
        mat[rr] = ([x / a for x in mat[rr][0]], mat[rr][1] / a)
        for i in range(len(mat)):
            if i != rr and mat[i][0][col] != 0:
                f = mat[i][0][col]
                mat[i] = ([x - f * y for x, y in zip(mat[i][0], mat[rr][0])],
                          mat[i][1] - f * mat[rr][1])
        piv.append(col)
        rr += 1
        if rr == len(mat):
            break
    for r, c in mat[rr:]:
        if all(x == 0 for x in r) and c != 0:
            return {}, False
    sol = {}
    for i, pc in enumerate(piv):
        row, c = mat[i]
        expr = {vs[j]: -row[j] / row[pc] for j in range(len(vs))
                if j != pc and row[j] != 0}
        sol[vs[pc]] = (c / row[pc], expr)
    return sol, True


# --------------------------------------------------------------- FM / sign -----
NEG_OP = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}


def to_constraint(l, r, op):
    form = (t_sub(l, r) if op in ("<=", "<") else t_sub(r, l))
    d = nf(form)
    if d is None:
        return None
    coefs = {m[0]: c for m, c in d.items() if m}
    const = d.get((), Fraction(0))
    if any(len(m) != 1 for m in d.keys()):
        return None
    return (coefs, -const, op in ("<", ">"))


def t_sub(a, b):
    return ("sub", a, b)


def fm_infeasible(cons, cap=8000):
    cons = list(cons)
    for x in sorted({v for c, _, _ in cons for v in c}):
        pos = [c for c in cons if c[0].get(x, 0) > 0]
        neg = [c for c in cons if c[0].get(x, 0) < 0]
        new = [c for c in cons if c[0].get(x, 0) == 0]
        for cp, rp, sp in pos:
            a = cp[x]
            for cn, rn, sn in neg:
                b = -cn[x]
                cf = {}
                for k, val in cp.items():
                    if k != x:
                        cf[k] = cf.get(k, 0) + val * b
                for k, val in cn.items():
                    if k != x:
                        cf[k] = cf.get(k, 0) + val * a
                new.append(({k: v for k, v in cf.items() if v != 0},
                            rp * b + rn * a, sp or sn))
                if len(new) > cap:
                    return None
        cons = new
    for cf, rhs, st in cons:
        if not cf and (rhs < 0 or (rhs == 0 and st)):
            return True
    return False


def real_roots(a, b, c):
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    sn, sd = math.isqrt(disc.numerator), math.isqrt(disc.denominator)
    if sn * sn != disc.numerator or sd * sd != disc.denominator:
        return None
    s = Fraction(sn, sd)
    return [(-b + s) / (2 * a), (-b - s) / (2 * a)]


# ------------------------------------------------------------- replay verdict --
def replay(premises, conclusion):
    """Re-derive the conclusion's verdict independently.
    premises/conclusion: (kind, op, lhs, rhs). Returns CERTIFIED / REFUTED /
    UNVERIFIABLE / UNCONFIRMED."""
    cong = {}
    eqp, ineqp = [], []
    for kind, op, l, r, is_prem in premises:
        if kind == "int":
            continue
        if kind == "eq" and l[0] == "mod" and l[2][0] == "c":
            m = _int(l[2][1])
            rr = _int(r[1]) if r[0] == "c" else None
            if m and m > 0 and rr is not None:
                ex = nf(l[1])
                if ex is not None and len(ex) == 1:
                    mon, coef = next(iter(ex.items()))
                    if len(mon) == 1:
                        c, k = _int(coef), _int(ex.get((), Fraction(0)))
                        if c is not None and k is not None:
                            g = math.gcd(abs(c), m)
                            if (rr - k) % g == 0:
                                m2 = m // g
                                res = ((rr - k) // g * pow((c // g) % m2, -1, m2)) % m2
                                cong.setdefault(mon[0], []).append((res, m2))
            continue
        if kind == "eq":
            d = nf(t_sub(l, r))
            row = lin_row(d) if d is not None else None
            if row is not None:
                eqp.append(row)
            eqp_poly = d
        else:
            ineqp.append((op, l, r))
            eqp_poly = None
    sol, consistent = gauss(eqp) if eqp else ({}, True)
    if not consistent:
        return "REFUTED"
    smap = {v: _term(c, e) for v, (c, e) in sol.items()}

    kind, op, l, r = conclusion
    if kind == "eq":
        if l[0] == "mod" and l[2][0] == "c" and r[0] == "c":
            m, rr = _int(l[2][1]), _int(r[1])
            if m and rr is not None:
                res = mod_residue(l[1], m, cong)
                if res is not None:
                    return ("CERTIFIED" if (0 <= rr < m and res == rr)
                            else "REFUTED")
        d = nf(subst(t_sub(l, r), smap))
        if d is None:
            return "UNCONFIRMED"
        if not d:
            return "CERTIFIED"
        F = []
        for kind2, _op2, lp, rp, _isp in premises:
            if kind2 == "eq":
                dd = nf(subst(t_sub(lp, rp), smap))
                if dd is not None and dd:
                    F.append(dd)
        if F:
            G = p_grobner(F)
            if G:
                rem = p_divide(d, G)
                if rem is not None and not rem:
                    return "CERTIFIED"
        return "UNCONFIRMED"
    # ---- inequality ----
    l2, r2 = subst(l, smap), subst(r, smap)
    lc = []
    bnd = {}
    for opx, lp, rp in ineqp:
        a2, b2 = subst(lp, smap), subst(rp, smap)
        c = to_constraint(a2, b2, opx)
        if c is not None:
            lc.append(c)
        # var-vs-const bound for the domain
        if a2[0] == "v" and b2[0] == "c":
            _setb(bnd, a2[1], opx, b2[1])
        elif b2[0] == "v" and a2[0] == "c":
            _setb(bnd, b2[1], {"<": ">", ">": "<", "<=": ">=", ">=": "<="}[opx],
                  a2[1])
    if lc:
        neg = NEG_OP.get(op)
        if neg is not None:
            nc = to_constraint(l2, r2, neg)
            if nc is not None and fm_infeasible(lc + [nc]) is True:
                return "CERTIFIED"
    d = (nf(t_sub(r2, l2)) if op in ("<", "<=") else nf(t_sub(l2, r2)))
    if d is not None:
        vs = {x for mm in d for x in set(mm)}
        if len(vs) == 1 and all(len(set(m)) == len(m) for m in d):
            v = next(iter(vs))
            a = d.get((v, v), Fraction(0))
            b = d.get((v,), Fraction(0))
            c = d.get((), Fraction(0))
            if a != 0:
                roots = real_roots(a, b, c)
                if roots == []:
                    disc = b * b - 4 * a * c
                    if a > 0 and (disc < 0 or (disc == 0 and op in ("<=", ">="))):
                        return "CERTIFIED"
                elif roots is not None:
                    strict = op in ("<", ">")
                    inside = [rt for rt in set(roots) if _in_box(rt, bnd.get(v))]
                    if not inside:
                        val = _eval(d, v, bnd.get(v))
                        if val is not None and (val > 0 if strict else val >= 0):
                            return "CERTIFIED"
    return "UNCONFIRMED"


def _setb(bnd, v, op, k):
    cur = bnd.setdefault(v, [None, None, False, False])
    k = Fraction(k)
    if op in (">=", ">"):
        if cur[0] is None or k > cur[0]:
            cur[0], cur[2] = k, op == ">"
        elif k == cur[0]:
            cur[2] = cur[2] or op == ">"
    else:
        if cur[1] is None or k < cur[1]:
            cur[1], cur[3] = k, op == "<"
        elif k == cur[1]:
            cur[3] = cur[3] or op == "<"


def _in_box(q, b):
    if not b:
        return True
    lo, hi, ls, hs = b
    if lo is not None and (q < lo or (q == lo and ls)):
        return False
    if hi is not None and (q > hi or (q == hi and hs)):
        return False
    return True


def _eval(d, v, b):
    lo, hi = (b[0], b[1]) if b else (None, None)
    if lo is not None:
        s = lo + (1 if b[2] else 0)
    elif hi is not None:
        s = hi - (1 if b[3] else 0)
    else:
        s = Fraction(0)
    tot = d.get((), Fraction(0))
    for m, c in d.items():
        if m:
            tot += c * s ** len(m)
    return tot


def _term(c, expr):
    t = ("c", c)
    for v, cf in expr.items():
        t = ("add", t, ("mul", ("c", cf), ("v", v)))
    return t


def replay_text(claims):
    """claims: list of dicts {kind, op, lhs, rhs, is_premise}."""
    prem = [(c["kind"], c["op"], c["lhs"], c["rhs"], c["is_premise"])
            for c in claims if c["is_premise"]]
    concl = [c for c in claims if not c["is_premise"]]
    if not concl:
        return "UNVERIFIABLE"
    return replay(prem, (concl[0]["kind"], concl[0]["op"],
                         concl[0]["lhs"], concl[0]["rhs"]))
