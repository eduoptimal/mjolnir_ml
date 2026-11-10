"""mjolnir_prover.lib.certificate — certificate.json v0 schema + provenance gate (L0).

The certificate is the I/O boundary of the whole system: everything the proposer,
gate, and encoder produce is serialized here, and everything downstream (kernel,
audit, online mode) consumes ONLY this artifact.

Schema v0 (science_usability_v1.md, Sequence 1 / L0):
{
  "schema": "certificate/v0",
  "unit":     {"path", "name", "kind", "code_hash"},
  "ast":      {"hash"},
  "spec":     {"expr", "source", "confirmed", "hash"},
  "program":  {"inputs", "steps", "outputs"},
  "obligations": [{"kind", "premise", "conclusion", "method", "status"}],
  "assumptions": [], "tcb": [], "tool": {},
  "encoder":  {"role": "untrusted"},
  "status":   "CERTIFIED" | "UNVERIFIABLE" | "REFUTED",
  "reason":   ..., "witness": ...
}

Provenance rule (enforced here, non-negotiable):
  CERTIFIED requires spec.source ∈ {"assert", "type_hint"} AND spec.confirmed.
  Anything else -> UNVERIFIABLE("unconfirmed_spec"). Inferred/NL specs never certify.
"""
from __future__ import annotations
import hashlib, json
from . import term as T
from . import intervals as IV
from . import theory as TH
from . import countermodel as CM
from fractions import Fraction as _Fr

SCHEMA = "certificate/v0"
SOURCES_OK = ("assert", "type_hint")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- spec hash ---
def spec_hash(spec) -> str:
    """Content hash binding expr + provenance, so a certificate is pinned to the
    exact confirmed spec that produced it."""
    return sha(json.dumps({"expr": spec["expr"], "source": spec["source"],
                           "confirmed": bool(spec["confirmed"])}, sort_keys=True))


# ------------------------------------------------------- obligation builder ---
def straight_line_obligations(spec_term, steps, outputs):
    """Derive the obligations for a straight-line unit by forward substitution.

    steps: [("assign", name, term)]; outputs: [name]; spec_term: ("eq", lhs, rhs)
    with rhs the post-condition over outputs. The obligation set is:
      kind="post":  conclusion = post with outputs replaced by their final values.
    Pure gate-layer derivation (term algebra only) — no search, no encoder.
    """
    env = {}
    for st in steps:
        k = st[0]
        if k == "assign":
            env[st[1]] = st[2]
        else:
            raise ValueError(f"certificate v0 covers straight-line only, got {k}")
    target = spec_term[1][1] if spec_term[1][0] == "v" else None
    if target is None:
        raise ValueError("v0 fragment: spec must be Eq(var, expr)")
    out_term = T.sub_all(env.get(target, T.var(target)), env)
    kind = spec_term[0]
    return [{"kind": kind, "premise": T.tstr(spec_term[2]),
             "conclusion": T.tstr(out_term), "method": "nf+decide_eq",
             "status": "ok" if T.decide_eq(spec_term[2], out_term) else "fail"}]


def status_of(spec, obligations) -> tuple[str, str]:
    """Provenance gate + obligation gate. Returns (status, reason)."""
    src = spec.get("source")
    if src not in SOURCES_OK or not spec.get("confirmed"):
        return "UNVERIFIABLE", "unconfirmed_spec"
    if any(o["status"] != "ok" for o in obligations):
        return "UNVERIFIABLE", "obligation_failed"
    return "CERTIFIED", ""


# ----------------------------------------------------------------- emitter ----
def emit(unit, spec, spec_term, steps, outputs, assumptions=(), tcb=(),
         tool=None) -> dict:
    """Build a v0 certificate dict from the trusted inputs."""
    if spec_term[0] != "eq":
        raise ValueError("v0 fragment: eq spec only")
    if spec.get("expr") != T.tstr(spec_term):
        raise ValueError("spec.expr must be the canonical term string (tstr) — "
                         "the kernel re-parses it; provenance binds to it")
    code_hash = sha(unit.get("code", ""))
    obligations = straight_line_obligations(spec_term, steps, outputs)
    status, reason = status_of(spec, obligations)
    return {
        "schema": SCHEMA,
        "unit": {"path": unit.get("path", ""), "name": unit.get("name", ""),
                 "kind": unit.get("kind", "function"), "code_hash": code_hash},
        "ast": {"hash": sha(json.dumps([[s[0], s[1], T.key(s[2])] for s in steps]))},
        "spec": {"expr": spec["expr"], "source": spec["source"],
                 "confirmed": bool(spec["confirmed"]), "hash": spec_hash(spec)},
        "program": {"inputs": list(unit.get("inputs", [])),
                    "steps": [[s[0], s[1], T.tstr(s[2])] for s in steps],
                    "outputs": list(outputs)},
        "obligations": obligations,
        "assumptions": list(assumptions),
        "tcb": list(tcb),
        "tool": dict(tool or {"name": "mjolnir", "certificate_schema": SCHEMA}),
        "encoder": {"role": "untrusted"},
        "status": status,
        "reason": reason,
        "witness": None,
    }


def dump(cert) -> str:
    return json.dumps(cert, indent=2, sort_keys=True)


def load(text: str) -> dict:
    c = json.loads(text)
    if c.get("schema") != SCHEMA:
        raise ValueError(f"unknown schema {c.get('schema')!r}")
    return c


# ------------------------------------------------- assert-spec acquisition ----
# ------------------------------------------- premise-derived discharges ----
# A declared inequality premise `a <= b` is the FACT `b - a >= 0`: a direction
# in the space of linear forms, not a per-variable bound. `facts_iv` cannot hold
# that (it is per-variable), which is why a param-param premise used to be
# invisible to the box — and why `lo <= hi |- lo + x <= hi + x` was refuted with
# a witness violating its own premise.
#
# The fix is not a wider bound. It is the obvious one: a claim `c <= d` is
# proved when `d - c` is a NON-NEGATIVE INTEGER COMBINATION of premise
# differences. For `lo <= hi` and the claim `lo + x <= hi + x`, both
# differences are `hi - lo` and the claim IS the premise restated — the x
# cancels. Two premises `a <= b`, `b <= c` discharge `a <= c`.
#
# This is the maximal NON-VACUOUS form: each such certificate is a real
# theorem about the real contract, proved from the contract. Nothing is
# assumed that was not written down, and nothing is concluded that was not
# asserted.
_CONE_MAX = 4          # largest multiplier tried per premise
_CONE_BUDGET = 4       # largest TOTAL sum of multipliers searched


def _poly_of(term_):
    """term -> {atom: exact coefficient} or None if not linear."""
    try:
        d = T.nf(term_)
    except Exception:
        return None
    if d is None:
        return None
    out = {}
    for key, coeff in d.items():
        try:
            from fractions import Fraction as _FQ
            out[tuple(key)] = _FQ(coeff)
        except Exception:
            return None
    return out


def _premise_poly(l_, r_, sign_):
    """The `>= 0` difference polynomial a premise of `sign_` asserts."""
    # normalise every premise to "lhs >= 0" by ordering the two sides
    if sign_ in ("ge", "gt"):
        hi_, lo_ = l_, r_
    else:
        hi_, lo_ = r_, l_
    return _poly_of(T.csub(hi_, lo_))


def _is_nonneg_poly(d) -> bool:
    if d is None:
        return False
    for k, c in d.items():
        if c < 0:
            return False
    return True


def _poly_eq(a, b) -> bool:
    return a is not None and b is not None and a == b


def _conic_combo(target, premises):
    """target = sum k_i * p_i with k_i in N (0.._CONE_MAX).

    Returns a multiplier VECTOR of length len(premises), or None. The length is
    load-bearing: the kernel rebuilds the sum from the recorded vector, so a
    short list would read as "these two premises only" and fail to reproduce
    the rest. (It did exactly that at first: an early exact match returned the
    path taken rather than the full vector, and the kernel — correctly —
    rejected the certificate as malformed.)

    A small exact search over rational coefficient vectors, not a solver: the
    budget is tiny, the arithmetic is exact, and an approximate answer would be
    unsound in the direction that matters (proving too much).
    """
    if target is None:
        return None
    # NO "nonnegative coefficients => axioms alone" shortcut. Over FREE
    # parameters a nonnegative-coefficient polynomial is not nonnegative
    # ({high:1} is negative at high=-5), and the kernel re-derives by exact
    # reproduction: a [0]-vector against a nonzero target came back as
    # false_certificate (FAULT on `low>=0, high>=low |- high>=0`, found by the
    # policy-language prototype, 2026-10-07). A ZERO target is handled by
    # search's first step (remaining all zeros -> [0]*n), which the kernel
    # reproduces exactly. Everything else: produce a real combination or
    # return None and let the sound paths decide.
    keys = set()
    for p in premises:
        keys |= set(p.keys())
    keys |= set(target.keys())
    if not keys:
        return None
    order = sorted(keys)
    n = len(premises)

    def vec(p):
        return [p.get(k, 0) for k in order]

    pv = [vec(p) for p in premises]
    tv = vec(target)
    width = len(order)

    def search(ks, remaining, spent):
        # `remaining` is target - sum(ks_i * p_i); the multiplier VECTOR is
        # carried explicitly, and `spent` bounds the total so the search stays a
        # bounded exact enumeration rather than an exponential walk.
        if all(x == 0 for x in remaining):
            return list(ks)
        if spent >= _CONE_BUDGET:
            return None
        for i in range(n):
            if ks[i] >= _CONE_MAX:
                continue
            ks[i] += 1
            # remaining tracks (target - sum k_i p_i); adding premise i
            # SUBTRACTS it. Accumulating here instead would search for
            # sum(k_i p_i) == -target and silently find nothing (or, at an
            # earlier iteration with a short multiplier list, something wrong).
            nxt = [remaining[j] - pv[i][j] for j in range(width)]
            got = search(ks, nxt, spent + 1)
            if got is not None:
                return got
            ks[i] -= 1
        return None

    got = search([0] * n, list(tv), 0)
    return got


# ---------------------------------------------- premise contradiction ------
# POLICY IS IN THE TABLE (theories/classification.json ->
# `premise_contradiction`), read through theory.premise_contradiction_verdict.
# This function only supplies FACTS and obeys the verdict; it never decides.
# A DISABLED rule matches nothing, which is how a rejected directive stays in
# the table — hashed, reviewable, with its measurement — without being live.
_EQ_SPLIT = __import__("re").compile(r"^([^=<>!]+)==(.*)$")
_NUMERIC = __import__("re").compile(r"^-?\d+(\.\d+)?$")


def _check_premise_contradiction(claim: str, obs: list) -> tuple | None:
    """(verdict, reason, meta) if a DECLARED premise contradicts `claim`.

    "Declared" means an obligation mjolnir itself recorded as method='premise'
    — a name-vs-constant equality the runtime enforces (E1b). An assert that
    mjolnir abstains on is NOT a premise and cannot refute anything; treating
    one as a premise is the whole reason this rule is narrow.
    """
    from . import theory as _TH
    cs = _EQ_SPLIT.match(claim.strip())
    if not cs:
        return None
    c_lhs, c_rhs = cs.group(1).strip(), cs.group(2).strip()
    if not _NUMERIC.match(c_rhs):
        return None
    for o in obs:
        if o.get("method") != "premise":
            continue
        ps = _EQ_SPLIT.match(str(o.get("premise", "")).strip())
        if not ps:
            continue
        p_lhs, p_rhs = ps.group(1).strip(), ps.group(2).strip()
        same_lhs = p_lhs == c_lhs
        both_num = bool(_NUMERIC.match(p_rhs)) and bool(_NUMERIC.match(c_rhs))
        differ = both_num and p_rhs != c_rhs
        verdict, reason = _TH.premise_contradiction_verdict(
            premise_method="premise", lhs_identical=same_lhs,
            rhs_both_numeric=both_num, rhs_differ=differ)
        if verdict is None:
            continue
        return verdict, reason, {"premise": o.get("premise", ""), "lhs": p_lhs,
                                 "rhs_claim": c_rhs, "rhs_premise": p_rhs}
    return None




def _negate_cmp(node):
    """`not (a OP b)` -> the equivalent comparison, or None if not expressible.

    SOUND over the integers, which is the domain `nf` reasons in:
        not (a <= b)  <->  b < a        not (a == b)  <->  a != b
        not (a <  b)  <->  a >= b       and the reverse four.
    Anything else (`not f(x)`, `not x is None`) has no comparison form and is
    left as a negation -- which is now VISIBLE as a claim rather than invisible.
    """
    import ast as _ast
    # SOUNDNESS (2026-10-06): `not(a < b)` is `a >= b` and `not(a > b)` is
    # `a <= b` -- the entries used to read (LtE, ...) / (GtE, ...), which assert
    # the WEAKER, non-equivalent claim: `assert not (5 > 3)` lowered to
    # `5 >= 3` and CERTIFIED a false assertion. The order-negations are
    # equivalences only with the right target relation; a wrong entry is a
    # false certificate, pinned by the negation twins in selfcheck.
    table = {_ast.LtE: (_ast.Lt, True), _ast.Lt: (_ast.GtE, False),
             _ast.GtE: (_ast.Gt, True), _ast.Gt: (_ast.LtE, False),
             _ast.Eq: (_ast.NotEq, False), _ast.NotEq: (_ast.Eq, False)}
    if not (isinstance(node, _ast.Compare) and len(node.ops) == 1
            and len(node.comparators) == 1):
        return None
    m = table.get(type(node.ops[0]))
    if m is None:
        return None
    op, flip = m
    left, right = node.left, node.comparators[0]
    if flip:
        left, right = right, left
    return _ast.Compare(left=left, ops=[op()], comparators=[right])


def _expand_conjuncts(body):
    """Split `assert a and b`, and rewrite `assert not (a <= b)`.

    Two silent holes, both found by pointing the pipeline at claims that were
    visible in the source and absent from the certificate. A claim using `and`
    or `not` matched no `ast.Compare`, produced ZERO obligations, and was
    reported as `no_assert_spec` -- indistinguishable from a unit that states no
    claim at all. On a 6,000-unit corpus sample that misclassification covered
    more units than any capability gap measured today.

    SOUND, both halves: asserting a conjunction asserts each conjunct, and the
    order-negations are equivalences over the integers.

    `or` is NOT split. Asserting `a or b` does not assert `a`, and treating a
    disjunction as two independent claims would manufacture certificates for
    both -- the A-combobound shape. A disjunction stays whole and abstains.
    """
    import ast as _ast
    out = []
    for st in body:
        if (isinstance(st, _ast.Assert) and isinstance(st.test, _ast.BoolOp)
                and isinstance(st.test.op, _ast.And)):
            for v in st.test.values:
                out.append(_ast.Assert(test=v, lineno=st.lineno))
            continue
        if isinstance(st, _ast.Assert) and isinstance(st.test, _ast.UnaryOp) \
                and isinstance(st.test.op, _ast.Not):
            inner = st.test.operand
            neg = _negate_cmp(inner)
            # When no comparison form exists (`not (a and b)`, `not f(x)`,
            # `not not (...)`), keep the ORIGINAL negation. Appending `inner`
            # stripped the `not` and asserted the opposite claim -- a false
            # certificate whenever a later assert carried the unit. The kept
            # form is recorded as an undecidable claim downstream (visible),
            # never silently as its negation.
            out.append(_ast.Assert(test=neg if neg is not None else st.test,
                                   lineno=st.lineno))
            continue
        out.append(st)
    return out

def _ast2term(node, ctx=None):
    """stdlib ast -> term (exact fragment: +, -, *, //, %, unary -, ints, names,
    len(name), literal lists, name[const-index]).
    No external algebra dependency; the term algebra is the decision procedure."""
    import ast
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return T.const(node.value)          # ints exact; E5: floats as exact rationals
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return T.const(node.value)          # E2b: inert string (keys/content)
    if isinstance(node, ast.Name):
        return T.var(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return T.cneg(_ast2term(node.operand, ctx))
    if isinstance(node, (ast.List, ast.Tuple)):
        return T.clist(*[_ast2term(e, ctx) for e in node.elts])
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            raise ValueError("out_of_fragment:slice")
        return T.celem(_ast2term(node.value, ctx), _ast2term(node.slice, ctx))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "len" and len(node.args) == 1 and not node.keywords:
        return T.clen(_ast2term(node.args[0], ctx))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base = _ast2term(node.left, ctx)
        exp = node.right
        if isinstance(exp, ast.Constant) and isinstance(exp.value, int) \
                and 0 <= exp.value <= 8:                 # bounded: E-pow MVF
            out = T.const(1)
            for _ in range(exp.value):
                out = T.cmul(out, base)
            return out
        raise ValueError("out_of_fragment:Pow-unbounded")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("min", "max") and len(node.args) == 2 \
            and not node.keywords:
        f = T.cmin if node.func.id == "min" else T.cmax
        return f(_ast2term(node.args[0], ctx), _ast2term(node.args[1], ctx))
    _cts = (ctx or {}).get("contracts") or {}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _cts and not node.keywords \
            and len(node.args) == len(_cts[node.func.id][0]):
        # E3: inline the callee's return term under argument substitution
        gp, ret = _cts[node.func.id]
        t = ret
        for pname, argnode in zip(sorted(gp), node.args):
            t = T.sub(t, pname, _ast2term(argnode, ctx))
        return t
    op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul",
          ast.Div: "rdiv", ast.FloorDiv: "div", ast.Mod: "mod"}.get(type(node.op)) \
        if isinstance(node, ast.BinOp) else None
    if op is not None:
        return _BUILD_SAFE[op](_ast2term(node.left, ctx), _ast2term(node.right, ctx))
    raise ValueError(f"out_of_fragment:{type(node).__name__}")


_BUILD_SAFE = {"add": T.cadd, "sub": T.csub, "mul": T.cmul,
               "div": T.cdiv, "mod": T.cmod, "rdiv": T.crdiv}


def _const_lit(node):
    """A closed literal (int/str/bool/None or a tuple of those) -> its value."""
    import ast
    if isinstance(node, ast.Constant) and (node.value is None or
                                           isinstance(node.value, (int, str, bool))):
        return node.value
    if isinstance(node, ast.Tuple):
        vals = []
        for e in node.elts:
            v = _const_lit(e)
            if v is _NONE: return _NONE
            vals.append(v)
        return tuple(vals)
    return _NONE


_NONE = object()


def unit_vc(code: str, pre_bounds=None, module_source=None, tier=None):
    from . import trace as TR
    TR.start()
    """E3: module_source enables call contracts — in-module functions that are
    themselves straight-line pure arithmetic get INLINED into slices, so a
    unit's coverage is transitive through the utilities it calls."""
    """Claim-level VC discharge for straight-line pure-arithmetic stretches.

    For each assert (Eq/Le/Ge/Lt/Gt), BACKWARD-SLICE the assigned variables.
    Every sliced assignment must be exact integer arithmetic (calls, floats,
    subscripts, loops poison the slice -> labeled abstain); the assert then
    becomes an obligation over the unit's inputs, discharged by the term
    algebra (equations: decide_eq; inequalities: sound sign analysis).
    BOUNDED: substitution goes one variable at a time via T.sub with a node
    budget -- no composed-mapping recursion (that was the memory runaway).
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return dict(source="assert", confirmed=True, status="UNVERIFIABLE",
                    obligations=[], reason=f"syntax_error:{e.lineno}")
    fn = next((x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)), None)
    if fn is None:
        return dict(source="assert", confirmed=True, status="UNVERIFIABLE",
                    obligations=[], reason="no_function")
    params = {a.arg for a in fn.args.args}
    body = _expand_conjuncts(fn.body)
    # The CLAIM is the unit's last assert: it must be DISCHARGED, never used as a
    # premise for itself. Only asserts before it may feed substitutions forward.
    # (Without this, e.g. `assert n + 1 == 2` is solved for n and re-substituted,
    # turning the claim into the tautology `2 == 2` — a false certificate.)
    _asserts_ = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
    _last_assert = max((a.lineno for a in _asserts_), default=None)
    # E3: in-module pure-arithmetic function contracts (name -> (params, ret))
    contracts = {}
    if module_source:
        try:
            mtree = ast.parse(module_source)
            other = [n for n in ast.walk(mtree)
                     if isinstance(n, ast.FunctionDef) and n.name != fn.name]
            for g in other:
                if g.args.vararg or g.args.kwarg or g.args.defaults:
                    continue
                rets = [n for n in ast.walk(g) if isinstance(n, ast.Return)]
                assigns = [n for n in ast.walk(g)
                           if isinstance(n, (ast.Assign, ast.AugAssign,
                                             ast.If, ast.For, ast.While,
                                             ast.Raise, ast.Try))]
                if len(rets) == 1 and not assigns and rets[0].value is not None:
                    gp = {a.arg for a in g.args.args}
                    try:
                        contracts[g.name] = (gp, _ast2term(rets[0].value))
                    except ValueError:
                        pass
        except SyntaxError:
            pass
    ctx = {"contracts": contracts}             # E3: callee contracts

    # ------------------------------------------------ R2: linear system ----
    def _linear_rows(fn_node):
        """Rows of the linear system from param-only Eq PREMISES:
        each row = ({var: coeff}, const). Fraction-exact.
        The unit's CLAIM (last assert) is excluded — a claim must never be
        solved from itself and re-substituted (that launders a false claim into
        a tautology, e.g. `n + 1 == 2` -> `2 == 2`)."""
        rows = []
        lines = set()
        for node in ast.walk(fn_node):
            if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.ops[0], ast.Eq)):
                continue
            if node.lineno == _last_assert:
                continue                          # the claim is not a premise
            try:
                l_ = _ast2term(node.test.left, ctx)
                r_ = _ast2term(node.test.comparators[0], ctx)
            except ValueError:
                continue
            d = T.nf(T.csub(l_, r_))
            if d is None:
                continue
            row, const = {}, _Fr(0)
            ok = True
            for mon, coef in d.items():
                vs = [v for v in mon]
                if len(vs) != 1:
                    if not vs:
                        const += _Fr(coef)
                        continue
                    ok = False
                    break
                if vs[0] not in params:
                    ok = False
                    break
                row[vs[0]] = row.get(vs[0], _Fr(0)) + _Fr(coef)
            if ok and (row or const):
                rows.append((row, const))
                lines.add(node.lineno)            # premise assert -> declared
        return rows, lines

    def _gauss(rows):
        """Exact RREF over Fractions. Returns {var: Fraction} for determined
        vars, or None if the system is inconsistent."""
        import itertools
        vars_ = sorted({v for row, _ in rows for v in row})
        mat = [([_Fr(row.get(v, 0)) for v in vars_], _Fr(const))
               for row, const in rows]
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
        for row, const in mat[r:]:
            if all(x == 0 for x in row) and const != 0:
                return None, {}                        # inconsistent: contradiction
        sol, subs = {}, {}
        for i, pc in enumerate(pivot_cols[:len(mat)]):
            row, const = mat[i]
            if len(row) and row[pc] != 0:
                others = [x for j, x in enumerate(row) if j != pc]
                if all(x == 0 for x in others) and row[pc] != 0:
                    # fully determined: x = -const / coefficient
                    sol[vars_[pc]] = -const / row[pc]
                else:
                    # partially determined: x = (-const - sum(other.coef * other)) / coef
                    t = T.const(_Fr(-const) / _Fr(row[pc]))
                    for j, coef in enumerate(row):
                        if j != pc and coef != 0:
                            t = T.csub(t, T.cmul(T.const(_Fr(coef) / _Fr(row[pc])),
                                                 T.var(vars_[j])))
                    if not (T.vars_of(t) - set(vars_)):
                        subs[vars_[pc]] = t
        return sol, subs

    oof, poison_line = False, None              # prefix poison state (survives phases)
    facts = {}
    eq_subst = {}                               # R1: var -> term (linear chain)                                   # E1: var -> lower bound (ints)
    facts_iv = {}                                # E5: var -> [lo, hi]
    for v_, b_ in (pre_bounds or {}).items():
        if isinstance(b_, (tuple, list)):
            facts_iv[v_] = tuple(b_)
        else:
            facts[v_] = b_
            facts_iv[v_] = (b_, IV.INF)
    def _uses_floats(t):
        if not isinstance(t, tuple):
            return False
        if t[0] == "c" and isinstance(t[1], _Fr):
            return True
        return any(_uses_floats(x) for x in t[1:] if isinstance(x, tuple))

    def _bounds_from(lhs, rhs, strict, upper=False):
        """Direction-aware declared bounds from param-vs-constant asserts:
        x >= c -> lo = c ; x > c -> lo = c+1 ; x <= c -> hi = c ; x < c -> hi = c-1.
        facts (poly path) keeps lower bounds only; facts_iv keeps the box."""
        if lhs[0] == "v" and rhs[0] == "c":
            var, c = lhs[1], rhs[1]
        elif rhs[0] == "v" and lhs[0] == "c":
            var, c = rhs[1], lhs[1]
        else:
            return
        lo, hi = facts_iv.get(var, (None, IV.INF))
        if upper:
            new_hi = c - (1 if strict else 0)
            cur_lo = lo if lo is not None else -IV.INF
            facts_iv[var] = (cur_lo,
                             new_hi if hi == IV.INF else min(hi, new_hi))
        else:
            new_lo = c + (1 if strict else 0)
            facts[var] = max(facts.get(var, new_lo), new_lo)
            cur_hi = hi
            facts_iv[var] = (max(lo if lo is not None else -IV.INF, new_lo),
                             cur_hi)

    def _param_vs_const(a, b):
        return a[0] == "v" and a[1] in params and b[0] == "c"

    # ---- E-mod: modular arithmetic --------------------------------------
    _residues = {}                                 # var -> (residue, modulus)
    _declared_premises = []                        # (line, claim) the runtime enforces
    _premise_diffs = []                            # live difference polys (see E-premise)

    def _is_mod_eq(t):
        """('c'/'v', cmod-term, const) pattern for `expr % m == r`."""
        return t[0] == "eq" and t[1][0] == "mod" and t[2][0] == "c"

    def _mod_claim(l_, r_, op):
        """Dispose `expr % m == r` (or <=,>=). Returns (verdict, reason) or
        None if not a mod claim."""
        if l_[0] != "mod":
            return None
        inner, m = l_[1], l_[2][1]
        if not isinstance(m, int) or m <= 0:
            return None
        if op != "eq":
            return None                             # mod inequalities: later
        r_ = r_[1] if r_[0] == "c" else None
        if r_ is None or not isinstance(r_, int):
            return None
        vs = T.vars_of(inner)
        if not vs:
            return (T.cmod(inner, T.const(m))[1] == r_, "mod-fold")
        # single var with a residue premise (mod-m congruence) or exact value?
        if len(vs) == 1:
            v = next(iter(vs))
            if v in vals:
                folded = T.cmod(T.sub(inner, v, T.const(vals[v])), T.const(m))
                return (folded[1] == r_, "mod-exact")
            if v in _residues and _residues[v][1] == m:
                # LEMMA (exact): a linear expression is constant on a residue
                # class mod m: n ≡ res (mod m) ⟹ lin(n) ≡ lin(res) (mod m)
                res = _residues[v][0]
                folded = T.cmod(T.sub(inner, v, T.const(res)), T.const(m))
                out = folded[1] == r_
                TR.lemma("mod-residue-constancy", 1,
                         (f"{v} ≡ {res} mod {m}", T.tstr(inner)), out)
                return (out, "mod-residue")
        # free vars: fail-closed corner test — the box corner is an allowed
        # input; if the assert fails there, the unit is broken on its domain.
        corner_env = {}
        for v in vs:
            if v in vals:
                corner_env[v] = vals[v]
            elif v in facts_iv and isinstance(facts_iv[v], tuple):
                corner_env[v] = facts_iv[v][0]      # lo corner
            else:
                corner_env[v] = 0
        try:
            _repl = {k: T.const(x) for k, x in corner_env.items()}
            corner = T.ev(T.cmod(T.sub_all(inner, _repl), T.const(m)), {})
            if corner != r_:
                return (False, "mod-corner-refute")
        except Exception:
            return None
        return None                                  # corner matches: abstain

    def _relational_premise(l_, r_):
        """E-rel: name-vs-name inequalities are runtime-enforced relations."""
        return (l_[0] == "v" and r_[0] == "v"
                and l_[1] in params and r_[1] in params)

    def _lin_box_decide(l_, r_, op):
        """E-rel: exact disposal of LINEAR inequality claims over param boxes.
        A linear function attains its extrema at the box corners (exact)."""
        d = T.nf(T.csub(r_, l_))          # need r - l >= 0 (le) etc.
        if d is None:
            return None
        # linear = every monomial is a single var (no products, no atoms)
        coeffs = {}
        const = _Fr(0)
        for mon, coef in d.items():
            if not mon:
                const += _Fr(coef)
                continue
            if len(mon) != 1:
                return None
            v = mon[0]
            if str(v).startswith(("clen:", "elem:", "fact:")):
                return None               # opaque atoms: not linear-over-box
            coeffs[v] = coeffs.get(v, _Fr(0)) + _Fr(coef)
        # box per var. Infinities stay as sentinels: IV.INF is float("inf") and
        # Fraction(inf) raises OverflowError. The old code called _Fr() on the
        # sentinel, so box() raised on EVERY one-sided box; the caller's
        # try/except turned that into `_lin = None`, and every upper-bounded
        # claim silently abstained. That was the A-combobound obstruction.
        def box(v):
            if v in vals:
                x = _Fr(vals[v])
                return (x, x)
            if v in facts_iv and isinstance(facts_iv[v], tuple):
                lo, hi = facts_iv[v]
                flo = -IV.INF if lo is None else (
                    lo if lo in (-IV.INF, IV.INF) else _Fr(lo))
                fhi = IV.INF if hi is None else (
                    hi if hi in (-IV.INF, IV.INF) else _Fr(hi))
                return (flo, fhi)
            return (-IV.INF, IV.INF)
        # A linear form attains its extremes at the box corners. One-sided
        # boxes are fine: the extremum is at the FINITE corner when the
        # coefficient points that way; otherwise that end is unbounded.
        lo_sum, hi_sum = const, const
        lo_unb = hi_unb = False           # min == -inf / max == +inf
        for v, c in coeffs.items():
            blo, bhi = box(v)
            lo_end, hi_end = (blo, bhi) if c >= 0 else (bhi, blo)
            if lo_end in (-IV.INF, IV.INF):
                lo_unb = True
            else:
                lo_sum += c * lo_end
            if hi_end in (-IV.INF, IV.INF):
                hi_unb = True
            else:
                hi_sum += c * hi_end
        # d == r - l. Claim `l op r` must hold for ALL points of the box:
        #   l <= r  <=>  min d >= 0        l <  r  <=>  min d >  0
        #   l >= r  <=>  max d <= 0        l >  r  <=>  max d <  0
        # A violating input EXISTS when the opposite holds, so refutation is
        # min-based for le/lt and max-based for ge/gt — the old table used
        # lo_sum for every op and hi_sum for every op, which is wrong in both
        # directions and had never been exercised (box() always raised first).
        holds = {"le": (not lo_unb) and lo_sum >= 0,
                 "lt": (not lo_unb) and lo_sum > 0,
                 "ge": (not hi_unb) and hi_sum <= 0,
                 "gt": (not hi_unb) and hi_sum < 0}[op]
        fails = {"le": lo_unb or lo_sum < 0,
                 "lt": lo_unb or lo_sum <= 0,
                 "ge": hi_unb or hi_sum > 0,
                 "gt": hi_unb or hi_sum >= 0}[op]
        if holds:
            return (True, "rel-box")
        if fails:
            return (False, "rel-box-refute")
        return None

    def _eq_solve(l_, r_):
        """R1: solve a linear equality for one variable (sequential Gauss step).
        Returns (var, term) or None. Existing substitutions applied first via
        the resolver (eq_subst consulted in slice_expr)."""
        d = T.nf(T.csub(l_, r_))
        if d is None:
            return None
        for mon, coef in d.items():
            cnt = {}
            for v in mon:
                cnt[v] = cnt.get(v, 0) + 1
            if list(cnt.values()) != [1]:
                continue
            v = next(iter(cnt))
            rest = {m: c for m, c in d.items() if m != mon}
            t = T.const(_Fr(0))
            for m, c2 in rest.items():
                mt = T.const(_Fr(c2))
                for vv in m:
                    mt = T.cmul(mt, T.var(vv))
                t = T.cadd(t, T.cmul(T.const(_Fr(-1)), mt))
            # multiply by the exact Fraction inverse (never emit a div node:
            # nf cannot fold cdiv with vars, and the substitution must stay
            # inside the fragment for every downstream slice)
            t = T.cmul(T.const(_Fr(1) / _Fr(coef)), t)
            if not (T.vars_of(t) - params):
                return v, t
        return None

    def _vars_of_side(t):
        return {x for x in T.vars_of(t)}

    def _premise_shape(l, r, op_name):
        """E1c via THEORY TABLE (theories/classification.json): premise shape =
        single Name-vs-constant comparison (the confirmed mined shape V >= N).
        Compound comparisons over params stay CLAIMS — they can be false
        theorems (e.g. n + 1 <= n) and must remain refutable."""
        lk = "param" if (l[0] == "v" and l[1] in params) else (
            "const" if l[0] == "c" else "other")
        rk = "param" if (r[0] == "v" and r[1] in params) else (
            "const" if r[0] == "c" else "other")
        return TH.classify_premise(lk, rk, op_name)

    prefix = []                                # straight-line prefix (MVF)
    premise_obs = []
    consumed_premise_lines = set()
    claim_start = len(body)
    for idx, st in enumerate(body):
        if isinstance(st, ast.Assert) and isinstance(st.test, ast.Compare):
            # chained premise (0 <= px <= 255): middle operands are params,
            # edges may be constants -> consume as premise, keep going
            if len(st.test.ops) > 1:
                operands = [st.test.left] + list(st.test.comparators)
                middles_ok = all(isinstance(o, ast.Name) and o.id in params
                                 for o in operands[1:-1])
                pairs_ok = True
                if middles_ok:
                    for i, o_ in enumerate(st.test.ops):
                        a_, b_ = operands[i], operands[i + 1]
                        a_p = isinstance(a_, ast.Name) and a_.id in params
                        b_p = isinstance(b_, ast.Name) and b_.id in params
                        a_c = isinstance(a_, ast.Constant) and isinstance(a_.value, (int, float))
                        b_c = isinstance(b_, ast.Constant) and isinstance(b_.value, (int, float))
                        if a_p and b_p:
                            continue
                        if a_p and b_c:
                            _bounds_from(T.var(a_.id), T.const(b_.value),
                                         isinstance(o_, (_ast.Gt, _ast.Lt)),
                                         upper=isinstance(o_, (_ast.LtE, _ast.Lt)))
                        elif b_p and a_c:
                            _bounds_from(T.var(b_.id), T.const(a_.value),
                                         isinstance(o_, (_ast.Gt, _ast.Lt)),
                                         upper=isinstance(o_, (_ast.GtE, _ast.Gt)))
                        else:
                            pairs_ok = False
                if middles_ok and pairs_ok:
                    consumed_premise_lines.add(st.lineno)
                    consumed_premise_lines.add(st.lineno)
                    premise_obs.append(dict(kind="premise", line=st.lineno,
                                            claim=ast.unparse(st.test)[:80],
                                            premise=ast.unparse(st.test)[:80],
                                            conclusion="declared",
                                            method="premise", status="ok"))
                    continue
                claim_start = idx
                break
            # single-op premise-shaped? consume here and keep going
            if True:
                try:
                    pl = _ast2term(st.test.left, ctx)
                    pr = _ast2term(st.test.comparators[0], ctx)
                except ValueError:
                    claim_start = idx
                    break
                _op = st.test.ops[0]
                _opn = _OP_NAMES.get(type(_op))
                rule = _premise_shape(pl, pr, _opn) if _opn else None
                if rule:
                    b = rule["bound"]
                    strict = isinstance(_op, (_ast.Gt, _ast.Lt))
                    # the CONST side provides c; the VAR side is the subject
                    if pl[0] == "c":
                        c, _var_term = pl[1], pr
                    elif pr[0] == "c":
                        c, _var_term = pr[1], pl
                    else:
                        c, _var_term = None, None
                    if c is not None and isinstance(c, int):
                        _bounds_from(_var_term, T.const(c), strict,
                                     upper=(b["kind"] == "upper"))
                    # E-PREMISE COMPLETENESS: a prefix-consumed inequality is
                    # a DECLARED premise like any other, so its difference
                    # polynomial must reach the conic solver. The main loop
                    # never sees this line (consumed_premise_lines skips it),
                    # so without this registration the solver had an
                    # incomplete premise set, could not find the real
                    # combination, and the (now removed) [0]-shortcut emitted
                    # one the kernel could not reproduce -> FAULT on
                    # `low>=0, high>=low |- high>=0`.
                    if _opn in ("ge", "gt", "le", "lt"):
                        _pd_pre = _premise_poly(pl, pr,
                                                _OP_REL.get(type(_op)))
                        if _pd_pre is not None:
                            _premise_diffs.append(_pd_pre)
                    consumed_premise_lines.add(st.lineno)
                    premise_obs.append(dict(kind="premise", line=st.lineno,
                                            claim=ast.unparse(st.test)[:80],
                                            premise=ast.unparse(st.test)[:80],
                                            conclusion="declared",
                                            method="premise", status="ok"))
                    continue
                # E2c LENGTH bound: `len(param) <= N` is a declared input
                # contract -- SQL VARCHAR(n)/CHAR(n) is enforced on write and
                # a Python runtime assert enforces it. The length atom is not
                # a NUMERIC bound, so it deliberately feeds nothing into
                # `facts`/`bounds` (a length bound on a string says nothing
                # about it as a number): the contract is recorded so it is
                # visible, and the disposition is declared -- never proved
                # (`_proved`/the theorem guard demote a theorem built only
                # from it). computed locals are excluded: that shape is the
                # A-combobound claim, which must stay decidable-or-abstain.
                if ((pl[0] == "clen" and pr[0] == "c"
                     and isinstance(pr[1], int))
                        or (pr[0] == "clen" and pl[0] == "c"
                            and isinstance(pl[1], int))) and \
                        not ((T.vars_of(pl) | T.vars_of(pr)) - params):
                    consumed_premise_lines.add(st.lineno)
                    premise_obs.append(
                        dict(kind="premise", line=st.lineno,
                             claim=ast.unparse(st.test)[:80],
                             premise=ast.unparse(st.test)[:80],
                             conclusion="declared (length bound)",
                             method="premise", status="ok"))
                    continue
            claim_start = idx
            break
        if isinstance(st, ast.If) and any(
                isinstance(x, ast.Raise) for x in st.body):
            # E1c-raise: guard clauses are runtime-ENFORCED premises.
            # `if x < 0: raise` declares x >= 0; Or of violations = each negated.
            gtest = st.test
            negs = []                            # (pred, already_negated)
            if isinstance(gtest, ast.UnaryOp) and isinstance(gtest.op, ast.Not):
                # `if not P: raise` -> the premise IS P (flag: no further flip)
                if isinstance(gtest.operand, ast.Compare):
                    negs = [(gtest.operand, True)]
            elif isinstance(gtest, ast.BoolOp) and isinstance(gtest.op, ast.Or):
                negs = [(v, False) for v in gtest.values
                        if isinstance(v, ast.Compare)]
            elif isinstance(gtest, ast.Compare):
                negs = [(gtest, False)]          # violation-pred: premise = NOT g

            ok_all, recs = True, []
            for g, g_negated in negs:
                if not (isinstance(g, ast.Compare) and len(g.ops) == 1):
                    ok_all = False
                    break
                try:
                    gl = _ast2term(g.left, ctx)
                    gr = _ast2term(g.comparators[0], ctx)
                except ValueError:
                    ok_all = False
                    break
                # premise operator: NOT(violation) — flip unless already negated
                # negation table: not(a>b)=a<=b, not(a>=b)=a<b,
                #                 not(a<b)=a>=b, not(a<=b)=a>b
                _flip = {_ast.Gt: _ast.LtE, _ast.GtE: _ast.Lt,
                         _ast.Lt: _ast.GtE, _ast.LtE: _ast.Gt}.get(type(g.ops[0]))
                # class of the premise operator (flip-map returns a class;
                # a negated guard keeps its own operator INSTANCE)
                _prem_op_cls = type(g.ops[0]) if g_negated else _flip
                if _prem_op_cls is None:
                    ok_all = False
                    break
                _opn = _OP_NAMES[_prem_op_cls]
                strict = _opn in ("gt", "lt")
                lk = "param" if (gl[0] == "v" and gl[1] in params) else (
                    "const" if gl[0] == "c" else "other")
                rk = "param" if (gr[0] == "v" and gr[1] in params) else (
                    "const" if gr[0] == "c" else "other")
                rule = TH.classify_premise(lk, rk, _opn)
                if rule is None:
                    ok_all = False
                    break
                recs.append((gl, gr, rule, strict, g))
            if ok_all and recs:
                consumed_premise_lines.add(st.lineno)
                for gl, gr, rule, strict, g in recs:
                    # const side feeds the bound
                    if gl[0] == "c":
                        c, vterm = gl[1], gr
                    elif gr[0] == "c":
                        c, vterm = gr[1], gl
                    else:
                        continue                # param-param guard: no const bound
                    if not isinstance(c, (int, float)):
                        continue
                    _bounds_from(vterm, T.const(c), strict,
                                 upper=(rule["bound"]["kind"] == "upper"))
                    premise_obs.append(dict(
                        kind="premise", line=g.lineno,
                        claim=ast.unparse(g)[:80],
                        premise="guard: " + ast.unparse(g)[:70],
                        conclusion="declared", method="premise", status="ok"))
                continue
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) \
                and isinstance(st.value.value, str):
            continue                               # docstring: inert
        if isinstance(st, ast.Return):
            claim_start = idx
            break                                  # region ends; no poison
        if isinstance(st, (ast.If, ast.While, ast.With, ast.Try,
                           ast.Raise, ast.Match, ast.Expr)):
            # ast.Expr: bare call statements MUTATE state (e.g. sort(xs)) —
            # between assignment and assert they invalidate the literals
            oof, poison_line = True, st.lineno
            claim_start = idx
            break
        prefix.append(st)

    NODE_BUDGET = 512                          # per-sliced-term cap (nodes)
    env_terms = {}                             # name -> term | None(poison)
    vals = {}                                  # E1: var -> exact value
    # Keys of `vals` that came from a DECLARED premise (an assert the runtime
    # enforces, or the R2 solve over them) rather than from a program
    # assignment. The refute-by-declaration gate reads this: an assignment
    # (`b = 5`) is execution semantics and may refute; a declaration may not.
    _declared_vars = set()

    # R2: linear-system solve — param-only Eq premises form a system; the
    # exact solution feeds every claim (all vars determined -> claims become
    # constant comparisons). Inconsistent premises -> REFUTED with the system
    # as witness.
    _sys_rows, _r2_lines = _linear_rows(fn)
    if _sys_rows:
        _sol, _subs = _gauss(_sys_rows)
        # Premises form a system that MAY be solved jointly and substituted into
        # the claim (A-system rung: a+b=10, a-b=2 |- a=6). But `_linear_rows`
        # excludes the unit's CLAIM (last assert), so a claim can never solve
        # itself and re-substitute — that is what laundered `n + 1 == 2` into
        # `2 == 2`. Premise solutions are applied ONLY to later asserts.
        if _sol is None:
            obs_pre = [dict(kind="premise", line=0, claim="linear system",
                            premise="param-only Eq premises",
                            conclusion="inconsistent", method="linarith-system",
                            status="fail",
                            witness=dict(system="inconsistent premises"))]
            trace = TR.finish()
            return dict(source="assert", confirmed=True, status="REFUTED",
                        obligations=_sys_rows and [dict(
                            kind="system", premise="linear system",
                            conclusion="inconsistent", method="linarith-system",
                            status="fail")] + obs_pre,
                        reason="inconsistent_premises", witness=None,
                        trace=trace, trace_digest=TR.digest(trace))
        for v_, t_ in _subs.items():
            if v_ in params and v_ not in _sol and T.size(t_) <= NODE_BUDGET:
                eq_subst[v_] = t_
        for v_, c_ in _sol.items():
            if v_ in params:
                vals[v_] = c_
                _declared_vars.add(v_)
                facts_iv[v_] = (float(c_), float(c_))

    def slice_expr(node):
        t = _ast2term(node, ctx)               # raises on out-of-fragment
        for _ in range(64):                    # bounded resolution loop
            vs = [v for v in T.vars_of(t)
                  if (v in env_terms and env_terms[v] is not None)
                  or v in vals or v in eq_subst]
            if not vs:
                break
            v = vs[0]
            if v in eq_subst:
                rep = eq_subst[v]
            elif v in vals:
                rep = T.const(vals[v])
            else:
                rep = env_terms[v]
            if rep is None or T.size(rep) > NODE_BUDGET or T.size(t) > NODE_BUDGET:
                raise ValueError("sliced_var_out_of_fragment")
            prev = t
            t = T.sub(t, v, rep)               # canonicalizing single-var sub
            TR.rule("resolve", 1, (T.tstr(prev), v,
                                   T.tstr(rep) if isinstance(rep, tuple) else rep),
                    T.tstr(t))
        if T.size(t) > NODE_BUDGET:
            raise ValueError("slice_budget")
        return t

    from . import trace as TR
    TR.start()
    obs = list(premise_obs)                    # prefix premises travel in the cert
    refuted, undecided = False, False          # NOTE: oof/poison_line survive the prefix!
    # E1b: injected premises are recorded as declared obligations (visible)
    for v_, lb_ in sorted((pre_bounds or {}).items()):
        obs.append(dict(kind="premise", line=0,
                        claim=f"assumed {v_} >= {lb_} (confirmed directive)",
                        premise=f"assumed {v_} >= {lb_}", conclusion="declared",
                        method="premise", status="ok"))
    poison_line = None

    list_agnostic = isinstance(st, ast.Assign)  # placeholder (replaced below)

    for st in prefix:
        if oof:
            break
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name) \
                and isinstance(st.value, ast.List):
            # E2: literal collection -> exact clist (length + elements known)
            try:
                env_terms[st.targets[0].id] = T.clist(
                    *[_ast2term(e, ctx) for e in st.value.elts])
            except ValueError:
                env_terms[st.targets[0].id] = None
            continue
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name) \
                and isinstance(st.value, ast.Dict):
            try:                                   # E2b: literal dict
                kv = []
                for k_, v_ in zip(st.value.keys, st.value.values):
                    kv.extend((_ast2term(k_, ctx), _ast2term(v_, ctx)))
                env_terms[st.targets[0].id] = T.cdict(*kv)
            except ValueError:
                env_terms[st.targets[0].id] = None
            continue
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name) \
                and isinstance(st.value, ast.Constant) \
                and isinstance(st.value.value, str):
            env_terms[st.targets[0].id] = T.const(st.value.value)  # E2b: string
            continue
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name):
            name = st.targets[0].id
            try:
                t = slice_expr(st.value)
                if T.vars_of(t) - params:      # unresolved deps remain
                    env_terms[name] = None
                else:
                    env_terms[name] = t
                    if t[0] == "c" and isinstance(t[1], int):
                        vals[name] = t[1]      # exact-value fact
            except ValueError:
                env_terms[name] = None         # poison: consumers must abstain
        elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
            name = st.target.id
            f = _BUILD_SAFE.get(_OP_MAP.get(type(st.op)))
            try:
                t = f(T.var(name), slice_expr(st.value)) if f else None
                env_terms[name] = t if (t is not None and
                                        not (T.vars_of(t) - params)) else None
            except ValueError:
                env_terms[name] = None
        elif isinstance(st, ast.For):
            # E4: bounded-loop summary (exact closed forms, MVF-guarded)
            try:
                for name, acc in _loop_summary(st).items():
                    vals.pop(name, None)           # stale exact value dies here
                    cur = env_terms.get(name)
                    if cur is None or T.vars_of(cur) - params:
                        env_terms[name] = None     # unknown initial value
                    else:
                        env_terms[name] = T.cadd(cur, acc)
                    if T.size(env_terms[name]) > NODE_BUDGET:
                        env_terms[name] = None
            except (ValueError, KeyError):
                oof, poison_line = True, st.lineno
                break
        elif isinstance(st, (ast.If, ast.While, ast.With, ast.Try,
                             ast.Return, ast.Raise, ast.Match)):
            # control flow / effects in the slice's prefix: nothing after it is
            # a straight-line claim — the whole unit's asserts must abstain,
            # and we point at the statement that caused it.
            oof, poison_line = True, st.lineno
            break

    returned_at = None
    for st in body:
        if st.lineno in consumed_premise_lines:
            continue                               # already recorded as a premise
        if isinstance(st, ast.Return):
            returned_at = st.lineno                # claims after this are dead
        if oof or (returned_at is not None and isinstance(st, ast.Assert)
                   and st.lineno > returned_at):
            break
        if not isinstance(st, ast.Assert):
            continue
        if not isinstance(st.test, ast.Compare):
            # A VISIBLE claim, not a silent skip. `assert x`,
            # `assert not (a and b)`, `assert f(x)` state something; dropping
            # them reported no_assert_spec ("add asserts to make this unit
            # checkable") on units that DID assert, and let a mixed unit
            # certify while one of its own claims was never examined. Every
            # asserted claim must reach an obligation or an explicit abstain.
            _cl = ast.unparse(st.test)[:80]
            undecided = True
            obs.append(dict(kind="vc", line=st.lineno, claim=_cl, premise=_cl,
                            conclusion="?", method="out_of_fragment",
                            detail="assert shape outside the comparison fragment",
                            status="abstain"))
            continue
        # chained compares (0 <= x <= 255): split into per-pair premise checks.
        # Middle operands must be params; edges may be constants.
        if len(st.test.ops) > 1:
            operands = [st.test.left] + list(st.test.comparators)
            middles_ok = all(isinstance(o, ast.Name) and o.id in params
                             for o in operands[1:-1])
            pairs_ok = True
            if middles_ok:
                for i, o_ in enumerate(st.test.ops):
                    a_, b_ = operands[i], operands[i + 1]
                    a_param = isinstance(a_, ast.Name) and a_.id in params
                    b_param = isinstance(b_, ast.Name) and b_.id in params
                    a_const = isinstance(a_, ast.Constant) and isinstance(a_.value, (int, float))
                    b_const = isinstance(b_, ast.Constant) and isinstance(b_.value, (int, float))
                    if a_param and b_param:
                        continue                # param-param pair: no const bound
                    if a_param and b_const:
                        _bounds_from(T.var(a_.id), T.const(b_.value),
                                     isinstance(o_, (_ast.Gt, _ast.Lt)),
                                     upper=isinstance(o_, (_ast.LtE, _ast.Lt)))
                    elif b_param and a_const:
                        _bounds_from(T.var(b_.id), T.const(a_.value),
                                     isinstance(o_, (_ast.Gt, _ast.Lt)),
                                     upper=isinstance(o_, (_ast.GtE, _ast.Gt)))
                    else:
                        pairs_ok = False        # const-const or unknown names
            if middles_ok and pairs_ok:
                obs.append(dict(kind="premise", line=st.lineno,
                                claim=ast.unparse(st.test)[:80],
                                premise=ast.unparse(st.test)[:80],
                                conclusion="declared", method="premise", status="ok"))
                continue
            oof, poison_line = True, st.lineno
            break
        claim = ast.unparse(st.test)[:80]
        op = st.test.ops[0]

        # --- nullness rung: `x is None` / `x is not None` -------------------
        # Exactness: nullness of a LITERAL is evaluation (total, decided
        # here); nullness of a free parameter is an input contract (declared,
        # never proved -- `_proved`/the theorem guard demote it); nullness of
        # anything COMPUTED abstains, because SQL NULL propagates through
        # expressions and inferring it would guess.
        if isinstance(op, (_ast.Is, _ast.IsNot)):
            _UNSET = object()
            _lnode, _rnode = st.test.left, st.test.comparators[0]

            def _none_lit(n):
                return isinstance(n, _ast.Constant) and n.value is None

            # refutation gate: a declared premise (`b == 5` recorded, an
            # R1 solve) may CERTIFY downstream claims by implication, but may
            # never REFUTE one -- a premise may not spend itself with the
            # opposite sign (the linear discipline, selfcheck #25).
            # Per-node: the `None` side of a nullness compare has no term
            # representation, and one bad side must not erase the other
            # side's variable set (that reopened the refute-by-declaration
            # path on `b == 5; b is None`).
            def _raw_vars(*_nodes):
                _out = set()
                for _n in _nodes:
                    try:
                        _out |= T.vars_of(_ast2term(_n))
                    except Exception:
                        continue
                return _out

            _declared_hit = bool(
                _raw_vars(_lnode, _rnode) & (_declared_vars | set(eq_subst)))

            _l_nn, _r_nn = _none_lit(_lnode), _none_lit(_rnode)
            _val, _declared = _UNSET, False

            def _norm_lit(v):
                # term-algebra constants may be Fraction (R2 solves); store a
                # plain literal so the artifact repr re-parses in the kernel.
                if hasattr(v, "denominator"):
                    return int(v) if v.denominator == 1 else float(v)
                return v

            if _l_nn and _r_nn:
                _val = None                          # None is (not) None
            elif _l_nn or _r_nn:
                _other = _rnode if _l_nn else _lnode
                if isinstance(_other, _ast.Constant) and isinstance(
                        _other.value, (int, str, bool, float)):
                    _val = _other.value
                else:
                    try:
                        _t = slice_expr(_other)
                        if _t[0] == "c":
                            _val = _norm_lit(_t[1])
                        elif not (T.vars_of(_t) - params) and isinstance(
                                _other, _ast.Name) and _other.id in params:
                            _declared = True         # declared input contract
                    except ValueError:
                        _val = _UNSET
            else:
                undecided = True
                obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                                premise=claim, conclusion="?",
                                method="out_of_fragment",
                                detail="identity (is) outside the fragment",
                                status="abstain"))
                continue
            if _declared:
                obs.append(dict(kind="premise", line=st.lineno, claim=claim,
                                premise=claim, conclusion="declared (nullness)",
                                method="premise", status="ok"))
                continue
            if _val is _UNSET:
                undecided = True
                obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                                premise=claim, conclusion="?",
                                method="out_of_fragment",
                                detail="nullness not statically decided",
                                status="abstain"))
                continue
            _holds = (_val is not None) if isinstance(op, _ast.IsNot) \
                else (_val is None)
            if _holds:
                obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                                premise=repr(_val), conclusion="not null",
                                method="nullness", lhs=repr(_val),
                                status="ok"))
            elif _declared_hit:
                # contradicted only by a DECLARED premise: record the
                # conflict, never a refutation (refute-by-declaration).
                undecided = True
                obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                                premise=repr(_val), conclusion="?",
                                method="out_of_fragment",
                                detail="nullness contradicts a declared "
                                       "premise, not the program",
                                status="abstain"))
            else:
                obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                                premise=repr(_val), conclusion="null",
                                method="nullness", lhs=repr(_val),
                                status="fail",
                                witness=dict(line=st.lineno, claim=claim)))
                refuted = True
            continue

        # --- NotEq: the algebra decides equality, so it decides its negation
        # too. This op used to be SKIPPED at the whitelist below, so
        # `not (a == b)` (lowered to `a != b`) produced ZERO obligations and
        # the unit could certify while that claim was never examined -- the
        # third false-certificate path of the negation defect.
        if isinstance(op, _ast.NotEq):
            def _raw_vars2(*_nodes):
                _out = set()
                for _n in _nodes:
                    try:
                        _out |= T.vars_of(_ast2term(_n))
                    except Exception:
                        continue
                return _out

            # same linear discipline as nullness: identity reached through a
            # DECLARED premise (`a == b` recorded, an R1 solve) may not refute
            # -- that is refutation by declaration (selfcheck #25). Program
            # assignments (env_terms) are execution semantics and may.
            _declared_ne = bool(_raw_vars2(st.test.left,
                                           st.test.comparators[0])
                                & (_declared_vars | set(eq_subst)))
            try:
                _l, _r = slice_expr(st.test.left), \
                    slice_expr(st.test.comparators[0])
                if (T.vars_of(_l) - params) or (T.vars_of(_r) - params):
                    raise ValueError("unresolved_slice")
            except ValueError as _e:
                oof = True
                obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                                premise=claim, conclusion="?",
                                method="out_of_fragment", detail=str(_e),
                                status="abstain"))
                continue
            _ne = T.decide_eq(_l, _r)
            _closed = not (T.vars_of(_l) or T.vars_of(_r))
            if _ne is True and not _declared_ne:      # l == r identically
                obs.append(dict(kind="eq", line=st.lineno, claim=claim,
                                premise=T.tstr(_r), conclusion=T.tstr(_l),
                                method="nf+decide_ne", status="fail",
                                witness=dict(line=st.lineno, claim=claim)))
                refuted = True
                continue
            if _ne is True:
                undecided = True
                obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                                premise=claim, conclusion="?",
                                method="out_of_fragment",
                                detail="contradicts a declared premise, not "
                                       "the program",
                                status="abstain"))
                continue
            if _ne is False and _closed:
                # distinct CLOSED values: exact, never a guess -- `l != r`
                # holds. With free variables, decide_eq's False only means
                # "not identically equal" (x != 5 fails at x = 5), so it can
                # never discharge a universal inequality.
                obs.append(dict(kind="eq", line=st.lineno, claim=claim,
                                premise=T.tstr(_r), conclusion=T.tstr(_l),
                                method="nf+decide_ne", status="ok"))
                continue
            if _ne is False:
                # exact universal inequality: the difference normalizes to a
                # NONZERO CONSTANT (x != x + 1 -- difference is -1), which
                # never vanishes. A difference that still carries a variable
                # can be zero somewhere (x != 5 at x = 5) and must not certify.
                _diff = T.nf(T.csub(_l, _r))
                if _diff is not None and set(_diff.keys()) == {()} \
                        and _diff.get((), 0) != 0:
                    obs.append(dict(kind="eq", line=st.lineno, claim=claim,
                                    premise=T.tstr(_r),
                                    conclusion=T.tstr(_l),
                                    method="nf+decide_ne", status="ok"))
                    continue
            # undecidable over free parameters: an input contract the runtime
            # enforces -- declared, inert, never fed into a later proof.
            obs.append(dict(kind="premise", line=st.lineno, claim=claim,
                            premise=claim,
                            conclusion="declared (negated equality)",
                            method="premise", status="ok"))
            continue

        if not isinstance(op, (_ast.Eq, _ast.LtE, _ast.GtE, _ast.Lt, _ast.Gt)):
            undecided = True
            obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                            premise=claim, conclusion="?",
                            method="out_of_fragment",
                            detail=f"operator {type(op).__name__} outside "
                                   f"the fragment",
                            status="abstain"))
            continue
        # A premise assert that fed the R2 linear system is a DECLARED assumption:
        # record it as a premise obligation and do NOT slice/decide it under its own
        # solution (that would launder the premise into a hidden tautology and hide
        # it from the conditional/information filter). Later claims still consume the
        # substitution.
        if st.lineno in _r2_lines:
            obs.append(dict(kind="premise", line=st.lineno, claim=claim,
                            premise=claim, conclusion="declared (linear premise)",
                            method="premise", status="ok"))
            continue
        try:
            l, r = slice_expr(st.test.left), slice_expr(st.test.comparators[0])
            if (T.vars_of(l) - params) or (T.vars_of(r) - params):
                raise ValueError("unresolved_slice")
        except ValueError as e:
            oof = True
            obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                            premise=claim, conclusion="?", method="out_of_fragment",
                            detail=str(e), status="abstain"))
            continue
        # E1: declared premises from earlier discharged asserts (bounds)
        def _const_side(t):
            return t[1] if t[0] == "c" else None
        # A claim that reads a computed local is a claim about the computation,
        # not a declared input premise: it must be PROVEN or REFUTED, never
        # assumed as a premise. (Assuming a false claim as a premise and emitting
        # a chain step produced spurious FAULTs instead of REFUTED.)
        _assigned = set(env_terms.keys())
        _claim_reads_local = any(isinstance(n, ast.Name) and n.id in _assigned
                                 for n in ast.walk(st.test))
        # The unit's last assert is the CLAIM: it may not register itself as a
        # premise (E-mod residue, R1 chain, E1b value) — it must be decided below.
        _is_last = (st.lineno == _last_assert)
        # E-mod: modular claims — residue premises, residue-class lemmas,
        # fail-closed corner refutations
        if isinstance(op, ast.Eq):
            _lv, _rv = _vars_of_side(l), _vars_of_side(r)
            if _lv <= params and _rv <= params:
                # premise-shape first: `param % m == const` = RESIDUE PREMISE
                # (runtime-enforced input contract; feeds the residue chain)
                if not _claim_reads_local and not _is_last and l[0] == "mod" \
                        and l[1][0] == "v" and l[1][1] in params \
                        and r[0] == "c" and isinstance(r[1], int) \
                        and isinstance(l[2][1], int) and l[2][1] > 0:
                    _residues[l[1][1]] = (r[1] % l[2][1], l[2][1])
                    obs.append(dict(kind="premise", line=st.lineno, claim=claim,
                                    premise=claim, conclusion="declared",
                                    method="premise", status="ok"))
                    continue
                _mv = _mod_claim(l, r, "eq")
                if _mv is not None:
                    verdict_m, reason_m = _mv
                    if verdict_m is True:
                        obs.append(dict(kind="eq", line=st.lineno, claim=claim,
                                        premise=claim, conclusion="verified",
                                        method=f"mod:{reason_m}", status="ok"))
                        # single-var mod eq: residue premise for the chain
                        if l[0] == "mod" and l[1][0] == "v" and len(_lv) == 1:
                            _residues[next(iter(_lv))] = (r[1], l[2][1])
                        continue
                    if verdict_m is False:
                        obs.append(dict(kind="eq", line=st.lineno, claim=claim,
                                        premise=claim, conclusion="violated at box corner",
                                        method=f"mod:{reason_m}", status="fail",
                                        witness=dict(line=st.lineno, claim=claim)))
                        refuted = True
                        continue
                    undecided = True
                    obs.append(dict(kind="eq", line=st.lineno, claim=claim,
                                    premise=claim, conclusion="mod claim undecidable",
                                    method=f"mod:{reason_m}", status="abstain"))
                    continue

        # R1: compound Eq over params -> try the linear-chain solve first.
        # Claims reading a computed local are excluded: they are claims, not
        # premises, and must reach the decide_eq proof/refute path below.
        if not _claim_reads_local and not _is_last and isinstance(op, ast.Eq):
            _lv, _rv = _vars_of_side(l), _vars_of_side(r)
            if _lv and _rv and _lv <= params and _rv <= params and (
                    _param_vs_const(l, r) is False and _param_vs_const(r, l) is False):
                sol = _eq_solve(l, r)
                if sol is not None:
                    var_, term_ = sol
                    eq_subst[var_] = term_
                    obs.append(dict(kind="premise", line=st.lineno, claim=claim,
                                    premise=claim, conclusion=f"{var_} = {T.tstr(term_)}",
                                    method="premise", status="ok"))
                    continue

        # E5: interval route — floats present or interval premises declared.
        # Exact rational box arithmetic; unknown vars/zero divisors abstain.
        _env = IV.env_from_bounds({**{v: lb for v, lb in facts.items()},
                                   **facts_iv})
        # exact-poly first: n*0.5 + n*0.5 == n is pure algebra (works for ALL n)
        _pv = T.decide_eq(l, r)
        if _pv is True:
            obs.append(dict(kind="eq", line=st.lineno, claim=claim,
                            premise=T.tstr(r), conclusion=T.tstr(l),
                            method="nf+decide_eq", status="ok"))
            continue
        if _uses_floats(l) or _uses_floats(r) or any(
                isinstance(b, tuple) or isinstance(b, float)
                for b in facts.values()):
            _REL = {type(op).__name__: {"LtE": "le", "GtE": "ge", "Lt": "lt",
                                        "Gt": "gt", "Eq": "eq"}[type(op).__name__]}
            try:
                v = IV.decide_rel(l, r, _REL[type(op).__name__], _env)
                kind = "interval"
                rel = f"interval:{_REL[type(op).__name__]}"
            except ValueError as e:
                v = None
                kind, rel = "interval", f"interval_out_of_model:{e}"
            if v is True:
                obs.append(dict(kind=kind, line=st.lineno, claim=claim,
                                premise=claim, conclusion="holds over the declared box",
                                method=rel, bounds=dict(facts), status="ok"))
                continue
            if v is False:
                obs.append(dict(kind=kind, line=st.lineno, claim=claim,
                                premise=claim, conclusion="violated",
                                method=rel, bounds=dict(facts), status="fail",
                                witness=dict(line=st.lineno, claim=claim,
                                             box={k: [str(a), str(b)] for k, (a, b) in _env.items()})))
                refuted = True
                continue
            undecided = True
            obs.append(dict(kind=kind, line=st.lineno, claim=claim,
                            premise=claim, conclusion="box does not decide",
                            method=rel, bounds=dict(facts), status="abstain"))
            continue

        # E1: Name-vs-constant comparisons are DECLARED PREMISES, not theorems:
        # the runtime assert enforces them, so the certificate certifies the
        # implication "if this assert holds at runtime, then the rest holds".
        # They are recorded as premise obligations and feed facts downstream.
        def _name_vs_const(a, b):
            return a[0] == "v" and b[0] == "c" and a[1] in params

        # TIER-GATED (2026-10-06): `not _is_last` protects the THEOREM
        # reading — the last assert is the claim and must be decided, never
        # assumed (the A-combobound lesson, restated below). A premise-tier
        # unit is the opposite statement: its last assert IS the declaration
        # (`premise scale == 2`), and refusing to declare it made every
        # equality contract decline as `premise_refuted` while inequalities
        # declared — half the DSL's premise feature, asymmetric. The reading
        # keeps this safe: a premise obligation can only ever surface as
        # `declared`, never as a proof (certify_claim never counts it for a
        # theorem), and the shape still requires a PARAM against a CONST —
        # a closed falsehood (`3 == 5`) and computed locals stay out.
        if isinstance(op, ast.Eq) and (not _is_last or tier == "premise") \
                and (_param_vs_const(l, r) or _param_vs_const(r, l)):
            # E1b: declared equality premise (runtime-enforced); feeds exact value
            if _param_vs_const(r, l):
                l, r = r, l
            vals[l[1]] = r[1]
            _declared_vars.add(l[1])
            obs.append(dict(kind="premise", line=st.lineno, claim=claim,
                            premise=claim, conclusion="declared",
                            method="premise", status="ok"))
            _declared_premises.append((st.lineno, claim))
            continue
        if isinstance(op, ast.Eq):
            # A claim may be refuted when a DECLARED premise contradicts it.
            # The policy lives in classification.json (theory.
            # premise_contradiction_verdict); this site supplies facts and
            # obeys. The rule ships DISABLED with its measurement attached, so
            # today it decides nothing and behaviour is unchanged — the path is
            # here so that enabling it is a table edit, not a code change.
            _contra = _check_premise_contradiction(claim, obs)
            if _contra is not None:
                _v, _reason, _meta = _contra
                if _v == "REFUTED":
                    obs.append(dict(kind="eq", line=st.lineno, claim=claim,
                                    premise=claim,
                                    conclusion=f"contradicts declared premise "
                                               f"({_meta['premise']})",
                                    method=f"premise_contradiction:{_reason}",
                                    premise_obligation=_meta["premise"],
                                    lhs=_meta["lhs"], rhs_claim=_meta["rhs_claim"],
                                    rhs_premise=_meta["rhs_premise"],
                                    status="fail",
                                    witness=dict(line=st.lineno, claim=claim)))
                    refuted = True
                    continue
        if isinstance(op, ast.Eq):
            dep_l, dep_r = TR.atom(T.tstr(l), T.nf(l)), TR.atom(T.tstr(r), T.nf(r))
            v, kind, rel = T.decide_eq(l, r), "eq", "decide_eq"
            TR.rule("decide-eq", 1, (T.tstr(l), T.tstr(r)), v,
                    deps=(dep_l, dep_r))
            # Premises restrict the claim domain: a full-domain refutation is
            # not evidence on the restricted domain. Exact-constant disagreement
            # stays REFUTED; otherwise abstain when facts are in play.
            if v is False and facts and not (l[0] == "c" and r[0] == "c"):
                v, rel = None, "decide_eq+premises"
            # Opaque atoms (clen/elem/fact) are RUNTIME values: disagreement is
            # a runtime claim, not a theorem -> abstain (never refute).
            if v is False and (_has_atom(l) or _has_atom(r)):
                v, rel = None, "decide_eq+atoms"
        elif not _is_last \
                and isinstance(op, (_ast.LtE, _ast.GtE, _ast.Lt, _ast.Gt)) \
                and (_premise_shape(l, r, _OP_NAMES[type(op)])
                     or _relational_premise(l, r)):
            # E1c premise: a bound over PARAMETERS as a PRECONDITION (not the
            # last assert). Never REFUTED — it IS an input contract — and it
            # feeds bounds facts downstream.
            #
            # `not _is_last` is load-bearing, and its absence was the
            # A-combobound defect. The last assert is the CLAIM (see the note
            # above `_is_last`), and it must be DECIDED, never assumed. Without
            # the guard, a contract over a computed local falls in here as soon
            # as the local is substituted: `result <= 50` with `result = n`
            # becomes `n <= 50`, which matches the premise shape, so the claim
            # was recorded as a second declared premise — obligations
            # ['premise','premise'], nothing proved, verdict UNVERIFIABLE
            # ("assumed_not_proved"), and 6 such vacuous certificates landed in
            # the corrective teaching set as targets. The Eq branch above has
            # always carried this guard; the inequality branch did not.
            if l[0] == "v" and r[0] == "c":
                _bounds_from(l, r, isinstance(op, (_ast.Gt, _ast.Lt)),
                             upper=isinstance(op, (_ast.LtE, _ast.Lt)))
            elif r[0] == "v" and l[0] == "c":
                _bounds_from(r, l, isinstance(op, (_ast.Gt, _ast.Lt)),
                             upper=isinstance(op, (_ast.GtE, _ast.Gt)))
            else:
                _bounds_from(l, r, False)   # param-param: no const bound yet
            # Record the premise as a DIFFERENCE POLYNOMIAL. `a <= b` is not a
            # bound on either variable — it is the fact `b - a >= 0`, a
            # direction in the space of linear forms. facts_iv cannot hold
            # that (it is per-variable), which is exactly why a param-param
            # premise used to be invisible to the box. Keeping the polynomial
            # lets a later claim be proved as a NON-NEGATIVE COMBINATION of
            # premise differences, which is what actually discharges
            # `lo <= hi |- lo + x <= hi + x`.
            _pd = _premise_poly(l, r, _OP_REL.get(type(op)))
            if _pd is not None:
                _premise_diffs.append(_pd)      # live list; obs carries the
                                               # JSON-safe copy for the artifact
            # Carry the box this premise established. A certificate that does
            # not record what the verifier needs is a defective certificate:
            # without it, `rel:rel-box` claims derived from a SUBSTITUTED bound
            # (`result = n`, so `result <= 5` becomes `n <= 5`) record
            # bounds=None, and the kernel has nothing to re-decide with — so the
            # main arithmetic discharge was unverified while still scoring
            # agreement 1.0. JSON-safe: infinities become None (unbounded).
            _bnd = {}
            for _v, _b in facts_iv.items():
                _lo, _hi = _b
                _bnd[_v] = [None if _lo == -IV.INF else _lo,
                            None if _hi == IV.INF else _hi]
            obs.append(dict(kind="premise", line=st.lineno, claim=claim,
                            premise=claim, conclusion="declared",
                            method="premise", status="ok",
                            bounds=_bnd or None,
                            **({"difference": {" ".join(k): str(c)
                                                   for k, c in sorted(_pd.items())}}
                               if _pd is not None else {})))
            continue
        else:
            sign = _OP_REL.get(type(op))
            if sign is None:
                oof = True
                obs.append(dict(kind="vc", line=st.lineno, claim=claim,
                                premise=claim, conclusion="?",
                                method="out_of_fragment", status="abstain"))
                continue
            # E-quad: univariate quadratic sign analysis (exact, unbounded)
            _qd = None
            try:
                _qd = T._quad_sign(T.nf(T.csub(r, l)) if True else None)
            except Exception:
                _qd = None
            if _qd == 'pos' and not facts:
                obs.append(dict(kind="ineq", line=st.lineno, claim=claim,
                                premise=claim, conclusion="non-negative for all x (disc <= 0)",
                                method="quad:disc-sign", status="ok"))
                continue
            if _qd == 'neg' and not facts:
                obs.append(dict(kind="ineq", line=st.lineno, claim=claim,
                                premise=claim, conclusion="non-positive for all x (disc <= 0)",
                                method="quad:disc-sign", status="ok"))
                continue
            # A DECLARED premise that the box cannot represent must still
            # forbid refutation. `lo <= hi` is recorded as a premise but
            # contributes NO entry to `facts` (param-param yields no const
            # bound), so `facts` being empty did NOT mean "no domain
            # restriction" — it meant "a restriction we cannot represent".
            # Refuting there produced a FALSE REFUTATION with a witness that
            # violates the premise: `lo <= hi` together with the true theorem
            # `lo + x <= hi + x` was reported as violated, at lo=1, hi=0.
            # Same class as the A-combobound root fix: a contract the checker
            # cannot see must never be treated as absent.
            _premise_seen = any(o.get("method") == "premise" for o in obs)
            # ...and the precise question is not "are there premises" but
            # "did the BOX capture them". `_lin_box_decide` reads `facts_iv`,
            # which does record a premise like `n <= 3` — that refutation is
            # sound and carries a valid witness. A param-param premise like
            # `lo <= hi` yields NO bound in `facts_iv` (`_bounds_from` needs a
            # constant), so the box is unrestricted while the contract is not.
            _word = __import__("re").compile(r"[A-Za-z_]\w*")
            _unrepresented = False
            for _o in obs:
                if _o.get("method") != "premise":
                    continue
                for _v in set(_word.findall(str(_o.get("premise", "")))):
                    if _v not in facts_iv:
                        _unrepresented = True
                        break
                if _unrepresented:
                    break
            # E-premise: a claim discharged from the DECLARED premises as a
            # non-negative integer combination of their difference polys. Tried
            # BEFORE the box, because the box cannot represent a param-param
            # premise and would otherwise either refute (unsound) or abstain.
            # This is the maximal non-vacuous discharge: it proves a real
            # theorem from a real contract, and it proves nothing that was not
            # written down.
            _pdpre = _premise_diffs
            if _pdpre:
                _hi, _lo = (l, r) if sign in ("ge", "gt") else (r, l)
                _cdiff = _poly_of(T.csub(_hi, _lo))
                _combo = _conic_combo(_cdiff, _pdpre)
                if _combo is not None:
                    _strict = sign in ("gt", "lt")
                    if _strict:
                        # `a > b` needs a STRICT step the integer polynomials
                        # cannot supply from a non-strict premise alone, so a
                        # strict claim is only proved when its difference is
                        # non-negative on its own (already >= 0 is not enough
                        # for >); fall through to the sound paths.
                        if not (_cdiff is not None and _is_nonneg_poly(_cdiff)
                                and any(v == 0 for v in _cdiff.values())
                                and all(c > 0 for k, c in _cdiff.items()
                                        if any(x not in (0,) for x in k))):
                            _combo = None
                    if _combo is not None:
                        obs.append(dict(kind="ineq", line=st.lineno, claim=claim,
                                        premise=claim,
                                        conclusion="follows from declared "
                                                   "premises",
                                        lhs=T.tstr(l), rhs=T.tstr(r),
                                        method=f"premise_derive:{sign}",
                                        premise_combination=_combo,
                                        premise_differences=[
                                            {" ".join(k): str(c)
                                             for k, c in sorted(d.items())}
                                            for d in _pdpre],
                                        bounds=dict(facts_iv) or None,
                                        status="ok"))
                        continue
            # E-rel: exact linear box disposal first (corners are exact)
            _lin = None
            try:
                _lin = _lin_box_decide(l, r, _OP_NAMES[type(op)])
            except Exception:
                _lin = None
            if _lin is not None:
                v_lin, why = _lin
                # `lhs`/`rhs` record the CANONICAL, POST-SUBSTITUTION sides the
                # prover decided on (e.g. `n >= 12`, not the source claim
                # `result >= 12`). They are computed here and would otherwise be
                # discarded, leaving a refutation with no recoverable input-space
                # countermodel — which is why 348 refuted rows carried none. The
                # sides are recorded as a PROPOSAL: countermodel.py still has to
                # re-decide the claim at any point it emits from them.
                # The box this discharge used is RECORDED, so the kernel can
                # re-decide it. Without it the kernel had no `rel:` branch at
                # all: `bounds` was None whenever the box came from program
                # substitution (`result = n`), so the main arithmetic discharge
                # returned None and scored agreement 1.0 while being verified
                # nowhere. This mirrors how `interval:`/`sign_analysis:` already
                # carry their bounds, and the premises that produced them are
                # separately recorded as obligations.
                _used_box = {v: [None if b[0] == -IV.INF else b[0],
                                 None if b[1] == IV.INF else b[1]]
                             for v, b in facts_iv.items()}
                if v_lin is True:
                    obs.append(dict(kind="ineq", line=st.lineno, claim=claim,
                                    premise=claim, conclusion="holds over the declared box",
                                    lhs=T.tstr(l), rhs=T.tstr(r),
                                    bounds=_used_box or None,
                                    method=f"rel:{why}:{_OP_REL[type(op)]}",
                                    status="ok"))
                    continue
                if v_lin is False and not (facts or _unrepresented):
                    obs.append(dict(kind="ineq", line=st.lineno, claim=claim,
                                    premise=claim, conclusion="violated over the box",
                                    lhs=T.tstr(l), rhs=T.tstr(r),
                                    bounds=_used_box or None,
                                    method=f"rel:{why}:{_OP_REL[type(op)]}",
                                    status="fail",
                                    witness=dict(line=st.lineno, claim=claim)))
                    refuted = True
                    continue
                if v_lin is False and (facts or _premise_seen):
                    pass   # premises restrict the domain: full-domain refute
                           # invalid — unless the box actually captured them,
                           # in which case the refute stands and the witness is
                           # a real point in the restricted domain
            # sound direction; REFUTED only when the NEGATION is provable
            ok, neg = {
                "le": (lambda: T.provably_ge(r, l, facts),
                       lambda: T.provably_gt(l, r, facts)),
                "ge": (lambda: T.provably_ge(l, r, facts),
                       lambda: T.provably_lt(l, r, facts)),
                "lt": (lambda: T.provably_gt(r, l, facts),
                       lambda: T.provably_ge(l, r, facts)),
                "gt": (lambda: T.provably_gt(l, r, facts),
                       lambda: T.provably_ge(r, l, facts)),
            }[sign]
            raw = True if ok() else (False if neg() else None)
            if raw is True:
                v, abstain_reason = True, ""
            else:
                verdict, abstain_reason = TH.dispose_claim(
                    decide_eq=raw,                # True/False/None: table distinguishes
                    facts_present=bool(facts) or _premise_seen,
                    atoms_present=_has_atom(l) or _has_atom(r))
                v = {"CERTIFIED": True, "REFUTED": False}.get(verdict)
            kind, rel = "ineq", f"sign_analysis:{sign}"
            TR.rule(f"sign-{sign}", 1, (T.tstr(l), T.tstr(r), dict(facts)), v)
        bnd = dict(facts) if facts else None
        if v is True:
            obs.append(dict(kind=kind, line=st.lineno, claim=claim,
                            premise=T.tstr(r), conclusion=T.tstr(l),
                            method=rel, bounds=dict(facts_iv) or bnd, status="ok"))
            # E1: discharged Name-vs-constant comparisons become sound premises
            if l[0] == "v" and r[0] == "c":
                _bounds_from(l, r, isinstance(op, (_ast.Gt, _ast.Lt)),
                             upper=isinstance(op, (_ast.LtE, _ast.Lt)))
            elif r[0] == "v" and l[0] == "c":
                _bounds_from(r, l, isinstance(op, (_ast.Gt, _ast.Lt)),
                             upper=isinstance(op, (_ast.GtE, _ast.Gt)))
        elif v is False:
            obs.append(dict(kind=kind, line=st.lineno, claim=claim,
                            premise=T.tstr(r), conclusion=T.tstr(l),
                            method=rel, bounds=dict(facts_iv) or bnd, status="fail",
                            witness=dict(line=st.lineno, claim=claim)))
            refuted = True
        else:
            undecided = True
            obs.append(dict(kind=kind, line=st.lineno, claim=claim,
                            premise=T.tstr(r), conclusion=T.tstr(l),
                            method=rel, bounds=dict(facts_iv) or bnd, status="abstain"))
    trace = TR.finish()
    # Poison-aware premise-only units: if the body has NO claim-shaped asserts
    # at all (every assert is a declared premise), the poison never affected a
    # claim -> the unit certifies on its premises.
    if oof and obs and all(o.get("kind") == "premise" for o in obs):
        has_claim = False
        for n2 in ast.walk(fn):
            if isinstance(n2, ast.Assert) and isinstance(n2.test, ast.Compare):
                if len(n2.test.ops) > 1:
                    continue                   # chained premises: not claims
                try:
                    _l = _ast2term(n2.test.left, ctx)
                    _r = _ast2term(n2.test.comparators[0], ctx)
                except ValueError:
                    has_claim = True
                    break
                _opn2 = _OP_NAMES.get(type(n2.test.ops[0]), "")
                if not (_premise_shape(_l, _r, _opn2)
                        and isinstance(n2.test.ops[0],
                                       (_ast.LtE, _ast.GtE, _ast.Lt, _ast.Gt))):
                    has_claim = True
                    break
        if not has_claim:
            oof = False
    if not obs:
        return dict(source="assert", confirmed=True, status="UNVERIFIABLE",
                    obligations=[], trace=trace,
                    reason="out_of_fragment" if oof else "no_assert_spec",
                    location=dict(line=poison_line) if poison_line else None)
    if refuted:
        status, reason = "REFUTED", "assert_refuted"
    elif oof:
        status, reason = "UNVERIFIABLE", "out_of_fragment"
    elif undecided:
        status, reason = "UNVERIFIABLE", "undecidable_assert"
    else:
        status, reason = "CERTIFIED", ""
    return dict(source="assert", confirmed=True, status=status,
                obligations=obs, trace=trace, reason=reason,
                trace_digest=TR.digest(trace),
                # certificate/v0 declares a `witness` field; the AST path used
                # to leave it absent, so every REFUTED artifact was
                # non-conformant with its own schema. Populated ONLY on
                # REFUTED, and only with a point the trusted decision
                # procedure re-decided as false. Read-only w.r.t. verdicts.
                witness=CM.build(obs, status, reason, params=params))


import ast as _ast
_OP_NAMES = {getattr(_ast, k): v for k, v in
             [("GtE", "ge"), ("Gt", "gt"), ("LtE", "le"), ("Lt", "lt"), ("Eq", "eq")]}
_OP_MAP = {_ast.Add: "add", _ast.Sub: "sub", _ast.Mult: "mul",
           _ast.Div: "rdiv", _ast.FloorDiv: "div", _ast.Mod: "mod"}
# rel names the PROVABLE direction on (l, r): le means provably_ge(r, l), etc.
_OP_REL = {_ast.LtE: "le", _ast.GtE: "ge", _ast.Lt: "lt", _ast.Gt: "gt"}


def _has_atom(t):
    """True iff the term contains opaque runtime atoms (clen/elem/fact)."""
    if not isinstance(t, tuple):
        return False
    if t[0] in ("clen", "elem", "fact"):
        return True
    return any(_has_atom(x) for x in t[1:] if isinstance(x, tuple))


# ----------------------------------------------------- E4: loop summaries ----
def _mono_term(mon, coef):
    """Rebuild a monomial term: coef * product(vars)."""
    t = T.const(coef)
    for v in mon:
        t = T.cmul(t, T.var(v))
    return t


def _poly_in(expr_term, loop_var):
    """Decompose expr into loop-invariant monomial terms + [c0, c1, c2]
    coefficients of the loop-var polynomial (degree <= 2). None = reject."""
    d = T.nf(expr_term)
    if d is None:
        return None
    c, inv = [0, 0, 0], []
    for mon, coef in d.items():
        k = mon.count(loop_var)
        if k > 2:
            return None
        if k == 0:
            inv.append(_mono_term(mon, coef))     # loop-invariant term
        else:
            rest = tuple(v for v in mon if v != loop_var)
            if rest:
                return None                # mixed outer*loopvar: MVF reject
            c[k] += coef
    return inv, c


def _faulhaber(inv, c, N):
    """Exact closed form of sum_{i=0}^{N-1} expr(i) where
    expr = inv_terms + c0 + c1*i + c2*i^2:
    invariant terms contribute term*N ; S0 = N ; S1 = N(N-1)/2 ;
    S2 = N(N-1)(2N-1)/6 (all exact for integers)."""
    S0 = N
    S1 = T.cdiv(T.cmul(N, T.csub(N, T.const(1))), T.const(2))
    S2 = T.cdiv(T.cmul(T.cmul(N, T.csub(N, T.const(1))),
                       T.csub(T.cmul(T.const(2), N), T.const(1))), T.const(6))
    t = T.const(0)
    deps = []
    from . import trace as TR
    for it in inv:
        t = T.cadd(t, T.cmul(it, S0))
    if c[0]: t = T.cadd(t, T.cmul(T.const(c[0]), S0))
    if c[1]:
        deps.append(TR.lemma("faulhaber-s1", 1, (N,), S1))
        t = T.cadd(t, T.cmul(T.const(c[1]), S1))
    if c[2]:
        deps.append(TR.lemma("faulhaber-s2", 1, (N,), S2))
        t = T.cadd(t, T.cmul(T.const(c[2]), S2))
    return t


def _loop_summary(for_node):
    """Summarize `for v in range(N): <pure += updates>` into exact env updates.
    Raises ValueError on anything outside the MVF (nested loops, ifs, breaks,
    overwrites, non-polynomial bodies)."""
    import ast as _a
    tgt = for_node.target
    if not (isinstance(tgt, _a.Name) and isinstance(for_node.iter, _a.Call)
            and isinstance(for_node.iter.func, _a.Name)
            and for_node.iter.func.id == "range"
            and len(for_node.iter.args) == 1 and not for_node.iter.keywords):
        raise ValueError("loop_out_of_mvf")
    lv = tgt.id
    N = _ast2term(for_node.iter.args[0])
    updates = {}
    for st in for_node.body:
        if isinstance(st, _a.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], _a.Name):
            name, val = st.targets[0].id, st.value
            if isinstance(val, _a.BinOp) and isinstance(val.op, _a.Add) \
                    and isinstance(val.left, _a.Name) and val.left.id == name:
                updates.setdefault(name, []).append(_ast2term(val.right))
            else:
                raise ValueError("loop_overwrite")
        elif isinstance(st, _a.AugAssign) and isinstance(st.target, _a.Name) \
                and isinstance(st.op, _a.Add):
            updates.setdefault(st.target.id, []).append(_ast2term(st.value))
        else:
            raise ValueError("loop_body_out_of_mvf")
    out = {}
    for name, deltas in updates.items():
        acc = T.const(0)
        for dterm in deltas:
            dec = _poly_in(dterm, lv)
            if dec is None:
                raise ValueError("loop_delta_not_poly2")
            inv, c = dec
            acc = T.cadd(acc, _faulhaber(inv, c, N))
        out[name] = acc
    return out


# ----------------------------------------------------- coder-facing CTA map ---
_CTA = {
    "": "claim discharged: attach certificate.json to CI; strengthen trivial-tier claims",
    "assert_refuted": "paste the witness as a failing test — real bug, repro attached",
    "unconfirmed_spec": "add an assert / test / type hints: provenance is missing, not proof",
    "no_assert_spec": "no claims found: add asserts to make this unit checkable",
    "out_of_semantics_model": "move floats/exceptions/mutation out of the claimed path, or accept abstain",
    "out_of_fragment": "inline the callee / hoist the loop / quantify via a parameter",
    "undecidable_assert": "add a domain premise (e.g. assert x >= 0 first), restate, or record UNKNOWN",
    "no_counter_update": "loop counter never moves: no variant can exist — restructure the loop",
    "variant_not_discharged": "variant does not decrease: prove a different ranking or restructure",
    "obligation_failed": "an obligation failed: fix the program or weaken the claim",
    "undischarged": "hypothesis not discharged: fix the base/step or extend the theory",
}


def cta_for(status: str, reason: str, witness=None) -> str:
    """Actionable next step for a verdict/reason — shared by check and ledgers.

    The REFUTED CTA is conditional on an actual countermodel. Measured on the
    stored corpus, only 22 of 243 REFUTED artifacts carry a point witness, so
    the unconditional "repro attached" promise was false 91% of the time.
    """
    if status == "CERTIFIED":
        return _CTA[""]
    if status == "REFUTED":
        kind = (witness or {}).get("kind") if isinstance(witness, dict) else None
        if kind == "point":
            env = ", ".join(f"{k}={v}" for k, v in
                            sorted((witness.get("env") or {}).items()))
            return (f"a counterexample is attached: at ({env}) the claim is "
                    f"provably false — run it as a failing test")
        if kind == "region":
            return ("claim is violated somewhere in the recorded region; "
                    "no single input extracted yet")
        if kind == "inconsistency":
            return ("premises contradict: no falsifying input exists by "
                    "construction — fix the premises")
        return ("claim is refuted but no counterexample was extracted "
                "(the violation is over a computed value, not an input) — "
                "read the obligation; do not assume a repro exists")
    r = reason.split("(")[0]
    return _CTA.get(r, "record UNKNOWN with this reason in the ledger")


def assert_spec(code: str):
    """Spec acquisition from {assert, type_hint} only (provenance gate input).

    An assert comparing two arithmetic expressions yields a CONFIRMED Eq spec
    (source=assert). Type hints alone never produce an expr. Returns
    (spec dict, spec_term or None) — abstains honestly otherwise.
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare) \
                and len(n.test.ops) == 1 and isinstance(n.test.ops[0], ast.Eq):
            try:
                return ({"expr": None, "source": "assert", "confirmed": True},
                        ("eq", _ast2term(n.test.left), _ast2term(n.test.comparators[0])))
            except ValueError:
                return ({"expr": None, "source": "assert", "confirmed": True}, None)
    args = getattr(tree.body[0], "args", None) if tree.body else None
    fn = next((x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)), None)
    if fn is not None and all(a.annotation for a in fn.args.args) and fn.returns:
        return ({"expr": None, "source": "type_hint", "confirmed": True}, None)
    return ({"expr": None, "source": "inferred", "confirmed": False}, None)
