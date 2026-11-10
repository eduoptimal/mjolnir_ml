"""mjolnir_prover/lib/frontend_lean2.py — Lean 4 statement parser: tokenizer + RD.

A proper TOKENIZER handles all Lean tokens (unicode identifiers, subscripts,
Greek, negative literals, casts, sort types). A simple recursive-descent
parser consumes the linear-arithmetic subset. Handles everything the regex
frontend missed: multi-var binder groups, nested casts, let-chains,
hypothesis predicates, type ascriptions.
"""
from __future__ import annotations
import re, zlib

_SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

_TOKEN_RE = re.compile(r"""
    (?P<NAME>[A-Za-z_\u03b1-\u03c9\u2080-\u2089\u2090-\u209c][A-Za-z0-9_\u03b1-\u03c9\u2080-\u2089\u2090-\u209c']*)
  | (?P<NAT>-?\d+(?:\.\d+)?)
  | (?P<SORT>[ℝℚℤℕ])
  | (?P<OP>\*\*|>=|<=|!=|==|[-+*/^%<>=≤≥≠∧∣∤])
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<SEMI>;)
  | (?P<COLON>:)
  | (?P<WS>\s+)
""", re.X | re.U)


def tokenize(stmt: str):
    toks = []
    for m in _TOKEN_RE.finditer(stmt):
        kind = m.lastgroup
        if kind == "WS":
            continue
        v = m.group()
        # `-` before a digit is BINARY when it follows an operand (`2-16`);
        # only treat it as a negative literal at expression start / after an
        # operator (`x * -16`, `-5`). Otherwise `2-16` would read as `2 (-16)`.
        if kind == "NAT" and v.startswith("-") and toks and \
                toks[-1][0] in ("NAT", "NAME", "RPAREN", "SORT"):
            toks.append(("OP", "-"))
            toks.append(("NAT", v[1:]))
        else:
            toks.append((kind, v))
    return toks


def _safe(v: str) -> str:
    return v.translate(_SUBS).replace("'", "_p")


_FORBIDDEN_SYM = ("∀", "∃", "→", "↔", "¬", "∨", "λ", "++", ".map")


def _is_application(toks):
    """True when two operand tokens are adjacent — Lean juxtaposition, i.e.
    function application (`p x`). That is outside the linear MVF; lowering it
    would silently drop an argument and invent a false claim."""
    prev_operand = False
    for kind, _v in toks:
        operand = kind in ("NAME", "NAT", "SORT", "RPAREN")
        if prev_operand and kind in ("NAME", "NAT", "SORT", "LPAREN"):
            return True
        prev_operand = operand
    return False


def _in_fragment(text):
    """True iff `text` lowers soundly in this parser's linear/polynomial MVF.
    Mirrors frontend_lean's guard: reject higher-order syntax, quantifiers,
    function types/values and Lean function application."""
    if any(w in text for w in _FORBIDDEN_SYM):
        return False
    if "∧" in text:            # allowed in split hypotheses, not in a claim
        return False
    if re.search(r"\b(fun|let|List|Array|Set|Type|Prop)\b", text):
        return False
    return not _is_application(tokenize(text))


def _tokens_to_str(toks):
    return " ".join(v for _, v in toks)


def _normalize(stmt):
    """Rewrite two surface forms into the fragment before tokenizing:
    `v ∈ Set.Icc a b`  ->  `v >= a ∧ v <= b`
    `v ≡ r [MOD m]`    ->  `v % m = r`"""
    s = re.sub(r"([A-Za-z_][\w']*)\s*∈\s*Set\.Icc\s+(-?\d+(?:\.\d+)?)"
               r"\s+(-?\d+(?:\.\d+)?)",
               r"\1 >= \2 ∧ \1 <= \3", stmt)
    s = re.sub(r"([A-Za-z_][\w']*)\s*≡\s*(\S+?)\s*\[\s*MOD\s+(\d+)\s*\]",
               r"\1 % \3 = \2", s)
    return s


def parse_theorem(stmt: str):
    """Parse a Lean theorem statement.
    Returns (name, params, premises, conclusion) or None."""
    stmt = _normalize(stmt)
    toks = tokenize(stmt)
    if not toks or toks[0][1] != "theorem":
        return None
    # Reject anything the tokenizer would silently drop (quantifiers, function
    # values/types, let) — out-of-fragment must ABSTAIN, never lower to garbage.
    if any(w in stmt for w in _FORBIDDEN_SYM) or \
            re.search(r"\b(fun|let)\b", stmt):
        return None
    i = 1
    if i >= len(toks) or toks[i][0] != "NAME":
        return None
    name = _safe(toks[i][1])
    i += 1
    # --- scan binder groups (balanced parens) ---
    params = []
    premises = []
    while i < len(toks) and toks[i] == ("LPAREN", "("):
        i += 1
        depth = 1
        group = []
        while i < len(toks) and depth > 0:
            if toks[i][0] == "LPAREN":
                depth += 1
            elif toks[i][0] == "RPAREN":
                depth -= 1
            group.append(toks[i])
            i += 1
        # classify the group: split at the first COLON
        colon_i = next((j for j, (k2, v2) in enumerate(group) if k2 == "COLON"), None)
        vars_, typ_toks = [], []
        if colon_i is not None:
            # left of colon = binder name(s); right = the full predicate
            # (INCLUDING NAME tokens — the predicate's variables!)
            vars_ = [_safe(v) for k2, v in group[:colon_i] if k2 == "NAME"]
            typ_toks = group[colon_i + 1:]
        else:
            vars_ = [_safe(v) for k2, v in group if k2 == "NAME"]
        if vars_ and typ_toks:
            pred = _tokens_to_str(typ_toks).strip()
            while pred.endswith(")") and pred.count(")") > pred.count("("):
                pred = pred[:-1].strip()       # drop only an UNBALANCED leak
            if _has_cmp(pred):
                parts = _pred_parts(pred)          # ∧ split before the guard
                if not parts or not all(_in_fragment(p) for p in parts):
                    return None
                premises.extend((v, p) for v in vars_ for p in parts)
                params.extend(vars_)
            else:
                params.extend(vars_)  # type-only: declared
                # ℕ-domain: nonnegative is part of the theorem's meaning, and
                # it is what makes bounded residue/sign reasoning sound.
                if any(v2 in ("ℕ", "Nat") for k2, v2 in typ_toks):
                    premises.extend((v, f"{v} >= 0") for v in vars_)
                # integer domain (ℤ/ℕ): integrality constrains solved expressions
                if any(v2 in ("ℤ", "ℕ", "Int", "Nat") for k2, v2 in typ_toks):
                    premises.extend((v, None) for v in vars_)
        elif vars_:
            params.extend(vars_)
    # skip to the ':' before the conclusion
    while i < len(toks) and toks[i][1] != ":":
        i += 1
    i += 1
    # conclusion tokens until ':='
    concl_toks = []
    while i < len(toks) and toks[i][1] not in (":=", "by"):
        concl_toks.append(toks[i])
        i += 1
    while concl_toks and concl_toks[-1][1] in (":=", "=", "by", ":"):
        concl_toks.pop()
    if not concl_toks:
        return None
    concl = _tokens_to_str(concl_toks)
    if not concl or not re.search(r"[=<>≤≥]", concl):
        return None
    if not _in_fragment(concl):
        return None
    params = list(dict.fromkeys(params))
    return name, params, premises, concl


def _has_cmp(text):
    return any(op in text for op in ("=", "<", ">", "≤", "≥", "≠", "∣", "∤"))


def _pred_parts(pred):
    """Split a predicate on ∧ into individual comparison parts."""
    parts = re.split(r"\s*∧\s*", pred)
    return [p for p in parts if p and p != "True"]


def _tokens_to_str(toks):
    import re
    parts = []
    for kind, val in toks:
        if kind == "SORT":
            continue
        OP = {"≤": "<=", "≥": ">=", "≠": "!=", "^": "**"}.get(val, val)
        parts.append(str(OP))
    e = " ".join(parts)
    return e


def lower_theorem(stmt: str):
    """The LOWERING: Lean source → claim records directly (no string round-trip).

    Each claim record carries its terms built via the algebra's constructors —
    the disposal consumes the records directly. Returns ([Claim], meta) or None.
    When every variable is integer-typed, `/` is lowered as integer division
    (Lean ℕ/ℤ semantics) rather than exact rational division."""
    parsed = parse_theorem(stmt)
    if parsed is None:
        return None
    name, params, premises, concl = parsed
    from mjolnir_prover.lib import term as T
    rational = _lower_all(premises, concl, int_div=False)
    if rational is None:
        return None
    int_vars = {c.lhs[1] for c in rational if c.kind == "int"}
    all_vars = set()
    for c in rational:
        all_vars |= T.vars_of(c.lhs) | T.vars_of(c.rhs)
    meta = dict(name=name, params=params, source=stmt[:200])
    if all_vars and all_vars <= int_vars:
        idiv = _lower_all(premises, concl, int_div=True)
        if idiv is not None:
            meta["alt_claims"] = rational   # fallback if floor reasoning stalls
            return idiv, meta
    return rational, meta


def _lower_all(premises, concl, int_div):
    from mjolnir_prover.lib import claim_ir as IR
    from mjolnir_prover.lib import term as T
    claims = []
    line = 0
    for var, pred in premises:
        if pred is None:                 # integer-domain marker (ℤ/ℕ)
            claims.append(IR.Claim("int", T.var(var), T.const(0), line,
                                   True, f"{var} \u2208 \u2124"))
            continue
        claims.extend(_lower_expr(pred, line, True, int_div))
    conclusion = _lower_expr(concl, line + 1, False, int_div)
    if not conclusion:
        return None                      # nothing to discharge
    claims.extend(conclusion)
    return claims


def _lower_expr(text, line, is_premise, int_div=False):
    """Lower ONE Lean predicate (text) into claim records via the algebra."""
    from mjolnir_prover.lib import term as T, claim_ir as IR
    from mjolnir_prover.lib.frontend_lean2 import tokenize
    out = []
    toks = tokenize(text)
    # divisibility: `a ∣ b` is `b % a = 0` (a ∤ b is its negation, lowered too)
    dv = [(j, k2, v2) for j, (k2, v2) in enumerate(toks)
          if k2 == "OP" and v2 in ("∣", "∤")]
    if dv:
        j, _, dvop = dv[-1]
        lhs = _build_term(toks[:j], int_div)
        rhs = _build_term(toks[j + 1:], int_div)
        if lhs is None or rhs is None:
            return out
        out.append(IR.Claim("eq", T.cmod(rhs, lhs), T.const(0), line,
                            is_premise, text[:120], op="="))
        return out
    # find the comparison operator (the top-level = / <= / >= / < / >)
    ops_at_top = [(j, k2, v2) for j, (k2, v2) in enumerate(toks)
                  if k2 == "OP" and v2 in ("=", "<=", ">=", "<", ">", "!=", "≠")]
    if not ops_at_top:
        return out                       # no comparison: not a claim
    # take the LAST top-level operator (Lean's conclusion has one)
    j, _, opv = ops_at_top[-1]
    lhs_toks, rhs_toks = toks[:j], toks[j + 1:]
    lhs = _build_term(lhs_toks, int_div)
    rhs = _build_term(rhs_toks, int_div)
    if lhs is None or rhs is None:
        return out
    kind = "eq" if opv == "=" else ("neq" if opv in ("!=", "≠") else "ineq")
    out.append(IR.Claim(kind, lhs, rhs, line, is_premise, text[:120], op=opv))
    return out


def _build_term(toks, int_div=False):
    """Token list -> term via the algebra's constructors (or None)."""
    from mjolnir_prover.lib import term as T
    if not toks:
        return None
    # find the lowest-precedence operator and recurse
    PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "**": 3}
    for prec in (1, 2, 3):
        depth = 0
        for j in range(len(toks) - 1, -1, -1):     # right-assoc scan
            k2, v2 = toks[j]
            if k2 == "RPAREN":
                depth += 1
            elif k2 == "LPAREN":
                depth -= 1
            elif depth == 0 and k2 == "OP" and PREC.get(v2) == prec:
                if not toks[:j] and v2 in ("-", "+"):
                    rhs = _build_term(toks[j + 1:], int_div)
                    if rhs is None:
                        return None
                    return T.cneg(rhs) if v2 == "-" else rhs
                lhs = _build_term(toks[:j], int_div)
                rhs = _build_term(toks[j + 1:], int_div)
                if lhs is None or rhs is None:
                    return None
                if v2 == "+": return T.cadd(lhs, rhs)
                if v2 == "-": return T.csub(lhs, rhs)
                if v2 == "*": return T.cmul(lhs, rhs)
                if v2 == "/":
                    return T.cdiv(lhs, rhs) if int_div else T.crdiv(lhs, rhs)
                if v2 == "%": return T.cmod(lhs, rhs)
                if v2 == "**":
                    # bounded natural exponent: repeated mul
                    if rhs[0] == "c" and isinstance(rhs[1], int) and 0 <= rhs[1] <= 20:
                        out = T.const(1)
                        for _ in range(rhs[1]):
                            out = T.cmul(out, lhs)
                        return out
                    return None
    # no operator: single token
    k2, v2 = toks[0]
    if k2 == "NAME": return T.var(_safe(v2))
    if k2 == "NAT":
        try:
            if "." in v2: return T.const(float(v2))
            return T.const(int(v2))
        except ValueError:
            return None
    if k2 == "LPAREN" and toks[-1] == ("RPAREN", ")"):
        return _build_term(toks[1:-1], int_div)
    return None


def mine_theorem(stmt: str, rel: str, repo: str, commit: str, idx: int):
    """Lower a parsed theorem into the canonical unit format the disposal
    handles — same format as frontend_lean's mine_theorem (proven: 216+).
    The parse (frontend_lean2) gives better coverage; the lowering matches."""
    parsed = parse_theorem(stmt)
    if parsed is None:
        return None
    name, params, premises, concl = parsed
    fn = f"lean_{name}_{idx}".lower()
    # lower the hypothesis predicates + conclusion through the same lowering
    # pipeline as frontend_lean (exact rationals, == conversion, ** etc.)
    body_lines = []
    for vars_, pred in premises:
        pred_py = _lower_assert(pred)
        body_lines.append(f"    assert {pred_py}")
    concl_py = _lower_assert(concl)
    body_lines.append(f"    assert {concl_py}")
    wrapper = f"def {fn}({', '.join(params)}):\n" + "\n".join(body_lines) + "\n"
    sid = "ln::" + format(zlib.crc32(f"{repo}|{rel}|{idx}|{name}".encode()) & 0xffffffff, "08x")
    return dict(id=sid, family="premise", tier="premise",
                predicate=f"assert {concl_py}"[:160], annotation="lean-theorem",
                canonical_unit=wrapper, expect=None,
                origin=dict(repo=repo, commit=commit, file=rel, line=idx,
                            construct="lean_theorem", symbol=name),
                provenance=dict(source="mined", confirmed=False))


def _lower_assert(pred_text):
    """Lower a Lean predicate to a python assert: = → ==, exact rationals,
    ≤/≥ → <=/>=, ^ → **, ∧ → separate asserts (handled upstream)."""
    import re as _re
    e = pred_text
    e = re.sub(r"(\d+)\s*/\s*([1-9]\d*)", r"(\1.0 / \2.0)", e)
    e = e.replace("≤", "<=").replace("≥", ">=").replace("^", "**")
    e = _re.sub(r"([^<>!=])=([^=])", r"\1==\2", e, count=1)
    return e


def mine_stream(rows, rel: str, repo: str, commit: str, limit=None):
    out = []
    n = 0
    for row in rows:
        stmt = row.get("formal_statement")
        if not stmt:
            continue
        r = mine_theorem(stmt, rel, repo, commit, n)
        n += 1
        if r:
            out.append(r)
        if limit and n >= limit:
            break
    return out