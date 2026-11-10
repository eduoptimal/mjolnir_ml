"""mjolnir_prover.lib.policy — the Tier-1 policy language (claims file).

The LEAST segmentation that covers the most: ONE section, FOUR keys, ONE
expression grammar. You write CLAIMS (specific, disposable); the engine
derives the POLICY (the general form you sign in shape_ui — the waterfall
tested in mjolnir_experiments/shape_ui). Same candidate schema as the SQL /
Python / dsl frontends, so collapse -> confirm -> transfer is unchanged.

SPEC (locked 2026-10-07)
-------------------------
    claims:                        # the only root section
      - name: TI_CASH.AMOUNT       # subject, declared ONCE (dotted = scope)
        is: "!= null and >= 0"     # property only; subject omitted
      - name: currency_amt
        is: "money(scale=2, digits=10)"    # definition
      - name: credit_rule
        when: "grace_days >= 0"            # conditional: when may be a list
        then: "grace_days + 30 >= 30"      #   (YAML list = and-chain)

Grammar rules, all fail-closed with a line number:
  * keys: name / is / when / then (+ version at root). Anything else: error.
  * `is` starts with an operator or is a type call; the subject is expanded
    into every conjunct. Repeating the subject is an error.
  * banned everywhere: `or`, `not`, parentheses outside len(...) / sum(...).
  * numbers: integers; decimals only when binary-exact (the authority's own
    numeric-domain check — one source of truth).
  * functions: len(x), sum(a.b, ...) — sum/enum route to the OBSERVATION
    venue automatically; venues are derived, never authored.
  * when without then (or the reverse): error.

Compiles to EXACTLY the mined-candidate schema (constructs dsl_* so policies
collapse across tiers). Dispositions come from the single authority at
compile time (printed as the round-trip), never from this module.
"""
from __future__ import annotations

import ast
import re
import zlib

from . import knowledge as KN
from . import dsl as _dsl

_ROOT_KEYS = {"version", "claims"}
_ITEM_KEYS = {"name", "is", "when", "then"}
_CALLS = ("money", "varchar", "enum")
_NUM = re.compile(r"-?\d+\.\d+")
_WORD = r"(?<![A-Za-z0-9_])(?:%s)(?![A-Za-z0-9_])"
_BANNED = {"or": re.compile(_WORD % "or"),
           "not": re.compile(_WORD % "not")}


class PolicyError(ValueError):
    """A compile error with a line number: refuse, never guess."""


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _unquote(v: str, line: int, what: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    raise PolicyError(f"line {line}: {what} must be a quoted string: {v!r}")


def _check_expr(expr: str, line: int) -> None:
    """Banned syntax + exact numbers + parses. One function for is/when/then."""
    masked = re.sub(r"(len|sum)\s*\([^()]*\)", "call", expr)
    for m in ("(", ")"):
        if m in masked:
            raise PolicyError(f"line {line}: parentheses are Tier-2 only: {expr!r}")
    for word, rx in _BANNED.items():
        if rx.search(expr):
            raise PolicyError(f"line {line}: `{word}` is not in the grammar "
                              f"(domains use enum(...); flip the operator "
                              f"instead of `not`): {expr!r}")
    for lit in _NUM.findall(expr):
        if not KN.exactly_representable(lit):
            raise PolicyError(
                f"line {line}: {lit} is not exactly representable — use an "
                f"integer (or an exact binary decimal like .5): {expr!r}")
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise PolicyError(f"line {line}: bad expression {expr!r}: {e}") from None


def _parse(text: str, source: str) -> list:
    """Strict subset of YAML -> raw items [{name, is|when/then, line}]."""
    items, root, cur, pending_when = [], None, None, False
    for i, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        s = line.strip()
        if indent == 0:
            m = re.fullmatch(r"(\w+)\s*:\s*(.*)", s)
            if not m:
                raise PolicyError(f"{source}:{i}: not a directive: {s!r}")
            k, v = m.group(1), m.group(2).strip()
            if k not in _ROOT_KEYS:
                raise PolicyError(f"{source}:{i}: unknown root key {k!r} "
                                  f"(expected: claims, version)")
            if k == "claims":
                if v:
                    raise PolicyError(f"{source}:{i}: `claims:` opens a list")
                root = "claims"
            continue
        if root != "claims":
            raise PolicyError(f"{source}:{i}: content before `claims:`")
        if indent == 2 and s.startswith("- "):
            cur, pending_when = {"line": i}, False
            items.append(cur)
            s = s[2:].strip()
        elif indent >= 4 and cur is None:
            raise PolicyError(f"{source}:{i}: indented line before any item")
        # now: a key line (possibly a block-list member of `when:`)
        if pending_when and s.startswith("- "):
            cur.setdefault("when_list", []).append(
                _unquote(s[2:], i, "when"))
            continue
        m = re.fullmatch(r"(name|is|when|then)\s*:\s*(.*)", s)
        if not m:
            raise PolicyError(f"{source}:{i}: unknown key in item: {s!r} "
                              f"(expected: name, is, when, then)")
        k, v = m.group(1), m.group(2).strip()
        pending_when = False
        if k in cur:
            raise PolicyError(f"{source}:{i}: duplicate key {k!r}")
        if k == "name":
            cur[k] = _unquote(v, i, "name") if v[:1] in "'\"" else v
        elif k in ("when", "then"):
            if not v:
                pending_when = (k == "when")
                cur[k + "_block"] = True
                continue
            if v.startswith("["):
                if k != "when":
                    raise PolicyError(f"{source}:{i}: only `when` takes a list")
                inner = v[1:-1] if v.endswith("]") else None
                if inner is None:
                    raise PolicyError(f"{source}:{i}: unterminated [ list")
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                cur["when_list"] = [_unquote(p, i, "when member")
                                    for p in parts]
            else:
                cur[k] = _unquote(v, i, k)
        else:  # is
            cur[k] = _unquote(v, i, "is")
    if root != "claims":
        raise PolicyError(f"{source}: missing root key `claims:`")
    for n, it in enumerate(items, 1):
        if "name" not in it:
            raise PolicyError(f"{source}: item {n}: missing `name`")
        has_is = "is" in it
        has_rule = ("when" in it or "when_list" in it or "when_block" in it)
        if has_is and has_rule:
            raise PolicyError(f"{source}: item {n} ({it['name']}): `is` "
                              f"cannot mix with when/then")
        if has_rule and "then" not in it:
            raise PolicyError(f"{source}: item {n} ({it['name']}): when "
                              f"requires then")
        if "then" in it and not has_rule:
            raise PolicyError(f"{source}: item {n} ({it['name']}): then "
                              f"requires when")
        if not has_is and not has_rule:
            raise PolicyError(f"{source}: item {n} ({it['name']}): needs "
                              f"`is` or when/then")
    return items




def _nulls(expr: str) -> str:
    """`!= null` / `== null` are the nullness rung's spelling: the unit must
    say `is not None` / `is None` (the keyword `null` is not a Python name —
    measured: it became a free parameter and the claim declined). The
    predicate keeps `!= null`: it is the human-facing form."""
    e = re.sub(r"!=\s*null\b", "is not None", expr)
    e = re.sub(r"==\s*null\b", "is None", e)
    if re.search(r"(?<![A-Za-z0-9_])null(?![A-Za-z0-9_])", e):
        raise PolicyError(f"`null` is only valid with == or !=: {expr!r}")
    return e

def _safe(name: str) -> str:
    """Dotted scope (TI_CASH.AMOUNT) is display truth; the unit needs a real
    identifier — an Attribute node is outside the fragment (measured:
    out_of_fragment). Aliased only inside the unit; predicates/shapes keep
    the dotted form."""
    return re.sub(r"\W", "_", name)


def _safe_expr(expr: str, names: list) -> str:
    for n in sorted(names, key=len, reverse=True):
        if "." in n:
            expr = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(n) +
                          r"(?![A-Za-z0-9_])", _safe(n), expr)
    return expr

def _sid(repo: str, source: str, line: int, text: str) -> str:
    return "pol::" + format(zlib.crc32(
        f"{repo}|{source}|{line}|{text}".encode()) & 0xffffffff, "08x")


def _base(cand_id, tier, family, pred, unit, construct, origin):
    return dict(id=cand_id, family=family, tier=tier, predicate=pred,
                premise=None, canonical_unit=unit, expect=None,
                annotation="policy",
                origin=dict(origin, construct=construct),
                provenance=dict(source="mined", confirmed=False))


def compile_text(text: str, source: str = "claims.yaml",
                 repo: str = "policy") -> list:
    """Policy text -> registry candidates (the exact mined schema)."""
    out = []
    for it in _parse(text, source):
        name = it["name"]
        raw_line = it.get("line") or 1
        origin = dict(repo=repo, commit="", file=source, line=raw_line)
        sid = _sid(repo, source, raw_line, name + str(sorted(it)))
        if "is" in it:
            prop = it["is"].strip()
            # type call (definition)
            cm = re.fullmatch(r"(%s)\s*\((.*)\)" % "|".join(_CALLS),
                              prop, re.S)
            if cm:
                fn, args = cm.group(1), cm.group(2)
                if fn == "money":
                    am = re.fullmatch(r"scale\s*=\s*(\d+)\s*,\s*"
                                      r"digits\s*=\s*(\d+)", args.strip())
                    if not am:
                        raise PolicyError(
                            f"{source}:{raw_line}: money(scale=K, digits=D)")
                    scale, digits = int(am.group(1)), int(am.group(2))
                    e = digits - scale
                    if not (0 <= e <= 60):
                        raise PolicyError(
                            f"{source}:{raw_line}: digits must exceed scale")
                    bound = "1" + "0" * e
                    fnm = _dsl._fnname(None, sid)
                    sn = _safe(name)
                    unit = _dsl._unit(
                        [f"{sn} < {bound}", f"{sn} > -{bound}"], fnm)
                    out.append(_base(
                        sid, "premise", "premise",
                        f"range {name} |v| < 10^{e}", unit,
                        "dsl_premise", origin))
                elif fn == "varchar":
                    if not re.fullmatch(r"\d+", args.strip()):
                        raise PolicyError(
                            f"{source}:{raw_line}: varchar(N) needs an integer")
                    n = int(args.strip())
                    fnm = _dsl._fnname(None, sid)
                    unit = _dsl._unit([f"len({_safe(name)}) <= {n}"], fnm)
                    out.append(_base(
                        sid, "premise", "premise",
                        f"range {name} len <= {n}", unit, "dsl_premise",
                        origin))
                else:  # enum -> observation venue, derived not authored
                    vals = [v.strip() for v in args.split(",") if v.strip()]
                    if len(vals) < 2 or any(
                            not (len(v) >= 2 and v[0] == v[-1]
                                 and v[0] in "'\"")
                            for v in vals):
                        raise PolicyError(
                            f"{source}:{raw_line}: enum('A', 'B') with "
                            f"quoted values")
                    out.append(_base(
                        sid, "instance", "vc",
                        f"enum {name} in {{{','.join(vals)}}}", None,
                        "dsl_claim", origin))
                continue
            # operator property: expand the subject into every conjunct
            conjuncts = [c.strip() for c in
                         re.split(r"\s+and\s+", prop)]
            for c in conjuncts:
                if not c:
                    raise PolicyError(
                        f"{source}:{raw_line}: empty conjunct in {prop!r}")
                if not re.match(r"(==|!=|>=|<=|>|<)", c):
                    raise PolicyError(
                        f"{source}:{raw_line}: `is` must start with an "
                        f"operator or be a type call: {c!r}")
                if re.search(_WORD % re.escape(name), c) or (
                        "." in name and
                        re.search(_WORD % re.escape(name.split(".")[-1]), c)):
                    raise PolicyError(
                        f"{source}:{raw_line}: subject {name!r} repeats in "
                        f"`is` (it is declared by `name:`) — {c!r}")
                _check_expr(f"{name} {c}", raw_line)
            full = [f"{name} {c}" for c in conjuncts]
            unit_full = [_nulls(_safe_expr(f, [name])) for f in full]
            # UNIT ORDER IS NORMALIZED (display keeps the author's order):
            # the engine's prefix pass consumes leading premise-shaped
            # bounds and stops at the first other statement, so a nullness
            # assert in front of a bound would hide the bound from the
            # declared-premise path (measured: `!= null and >= 0` declined
            # as undecidable while `>= 0 and != null` certified). Bounds
            # first, nullness last — the conjunction is order-independent.
            unit_full = ([f for f in unit_full
                          if " is not None" not in f and " is None" not in f]
                         + [f for f in unit_full
                            if " is not None" in f or " is None" in f])
            if any("sum(" in f for f in full):
                # aggregate: the table settles it -> observation venue
                out.append(_base(
                    sid, "instance", "vc", " and ".join(full), None,
                    "dsl_claim", origin))
                continue
            fnm = _dsl._fnname(None, sid)
            unit = _dsl._unit(unit_full, fnm)
            out.append(_base(
                sid, "premise", "premise", " and ".join(full), unit,
                "dsl_premise", origin))
            continue
        # rule: when(+list) + then
        whens = list(it.get("when_list") or ([it["when"]]
                                             if "when" in it else []))
        if not whens:
            raise PolicyError(
                f"{source}:{raw_line}: when has no conditions")
        then = it["then"]
        for e in whens + [then]:
            _check_expr(e, raw_line)
        dotted = [name] + [x for x in (name, *whens, then) if "." in x]
        exprs = [_nulls(_safe_expr(e, dotted)) for e in whens] + \
            [_nulls(_safe_expr(then, dotted))]
        fnm = _dsl._fnname(name, sid)
        unit = _dsl._unit(exprs, fnm)
        out.append(dict(
            id=sid, family="vc", tier="static", predicate=then,
            premise="; ".join(whens),
            canonical_unit=unit, expect=None, annotation="policy",
            origin=dict(origin, construct="dsl_rule"),
            provenance=dict(source="mined", confirmed=False)))
    return out


def compile_file(path: str, repo: str | None = None) -> list:
    src = str(path)
    name = repo or src.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    with open(src, encoding="utf-8") as fh:
        return compile_text(fh.read(), source=src, repo=name)
