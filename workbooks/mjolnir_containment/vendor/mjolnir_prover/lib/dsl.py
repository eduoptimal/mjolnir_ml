"""mjolnir_prover.lib.dsl — a small domain-rule DSL that compiles to registry candidates.

WHY
---
Authoring rules directly in the candidate JSON (id/family/tier/predicate/
canonical_unit/origin/provenance) is tedious and error-prone. This DSL is the
syntactic sugar: it compiles to EXACTLY that format, so everything downstream
(collapse -> confirm -> typed transfer -> report) is unchanged.

GRAMMAR (line-oriented; `#` comments; blank lines ignored)
---------------------------------------------------------
    premise <expr>                     declared contract over typed symbols
    rule <name>: <premise> => <conclusion>   conditional theorem (premise |- conclusion)
    claim  <name>: <expr>              unconditional claim (may be filtered as vacuous)

`<name>` is optional (a stable id is derived if omitted). `<expr>` is a Python
expression over its free names, e.g. `amount >= 0`, `v * m == k`.

COMPILES TO
-----------
    premise -> tier=premise, predicate=<expr>,   canonical_unit = def f(...): assert <expr>
    rule    -> tier=static,  predicate=<conclusion>,
                             canonical_unit = def f(...): assert <premise>; assert <conclusion>
    claim   -> tier=static,  predicate=<expr>,   canonical_unit = def f(...): assert <expr>

The unit of record keeps the PREMISE, which is what lets the typed transfer
certify a conditional fact (informative) instead of a vacuous identity.
"""
from __future__ import annotations

import ast
import os
import re
import zlib

from . import certify

_KEYWORDS = {"assert", "and", "or", "not", "in", "is", "if", "else", "True",
             "False", "None", "lambda"}


def _names(expr: str) -> list:
    return certify._param_names(expr)


def _fnname(name: str | None, sid: str) -> str:
    base = re.sub(r"\W+", "_", (name or "")).strip("_").lower()
    if not base:
        base = "r_" + sid[5:13]
    if base[0].isdigit():
        base = "r_" + base
    return base


def _unit(exprs: list, fn: str) -> str:
    vars_ = []
    for e in exprs:
        for v in _names(e):
            if v not in vars_:
                vars_.append(v)
    body = "".join(f"    assert {e}\n" for e in exprs)
    return f"def {fn}({', '.join(vars_)}):\n{body}"


def compile_text(text: str, source: str = "rules.dsl", repo: str = "domain_rules"):
    """DSL text -> list of registry candidates (the exact mined schema)."""
    out = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        sid = "dom::" + format(zlib.crc32(f"{repo}|{source}|{i}|{line}".encode())
                               & 0xffffffff, "08x")
        m = re.match(r"(premise|rule|claim)\b\s*(.*)$", line, re.I)
        if not m:
            raise ValueError(f"{source}:{i}: not a directive: {line!r}")
        kind, rest = m.group(1).lower(), m.group(2).strip()
        name = None
        if kind in ("rule", "claim"):
            nm = re.match(r"([A-Za-z_]\w*)\s*:\s*(.*)$", rest)
            if not nm:
                raise ValueError(f"{source}:{i}: `{kind} <name>: ...` required")
            name, rest = nm.group(1), nm.group(2)
        if kind == "premise":
            exprs, tier, family, construct, pred = [rest], "premise", "premise", "dsl_premise", rest
        elif kind == "rule":
            parts = re.split(r"=>", rest)
            if len(parts) != 2:
                raise ValueError(f"{source}:{i}: rule needs `<premise> => <conclusion>`")
            p, c = (x.strip() for x in parts)
            # MULTI-PREMISE: separate premises with `;` -> each becomes its own assert
            # (the VC consumes each bound; a single `and` premise is NOT consumed).
            premises = [x.strip() for x in p.split(";") if x.strip()]
            if not premises:
                raise ValueError(f"{source}:{i}: rule needs at least one premise")
            exprs = premises + [c]
            tier, family, construct, pred = "static", "vc", "dsl_rule", c
        else:
            exprs, tier, family, construct, pred = [rest], "static", "vc", "dsl_claim", rest
        for e in exprs:                       # validate each expression parses
            try:
                ast.parse(e, mode="eval")
            except SyntaxError as ex:
                raise ValueError(f"{source}:{i}: bad expression {e!r}: {ex}")
        fn = _fnname(name, sid)
        out.append(dict(
            id=sid, family=family, tier=tier, predicate=pred,
            premise=("; ".join(exprs[:-1]) if kind == "rule" else None),
            annotation=f"dsl-{kind}", canonical_unit=_unit(exprs, fn), expect=None,
            origin=dict(repo=repo, commit="", file=source, line=i, construct=construct),
            provenance=dict(source="dsl", confirmed=False)))
    return out


def compile_file(path: str, repo: str | None = None):
    src = os.path.basename(path)
    name = repo or os.path.splitext(src)[0]
    with open(path, encoding="utf-8") as f:
        return compile_text(f.read(), source=src, repo=name)
