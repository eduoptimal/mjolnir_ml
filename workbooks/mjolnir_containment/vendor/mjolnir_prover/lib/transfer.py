"""mjolnir_prover.lib.transfer — Typed Shape Transfer (TST): the generalized domain
transfer protocol.

WHY THIS EXISTS
---------------
The earlier transfer substituted a rule's variables with the target's *raw tokens*
(positional), which produced junk like `INSERT > 0`. Reliability requires a TYPED
target environment and a type-preserving substitution. This module is that
protocol; `shapes.transfer_generate` remains the untyped primitive.

THE PROTOCOL (5 steps)
----------------------
1. RULE abstraction  : a confirmed member predicate -> (template, var sorts).
   sorts are inferred syntactically: arithmetic/`**` -> Num ; string compare -> Str.
2. DOMAIN model      : the target's TYPED environment = sites of sorted symbols
   (SQL: tables with typed columns; Python/Java: functions with typed params).
3. SUBSTITUTION σ    : a type-preserving, injective map of rule vars -> site
   symbols (Num->Num, Str->Str). Deterministic, bounded. No token soup.
4. ADOPTION gate     : the existing authority gate, unchanged —
   static -> oracle (adopt iff 0 REFUTED and >=1 CERTIFIED);
   premise/monitor -> structural parse.
5. INFORMATION filter: a certified instance is materialised only if its proof was
   CONDITIONAL (used a premise/bound). Unconditional identities are vacuous and
   dropped. Premise-tier instances are *declared contracts* (assumptions), labelled.

HONEST BOUND (stated, not hidden)
---------------------------------
TST transfers TYPED SHAPES, never semantics. With no typed target environment it
abstains (it does not fall back to token soup). A carried fact is:
  kind=premise    -> a declared contract over real target symbols (an ASSUMPTION), or
  kind=certified  -> a CONDITIONAL theorem (holds under a premise about the target).
Soundness is unchanged: every instance is disposed by the single verdict authority;
0 false certificates; positive claims are kernel-rechecked.
"""
from __future__ import annotations

import itertools
import os
import re

from . import certify
from . import shapes

_NUM_OPS = re.compile(r"(\*\*|[+\-*/%])")
_STR_CMP = re.compile(r"(?:==|!=|in)\s*['\"]")
_SQL_NUM = {"NUMERIC", "DECIMAL", "NUMBER", "MONEY", "SMALLMONEY", "INT", "INTEGER",
            "BIGINT", "SMALLINT", "TINYINT", "FLOAT", "REAL", "DOUBLE"}
_SQL_STR = {"STRING", "VARCHAR", "VARCHAR2", "CHAR", "CHARACTER", "TEXT", "CLOB",
            "NCHAR", "NVARCHAR"}
_SQL_OTHER = {"DATE", "DATETIME", "TIMESTAMP", "TIME", "BOOLEAN", "BIT"}


def sort_of_sql(type_name: str) -> str:
    t = (type_name or "").upper()
    if t in _SQL_NUM:
        return "Num"
    if t in _SQL_STR:
        return "Str"
    return "Other"          # dates, booleans, unknown -> never a Num/Str slot


def var_sorts(raw: str, names: list) -> dict:
    """Syntactic sort inference for a rule's variables (Num | Str | Any):
       Num  — used in arithmetic (`+ - * / // % **`) OR compared to a number;
       Str  — compared/`in` against a string literal;
       Any  — otherwise (matches any target sort)."""
    out = {}
    for n in names:
        num = (re.search(rf"\b{n}\b\s*(?:\*\*|[+\-*/%])|(?:[+\-*/%])\s*\b{n}\b", raw)
               or re.search(rf"\b{n}\b\s*(?:>=|<=|==|!=|>|<)\s*[-+]?\d", raw)
               or re.search(rf"[-+]?\d\s*(?:>=|<=|==|!=|>|<)\s*\b{n}\b", raw))
        if num:
            out[n] = "Num"
        elif re.search(rf"\b{n}\b\s*(?:==|!=|in)\s*['\"]", raw):
            out[n] = "Str"
        else:
            out[n] = "Any"
    return out


# ------------------------------------------------------------- domain models --
def sql_sites(root: str, cap_files: int = 4000):
    """Typed SQL environment: one site per table, symbols = typed columns."""
    from . import frontend_sql
    sites = {}
    seen = 0
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for fn in files:
            if not fn.lower().endswith(".sql"):
                continue
            seen += 1
            if seen > cap_files:
                break
            try:
                ddl = open(os.path.join(dp, fn), encoding="utf-8", errors="ignore").read()
                cols = frontend_sql.parse_columns_all(ddl)
            except Exception:
                continue
            for tbl, col, typ, _p, _s, _nn in cols:
                sites.setdefault(tbl or "?", {})[col] = sort_of_sql(typ)
    return [(name, syms) for name, syms in sorted(sites.items()) if syms]


# --------------------------------------------------------- typed substitution --
def _assignments(meta, sorts, symbols, cap):
    """Deterministic, type-preserving, injective var->symbol assignments."""
    names = meta["vars"]
    pools = []
    for n in names:
        s = sorts.get(n, "Any")
        if s == "Num":                       # strict: Num slot <- Num symbol only
            pool = [x for x, sx in symbols.items() if sx == "Num"]
        elif s == "Str":                     # strict: Str slot <- Str symbol only
            pool = [x for x, sx in symbols.items() if sx == "Str"]
        else:                                # Any slot <- any symbol
            pool = list(symbols)
        pools.append(sorted(pool))
    if any(not p for p in pools):
        return []
    out = []
    for combo in itertools.product(*pools):
        if len(set(combo)) != len(combo):        # injective: distinct rule vars
            continue
        out.append(combo)
        if len(out) >= cap:
            break
    return out


def _unit_template(unit: str):
    """Template the asserts of a unit-of-record with SHARED variables across
    asserts, so a rule `premise => conclusion` substitutes σ consistently into
    both (the premise is preserved -> the fact certifies conditionally)."""
    import ast
    from . import shapes as _sh
    fn = next((n for n in ast.walk(ast.parse(unit)) if isinstance(n, ast.FunctionDef)), None)
    exprs = [ast.unparse(st.test) for st in (fn.body if fn else [])
             if isinstance(st, ast.Assert)]
    vars_, nums = [], []

    def rep(m):
        if m.group("id"):
            n = m.group("id")
            if n in _sh._KEYWORDS:
                return n
            if n not in vars_:
                vars_.append(n)
            return "{v%d}" % vars_.index(n)
        n = m.group("num")
        if n not in nums:
            nums.append(n)
        return "{n%d}" % nums.index(n)

    return ([_sh._TOK.sub(rep, e) for e in exprs], {"vars": vars_, "nums": nums},
            "\n".join(exprs))


def transfer_typed(from_cands, from_ledger, sites, oracle_fn,
                   cap_per_rule=256, info_filter=True, collect=True):
    """Run the TST protocol. Returns (records, facts) where facts are the
    materialisable rows: certified (conditional, if info_filter) + premises."""
    confirmed = shapes._confirmed(from_ledger)
    by_sid = {}
    for c in from_cands:
        by_sid.setdefault(shapes.shape_id(c), []).append(c)
    records, facts = [], []

    for sid in sorted(confirmed):
        rec = confirmed[sid]
        # Representative must come from THIS rule (the confirmed repos), not from
        # another repo that happens to share the normalized shape — otherwise the
        # variable sorts are taken from the wrong rule (e.g. Num vs Any) and the
        # substitution picks symbols of the wrong sort.
        allm = by_sid.get(sid, [])
        repos = set((rec.get("evidence") or {}).get("repos") or [])
        own = [m for m in allm if m["origin"].get("repo") in repos]
        members = own or allm
        if not members and rec.get("sample"):
            members = [{"predicate": rec["sample"],
                        "origin": {"construct": rec["construct"]}}]
        if not members:
            continue
        rep = min(members, key=lambda m: (m["origin"].get("repo", ""),
                                          m["origin"].get("file", ""),
                                          m["origin"].get("line", 0)))
        raw = rep["predicate"]
        cu = rep.get("canonical_unit")
        if cu:                                   # unit-of-record: keep premises
            tpls, meta, sort_src = _unit_template(cu)
        else:
            tpl, meta = shapes._template(raw)
            tpls, sort_src = [tpl], raw
        sorts = var_sorts(sort_src, meta["vars"])   # infer over the whole unit
        num_sub = {f"n{i}": meta["nums"][i] for i in range(len(meta["nums"]))}

        inst = []                                # (code, display, site)
        for site, symbols in sites:
            for combo in _assignments(meta, sorts, symbols, cap_per_rule):
                sub = dict(num_sub)
                sub.update({f"v{i}": combo[i] for i in range(len(combo))})
                parts = [shapes.instantiate(t, sub) for t in tpls]
                if any(p is None for p in parts):
                    continue
                code = (f"def f({', '.join(combo)}):\n"
                        + "".join(f"    assert {p}\n" for p in parts))
                inst.append((code, parts[-1], site))
            if len(inst) >= cap_per_rule:
                break
        if not inst:
            continue
        members2 = [{"predicate": disp, "unit": code, "site": site,
                     "origin": {"construct": rec["construct"]}}
                    for code, disp, site in inst]
        collect_list = [] if collect else None
        val, adopt = shapes._validate(rec, members2, oracle_fn, collect=collect_list)
        val["generated"] = len(inst)
        rec_out = shapes._record(sid, rec, val, adopt,
                                 "typed transfer: type-preserving substitution",
                                 source="typed")
        records.append(rec_out)

        if adopt and collect:
            for (code, disp, site), status in zip(inst, [s for _x, s in collect_list]):
                if status == "CERTIFIED":
                    if not info_filter or _conditional(code):
                        facts.append({"kind": "certified", "site": site,
                                      "shape_id": sid, "rule": rec["shape"],
                                      "tier": rec["tier"], "predicate": disp})
                elif status == "parsed":
                    facts.append({"kind": "premise", "site": site, "shape_id": sid,
                                  "rule": rec["shape"], "tier": rec["tier"],
                                  "predicate": disp})
    return records, facts


def _conditional(code: str) -> bool:
    """Informative iff the certified proof USED a premise (conditional), not an
    unconditional identity (vacuous). Goes through the authority (certify) — this
    module must not call unit_vc directly (single-verdict-writer invariant)."""
    try:
        cert, _prof, _viol = certify.certify_unit("f", code)
    except Exception:
        return False
    return any(o.get("kind") == "premise" for o in (cert.get("obligations") or []))
