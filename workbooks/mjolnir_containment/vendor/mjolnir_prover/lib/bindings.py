"""mjolnir_prover.lib.bindings — explicit rule→target BINDINGS (the proper application).

WHY
---
A generic rule touches nothing until it is BOUND to real symbols. Blind
instantiation over every column is algebra, not evidence. The proper mechanism is
an explicit, reviewed binding: a rule + a target site + a substitution σ
(variable -> symbol). This module loads those, type-checks σ against the target's
typed environment, disposes the bound unit through the SINGLE verdict authority,
and returns certificates = {rule, site, sigma, status, kind}.

BINDING FILE (JSONL)
--------------------
    {"rule": "<rule_id | predicate text>", "site": "TI_CASH_RECONCILIATION",
     "sigma": {"sum_debit": "TOTAL_DEBIT", "sum_credit": "TOTAL_CREDIT"}}

The rule's unit-of-record (premise + conclusion) is substituted with σ verbatim —
no enumeration, no inference. σ must be total and TYPE-PRESERVING: a Num rule var
binds to a Num column, Str to Str (see transfer.sort_of_sql). A binding that fails
to type-check is rejected, never guessed.

CLASSIFICATION (what a bound rule yields)
-----------------------------------------
  CERTIFIED with a premise obligation -> kind "certified"  (conditional control)
  UNVERIFIABLE/assumed_not_proved     -> kind "premise"    (declared contract)
  CERTIFIED, all obligations non-premise (unconditional identity) -> filtered (vacuous)
  REFUTED / otherwise                 -> rejected (scope-narrowed), recorded only
"""
from __future__ import annotations

import json
import re

from . import certify
from . import shapes
from . import transfer


def load(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                b = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: bad JSON: {e}")
            if not b.get("rule") or not b.get("site") or not isinstance(b.get("sigma"), dict):
                raise ValueError(f"{path}:{i}: need rule, site, sigma{{}}")
            out.append(b)
    return out


def resolve(ref: str, groups: dict):
    """Find (shape_id, member). By id, normalized shape, predicate (exact member),
    or fn name. Returns (None, None) if unknown."""
    def rep(g):
        return min(g["members"], key=lambda m: (m["origin"].get("repo", ""),
                                                m["origin"].get("file", ""),
                                                m["origin"].get("line", 0)))
    if ref in groups:
        return ref, rep(groups[ref])
    for sid, g in groups.items():
        if g.get("shape") == ref:
            return sid, rep(g)
        for m in g["members"]:
            if m.get("predicate") == ref:
                return sid, m
            if (m.get("canonical_unit") or "").startswith(f"def {ref}("):
                return sid, m
    return None, None


def _subst(code: str, sigma: dict) -> str:
    """Whole-identifier substitution of σ into a unit's asserts."""
    for var, sym in sigma.items():
        code = re.sub(rf"\b{re.escape(var)}\b", sym, code)
    return code


def _classify(cert: dict):
    kinds = [o.get("kind") for o in cert.get("obligations", [])]
    st, reason = cert.get("status"), cert.get("reason", "")
    if st == "CERTIFIED":
        if any(k != "premise" for k in kinds) and any(k == "premise" for k in kinds):
            return "certified"
        if any(k != "premise" for k in kinds):
            return "vacuous"                      # unconditional identity -> filtered
        return None
    if st == "UNVERIFIABLE" and reason == "assumed_not_proved":
        return "premise"                          # all obligations are declared contracts
    return None                                   # refuted / other -> not materialised


def run(bindings: list, groups: dict, sites: dict, oracle=None):
    """Dispose every binding. Returns (results, facts). `sites` maps site->{sym:sort}."""
    oracle = oracle or certify.certify_unit
    results, facts = [], []
    for b in bindings:
        ref, site, sigma = b["rule"], b["site"], b["sigma"]
        sid, member = resolve(ref, groups)
        if sid is None:
            results.append({**b, "status": "REJECTED", "reason": "rule_not_found"})
            continue
        g = groups[sid]
        symbols = sites.get(site)
        if symbols is None:
            results.append({**b, "status": "REJECTED", "reason": "site_not_found"})
            continue
        cu = member.get("canonical_unit")
        if not cu:
            results.append({**b, "status": "REJECTED", "reason": "rule_has_no_unit"})
            continue
        tpls, meta, sort_src = transfer._unit_template(cu)
        sorts = transfer.var_sorts(sort_src, meta["vars"])
        # σ must be total and type-preserving over the site's typed symbols
        errs = []
        for var in meta["vars"]:
            if var not in sigma:
                errs.append(f"unbound:{var}")
                continue
            sym = sigma[var]
            if sym not in symbols:
                errs.append(f"no_such_symbol:{sym}")
                continue
            want, got = sorts.get(var, "Any"), symbols[sym]
            if want != "Any" and got != want:
                errs.append(f"type:{var}({want})<-{sym}({got})")
        if errs:
            results.append({**b, "status": "REJECTED", "reason": ";".join(errs)})
            continue
        syms = [sigma[v] for v in meta["vars"]]
        sub = {f"n{i}": meta["nums"][i] for i in range(len(meta["nums"]))}
        sub.update({f"v{i}": sigma[meta["vars"][i]] for i in range(len(meta["vars"]))})
        parts = [p for p in (shapes.instantiate(t, sub) for t in tpls) if p]
        if not parts:
            results.append({**b, "status": "REJECTED", "reason": "empty_unit"})
            continue
        code = (f"def bound({', '.join(syms)}):\n"
                + "".join(f"    assert {p}\n" for p in parts))
        cert, _p, _v = oracle("bound", code)
        kind = _classify(cert)
        results.append({"rule": ref, "rule_id": sid, "site": site, "sigma": sigma,
                        "status": cert["status"], "reason": cert.get("reason", ""),
                        "kind": kind, "unit": code})
        if kind in ("certified", "premise"):
            facts.append({"kind": kind, "site": site, "shape_id": sid,
                          "rule": g["shape"], "binding": sigma,
                          "predicate": parts[-1]})
    return results, facts
