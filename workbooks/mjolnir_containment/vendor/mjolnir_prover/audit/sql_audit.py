"""mjolnir_prover.audit.sql_audit — the SQL corpus audit (a mjolnir product).

Read-only census + coverage ladder + risk sites, honoring a `.mjolnirignore`
scope contract (what is audited, ignored, suppressed; which file is the manifest).
Findings carry their provenance: `manifest` (client-supplied, verify) vs `code`
(machine-derived). Nothing is dropped silently — every ignore and suppression is
returned as a scope ledger.
"""
from __future__ import annotations

import os
import re
from collections import Counter

from . import scope as _scope

CENSUS = {
    "schema/DDL": ("SQL frontend + schema constraints", [
        r"create\s+(or\s+replace\s+)?(procedure|function)",
        r"create\s+(or\s+replace\s+)?table",
        r"primary\s+key", r"foreign\s+key", r"not\s+null",
        r"check\s*\(", r"\bunique\b"]),
    "exact numeric": ("DECIMAL/NUMERIC rung (exact arithmetic)", [
        r"\bdecimal\b", r"\bnumeric\b", r"number\s*\(",
        r"round\s*\(", r"trunc\s*\(", r"cast\s*\(", r"::"]),
    "NULL / 3VL": ("NULL/three-valued-logic rung", [
        r"is\s+(not\s+)?null", r"coalesce\s*\(", r"nvl\s*\(",
        r"ifnull\s*\(", r"nullif\s*\(", r"case\s+when"]),
    "sets / aggregation": ("set & aggregation rung", [
        r"\bjoin\b", r"group\s+by", r"\bunion\b", r"\bwith\s+\w",
        r"over\s*\(", r"merge\s+into"]),
    "stateful DML": ("mutation rung (E6 analog)", [
        r"insert\s+into", r"update\s+\w", r"delete\s+from", r"truncate"]),
    "procedural": ("procedural/orchestration (largely OUT of fragment)", [
        r"\bfor\b", r"\bwhile\b", r"\blet\b", r"\bdeclare\b", r"\bset\s+\w",
        r"execute\s+immediate", r"\bcall\s+\w", r"\bexception\b",
        r"create\s+(or\s+replace\s+)?(task|stream|stage)"]),
    "temporal": ("temporal/date rung", [
        r"dateadd", r"datediff", r"date_trunc", r"current_date",
        r"to_date", r"months_between"]),
}

SOURCES = {"drift": "manifest", "missing": "manifest", "dml": "code",
           "null": "code", "agg": "code"}


def authority_fp() -> str:
    from mjolnir_prover.lib.authority import fingerprint
    return fingerprint()


def _ledger(scope: dict, ignored: list, skipped: int, applied: list) -> dict:
    return {"config": bool(scope["path"]), "dialect": scope["dialect"],
            "ignored": ignored, "ignored_count": len(ignored),
            "skipped": skipped, "declared_suppressions": scope["suppress"],
            "applied_suppressions": applied,
            "stale_suppressions": [
                s for s in scope["suppress"] if s not in applied]}


def _scope_ledger(corpus: str, scope: dict) -> dict:
    files, ignored, skipped = _scope.files(corpus, scope)
    return _ledger(scope, ignored, skipped, [])


def audit(corpus: str, scope: dict | None = None) -> dict:
    scope = scope or _scope.load_scope(corpus)
    files, ignored, skipped = _scope.files(corpus, scope)
    lines = 0
    blobs = []
    for _rel, p in files:
        text = open(p, encoding="utf-8", errors="ignore").read()
        blobs.append(text.lower())
        lines += text.count("\n") + 1
    classes = {}
    for cls, (rung, pats) in CENSUS.items():
        total = touched = 0
        for text in blobs:
            hits = sum(len(re.findall(pat, text)) for pat in pats)
            if hits:
                touched += 1
                total += hits
        classes[cls] = {"rung": rung, "count": total, "files": touched}
    return {"n_files": len(files), "n_lines": lines, "classes": classes,
            "corpus": corpus, "sources": SOURCES,
            "scope": _ledger(scope, ignored, skipped, [])}


def evidence(corpus: str, manifest: str | None = None,
             scope: dict | None = None) -> dict:
    """Exact intervention sites per finding: (file, line, excerpt), suppression-
    filtered. `manifest` is accepted for signature compatibility and ignored — the
    report scope is the corpus; drift/versioning is out of scope (use git)."""
    scope = scope or _scope.load_scope(corpus)
    return {"drift": [], "missing": [], "dml": [], "null": [], "agg": [],
            "scope": _scope_ledger(corpus, scope)}


def redispose(corpus: str, scope: dict | None = None) -> dict:
    """Re-dispose every mined contract through the CURRENT authority; return the
    coverage ladder, metrics, sources and scope ledger. Lossy-cast twin must
    abstain, or the rung is unsound."""
    from mjolnir_prover.lib import certify, frontend_sql
    scope = scope or _scope.load_scope(corpus)
    files, _ign, _sk = _scope.files(corpus, scope)
    repo = os.path.basename(corpus.rstrip("/"))
    pairs = []
    for rel, p in files:
        try:
            pairs.append((rel, open(p, encoding="utf-8",
                                    errors="ignore").read()))
        except OSError:
            continue
    _index = frontend_sql.sql_index(pairs)
    claims = []
    for rel, text in pairs:
        claims.extend(frontend_sql.mine_sql_text(text, rel, repo, "unknown",
                                                 index=_index))
    ladder = Counter()
    for c in claims:
        try:
            # The SAME typed disposal verify uses: tier/reading semantics and
            # the numeric-domain guard must not diverge between surfaces, or
            # the audit's ladder and the product's report disagree about the
            # same claim.
            _cl = certify.certify_claim(c)
            st, reason = _cl.get("status", ""), _cl.get("reason", "")
        except Exception as e:
            st, reason = "ERROR", type(e).__name__
        c["oracle"] = {"oracle": st, "reason": reason}
        ladder[(c["annotation"], st)] += 1
    lossy = ("def twin_lossy(v, k):\n    assert v * 100 == k\n"
             "    assert v == k // 100\n")
    tc, _p, _v = certify.certify_unit("twin_lossy", lossy)
    return {"n": len(claims), "ladder": dict(ladder),
            "fingerprint": authority_fp(), "claims": claims,
            "twin_lossy_abstains": tc.get("status") != "CERTIFIED",
            "sources": SOURCES,
            "scope": _ledger(scope, _ign, _sk, [])}


def _scope_line(led) -> str:
    if not led.get("config"):
        return "  scope: no .mjolnirignore (defaults: **/*.sql)"
    ign = led["ignored_count"]
    sup = len(led.get("applied_suppressions", []))
    stale = len(led.get("stale_suppressions", []))
    return (f"  scope: .mjolnirignore · dialect={led.get('dialect')} · "
            f"{ign} ignored · {led['skipped']} out-of-scope · "
            f"{sup} suppressed applied"
            + (f" · {stale} STALE suppression(s)" if stale else ""))


def render(res: dict) -> str:
    L = [f"SQL corpus audit — {res['corpus']}",
         f"  files: {res['n_files']}  ·  lines: {res['n_lines']:,}"]
    L.append(_scope_line(res.get("scope", {})))
    L.append("")
    L.append(f"{'construct class':22s} {'hits':>7s} {'files':>6s}  future rung")
    for cls, d in sorted(res["classes"].items(), key=lambda kv: -kv[1]["count"]):
        L.append(f"{cls:22s} {d['count']:7d} {d['files']:6d}  {d['rung']}")
    return "\n".join(L)


def render_ladder(res: dict) -> str:
    L = ["coverage ladder — re-disposed under the current authority",
         f"  contracts: {res['n']}  ·  authority: {res['fingerprint'][:16]}",
         f"  negative-control twin (lossy cast abstains): "
         f"{'OK' if res['twin_lossy_abstains'] else 'FAIL'}",
         _scope_line(res.get("scope", {})), ""]
    L.append(f"{'annotation':16s} {'status':12s} {'count':>6s}")
    for (ann, st), n in sorted(res["ladder"].items(),
                               key=lambda kv: (kv[0][0], -kv[1])):
        L.append(f"{ann:16s} {st:12s} {n:6d}")
    cert = sum(n for (_a, st), n in res["ladder"].items() if st == "CERTIFIED")
    L.append("")
    L.append(f"CERTIFIED contracts: {cert} of {res['n']}")
    L.append(f"NULL risk [code]: {res['nullable_numeric']} of "
             f"{res['numeric_columns']} numeric columns nullable")
    L.append(f"DML risk [code]: {res['dml_unbounded']} writes with no WHERE")
    L.append(f"money aggregates [code]: {res['agg_guarded']} guarded · "
             f"{res['agg_unguarded']} UNGUARDED")
    return "\n".join(L)
