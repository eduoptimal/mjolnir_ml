"""mjolnir_prover.audit.scope — the .mjolnirignore contract for an audit.

Declares what IS audited, what to ignore, which file is the (non-audited) manifest,
the SQL dialect, and explicit suppressions (a reason is required). Nothing is
dropped silently: `files()` returns a scope ledger listing everything excluded, and
`suppressed()` resolves a site to its recorded reason, for the report.
"""
from __future__ import annotations

import fnmatch
import os

FNAME = ".mjolnirignore"
DEFAULTS = {"audit": ["**/*.sql"], "ignore": [], "manifest": None,
            "dialect": None, "suppress": []}


def _match(rel: str, pat: str) -> bool:
    rel = rel.replace(os.sep, "/")
    pat = pat.replace(os.sep, "/")
    if pat.startswith("**/"):
        tail = pat[3:]
        return (fnmatch.fnmatch(rel, tail)
                or fnmatch.fnmatch(os.path.basename(rel), tail)
                or any(fnmatch.fnmatch(seg, tail) for seg in rel.split("/")))
    return (fnmatch.fnmatch(rel, pat)
            or fnmatch.fnmatch(os.path.basename(rel), pat))


def load_scope(corpus: str) -> dict:
    scope = {"audit": list(DEFAULTS["audit"]), "ignore": [], "manifest": None,
             "dialect": None, "suppress": [], "path": None}
    path = os.path.join(corpus, FNAME)
    if not os.path.exists(path):
        return scope
    scope["path"] = path
    seen_audit = False
    for raw in open(path, encoding="utf-8").read().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        reason = ""
        if "#" in line:
            line, reason = line.split("#", 1)
            line, reason = line.strip(), reason.strip()
        if ":" not in line:
            continue
        key, val = (x.strip() for x in line.split(":", 1))
        key = key.lower()
        if key == "audit":
            if not seen_audit:
                scope["audit"] = []
                seen_audit = True
            scope["audit"].append(val)
        elif key == "ignore":
            scope["ignore"].append(val)
        elif key == "manifest":
            scope["manifest"] = val
        elif key == "dialect":
            scope["dialect"] = val
        elif key == "suppress":
            parts = val.split()
            scope["suppress"].append({
                "category": parts[0] if parts else "",
                "site": parts[1] if len(parts) > 1 else "",
                "reason": reason})
    return scope


def files(corpus: str, scope: dict):
    """(in_scope [(rel, path)], ignored [rel], skipped_count). The ignore file
    itself is never audited."""
    in_scope, ignored, skipped = [], [], 0
    for root, _d, names in os.walk(corpus):
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, corpus)
            if rel == FNAME:
                continue
            if any(_match(rel, pat) for pat in scope["ignore"]):
                ignored.append(rel)
            elif any(_match(rel, pat) for pat in scope["audit"]):
                in_scope.append((rel, p))
            else:
                skipped += 1
    return in_scope, ignored, skipped


def suppressed(rel: str, line: int, scope: dict):
    """The recorded suppression for this site, if any. Match on basename or
    corpus-relative path + line number."""
    base = os.path.basename(rel)
    for s in scope["suppress"]:
        site = s["site"]
        if ":" not in site:
            continue
        f, _, ln = site.rpartition(":")
        try:
            if int(ln) != line:
                continue
        except ValueError:
            continue
        if (os.path.basename(f) == base
                or f.replace("\\", "/") == rel.replace(os.sep, "/")):
            return s
    return None
