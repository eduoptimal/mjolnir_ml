"""mjolnir_prover.audit.policy — declared interpretations and severities for findings.

A finding is split into a FACT (computed by mjolnir, or quoted from the client's
change report and marked unverified) and an INTERPRETATION — a *declared* policy
reading, labelled ``policy R-n``. A client may override any rule in a
``mjolnir_prover.policy`` file at the corpus root; absent one, the documented defaults ship.

Severity is computed, never hardcoded in the report: a finding with count 0 is
``ok`` ("no exposure detected"); otherwise it takes the severity declared by the rule.
"""
from __future__ import annotations

import os

POLICY_FNAME = "mjolnir_prover.policy"

# Default rules ship with the product and are cited in the report as policy R-n.
DEFAULTS = {
    "bound": {
        "rule": "R-6", "severity": "ok",
        "interpretation": "Policy that is NOT encoded in the code: each bound "
        "control is proven on the named columns, conditional on its declared contract "
        "(assumption). It is evidence only as strong as that contract.",
        "action": "Review the bindings; keep the premises enforced; retain as standing "
        "evidence."},
}


def load_policy(corpus: str) -> dict:
    """Rules (defaults overridden by ``mjolnir_prover.policy``), plus provenance so the
    report can say whether an interpretation is default or client-declared."""
    rules = {k: dict(v) for k, v in DEFAULTS.items()}
    path = os.path.join(corpus, POLICY_FNAME)
    declared = os.path.exists(path)
    if declared:
        current = None
        for raw in open(path, encoding="utf-8", errors="ignore").read().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, val = (x.strip() for x in line.split(":", 1))
            key = key.lower()
            if key == "finding":
                current = val.lower()
                rules.setdefault(current, {"rule": "R-0", "severity": "high",
                                           "interpretation": "", "action": ""})
            elif current:
                rules[current][key] = val
    return {"rules": rules, "path": path, "declared": declared}


def severity(count: int, rule: dict) -> str:
    """Computed severity: no occurrences is not a High finding."""
    return "ok" if count <= 0 else str(rule.get("severity", "high")).lower()
