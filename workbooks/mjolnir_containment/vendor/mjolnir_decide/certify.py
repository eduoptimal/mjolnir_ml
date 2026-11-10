"""mjolnir_decide.certify — the safety-policy gate over the estimator (0 params).

WHY THIS LAYER EXISTS
---------------------
The estimator generalizes: the flow head read `cancel_reservation(...)` as
in-regime and scored it safe (noul 1) — generalization is what a model is for,
and it is also how a novel irreversible verb reaches ALLOW.  This module is
the deterministic backstop: a closed, versioned, digested policy that can only
NARROW the estimator, never widen it:

    noul_certified = 0 if any rule fires else noul_head

Fail-closed, in both directions this layer owns:
  * a fired rule -> noul 0 (release blocked, queue for a human);
  * a state the policy cannot read (no visible tool call) or a verb outside
    its closed vocabulary -> `unverified_action` veto.  Unknown is never
    passed silently — certification certifies only what it can classify.

The receipt keeps BOTH numbers: `noul_head` (what the estimator said) and
`noul_certified` (what the policy allowed).  Fail-closed means block the
release, not erase the disagreement — the veto/head divergence is the
calibration signal for the policy itself.

WHAT THE RULES DO NOT SEE is declared in every response (BLIND_SPOT):
safeguards are keyword-declared text only (backup / approval / dry-run / ...),
prose approvals outside the lexicon are unread; the vocabulary is closed and
grows only by a versioned policy change; and no rule ever raises noul.

Policy `safety-policy@1`: RULES below are the contract.  Changing a rule
bumps the version; the digest travels in every response so a receipt can be
checked against the policy that issued it.
"""
from __future__ import annotations

import hashlib
import json
import re

POLICY_VERSION = "safety-policy@1"
RULES = ("destructive_without_backup",
         "irreversible_without_safeguard",
         "unverified_action")

# The trained dialect (model.KIND_BY_TOOL) — classification, not the parser's
# feature mapping: destructive here means "rule 1 applies".
DIALECT = {"select_rows": "known", "update_rows": "known",
           "insert_row": "known",
           "delete_rows": "destructive", "drop_table": "destructive"}

# Stem tables for verbs OUTSIDE the dialect.  Match is exact or stem + "_".
DESTRUCTIVE_STEMS = ("delete", "drop", "purge", "wipe", "truncate",
                     "erase", "destroy")
IRREVERSIBLE_STEMS = ("cancel", "revoke", "terminate", "kill", "expire",
                      "remove")
KNOWN_STEMS = ("refund", "export", "count", "fetch", "list", "get",
               "search", "show", "read", "select")

_CALL_RE = re.compile(r"^Proposed tool call:\s*([A-Za-z_]\w*)\s*\(", re.MULTILINE)
# Same negation-aware reading as model.parse_feats (kept beside it, torch-free).
_BACKUP_RE = re.compile(r"\bbackup\b", re.I)
_NO_BACKUP_RE = re.compile(r"\bno backup\b", re.I)
_SAFEGUARD_RE = re.compile(
    r"\b(approval|approved|approve|dry[\s_-]?run|rollback|revert|"
    r"sign[\s_-]?off|human[\s_-]?review)\b", re.I)

BLIND_SPOT = {
    "behaviour": "fail-closed: a fired rule sets noul to 0; rules never raise noul",
    "vocabulary": ("closed: verbs outside the dialect and the stem lists are "
                   "vetoed as unverified_action, never guessed"),
    "safeguards": ("keyword-declared text only (backup / approval / dry-run / "
                   "rollback / sign-off); prose approvals outside the lexicon "
                   "are not seen"),
}

_POLICY_BYTES = json.dumps({"version": POLICY_VERSION, "rules": list(RULES)},
                           sort_keys=True, separators=(",", ":"))
DIGEST = "sha256:" + hashlib.sha256(_POLICY_BYTES.encode()).hexdigest()[:16]


def policy() -> dict:
    """A fresh copy: receipts pin the policy, they do not share a mutable."""
    return {"version": POLICY_VERSION, "rules": list(RULES), "digest": DIGEST}


def classify(verb: str) -> str:
    """-> 'known' | 'destructive' | 'irreversible' | 'unknown' (fail-closed)."""
    v = (verb or "").lower()
    if v in DIALECT:
        return DIALECT[v]
    for table, kind in ((DESTRUCTIVE_STEMS, "destructive"),
                        (IRREVERSIBLE_STEMS, "irreversible"),
                        (KNOWN_STEMS, "known")):
        for stem in table:
            if v == stem or v.startswith(stem + "_"):
                return kind
    return "unknown"


def backup_declared(state: str) -> bool:
    """'no backup' wins over a bare 'backup' mention (negation-aware)."""
    if _NO_BACKUP_RE.search(state or ""):
        return False
    return bool(_BACKUP_RE.search(state or ""))


def safeguard_declared(state: str) -> bool:
    return backup_declared(state) or bool(_SAFEGUARD_RE.search(state or ""))


def _veto(rule: str, detail: str, facts: dict) -> dict:
    return {"rule": rule, "detail": detail, "facts": facts}


def certify(state: str) -> list:
    """Evaluate safety-policy@1 against one state -> the fired vetoes ([] = clear).

    Deterministic string work only: no model, no network, no registry.  A
    state with multiple call lines is certified per step by the batch path;
    this function reads the FIRST visible call, exactly as the parser does.
    """
    m = _CALL_RE.search(state or "")
    if not m:
        return [_veto("unverified_action",
                      "no proposed tool call visible; nothing can be certified",
                      {"verb": None, "classified": "missing"})]
    verb = m.group(1)
    kind = classify(verb)
    if kind == "unknown":
        return [_veto("unverified_action",
                      f"action {verb!r} is outside the policy vocabulary; "
                      "cannot certify",
                      {"verb": verb, "classified": "unknown"})]
    vetoes = []
    if kind == "destructive" and not backup_declared(state):
        vetoes.append(_veto(
            "destructive_without_backup",
            f"destructive action {verb!r} with no declared backup",
            {"verb": verb, "destructive": True, "backup": False}))
    if kind == "irreversible" and not safeguard_declared(state):
        vetoes.append(_veto(
            "irreversible_without_safeguard",
            f"irreversible action {verb!r} with no declared safeguard "
            "(backup / approval / dry-run)",
            {"verb": verb, "irreversible": True, "safeguard": False}))
    return vetoes


def certification_block(noul_head, noul_certified: float | int,
                        vetoes: list) -> dict:
    """The additive response block — both numbers, the policy, the vetoes,
    and the rule layer's own declared blind spot."""
    return {
        "noul_certified": noul_certified,
        "noul_head": noul_head,
        "policy": policy(),
        "rule_vetoes": vetoes,
        "blind_spot": dict(BLIND_SPOT),
    }
