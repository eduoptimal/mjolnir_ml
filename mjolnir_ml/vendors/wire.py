# CLONE of src/mjolnir_disposition/engine.py.
#
# DELTA vs the original (the only one): the import points at the vendored core
# (`vendors.core.verdict`) instead of `mjolnir_router.core.verdict`, so mjolnir_ml
# stays standalone. Everything below the import is byte-identical (pinned by
# the test suite).
"""mjolnir_disposition.engine — the wire layer over the ONE rule engine.

THE ENGINE ITSELF LIVES IN `mjolnir_router.core.verdict`. That package keeps its
core standalone (its own boundary test forbids the core importing any sibling),
so the dependency points this way: the router owns the frozen rule engine, and
this module re-exports it and adds ONLY the wire vocabulary the endpoint, the
agent, and the audit speak.

That means exactly one implementation of `evaluate`, `Rule`, and `Verdict`
exists in the repo — `kernel.evaluate is core.evaluate` — and the wire enum
(`release · ask · deny · abstain`) is a projection of the engine's verdicts, not
a second engine.
"""
from __future__ import annotations

from mjolnir_ml.vendors.core.verdict import (ABSTAIN, DENY, ESCALATE,  # noqa: F401
                                             ROUTE, UNSTATED, VERDICTS,
                                             _Declared, Rule, Verdict,
                                             evaluate)

# ---------------------------------------------------------------- wire enum --
RELEASE = "release"
ASK = "ask"
DENY_W = "deny"
ABSTAIN_W = "abstain"
WIRE = (RELEASE, ASK, DENY_W, ABSTAIN_W)
# The monotone ladder, strictest last. `ask` is stricter than `release`;
# `abstain` (undecidable) is stricter still; a rule that declares `deny` is the
# most restrictive and is never softened by an abstention elsewhere.
RANK = {RELEASE: 0, ASK: 1, ABSTAIN_W: 2, DENY_W: 3}


def project(value: str) -> str:
    """Engine verdict -> wire verdict. ROUTE is a release; ESCALATE is an ask
    (a human may still act); DENY is a refusal that is nobody's to grant."""
    return {ROUTE: RELEASE, ESCALATE: ASK, DENY: DENY_W, ABSTAIN: ABSTAIN_W}[value]


def strictest(a: str, b: str) -> str:
    """The stricter of two WIRE verdicts — the tier-combination law."""
    return a if RANK[a] >= RANK[b] else b
