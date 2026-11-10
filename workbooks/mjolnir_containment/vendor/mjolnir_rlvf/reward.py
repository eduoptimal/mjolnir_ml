"""rlvf.reward — typed verdict reward plus the calibration layer.

Two stacked mappings, both derived from the single authority's verdict:

  1. typed_reward(reading, status)      — the primitive (see rlvf.md §3).
  2. calibration_reward(outcome, oracle) — the mandate (rlvf.md §3.1):
     honest abstention on an undecidable task is a positive outcome, and
     overclaiming is the worst.
"""
from __future__ import annotations

TYPED = {
    "theorem": {"CERTIFIED": 1.0, "REFUTED": -1.0, "UNVERIFIABLE": 0.0},
    "strengthening": {"CERTIFIED": 0.5, "REFUTED": 0.0, "UNVERIFIABLE": 0.0},
    "premise": {"CERTIFIED": 0.5, "REFUTED": 0.0, "UNVERIFIABLE": 0.0},
    "instance": {"CERTIFIED": 0.0, "REFUTED": 0.0, "UNVERIFIABLE": 0.0},
}

CALIBRATION = {
    ("certified", "decidable"): 1.0,
    ("abstain", "undecidable"): 0.75,
    ("abstain", "decidable"): 0.0,
    ("refuted", "decidable"): -1.0,
    ("refuted", "undecidable"): -1.0,
    # H-R2' FABRICATED: the claim names a quantity the source does not contain.
    # Scored at the SAME penalty as asserting a wrong value, and never better than
    # honest abstention, so a learner has a gradient toward abstaining instead of
    # guessing. It was previously invisible: `UNVERIFIABLE` covered both "absent"
    # and "cannot decide", so the reward could not punish the first. The absence
    # is not a smaller sin than a wrong answer.
    ("fabricated", "decidable"): -1.0,
    ("fabricated", "undecidable"): -1.0,
    ("unverifiable", "undecidable"): -0.25,
    ("unverifiable", "decidable"): 0.0,
    ("no_claim", "decidable"): -0.25,
    ("no_claim", "undecidable"): 0.0,
    ("error", "decidable"): 0.0,
    ("error", "undecidable"): 0.0,
    ("fault", "decidable"): 0.0,
    ("fault", "undecidable"): 0.0,
    # P2: premise invention is an overclaim class the oracle cannot see —
    # the reward must punish it or RLVF TEACHES it (reward leakage through
    # the task-level grounding gap).
    ("premise_invention", "decidable"): -1.0,
    ("premise_invention", "undecidable"): -1.0,
}


OUTCOME_ORDER = ("certified", "abstain", "refuted", "unverifiable", "no_claim",
                 "error", "fault", "premise_invention", "fabricated")


def is_fabrication(cert: dict) -> bool:
    """The claim names a quantity the SOURCE does not contain.

    Read from the authority's own reason, never re-derived here: `tactics`
    preserves the router's `quantity_not_in_source:<q>` precisely so that
    fabrication is distinguishable from honest uncertainty, and this is the only
    place that turns that reason into an outcome.
    """
    if not cert:
        return False
    if str(cert.get("reason", "")).startswith("quantity_not_in_source"):
        return True
    return any(str(c.get("reason", "")).startswith("quantity_not_in_source")
               for c in (cert.get("claims") or []))


def typed_reward(reading, status) -> float:
    return TYPED.get(reading or "theorem", TYPED["theorem"]).get(status, 0.0)


def calibration_reward(outcome: str, oracle: str) -> float:
    return CALIBRATION.get((outcome, oracle), 0.0)


def is_non_vacuous(cert: dict) -> bool:
    """A certification counts only if at least one obligation is a real
    (non-premise) discharged proof."""
    for ob in (cert or {}).get("obligations", []) or []:
        if ob.get("status") == "ok" and ob.get("kind") != "premise":
            return True
    return False
