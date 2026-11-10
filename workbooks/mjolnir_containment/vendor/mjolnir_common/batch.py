"""mjolnir_common.batch — the joint-AND plan law, shared by route and decide.

WHY A JOINT AND AND NEVER A PRODUCT
-----------------------------------
A plan is SEQUENTIAL COMPOSITION: the caller intends step1 then step2 against
one world, in order. Releasing the plan means releasing every step, so the
batch verdict is the WORST step (ALLOW < ASK < DENY) and the batch scalar is
the MIN of the step scalars. A product (prod noul_i) would be the right
aggregation for INDEPENDENT claims; sequential steps are not independent —
they share world state, and the steps of a plan may not even be disjoint —
so min is the aggregation that is sound under arbitrary dependence (it is a
conservative lower bound on "no-unsafe across the plan").

PARTIAL RELEASE IS NOT OFFERED
------------------------------
Running the "safe" half of a non-commutative sequence is a DIFFERENT action
than the plan (cancel W1 without refunding it), and nobody triaged that
action. All-or-nothing is the sound default; a partial-release policy would
be a separate, explicit opt-in — never this function.

FAIL-CLOSED, EVERYWHERE
-----------------------
  * an empty step list raises (an empty plan is a contract error, not a
    release);
  * one step out of regime forces the batch scalar to 0.0 regardless of the
    other steps' values (the regime gate is joint too);
  * the splitter never guesses: a segment that does not look like a step
    makes the whole reading AMBIGUOUS and the caller ASKs.

No domain nouns: verdicts are the wire vocabulary every surface shares;
step recognition is INJECTED (`is_step`), so each service keeps its own
dialect.
"""
from __future__ import annotations

import re

ALLOW = "ALLOW"
ASK = "ASK"
DENY = "DENY"
SEVERITY = {ALLOW: 0, ASK: 1, DENY: 2}

# Longest-first so " and then " is not split as " and " + trailing "then".
CONNECTIVES = (" and then ", ", then ", " then ", " and ", "; ")


def worst_case(verdicts) -> str:
    """Joint-AND over wire verdicts: the batch is its worst step."""
    if not verdicts:
        raise ValueError("empty plan: no step to aggregate (fail-closed)")
    return max(verdicts, key=lambda v: SEVERITY[v])


def min_noul(values, regimes) -> float:
    """Joint-AND over scalars: min, with the regime gate applied jointly.

    One out-of-regime step -> 0.0 for the whole batch: a single unscorable
    step means the plan is not scorable, whatever its siblings scored.
    """
    vals = list(values)
    regs = list(regimes)
    if not vals or len(vals) != len(regs):
        raise ValueError("empty or ragged plan (fail-closed)")
    if not all(bool(r) for r in regs):
        return 0.0
    return min(float(v) for v in vals)


_SPLIT_RE = re.compile("|".join(re.escape(c) for c in CONNECTIVES))


def split_plan(text: str, is_step) -> dict:
    """Deterministic connective split with an injected step predicate.

    -> {"kind": "single"|"multi"|"ambiguous", "steps": [segment, ...]}

      * no connective                       -> single (the whole text)
      * every segment satisfies `is_step`   -> multi (the plan)
      * any segment fails `is_step`         -> AMBIGUOUS: the caller shows the
        candidate readings and ASKs. ("cancel order W1 and W2" splits, and
        "W2" is not a step — the reading is refused, never guessed.)

    The split itself is pure string work on a fixed connective table; nothing
    here consults a model or a registry.
    """
    t = (text or "").strip()
    if not t:
        return {"kind": "ambiguous", "steps": []}
    segs = [s.strip() for s in _SPLIT_RE.split(t)]
    segs = [s for s in segs if s]
    if len(segs) <= 1:
        return {"kind": "single", "steps": [t]}
    if all(is_step(s) for s in segs):
        return {"kind": "multi", "steps": segs}
    return {"kind": "ambiguous", "steps": segs}
