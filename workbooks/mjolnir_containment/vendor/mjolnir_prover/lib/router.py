"""mjolnir_prover.lib.router — the L2 interface, demoted from a rung to a ROUTER.

It may do exactly two things: route a claim to a decidable rung, or abstain.
It may NEVER certify and NEVER refute. There is no method here that can return a
verdict, and that absence is the guarantee — checkable with
`hasattr(Router, "certify") is False`, not by discipline.

WHY DEMOTED RATHER THAN DELETED. The original plan had a calibrated NLI model as
rung L2, deciding entailment. The evidence against that is real but narrower
than "delete it": the only fallible stage ever MEASURED was a regex formaliser
(0 certified, 297 refuted), and no NLI model was ever built. What the evidence
supports is a claim about the ROLE, not the model:

    a fallible stage must never DISPOSE.  A refutation costs more than an
    abstention, and a fallible stage manufactures refutations.  That holds for a
    perfectly calibrated NLI model too.

    T-NL9's delta is not a licence to dispose.  Calibration bounds the error
    RATE; it does not change the error's KIND.

EVIDENCE THAT THE INTERFACE IS SAFE. Routing must not WIDEN the set of
dispositions. Measured (`tactics/l2_router.py`): 0 new dispositions against
60/60 both ways. A deterministic router passing that means a fallible one
cannot do worse, because it can only route FEWER claims, and routing fewer is a
strict subset of the same disposition set.

THE RULE THE SAFETY TEST FOUND. A router must carry the EVIDENCE it routed on.
The first version bound it for L1 and dropped it for L3 (`assert x == v` over a
free variable), so every claim came back undecided and the router introduced 30
errors. A router that routes without evidence is not a router — it is a guess
that happens to invoke a checker.
"""
from __future__ import annotations

import re

# Destinations this router may produce. Both are decidable and exact.
DESTINATIONS = ("L1", "L3")

VERBATIM = re.compile(r'"[^"]{8,}"|`[^`]{3,}`')
NUMBER = re.compile(r"\b\d[\d,.]*\b")
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9 ]*")


class Router:
    """May ROUTE or ABSTAIN. No verdict method exists, by design."""

    name = "lexical-router"

    def route(self, quantity, value, known):
        """-> (destination, evidence) | (None, reason). NEVER a verdict.

        `known` maps a declared quantity identifier to the value the SOURCE
        states for it. The router may only point at something the source
        contains; it may not invent a destination for a quantity the source does
        not declare. That is what makes routing a subset operation.
        """
        if quantity not in known:
            return None, f"quantity_not_in_source:{quantity}"
        # LOCATE, DO NOT JUDGE. The first version returned abstain when the
        # claim's value was absent from the source cell — so a source that
        # DISAGREED with the claim produced no verdict at all, and the
        # refutation channel silently vanished. Abstaining on disagreement is
        # the router making a judgement, which is precisely what this interface
        # exists to forbid: the authority must be the thing that refutes.
        src = str(known[quantity])
        v = str(value).strip()
        if v and v in src:                       # the number is stated
            return "L3", src
        q = IDENT.search(str(quantity))
        if q and q.group(0).strip().lower() in src.lower():
            return "L1", src
        # The quantity IS declared but the value is not what the source says.
        # That is not a routing failure — it is the case the checker exists for.
        return "L3", src


def no_verdict_surface(cls) -> bool:
    """Assert the interface cannot express a verdict. Used by the selfcheck."""
    banned = {"certify", "verify", "refute", "judge", "verdict", "decide"}
    return not (banned & set(dir(cls)))
