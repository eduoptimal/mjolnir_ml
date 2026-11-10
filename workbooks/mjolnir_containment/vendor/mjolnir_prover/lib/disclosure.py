"""mjolnir_prover.lib.disclosure — the soft/disclosure certificate layer.

THIS IS NOT A SECOND VERIFIER. It cannot decide anything. It reports what the
authority (`mjolnir_prover.lib.certify`) already decided, over a SET of claims rather
than one artifact, and it refuses to look like a boolean.

WHY IT EXISTS. A checker certifies one artifact, so a chat-shaped output — which
is a bag of assertions — has no single artifact to certify. The disclosure
certificate covers the SET: a subset that certifies, plus the remainder, which is
not certified and is not claimed to be.

WHAT IS PROVEN, AND WHERE
--------------------------
    only Omega may dispose          this module has no verdict path
    a marked remainder is REQUIRED   without clause (iii) a laundering dump
                                    certifies at rate 1.000 (S0)
    k may not be relaxed alone      (ii) alone is defeated at every k >= 1 (S1)
    it reports DISCLOSURE not truth  S-Omega = 1.000 at a 0.992 false fraction
    the pair is gameable            reported pair constant while the false
                                    fraction moves 0.992 -> 0.333 (S1)
    decidable with no model         0.06 ms over 401 assertions (S1/T-NL14)

So the honest report is `(dispositions, remainder)`, and the number a reader
wants — the false fraction of the remainder — is NOT computable from the artifact
(T-NL18). That is a barrier, not a bug, and it is why no scalar is exposed.

`disclose()` returns a dict with the remainder attached and a `scalar_view` that
is deliberately not a score.
"""
from __future__ import annotations

SCHEMA = "disclosure/v1"

# Outcomes the authority may return. Anything else is a bug, not a fallback.
DISPOSITIONS = ("CERTIFIED", "REFUTED", "UNVERIFIABLE", "FAULT")


class Claim:
    """One assertion plus what the router resolved about it."""

    __slots__ = ("text", "quantity", "value", "evidence", "destination",
                 "verdict", "reason", "marked")

    def __init__(self, text, quantity=None, value=None, evidence=None,
                 destination=None, verdict=None, reason=""):
        self.text = text
        self.quantity = quantity          # the referent, resolved
        self.value = value                # what the claim asserts
        self.evidence = evidence          # WHAT THE ROUTER SUPPLIED
        self.destination = destination    # "L1" | "L3" | None (abstain)
        self.verdict = verdict            # set ONLY by the authority
        self.reason = reason
        self.marked = False        # set by mark(); (iii) reads this

    def certified(self) -> bool:
        return self.verdict == "CERTIFIED"

    def as_dict(self) -> dict:
        # `marked` is part of the artifact, not an implementation detail. Clause
        # (iii) is "every claim not in S is EXPLICITLY marked", and a reader who
        # cannot see the mark cannot check that. Omitting it made `unattributed`
        # vacuously `[]` in every emitted certificate -- T-NL14 stayed decidable
        # but stopped being checkable by anyone who does not trust mjolnir.
        return dict(text=self.text, quantity=self.quantity, value=self.value,
                    evidence=self.evidence, destination=self.destination,
                    verdict=self.verdict, reason=self.reason,
                    marked=bool(getattr(self, "marked", False)))

    def __repr__(self) -> str:                            # pragma: no cover
        return f"<Claim {self.quantity or self.text[:28]!r} -> {self.verdict}>"


def disclose(claims, *, level: int = 1) -> dict:
    """The disclosure certificate over a SET of claims.

    (i)   the certified subset
    (ii)  the subset SUFFICES for the whole set at cover level `level`
    (iii) every claim not in S is EXPLICITLY marked unverified

    Clause (iii) is REQUIRED, not advisory. Without it a claim that borrows a
    covered topic's vocabulary certifies at rate 1.000, because a hallucination
    inside a covered block is by construction covered. That was measured, and it
    is the whole reason the remainder is enumerated instead of summarised.
    """
    # `k` is a REACH knob, not a safety one, and the ladder has TWO rungs.
    # T-NL17 measured that (ii) alone is defeated identically at every relaxed
    # level, and with (iii) enforced every relaxed level is equally sound --
    # so there is no level at which the checker is MORE trusted, only one where
    # the emitter's disclosure carries MORE. An inert intermediate that reads
    # like a strength gradient is worse than an honest binary, so k > 1 is
    # refused rather than silently aliased to k == 1.
    if level not in (0, 1):
        raise ValueError(
            f"cover level must be 0 or 1; got {level}. There is no safe "
            "intermediate level -- see T-NL17 in docs/analsysis/"
            "hypothesis_validation.md")
    items = list(claims)
    certified_idx = [i for i, c in enumerate(items) if c.certified()]
    remainder = [c for i, c in enumerate(items) if i not in set(certified_idx)]

    # (iii): a remainder claim that is not marked is not a claim we can describe;
    # it is an unattributed one, and the disclosure says so rather than counting it.
    unmarked = [c for c in remainder if not getattr(c, "marked", False)]

    if level == 0:
        sufficient = len(certified_idx) == len(items)
    else:
        # a topic block counts as reached if S reaches it
        need = {_topic(c) for c in items}
        have = {_topic(c) for c in items if c.certified()}
        sufficient = need <= have

    return {
        "schema": SCHEMA,
        "level": level,
        "discloses": sufficient,
        "claims": [c.as_dict() for c in items],
        "n_claims": len(items),
        "n_certified": len(certified_idx),
        "remainder": [c.as_dict() for c in remainder],
        "n_remainder": len(remainder),
        "unattributed": [c.as_dict() for c in unmarked],
        "note": ("This is a DISCLOSURE certificate: it reports what was verified "
                 "and what was declared unverified. It is not a truth claim "
                 "about the remainder, and the false fraction of the remainder "
                 "is not computable from the artifact."),
    }


def mark(claims) -> list:
    """The harness attaches the marks. DETERMINISTIC — a function of the
    authority's output, not a learned behaviour.

    H-NL20 measured that no model retains a learned mark (acquisition 1.00,
    retention 0.00 after one capability round), so this must not be trained.
    A model-supplied mark is a soundness-critical step inside a policy, which is
    the A-combobound shape.
    """
    for c in claims:
        c.marked = True
    return claims


def _topic(c: Claim) -> str:
    """The cover block. Declared in the artifact, never inferred."""
    return c.quantity or ("__untagged__:" + c.text[:40])
