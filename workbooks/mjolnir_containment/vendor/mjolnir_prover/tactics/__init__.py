"""mjolnir_prover.tactics — the SOFT / disclosure certificate over a set of claims.

`mjolnir_prover.certify` answers one question about one artifact: does this claim follow
from these premises? A chat-shaped or table-shaped output is a BAG of claims, so
there is no single artifact. This package certifies the SET:

    extract -> route -> dispose -> mark -> disclose

with ONE invariant that governs the whole chain:

    ONLY `dispose` may return a verdict. `extract` and `route` return no
    verdict type, and the `Router` class has no `certify` method at all. That
    absence is the guarantee, and it is checkable rather than conventional.

WHAT IT IS NOT
    not a truth certificate — S-Omega = 1.000 at a 0.992 false fraction (H-NL13)
    not a quality score      — the false fraction is not computable (T-NL18), so
                               no scalar is exposed
    not prose                — prose is a DECLINED class, 0/297 on a measured
                               100% false-answer rate

The honest output is `(dispositions, remainder)`. If you want a number, the
number is the remainder's SIZE, and it is under the emitter's control (H-NL19) —
so treat it as a disclosure, not a rating.
"""
from __future__ import annotations

from ..lib import compose, frontend_table, router
from ..lib.disclosure import Claim, disclose, mark


def certify(claims, source: dict, *, level: int = 1, adapter=None) -> dict:
    """The whole chain. `claims` and `source` are SEPARATE inputs, and that is
    the whole design.

    THE SIGNATURE IS A CORRECTNESS REQUIREMENT. The first version took a single
    `source` and generated the claims from it. That is a SELF-CONSISTENCY check:
    a document whose value was mutated produced a claim matching the mutation,
    and it CERTIFIED — the pipeline could not detect a disagreement by
    construction, because it was the author of both sides.

    A claim comes from an EMITTER. The source is what it is checked AGAINST.
    Merging them makes the chain a tautology, and a tautology is exactly what a
    certificate must never be — the A-combobound shape at the architecture level
    rather than inside one rule.

    `adapter` replaces the router. Because the interface cannot dispose,
    substituting a fallible one cannot introduce a verdict.
    """
    r = (adapter or router.Router)()
    claims = list(claims)

    # stage 2: route (may abstain)
    for c in claims:
        dest, ev = r.route(c.quantity, c.value, source)
        c.destination, c.evidence = dest, (ev if dest else None)
        # Keep the router's REASON. The router distinguishes "the source does not
        # contain this quantity" from "I cannot decide this one", and flattening
        # that into "router abstained" made FABRICATION indistinguishable from
        # honest uncertainty -- which silently exited 0 on five fabricated metrics.
        if not dest and isinstance(ev, str) and ev:
            c.reason = ev

    # stage 3: ONLY here does a verdict appear, and only from the authority
    for c in claims:
        if c.destination is None:
            c.verdict = "UNVERIFIABLE"
            if not c.reason:
                c.reason = "router abstained"
            continue
        stated = source[c.quantity]
        if isinstance(stated, (list, tuple)):
            # a RELATION: one referent, several stated values. The authority
            # returns the MATCHED value as evidence, so a certified claim is
            # re-derivable as the single-value proposition the kernel knows.
            c.verdict, c.evidence = compose.dispose_relation(
                c.quantity, c.value, list(stated))
            c.reason = f"disposed at {c.destination} (relation)"
        else:
            c.verdict = compose.dispose_single(c.quantity, c.value, stated)
            c.reason = f"disposed at {c.destination}"

    # stage 4: the harness marks. Deterministic, never learned (H-NL20).
    mark(claims)
    return disclose(claims, level=level)


def self_check(source: dict, *, level: int = 1) -> dict:
    """Certificate the source against ITSELF. Every claim will agree, and that
    agreement is VACUOUS — this is a self-consistency check, useful for testing
    the plumbing and for showing the shape of the output. It is NOT evidence of
    grounding. Use `certify(claims, source)` for that."""
    return certify(frontend_table.extract(source), source, level=level)


__all__ = ["certify", "self_check", "Claim", "disclose", "mark", "router",
           "compose", "frontend_table"]
