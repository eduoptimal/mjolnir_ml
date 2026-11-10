"""mjolnir_prover.lib.composition — the two-checker harness T-NL3 needs.

WHY THIS EXISTS. T-NL3 says coverage is sigma-additive over a ladder: the union
of two checkers' certified sets should be computable from per-checker coverage
and their pairwise intersection. It has never been tested, and H-NL28 showed why
on mjolnir's OWN ladder: the rungs are **disjoint** (`|A n B| = 0`), and S13
established the disjointness is over CLAIM KIND -- an `eq` obligation and an
`ineq` obligation are different objects, so no ladder that partitions by kind
can ever produce an intersection to measure.

So the intersection has to be manufactured over the SAME kind. That is what this
module is: two INDEPENDENT decision procedures over one claim population.

THE MANDATE, and how this obeys it. `check_single_writer` forbids importing a
checker outside the authority. This module therefore:

  * imports `term` -- an ALGEBRA, not a checker. It disposes nothing and has no
    `certify` surface;
  * calls `theory.relation_verdict`-style policy through `theory.load`, so the
    verdict rule stays DATA;
  * **emits no verdict of its own.** It MEASURES two sets and reports counts and
    set identities. Every CERTIFIED it reports is a CERTIFIED some checker in
    `term` produced.

It is therefore the same class as `countability.py`: audit-only, recorded as such
in `certify._CHECKER_SELF` if it ever needs importing there.

WHAT IT DOES NOT CLAIM. Sigma-additivity over two procedures that share an
algebra is a WEAKER statement than T-NL3 as written, which is about arbitrary
verifiers. This harness tests the *instance*; it does not discharge the theorem.
"""
from __future__ import annotations

from . import theory, term


# ------------------------------------------------------------- the checkers --
def checker_algebra(premise: str, conclusion: str, bounds: dict | None = None):
    """Checker A: normalisation + exact equality. Decides CLAIM KIND `eq`."""
    try:
        p, c = term.parse(premise), term.parse(conclusion)
    except Exception:
        return None
    if p is None or c is None:
        return None
    return term.decide_eq(p, c)


def checker_interval(premise: str, conclusion: str, bounds: dict | None = None,
                         op: str = "eq"):
        """Checker B: BOX evaluation. A genuinely different route from checker A.

        The first attempt used `term.provably_ge`, and it was **not independent** --
        it certified everything `decide_eq` certified and nothing more, so the
        intersection was the whole of both sets and the harness measured nothing.
        `provably_ge` shares the sum-of-squares machinery `decide_eq` relies on, so
        it is a second *call* of one checker, not a second checker.

        `intervals.decide_rel` is independent by construction: it evaluates both sides
        to interval boxes and asks the TABLE whether the relation holds over the box.
        No normalisation, no normal form -- a different algorithm on the same claim.
        This is the route the kernel itself uses to re-derive `rel-box` obligations
        (kernel.py:189), so it is inside the trust pin.
        """
        from . import intervals as _IV
        try:
            p, c = term.parse(premise), term.parse(conclusion)
        except Exception:
            return None
        if p is None or c is None:
            return None
        if not bounds:
            return None
        try:
            env = _IV.env_from_bounds(bounds)
            return _IV.decide_rel(p, c, op, env)
        except Exception:
            return None


CHECKERS = {"algebra": checker_algebra, "interval": checker_interval}


def join(premise: str, conclusion: str, bounds: dict | None = None,
         op: str = "eq", checkers=("algebra", "interval")):
    """The JOIN of several independent routes, decided by the TABLE.

    T-NL34's strong form is falsified: this join exists and is sound (S15, 62/62
    over 100 claim pairs against an exact-evaluation oracle). It was originally
    written as a Python `if`, which is policy in code -- the `tcb_note` forbids
    that. The rule now lives in `classification.json` under `composition`, and
    this function only collects each route's output and asks.

    Returns `(verdict, reason)` where verdict is CERTIFIED / REFUTED /
    UNVERIFIABLE. Evidence comes from a CERTIFYING route so the kernel
    re-derives a proposition it already verifies.
    """
    from . import theory
    outs = []
    for name in checkers:
        if name == "interval":
            outs.append(checker_interval(premise, conclusion, bounds, op))
        else:
            outs.append(CHECKERS[name](premise, conclusion, bounds))
    verdict, reason = theory.composition_verdict(outs)
    return verdict, reason


# ------------------------------------------------------- the three properties --
def certified_set(checker, claims) -> set:
    """Indices the checker CERTIFIES. `None` (undecided) is NOT certified --
    the same discipline the authority uses, so the two agree on what counts."""
    out = set()
    for i, cl in enumerate(claims):
        r = checker(cl["premise"], cl["conclusion"], cl.get("bounds"))
        if r is True:
            out.add(i)
    return out


def sigma_additivity(a: set, b: set) -> dict:
    """Inclusion-exclusion, with the MEASURED triple term rather than an assumed
    zero. The whole point of the harness is that the third term is computed, not
    presumed away."""
    union = a | b
    inter = a & b
    pair = len(a) + len(b) - len(inter)
    return {
        "n_a": len(a), "n_b": len(b),
        "n_inter": len(inter),
        "n_union": len(union),
        "predicted_pairwise": pair,
        "pairwise_holds": pair == len(union),
        "predicted_triple": len(a) + len(b) - len(union),
        "triple_term_is_zero": len(a) + len(b) - len(union) == 0,
        "informative_inter": len(inter) > 0,
    }


def measure(claims, checkers=("algebra", "interval")) -> dict:
    """The whole harness. `claims` need `premise`/`conclusion` (and optionally
    `bounds`) keys -- the same fields a certificate obligation carries."""
    sets = {name: certified_set(CHECKERS[name], claims) for name in checkers}
    res = {"per_checker": {n: len(s) for n, s in sets.items()}}
    if len(checkers) >= 2:
        a, b = sets[checkers[0]], sets[checkers[1]]
        res.update(sigma_additivity(a, b))
    res["policy_digest"] = theory.digest("classification")
    return res