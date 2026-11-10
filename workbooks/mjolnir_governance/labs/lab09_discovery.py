"""Lab 9: the refused admission.

The proposer proposes; the oracle evaluates; the census reports; the gate
admits. Self-certification is refused, and a candidate that fails a labelled
test is declined by name.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (CENSUS_VERDICTS, CERTIFIED, DECLINED, Candidate,
                        DiscoverySession)


def corpus():
    return {f"claim{i}": DECLINED for i in range(10)}


def oracle(claim: str) -> str:
    return CERTIFIED if claim == "claim0" else DECLINED


def candidate(claims, tests, proposed_by="proposer"):
    return Candidate(
        kind="rung",
        shape="conjunction",
        claims=tuple(claims),
        tests=tuple(tests),
        twin=(("claim9", DECLINED),),
        proposed_by=proposed_by,
    )


def main() -> None:
    args = parse_args(__doc__)
    session = DiscoverySession(corpus())

    good = candidate(["claim0", "claim1"], [("claim0", CERTIFIED)])
    admission = session.promote(good, oracle, oracle_id="oracle")

    after = session.corpus
    census = {v: sum(1 for x in after.values() if x == v)
              for v in CENSUS_VERDICTS}

    bad = candidate(["claim2", "claim1"], [("claim2", CERTIFIED)])
    refused = session.promote(bad, oracle, oracle_id="oracle")

    self_cert = candidate(["claim3"], [("claim3", CERTIFIED)],
                          proposed_by="oracle-self")
    try:
        session.evaluate(self_cert, oracle, oracle_id="oracle-self")
        alarm = None
    except PermissionError as ex:
        alarm = str(ex)

    suspect = candidate(["claim4"], [("claim4", CERTIFIED)],
                        proposed_by="oracle-self")
    self_refusal = session.promote(suspect, oracle, oracle_id="oracle")

    emit(
        "lab09_discovery",
        {
            "census_keys": sorted(census),
            "census_has_four_counts": set(census) == set(CENSUS_VERDICTS),
            "census": census,
            "admission_approved": admission.approved,
            "refused_admission": not refused.approved,
            "refused_reason": refused.reason,
            "self_certification_alarm": alarm,
            "self_certification_refused": not self_refusal.approved,
            "journal_verifies": session.journal.verify(),
            "decline_frontier": {
                "reason": "ambiguous",
                "undecided": census.get(DECLINED, 0),
            },
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
