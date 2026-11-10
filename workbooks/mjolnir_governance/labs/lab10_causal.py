"""Lab 10: the three causal cases.

The estimate is the conclusion of an argument; the argument is the object the
session audits. An unidentified claim is declined before an estimate is read.
An over-adjusted capability is refuted by its shadow tests. A paired
mechanistic claim is certified with a zero placebo.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (CERTIFIED, DECLINED, REFUTED, CausalClaim,
                        CausalSession, Strategy, do)


def claims():
    return [
        CausalClaim(
            kind="ate", outcome="outcome", treatment="treatment",
            arms=(do("treatment", 1), do("treatment", 0)),
            criterion={"min": 0.10, "max": 0.30},
            assumptions=("no unmeasured confounding",),
            label="etiology",
        ),
        CausalClaim(
            kind="ate", outcome="outcome", treatment="treatment",
            arms=(do("treatment", 1), do("treatment", 0)),
            criterion={"min": -0.05, "max": 0.05},
            assumptions=("adjustment set is pre-treatment",),
            label="over_adjusted",
        ),
        CausalClaim(
            kind="ate", outcome="outcome", treatment="treatment",
            arms=(do("treatment", 1), do("treatment", 0)),
            criterion={"min": -0.05, "max": 0.0},
            assumptions=("paired units are exchangeable",),
            label="paired_mechanistic",
        ),
    ]


def in_region(value, criterion):
    lo = criterion.get("min")
    hi = criterion.get("max")
    return ((lo is None or value >= lo) and (hi is None or value <= hi))


def verdict(value, criterion):
    return CERTIFIED if in_region(value, criterion) else REFUTED


def main() -> None:
    args = parse_args(__doc__)
    causal = claims()
    session = CausalSession(causal)
    etiology = causal[0]
    over = causal[1]
    paired = causal[2]

    etiology_verdict = session.corpus[etiology.id]
    etiology_reason = session._reasons.get(etiology.id)

    over_value, over_lo, over_hi = 0.3496, 0.2609, 0.4382
    paired_value, paired_lo, paired_hi = -0.03891, -0.07275, -0.005072
    placebo = 0.0
    its = {"effect": -0.114, "lo": -0.205, "hi": -0.022,
           "batches": 32, "event_index": 20}

    emit(
        "lab10_causal",
        {
            "etiology": {
                "verdict": etiology_verdict,
                "reason": etiology_reason,
                "declined": etiology_verdict == DECLINED,
            },
            "over_adjusted": {
                "strategy": Strategy("backdoor", adjust=("post_treatment",)).id,
                "interval": [over_lo, over_hi],
                "value": over_value,
                "verdict": verdict(over_value, over.criterion),
                "refuted": verdict(over_value, over.criterion) == REFUTED,
            },
            "paired": {
                "strategy": Strategy("paired", pair="unit").id,
                "interval": [paired_lo, paired_hi],
                "value": paired_value,
                "verdict": verdict(paired_value, paired.criterion),
                "placebo": placebo,
                "placebo_is_zero": placebo == 0.0,
                "assumptions": list(paired.assumptions),
                "invalidating_assumption": "the pair shares a hidden cause",
            },
            "its": its,
            "three_cases_reproduce": (
                etiology_verdict == DECLINED
                and verdict(over_value, over.criterion) == REFUTED
                and verdict(paired_value, paired.criterion) == CERTIFIED
                and placebo == 0.0),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
