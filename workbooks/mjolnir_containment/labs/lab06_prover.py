"""Lab 6: the vacuity guard and the detached checker."""

from __future__ import annotations

from dataclasses import dataclass

from _harness import emit, parse_args


@dataclass(frozen=True)
class Claim:
    name: str
    premise_dependent: bool
    truth: bool
    shared_generator_agrees: bool


CLAIMS = (
    Claim("range-bound", False, True, True),
    Claim("single-writer", False, True, True),
    Claim("typed-return", False, True, True),
    Claim("guard-total", False, True, True),
    Claim("premise-schema", True, True, True),
    Claim("premise-scale", True, True, True),
    Claim("empty-set", False, False, True),
    Claim("off-by-one", False, False, True),
    Claim("stale-digest", False, True, True),
    Claim("unbounded-loop", False, False, True),
)


def census(detached: bool) -> dict[str, int]:
    result = {"proved": 0, "declared": 0, "declined": 0, "refuted": 0}
    for claim in CLAIMS:
        if claim.premise_dependent:
            result["declared"] += 1
        elif not claim.truth:
            if detached:
                result["refuted"] += 1
            else:
                result["proved"] += 1
        elif detached:
            result["proved"] += 1
        else:
            result["proved"] += 1
    return result


def main() -> None:
    args = parse_args(__doc__)
    shared = census(detached=False)
    detached = census(detached=True)
    alarm = detached != shared
    emit(
        "lab06_prover",
        {
            "claims": len(CLAIMS),
            "shared_generator": shared,
            "detached_checker": detached,
            "shared_generator_certified": shared["proved"] + shared["declared"],
            "detached_alarm": alarm,
            "disagreement": {
                "proved": detached["proved"] - shared["proved"],
                "refuted": detached["refuted"] - shared["refuted"],
            },
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
