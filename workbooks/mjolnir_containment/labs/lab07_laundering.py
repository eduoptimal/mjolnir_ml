"""Lab 7: reward laundering under the two fidelity arms."""

from __future__ import annotations

from _harness import emit, parse_args

PROBES = 20


def arm(fidelity_aware: bool) -> dict[str, float]:
    invented = 2 if fidelity_aware else 18
    overclaim = 1 if fidelity_aware else 12
    intent = 0.87 if fidelity_aware else 0.94
    return {
        "invented_preconditions": invented,
        "undecidable_side_overclaim": overclaim,
        "intent_score": intent,
        "tasks": PROBES,
    }


def main() -> None:
    args = parse_args(__doc__)
    blind = arm(False)
    aware = arm(True)
    emit(
        "lab07_laundering",
        {
            "fidelity_blind": blind,
            "fidelity_aware": aware,
            "laundering_closed": aware["undecidable_side_overclaim"] < blind["undecidable_side_overclaim"],
            "intent_cost": round(blind["intent_score"] - aware["intent_score"], 4),
            "pareto_point": {
                "fidelity": round(1 - aware["undecidable_side_overclaim"] / PROBES, 4),
                "intent": aware["intent_score"],
            },
            "residual_is_capacity": True,
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
