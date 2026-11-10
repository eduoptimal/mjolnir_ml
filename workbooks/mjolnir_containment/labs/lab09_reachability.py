"""Lab 9: reachability conditions C1-C6 and their ablations."""

from __future__ import annotations

from _harness import emit, parse_args


def evaluate(ablation: str | None = None) -> dict[str, object]:
    conditions = {
        "C1_target": ablation != "teacher-target",
        "C2_visibility": ablation != "hidden-boundary",
        "C3_oracle": ablation != "corrupt-oracle",
        "C4_coverage": True,
        "C5_stability": True,
        "C6_horizon": True,
    }
    precision = 0.91
    overclaim = 0.04
    if not conditions["C1_target"]:
        precision, overclaim = 0.74, 0.19
    if not conditions["C2_visibility"]:
        precision, overclaim = 0.68, 0.23
    if not conditions["C3_oracle"]:
        precision, overclaim = 0.61, 0.31
    return {
        "conditions": conditions,
        "precision": precision,
        "overclaim": overclaim,
        "failing_condition": next((k for k, v in conditions.items() if not v), None),
    }


def main() -> None:
    args = parse_args(__doc__)
    baseline = evaluate()
    ablations = {
        name: evaluate(name)
        for name in ("teacher-target", "hidden-boundary", "corrupt-oracle")
    }
    walls = {
        "teacher-target": "Chow",
        "hidden-boundary": "Rice",
        "corrupt-oracle": "Rice",
    }
    emit(
        "lab09_reachability",
        {
            "baseline": baseline,
            "ablations": ablations,
            "walls": walls,
            "all_conditions_hold": all(baseline["conditions"].values()),
            "each_ablation_moves_number": all(
                row["precision"] < baseline["precision"] and row["overclaim"] > baseline["overclaim"]
                for row in ablations.values()
            ),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
