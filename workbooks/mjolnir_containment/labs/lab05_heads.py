"""Lab 5: the density floor and the delegation to the ranker."""

from __future__ import annotations

import math

from _harness import emit, parse_args

IN_DOMAIN = [0.82 + 0.01 * i for i in range(20)]
BATTERY = [-0.20 - 0.01 * i for i in range(20)]
OUT_OF_DOMAIN = [-1.60 - 0.02 * i for i in range(20)]


def floor_from_tails(indomain: list[float], battery: list[float]) -> float:
    return (min(indomain) + max(battery)) / 2


def main() -> None:
    args = parse_args(__doc__)
    floor = floor_from_tails(IN_DOMAIN, BATTERY)
    blocked = [score for score in BATTERY + OUT_OF_DOMAIN if score < floor]
    passed = [score for score in IN_DOMAIN if score >= floor]
    leaks = [score for score in BATTERY + OUT_OF_DOMAIN if score >= floor]
    ranker_releases = [score for score in passed if score >= floor + 0.05]
    emit(
        "lab05_heads",
        {
            "in_domain_min": min(IN_DOMAIN),
            "battery_max": max(BATTERY),
            "out_of_domain_mean": sum(OUT_OF_DOMAIN) / len(OUT_OF_DOMAIN),
            "floor": floor,
            "battery_blocked": len(blocked),
            "battery_leaks": len(leaks),
            "passed_to_ranker": len(passed),
            "ranker_releases": len(ranker_releases),
            "head_is_filter": len(ranker_releases) <= len(passed) and len(leaks) == 0,
            "log_floor": round(math.log(max(floor, 1e-9)), 6),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
