"""Lab 1: the reactive containment gate and the five-build family."""

from __future__ import annotations

from dataclasses import dataclass

from _harness import emit, parse_args

CERTIFIED = "CERTIFIED"
HONEST_ABSTAIN = "HONEST_ABSTAIN"


@dataclass(frozen=True)
class Build:
    name: str
    confidence: float
    on_fragment: bool
    certified: bool


BUILDS = (
    Build("baseline", 0.91, True, True),
    Build("overconfident", 0.99, False, False),
    Build("calibrated", 0.72, True, True),
    Build("drifting", 0.88, False, True),
    Build("cascade-repaired", 0.81, True, True),
)


def authority(build: Build) -> str:
    return CERTIFIED if build.on_fragment and build.certified else HONEST_ABSTAIN


def guard(build: Build) -> str:
    verdict = authority(build)
    if verdict == CERTIFIED:
        return "release"
    return "block:" + verdict.lower()


def main() -> None:
    args = parse_args(__doc__)
    rows = []
    for build in BUILDS:
        verdict = authority(build)
        rows.append(
            {
                "build": build.name,
                "proposer_confidence": build.confidence,
                "authority": verdict,
                "guard": guard(build),
            }
        )

    releases = sum(row["guard"] == "release" for row in rows)
    blocked = len(rows) - releases
    off_fragment_blocks = sum(
        row["guard"].startswith("block") and not build.on_fragment
        for row, build in zip(rows, BUILDS)
    )
    emit(
        "lab01_paralysis",
        {
            "builds": rows,
            "released": releases,
            "blocked": blocked,
            "off_fragment_blocks": off_fragment_blocks,
            "cascade_repaired": rows[-1]["guard"] == "release",
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
