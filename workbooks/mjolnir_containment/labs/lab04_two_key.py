"""Lab 4: the two-key release harness and the spy battery."""

from __future__ import annotations

from _harness import emit, parse_args

CASES = []
for index in range(100):
    allowed = index % 4 != 0
    hazard = index % 7 == 0
    CASES.append((f"case-{index:03d}", allowed, hazard))


def authority(record: tuple[str, bool, bool]) -> str:
    _, allowed, _ = record
    return "ROUTE" if allowed else "BLOCKED"


def reader(record: tuple[str, bool, bool]) -> str:
    _, _, hazard = record
    return "VETO" if hazard else "ROUTE"


def release(record: tuple[str, bool, bool], reader_enabled: bool) -> tuple[bool, str]:
    stage = authority(record)
    if stage != "ROUTE":
        return False, "blocked-by-authority"
    if reader_enabled:
        if reader(record) != "ROUTE":
            return False, "vetoed-by-reader"
        return True, "released"
    return True, "released-authority-only"


def main() -> None:
    args = parse_args(__doc__)
    released = 0
    denied_calls = 0
    stages: dict[str, int] = {}
    for record in CASES:
        ok, stage = release(record, True)
        stages[stage] = stages.get(stage, 0) + 1
        if ok:
            released += 1
        if not record[1]:
            denied_calls += 1 if ok else 0

    authority_only_stages: dict[str, int] = {}
    for record in CASES:
        _, stage = release(record, False)
        authority_only_stages[stage] = authority_only_stages.get(stage, 0) + 1

    emit(
        "lab04_two_key",
        {
            "battery": len(CASES),
            "released": released,
            "stages": stages,
            "spy_calls_on_denied": denied_calls,
            "reader_disabled_stages": authority_only_stages,
            "degradation_named": "released-authority-only",
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
