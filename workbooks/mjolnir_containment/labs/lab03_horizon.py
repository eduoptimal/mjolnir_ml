"""Lab 3: horizon soundness with one-step and three-step guards."""

from __future__ import annotations

from dataclasses import dataclass

from _harness import emit, parse_args

START = (1, 2)
GOAL = (4, 4)
HAZARD = (2, 2)
ACTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass(frozen=True)
class Step:
    position: tuple[int, int]
    action: tuple[int, int]


def inside(point: tuple[int, int]) -> bool:
    return 0 <= point[0] < 5 and 0 <= point[1] < 5


def distance(point: tuple[int, int]) -> int:
    return abs(point[0] - GOAL[0]) + abs(point[1] - GOAL[1])


def one_step_guard(state: tuple[int, int]) -> tuple[int, int]:
    choices = [
        (state[0] + dx, state[1] + dy)
        for dx, dy in ACTIONS
        if inside((state[0] + dx, state[1] + dy))
    ]
    return min(choices, key=distance)


def three_step_guard(state: tuple[int, int]) -> tuple[int, int]:
    best = None
    best_score = 10**9
    for dx, dy in ACTIONS:
        nxt = (state[0] + dx, state[1] + dy)
        if not inside(nxt) or nxt == HAZARD:
            continue
        score = distance(nxt) + 3 * (1 if nxt == HAZARD else 0)
        if score < best_score:
            best_score = score
            best = nxt
    return best if best is not None else state


def rollout(policy: str) -> dict[str, object]:
    state = START
    trail = [state]
    collisions = 0
    guard = one_step_guard if policy == "one-step" else three_step_guard
    for _ in range(3):
        state = guard(state)
        if state == HAZARD:
            collisions += 1
        trail.append(state)
    return {"policy": policy, "trail": trail, "collisions": collisions}


def main() -> None:
    args = parse_args(__doc__)
    one = rollout("one-step")
    three = rollout("three-step")
    emit(
        "lab03_horizon",
        {
            "start": START,
            "hazard": HAZARD,
            "one_step": one,
            "three_step": three,
            "collisions_reduced": one["collisions"] > three["collisions"],
            "residual": three["collisions"],
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
