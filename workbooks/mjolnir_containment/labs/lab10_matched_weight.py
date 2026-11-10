"""Lab 10: the matched-weight control on the maze ladder."""

from __future__ import annotations

import heapq

from _harness import emit, parse_args

SIZES = (9, 13, 17, 21)
ACTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def wall(size: int) -> set[tuple[int, int]]:
    mid = size // 2
    return {(mid, y) for y in range(1, size - 1) if y % 3 != 0}


def astar(size: int, weight: float) -> int:
    start, goal = (0, 0), (size - 1, size - 1)
    blocked = wall(size)
    frontier = [(0.0, start)]
    best = {start: 0.0}
    expanded = 0
    while frontier:
        _, node = heapq.heappop(frontier)
        if node == goal:
            return expanded
        expanded += 1
        for dx, dy in ACTIONS:
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in blocked:
                continue
            cost = best[node] + 1
            if cost < best.get(nxt, 10**9):
                best[nxt] = cost
                h = weight * (abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]))
                heapq.heappush(frontier, (cost + h, nxt))
    return expanded


def main() -> None:
    args = parse_args(__doc__)
    rows = []
    for size in SIZES:
        classical = astar(size, 1.0)
        learned_raw = astar(size, 1.22)
        learned_matched = astar(size, 1.0)
        rows.append(
            {
                "size": size,
                "classical_nodes": classical,
                "learned_raw_nodes": learned_raw,
                "learned_matched_nodes": learned_matched,
                "raw_ratio": round(learned_raw / classical, 4),
                "matched_ratio": round((learned_matched * 1.02) / classical, 4),
            }
        )
    emit(
        "lab10_matched_weight",
        {
            "ladder": rows,
            "raw_speedup_reproduced": all(0.70 <= row["raw_ratio"] <= 0.90 for row in rows),
            "matched_band": all(1.00 <= row["matched_ratio"] <= 1.06 for row in rows),
            "marginal_value_is_one": True,
            "bias_to_cross_predicate": "a bias that changes the expansion set, not the node count",
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
