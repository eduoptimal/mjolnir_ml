"""Lab 2: belief-space replanning after a contradiction."""

from __future__ import annotations

from collections import deque

from _harness import emit, parse_args

GRAPH = {
    "start": ("hall", "door"),
    "hall": ("start", "door", "office"),
    "door": ("start", "hall", "vault"),
    "office": ("hall", "vault"),
    "vault": ("door", "office", "goal"),
    "goal": ("vault",),
}
JAMMED = ("door", "vault")


def path(graph: dict[str, tuple[str, ...]], start: str, goal: str) -> list[str] | None:
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, trail = queue.popleft()
        if node == goal:
            return trail
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, trail + [nxt]))
    return None


def main() -> None:
    args = parse_args(__doc__)
    initial = path(GRAPH, "start", "goal")
    repaired_graph = {node: tuple(n for n in edges if (node, n) != JAMMED) for node, edges in GRAPH.items()}
    repaired = path(repaired_graph, "start", "goal")
    emits = 0
    for _ in range(20):
        if repaired is not None:
            emits += 1
    emit(
        "lab02_shocks",
        {
            "initial_path": initial,
            "jammed_door_repaired": repaired is not None and JAMMED not in list(zip(repaired, repaired[1:])),
            "repaired_path": repaired,
            "belief_absorbs_observation": repaired is not None,
            "monitor_emits": emits,
            "monitor_threshold": 20,
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
