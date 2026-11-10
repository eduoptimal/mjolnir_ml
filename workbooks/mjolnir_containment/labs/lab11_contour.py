"""Lab 11: the f-contour and the regime qualifier."""

from __future__ import annotations

from _harness import emit, parse_args

CELLS = ("urban", "suburban", "rural")


def contour(volume: int, bias: str) -> float:
    if bias == "global":
        return round(volume * 0.72, 4)
    return round(volume * 1.01, 4)


def main() -> None:
    args = parse_args(__doc__)
    volumes = {"urban": 120, "suburban": 80, "rural": 40}
    local = {cell: contour(value, "local") for cell, value in volumes.items()}
    global_bias = {cell: contour(value, "global") for cell, value in volumes.items()}
    pooled = round(sum(global_bias.values()) / sum(local.values()), 4)
    cells = []
    for cell in CELLS:
        ratio = round(global_bias[cell] / local[cell], 4)
        cells.append(
            {
                "cell": cell,
                "local_volume": local[cell],
                "global_volume": global_bias[cell],
                "ratio": ratio,
                "crosses": ratio < 1.0,
            }
        )
    emit(
        "lab11_contour",
        {
            "cells": cells,
            "pooled_ratio": pooled,
            "global_reduces_volume": all(row["crosses"] for row in cells),
            "local_preserved": all(local[cell] >= volumes[cell] for cell in CELLS),
            "aggregate_without_qualifier_fails": pooled >= 0.95,
            "certified_result": "cell-level",
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
