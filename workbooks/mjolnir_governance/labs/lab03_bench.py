"""Lab 3: pre-register the head-to-head.

Twenty-five operating points are swept over one frozen batch. The pair is
computed at each point, and its precision is correlated against the held-out
failure rate. A scalar comparator is correlated on the same points. The sign
and the interval are fixed by construction before the result is read.
"""

from __future__ import annotations

import random

from _harness import emit, parse_args

from mjolnir_ml import RELEASE, omega_pair


def batch(n=200):
    rows = []
    for i in range(n):
        s = round((i * 0.6180339887498949) % 1.0, 6)
        rows.append({"score": s, "safe": s > 0.45})
    return rows


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def bootstrap_ci(xs, ys, *, seed=0, b=400):
    rng = random.Random(seed)
    n = len(xs)
    vals = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(pearson([xs[i] for i in idx], [ys[i] for i in idx]))
    vals.sort()
    return vals[int(0.025 * b)], vals[int(0.975 * b) - 1]


def main() -> None:
    args = parse_args(__doc__)
    rows = batch()
    thresholds = [0.30 + 0.30 * i / 24.0 for i in range(25)]
    points = []
    for t in thresholds:
        verdicts = [RELEASE if row["score"] < t else "deny" for row in rows]
        safe = [row["safe"] for row in rows]
        decidable = [True] * len(rows)
        pair = omega_pair(verdicts, safe, decidable)
        failures = sum(1 for v, s in zip(verdicts, safe)
                       if v == RELEASE and not s) / len(rows)
        prec = pair.prec if pair.prec is not None else 0.0
        scalar = prec * pair.rec
        points.append({
            "threshold": round(t, 6),
            "prec": pair.prec,
            "rec": pair.rec,
            "failure": round(failures, 6),
            "scalar": round(scalar, 6),
        })

    usable = [p for p in points if p["prec"] is not None]
    xs_pair = [p["prec"] for p in usable]
    ys_fail = [p["failure"] for p in usable]
    xs_scalar = [p["scalar"] for p in usable]

    rho_pair = pearson(xs_pair, ys_fail)
    rho_scalar = pearson(xs_scalar, ys_fail)
    ci = bootstrap_ci(xs_pair, ys_fail)

    emit(
        "lab03_bench",
        {
            "n_points": len(points),
            "n_usable": len(usable),
            "rho_pair": round(rho_pair, 6),
            "rho_pair_ci": [round(ci[0], 6), round(ci[1], 6)],
            "rho_scalar": round(rho_scalar, 6),
            "preregistered_sign": "negative",
            "sign_held": rho_pair < 0.0 and ci[1] < 0.0,
            "pair_beats_scalar": rho_pair < rho_scalar,
            "points": points,
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
