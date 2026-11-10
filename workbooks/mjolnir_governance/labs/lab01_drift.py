"""Lab 1: induce the drift flip.

A static conformal promise is a statement about an exchangeable stream. When
the label relation flips, static coverage collapses. Adaptive conformal
inference (ACI) moves the risk level from observed miscoverage, trading width
for coverage. The lab reports both arms on the same rows.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (DecisionModel, DecisionPolicyBase, RELEASE, dsl, rule)


@dsl
class DriftPolicy(DecisionPolicyBase):
    facts = ("score",)

    assert rule("release_certified",
                when=lambda certified: certified,
                verdict=RELEASE)


class ScoreClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def dataset(n, offset=0.0):
    rows = []
    for i in range(n):
        s = round((i * 0.6180339887498949 + offset) % 1.0, 6)
        rows.append({"score": s})
    return rows


def labels(rows, flip_after=None):
    out = []
    for i, row in enumerate(rows):
        truth = 1 if row["score"] > 0.5 else 0
        if flip_after is not None and i >= flip_after:
            truth = 1 - truth
        out.append(truth)
    return out


def run_arm(adaptive: bool):
    cal_rows = dataset(80, 0.13)
    cal_y = labels(cal_rows)
    dm = DecisionModel(ScoreClf(), DriftPolicy, features=("score",),
                       regime="off")
    if adaptive:
        dm.calibrate(cal_rows, cal_y, alpha=0.10, adaptive=True,
                     gamma=0.10, window=120)
    else:
        dm.calibrate(cal_rows, cal_y, alpha=0.10)

    stream = dataset(200, 0.37)
    stream_y = labels(stream, flip_after=70)
    pre = {"covered": 0, "n": 0, "size": 0}
    post = {"covered": 0, "n": 0, "size": 0}
    for i, (row, y) in enumerate(zip(stream, stream_y)):
        d = dm.dispose(row)
        prediction = d.prediction or {}
        covered = y in prediction.get("set", [])
        slot = pre if i < 70 else post
        slot["covered"] += int(covered)
        slot["size"] += len(prediction.get("set", []))
        slot["n"] += 1
        if adaptive:
            dm.observe(row, y)

    def summary(slot):
        return {"coverage": round(slot["covered"] / slot["n"], 6),
                "mean_set_size": round(slot["size"] / slot["n"], 6)}

    out = {"pre": summary(pre), "post": summary(post)}
    if adaptive:
        out["terminal_alpha"] = round(dm.kappa.alpha_at("__all__"), 6)
    return out


def main() -> None:
    args = parse_args(__doc__)
    static = run_arm(False)
    adaptive = run_arm(True)
    emit(
        "lab01_drift",
        {
            "static": static,
            "adaptive": adaptive,
            "static_coverage_collapses": (
                static["post"]["coverage"] < static["pre"]["coverage"]),
            "adaptive_falls_less": (
                adaptive["post"]["coverage"] > static["post"]["coverage"]),
            "adaptive_set_size_rises": (
                adaptive["post"]["mean_set_size"]
                > adaptive["pre"]["mean_set_size"]),
            "terminal_alpha_below_start": adaptive["terminal_alpha"] < 0.10,
            "claim": "adaptive tracks the stream; it does not restore 0.90",
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
