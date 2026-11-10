"""Lab 7: Mondrian calibration and continuous envelopes.

Pooled conformal quantiles can under-cover a small group. Mondrian calibration
computes the quantile per declared region and refuses a region below the
conformal floor rather than substituting the pooled value. The second half
compares a constant split interval with a CQR interval whose width follows the
input.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (CalibrationError, DecisionModel, DecisionPolicyBase,
                        RELEASE, dsl, rule)


@dsl
class GroupPolicy(DecisionPolicyBase):
    facts = ("score", "g")

    assert rule("certified_release",
                when=lambda certified, p_positive: certified and p_positive < 0.5,
                verdict=RELEASE)


@dsl
class IntervalPolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("safe_band",
                when=lambda lo, hi: lo >= -5.0 and hi <= 15.0,
                verdict=RELEASE)


class ScoreClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


class IdentityReg:
    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [row[0] for row in X]


class LowerQuantile:
    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [0.5 * row[0] - 1.0 for row in X]


class UpperQuantile:
    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1.5 * row[0] + 1.0 for row in X]


def group_rows():
    rows, y = [], []
    for g, offset in (("A", 0.0), ("B", 0.25)):
        for i in range(30):
            s = round((i * 0.6180339887498949 + offset) % 1.0, 6)
            rows.append({"score": s, "g": g})
            y.append(1 if s > 0.5 else 0)
    return rows, y


def coverage_by_group(dm, rows, y, *, use_row_group):
    out = {}
    for row, label in zip(rows, y):
        d = dm.dispose(row)
        prediction = d.prediction or {}
        group = row["g"] if use_row_group else prediction.get("calibrated_group")
        covered = label in prediction.get("set", [])
        slot = out.setdefault(group, {"covered": 0, "n": 0, "size": 0})
        slot["covered"] += int(covered)
        slot["size"] += len(prediction.get("set", []))
        slot["n"] += 1
    return {g: {"coverage": round(v["covered"] / v["n"], 6),
                "mean_set_size": round(v["size"] / v["n"], 6)}
            for g, v in out.items()}


def main() -> None:
    args = parse_args(__doc__)
    rows, y = group_rows()

    pooled = DecisionModel(ScoreClf(), GroupPolicy,
                           features=("score", "g"), regime="off")
    pooled.calibrate(rows, y, alpha=0.10)

    mondrian = DecisionModel(ScoreClf(), GroupPolicy,
                             features=("score", "g"), regime="off")
    mondrian.calibrate(rows, y, alpha=0.10, group="g")

    pooled_cov = coverage_by_group(pooled, rows, y, use_row_group=True)
    mondrian_cov = coverage_by_group(mondrian, rows, y, use_row_group=False)

    under = DecisionModel(ScoreClf(), GroupPolicy,
                          features=("score", "g"), regime="off")
    small_rows = rows[:30] + [{"score": 0.5, "g": "C"} for _ in range(3)]
    small_y = y[:30] + [1, 0, 1]
    try:
        under.calibrate(small_rows, small_y, alpha=0.10, group="g")
        undersampled = {"refused": False}
    except CalibrationError as ex:
        undersampled = {"refused": True, "error": str(ex), "certified": False}

    reg = DecisionModel(IdentityReg(), IntervalPolicy, features=("x",),
                        regime="off")
    cal_rows = [{"x": i / 10.0} for i in range(40)]
    cal_y = [row["x"] + 0.2 * ((i % 3) - 1) for i, row in enumerate(cal_rows)]
    reg.calibrate(cal_rows, cal_y, alpha=0.10, method="cqr",
                  lower=LowerQuantile(), upper=UpperQuantile())
    widths = {}
    for x in (1.0, 4.0):
        d = reg.dispose({"x": x})
        widths[str(x)] = round(
            (d.prediction or {}).get("hi", 0.0)
            - (d.prediction or {}).get("lo", 0.0), 6)

    emit(
        "lab07_groups",
        {
            "pooled_coverage": pooled_cov,
            "mondrian_coverage": mondrian_cov,
            "groups_differ": pooled_cov != mondrian_cov,
            "undersampled_group": undersampled,
            "cqr_widths": widths,
            "cqr_width_varies": len(set(widths.values())) > 1,
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
