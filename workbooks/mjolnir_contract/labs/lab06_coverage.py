"""Lab 6: coverage, set size, and the zero-claim cell.

Split-conformal calibration turns a point score into a set. The lab reports
empirical coverage, mean set size, and the singleton rate, then repeats with a
smaller calibration split. A zero-claim policy reports vacuous precision.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (DecisionModel, DecisionPolicyBase, RELEASE, dsl,
                        omega_pair, rule)


@dsl
class CoveragePolicy(DecisionPolicyBase):
    facts = ("score",)

    assert rule("certified_low_risk",
                when=lambda certified, p_positive: certified and p_positive < 0.50,
                verdict=RELEASE)


class ScoreClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def dataset(n):
    rows, y = [], []
    for i in range(n):
        s = round((i * 0.6180339887498949) % 1.0, 6)
        rows.append({"score": s})
        y.append(1 if s > 0.5 else 0)
    return rows, y


def evaluate(cal_n: int):
    rows, y = dataset(120)
    cal_rows, cal_y = rows[:cal_n], y[:cal_n]
    test_rows, test_y = rows[cal_n:cal_n + 60], y[cal_n:cal_n + 60]
    dm = DecisionModel(ScoreClf(), CoveragePolicy, features=("score",),
                       regime="off")
    dm.calibrate(cal_rows, cal_y, alpha=0.10)
    disps = [dm.dispose(row) for row in test_rows]
    covered = sum(
        y_i in (d.prediction or {}).get("set", [])
        for d, y_i in zip(disps, test_y)
    )
    sizes = [len((d.prediction or {}).get("set", [])) for d in disps]
    return {
        "n_cal": cal_n,
        "n_test": len(test_rows),
        "coverage": round(covered / len(test_rows), 6),
        "mean_set_size": round(sum(sizes) / len(sizes), 6),
        "singleton_rate": round(sum(s == 1 for s in sizes) / len(sizes), 6),
        "releases": sum(d.verdict == RELEASE for d in disps),
    }


def main() -> None:
    args = parse_args(__doc__)
    large = evaluate(80)
    small = evaluate(24)
    zero_claim = omega_pair(["abstain"] * 10, [True] * 10, [True] * 10)

    emit(
        "lab06_coverage",
        {
            "large_calibration": large,
            "small_calibration": small,
            "nominal_level": 0.90,
            "interval_width_note": "finite-sample band widens as n_cal falls",
            "singleton_rate_falls": small["singleton_rate"] < large["singleton_rate"],
            "zero_claim_cell": {
                "n_claim": zero_claim.n_claim,
                "precision": zero_claim.prec,
                "vacuous": zero_claim.prec is None,
            },
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
