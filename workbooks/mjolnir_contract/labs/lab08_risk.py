"""Lab 8: two tracks, one cohort.

Pointwise coverage and cohort risk are separate guarantees. The coverage track
asks whether the conformal set contains the label; the risk track asks whether
the release and deny cohorts satisfy their corrected empirical risk bounds.
The lab reports both on the same rows.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (DecisionModel, DecisionPolicyBase, RELEASE, crc_thresholds,
                        dsl, rule)


@dsl
class CoveragePolicy(DecisionPolicyBase):
    facts = ("score",)

    assert rule("certified_low_risk",
                when=lambda certified, p_positive: certified and p_positive < 0.5,
                verdict=RELEASE)


class ScoreClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def cohort(n=200):
    scores = [round(i / (n - 1), 6) for i in range(n)]
    events = [1 if s > 0.72 else 0 for s in scores]
    return scores, events


def coverage_track():
    rows, y = [], []
    for i in range(120):
        s = round((i * 0.6180339887498949) % 1.0, 6)
        rows.append({"score": s})
        y.append(1 if s > 0.5 else 0)
    dm = DecisionModel(ScoreClf(), CoveragePolicy, features=("score",),
                       regime="off")
    dm.calibrate(rows[:80], y[:80], alpha=0.10)
    test_rows, test_y = rows[80:], y[80:]
    disps = [dm.dispose(row) for row in test_rows]
    covered = sum(label in (d.prediction or {}).get("set", [])
                  for d, label in zip(disps, test_y))
    sizes = [len((d.prediction or {}).get("set", [])) for d in disps]
    return {
        "n": len(test_rows),
        "coverage": round(covered / len(test_rows), 6),
        "mean_set_size": round(sum(sizes) / len(sizes), 6),
        "singleton_rate": round(sum(s == 1 for s in sizes) / len(sizes), 6),
    }


def main() -> None:
    args = parse_args(__doc__)
    scores, events = cohort()
    lam_r, lam_d = crc_thresholds(scores, events,
                                  alpha_release=0.10, alpha_deny=0.10)

    release_cohort = [(s, e) for s, e in zip(scores, events) if s <= lam_r]
    deny_cohort = [(s, e) for s, e in zip(scores, events) if s > lam_d]
    ask_band = [s for s in scores if lam_r < s <= lam_d]

    release_risk = (sum(e for _, e in release_cohort) / len(release_cohort)
                    if release_cohort else None)
    deny_risk = (sum(1 - e for _, e in deny_cohort) / len(deny_cohort)
                 if deny_cohort else None)

    emit(
        "lab08_risk",
        {
            "coverage_track": coverage_track(),
            "risk_track": {
                "lambda_release": lam_r,
                "lambda_deny": lam_d,
                "thresholds_disjoint": lam_r <= lam_d,
                "release_cohort": len(release_cohort),
                "release_cohort_risk": round(release_risk, 6)
                if release_risk is not None else None,
                "deny_cohort": len(deny_cohort),
                "deny_cohort_risk": round(deny_risk, 6)
                if deny_risk is not None else None,
                "ask_band": len(ask_band),
            },
            "reported_separately": True,
            "ask_not_counted_as_release": True,
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
