"""Lab 1: induce and measure totality failure.

A fitted estimator is total: it answers every coordinate. The contract is
partial: an out-of-support coordinate must abstain. The lab disposes a
contaminated cohort through both paths and reports the census.
"""

from __future__ import annotations

from _harness import emit, normalize_basis, out_of_regime, parse_args

from mjolnir_ml import DecisionModel, DecisionPolicyBase, RELEASE, dsl, rule


@dsl
class TotalityPolicy(DecisionPolicyBase):
    facts = ("a", "b")

    assert rule("confident_release",
                when=lambda conf: conf >= 0.50,
                verdict=RELEASE)


class ConfidentClf:
    """A total classifier: it always returns p=0.93, in or out of support."""

    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 for _ in X]

    def predict_proba(self, X):
        return [[0.07, 0.93] for _ in X]


def supported_rows():
    grid = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    return [{"a": a, "b": b} for a in grid for b in grid]


def contaminated_cohort():
    rows = [{"a": 0.5, "b": 0.5} for _ in range(60)]
    rows += [{"a": 5.0 + 0.01 * i, "b": 0.5} for i in range(20)]
    rows += [{"a": 0.5, "b": -3.0 - 0.01 * i} for i in range(20)]
    return rows


def main() -> None:
    args = parse_args(__doc__)
    estimator = ConfidentClf()
    dm = DecisionModel(estimator, TotalityPolicy, features=("a", "b"))
    dm.fit_regime(supported_rows())

    cohort = contaminated_cohort()
    probs = [estimator.predict_proba([[r["a"], r["b"]]])[0][1] for r in cohort]
    high_confidence = sum(p > 0.80 for p in probs) / len(probs)

    dispositions = [dm.dispose(row) for row in cohort]
    verdicts = [d.verdict for d in dispositions]
    corrupted = [d for d, row in zip(dispositions, cohort)
                 if row["a"] > 1.0 or row["b"] < 0.0]
    corrupted_abstained = sum(out_of_regime(d.basis) for d in corrupted)

    emit(
        "lab01_totality",
        {
            "n_cohort": len(cohort),
            "ungated_mean_confidence": round(sum(probs) / len(probs), 6),
            "ungated_share_above_0.80": round(high_confidence, 6),
            "gated_census": {v: verdicts.count(v) for v in sorted(set(verdicts))},
            "corrupted_rows": len(corrupted),
            "corrupted_abstained_out_of_regime": corrupted_abstained,
            "corrupted_abstention_rate": round(
                corrupted_abstained / len(corrupted), 6),
            "sample_basis": normalize_basis(corrupted[0].basis),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
