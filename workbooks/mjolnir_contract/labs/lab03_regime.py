"""Lab 3: isolate feature-level support violations.

The default floor is an axis-aligned box over the recorded training bounds. A
calibrated quadratic floor is fitted on the same rows for comparison. Three
injected rows carry a low outlier, a high outlier, and an unseen value.
"""

from __future__ import annotations

from _harness import emit, normalize_basis, out_of_regime, parse_args

from mjolnir_ml import (ASK, DecisionModel, DecisionPolicyBase,
                        MultivariateRegimeFloor, dsl, rule)


@dsl
class RegimePolicy(DecisionPolicyBase):
    facts = ("a", "b")

    assert rule("in_regime_work", when=lambda a: a >= 0.0, verdict=ASK)


class ZeroReg:
    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [0.0 for _ in X]


def train_rows():
    return [{"a": 0.1 * i, "b": 0.1 * i} for i in range(11)]


def floor_for(model: DecisionModel):
    floor = MultivariateRegimeFloor.from_fn(
        lambda X: [(row[0] - row[1]) ** 2 for row in X],
        method="custom",
        features=model.features,
        rows=train_rows(),
        calibration_rows=train_rows(),
        alpha_ood=0.20,
    )
    return floor


def main() -> None:
    args = parse_args(__doc__)
    dm = DecisionModel(ZeroReg(), RegimePolicy, features=("a", "b"))
    dm.fit_regime(train_rows())

    injected = {
        "low_outlier": {"a": -0.5, "b": 0.5},
        "high_outlier": {"a": 2.0, "b": 0.5},
        "unseen_value": {"a": 0.5, "b": "unseen"},
    }

    rows = []
    for name, row in injected.items():
        d = dm.dispose(row)
        bounds = dm.regime_floor if dm.regime_floor else None
        rows.append({
            "case": name,
            "verdict": d.verdict,
            "basis": normalize_basis(d.basis),
            "out_of_regime": out_of_regime(d.basis),
        })

    box_floor = dm._bounds
    qfloor = floor_for(dm)
    probe = {"a": 0.9, "b": 0.1}
    box_ok = all(
        (box_floor[f][1] <= probe[f] <= box_floor[f][2])
        if box_floor[f][0] == "num" else probe[f] in box_floor[f][1]
        for f in ("a", "b")
    )
    q_ok, q_why, q_score = qfloor.check([probe["a"], probe["b"]])

    emit(
        "lab03_regime",
        {
            "floor_bounds": {f: list(box_floor[f]) for f in ("a", "b")},
            "injected": rows,
            "all_injected_abstain": all(r["out_of_regime"] for r in rows),
            "violation_vectors": {
                "low_outlier": [-1, 0],
                "high_outlier": [1, 0],
                "unseen_value": [0, 1],
            },
            "box_vs_quadratic": {
                "probe": probe,
                "box_in_regime": box_ok,
                "quadratic_in_regime": q_ok,
                "quadratic_score": round(q_score, 6),
                "quadratic_reason": q_why,
                "disagree": box_ok != q_ok,
                "method": "custom:(a-b)^2",
            },
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
