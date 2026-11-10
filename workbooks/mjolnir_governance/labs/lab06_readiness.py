"""Lab 6: the blocking gate.

Readiness is a conjunction of gates. A failing gate names its value, its
threshold, and the margin; the readiness identifier changes when a declared
threshold changes. The threshold is a policy decision, not a measurement.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (DecisionModel, DecisionPolicyBase, RELEASE,
                        ReadinessThresholds, dsl, readiness, rule)


@dsl
class ReadyPolicy(DecisionPolicyBase):
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


def main() -> None:
    args = parse_args(__doc__)
    rows, y = [], []
    for i in range(120):
        s = round((i * 0.6180339887498949) % 1.0, 6)
        rows.append({"score": s})
        y.append(1 if s > 0.5 else 0)

    dm = DecisionModel(ScoreClf(), ReadyPolicy, features=("score",),
                       regime="off")
    dm.calibrate(rows[:80], y[:80], alpha=0.10)
    test_rows = rows[80:]
    safe = [label == 0 for label in y[80:]]
    decidable = [True] * len(test_rows)

    strict = readiness(dm, test_rows, safe=safe, decidable=decidable,
                       key=b"readiness-key",
                       thresholds=ReadinessThresholds(min_precision=1.01,
                                                      min_recall=0.0))
    relaxed = readiness(dm, test_rows, safe=safe, decidable=decidable,
                        key=b"readiness-key",
                        thresholds=ReadinessThresholds(min_precision=0.0,
                                                       min_recall=0.0))

    def gates(report):
        return [{"name": g.name, "passed": g.passed, "value": g.value,
                 "threshold": g.threshold, "detail": g.detail}
                for g in report.gates]

    blocking = [g for g in strict.gates if not g.passed]
    emit(
        "lab06_readiness",
        {
            "strict_approved": strict.approved,
            "strict_id": strict.readiness_id,
            "relaxed_approved": relaxed.approved,
            "relaxed_id": relaxed.readiness_id,
            "id_changes_with_threshold": (
                strict.readiness_id != relaxed.readiness_id),
            "strict_gates": gates(strict),
            "blocking_gate": (
                {"name": blocking[0].name, "value": blocking[0].value,
                 "threshold": blocking[0].threshold,
                 "margin": (blocking[0].value - blocking[0].threshold)
                 if isinstance(blocking[0].value, (int, float))
                 else None} if blocking else None),
            "evidence_verifies": strict.signature is not None,
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
