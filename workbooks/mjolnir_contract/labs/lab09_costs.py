"""Lab 9: price the reject option.

The cost model is an institutional declaration, not an estimate. Compiling it
into the policy IR yields the analytic bands. The lab varies the ask cost and
records the movement of the ask volume, and reports expected cost separately
from realized cost.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (ASK, DENY, RELEASE, Costs, DecisionModel, cost_policy)


class ScoreClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def realized_cost(verdict, outcome, costs):
    if verdict == RELEASE:
        return costs.c_false_release if outcome == 1 else 0.0
    if verdict == DENY:
        return costs.c_false_deny if outcome == 0 else 0.0
    return costs.c_ask


def evaluate(costs: Costs, batch):
    policy = cost_policy(costs)
    dm = DecisionModel(ScoreClf(), policy, features=("p",), regime="off")
    verdicts = [dm.dispose({"p": p}).verdict for p, _ in batch]
    expected = sum(costs.expected_cost(v, p)
                   for v, (p, _) in zip(verdicts, batch)) / len(batch)
    realized = sum(realized_cost(v, y, costs)
                   for v, (_, y) in zip(verdicts, batch)) / len(batch)
    return {
        "c_ask": costs.c_ask,
        "release_threshold": costs.release_threshold,
        "deny_threshold": costs.deny_threshold,
        "ask_volume": sum(v == ASK for v in verdicts),
        "release_volume": sum(v == RELEASE for v in verdicts),
        "deny_volume": sum(v == DENY for v in verdicts),
        "expected_cost": round(expected, 6),
        "realized_cost": round(realized, 6),
    }


def main() -> None:
    args = parse_args(__doc__)
    batch = [(i / 100.0, 1 if i / 100.0 > 0.5 else 0) for i in range(100)]
    runs = [evaluate(Costs(10.0, 10.0, c_ask), batch)
            for c_ask in (0.5, 1.0, 2.0, 4.0)]

    analytic = Costs(10.0, 10.0, 2.0)
    ask_volumes = [r["ask_volume"] for r in runs]

    emit(
        "lab09_costs",
        {
            "analytic_bands": {
                "release_threshold": analytic.release_threshold,
                "deny_threshold": analytic.deny_threshold,
            },
            "runs": runs,
            "ask_volume_monotone_nonincreasing": all(
                b <= a for a, b in zip(ask_volumes, ask_volumes[1:])),
            "expected_and_realized_reported_separately": all(
                "expected_cost" in r and "realized_cost" in r for r in runs),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
