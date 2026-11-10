"""Lab 4: the uncalibrated contract.

A release rule that requires the conformal fact `certified` must abstain when
calibration was never bound. A stated-fact policy on the same row decides
deterministically without fabricating a calibrated fact.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (ASK, DENY, RELEASE, DecisionModel,
                        DecisionPolicyBase, dsl, rule)


@dsl
class CertifiedPolicy(DecisionPolicyBase):
    facts = ("approved",)

    assert rule("certified_release",
                when=lambda certified, p_positive: certified and p_positive < 0.50,
                verdict=RELEASE)
    assert rule("ambiguous",
                when=lambda certified: not certified,
                verdict=ASK)


@dsl
class StatedFactPolicy(DecisionPolicyBase):
    facts = ("approved",)

    assert rule("approve",
                when=lambda approved: approved,
                verdict=RELEASE)
    assert rule("decline",
                when=lambda approved: not approved,
                verdict=DENY)


class LowRiskClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [0 for _ in X]

    def predict_proba(self, X):
        return [[0.99, 0.01] for _ in X]


def main() -> None:
    args = parse_args(__doc__)
    features = ("x", "approved")
    row = {"x": 0.5, "approved": True}

    uncalibrated = DecisionModel(
        LowRiskClf(), CertifiedPolicy, features=features)
    uncalibrated.fit_regime([{"x": 0.0, "approved": True},
                             {"x": 1.0, "approved": False}])
    d_missing = uncalibrated.dispose(row)

    stated = DecisionModel(LowRiskClf(), StatedFactPolicy, features=features)
    stated.fit_regime([{"x": 0.0, "approved": True},
                       {"x": 1.0, "approved": False}])
    d_stated = stated.dispose(row)

    def fact_names(d):
        return sorted(d.facts)

    emit(
        "lab04_facts",
        {
            "uncalibrated": {
                "verdict": d_missing.verdict,
                "basis": list(d_missing.basis),
                "missing": d_missing.missing,
                "fact_names": fact_names(d_missing),
                "abstains_with_named_missing_fact": (
                    d_missing.missing == "certified"),
            },
            "stated_fact": {
                "verdict": d_stated.verdict,
                "basis": list(d_stated.basis),
                "fact_names": fact_names(d_stated),
                "fabricated_certified": "certified" in d_stated.facts,
            },
            "fact_namespaces_are_declared": {
                "uncalibrated": set(fact_names(d_missing))
                <= {"x", "approved", "pred", "conf", "p_positive", "certified",
                    "set_size", "set"},
                "stated": set(fact_names(d_stated))
                <= {"x", "approved", "pred", "conf", "p_positive", "certified",
                    "set_size", "set"},
            },
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
