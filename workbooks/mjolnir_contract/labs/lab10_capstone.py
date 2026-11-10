"""Lab 10: deploy and defend a decision contract.

The capstone binds a schema, a regime floor, a calibrated index, and a policy,
then issues and verifies one signed record. The replay path uses only the fact
table, the policy, and the key: it never loads the estimator.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (ASK, DENY, RELEASE, CertificateError, Costs,
                        DecisionModel, DecisionPolicyBase, certify, dsl,
                        report_data, rule, verify_certificate)


@dsl
class CapstonePolicy(DecisionPolicyBase):
    facts = ("amount", "limit")

    assert lambda amount: amount > 0, "amount_positive"
    assert lambda limit: limit > 0, "limit_positive"

    assert rule("deny_over_limit",
                when=lambda amount, limit: amount > limit,
                verdict=DENY)
    assert rule("ask_mid",
                when=lambda amount, limit: amount > limit * 0.5,
                verdict=ASK)
    assert rule("release_certified_small",
                when=lambda certified, p_positive: certified and p_positive < 0.20,
                verdict=RELEASE)


class RatioClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if min(1.0, 0.5 * row[0] / row[1]) >= 0.5 else 0
                for row in X]

    def predict_proba(self, X):
        out = []
        for amount, limit in X:
            p = min(1.0, 0.5 * amount / limit)
            out.append([1.0 - p, p])
        return out


def main() -> None:
    args = parse_args(__doc__)
    key = b"audit-key"
    features = ("amount", "limit")
    dm = DecisionModel(RatioClf(), CapstonePolicy, features=features)
    train = [{"amount": a, "limit": 100.0}
             for a in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
    dm.fit_regime(train)
    y = [1 if row["amount"] > 50 else 0 for row in train]
    dm.calibrate(train, y, alpha=0.10)

    row = {"amount": 20.0, "limit": 100.0}
    disposition = dm.dispose(row)
    certificate = certify(disposition, key=key)
    verified = verify_certificate(certificate, key=key,
                                  policy=CapstonePolicy, row=row)

    tampered = dict(certificate)
    tampered["facts"] = dict(certificate["facts"])
    tampered["facts"]["amount"] = 90.0
    try:
        verify_certificate(tampered, key=key, policy=CapstonePolicy)
        tamper_check = "accepted (unexpected)"
    except CertificateError as ex:
        tamper_check = str(ex)

    costs = Costs(10.0, 10.0, 2.0)
    report = report_data(dm, [row], title="capstone", safe=[True],
                         decidable=[True], sample=0)

    emit(
        "lab10_capstone",
        {
            "verdict": disposition.verdict,
            "basis": list(disposition.basis),
            "guarantee": disposition.guarantee,
            "assurance": disposition.assurance,
            "certificate_id": certificate["certificate_id"],
            "replayable": certificate["replayable"],
            "verified": verified,
            "tamper_check": tamper_check,
            "manifest": {
                "policy_digest": disposition.policy_digest,
                "model_digest": disposition.model_digest,
                "calibration_digest": disposition.calibration_digest,
                "regime_digest": disposition.regime_digest,
            },
            "cost": {
                "expected": round(costs.expected_cost(disposition.verdict,
                                                      (disposition.prediction or {}).get(
                                                          "p_positive", 0.0)), 6),
                "bands": costs.as_dict(),
            },
            "report_claim_rate": report["decisions"]["claim_rate"],
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
