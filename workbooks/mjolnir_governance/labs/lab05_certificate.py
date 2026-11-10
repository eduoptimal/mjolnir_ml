"""Lab 5: three tampering cases.

A signature proves the record was not altered; a replay proves the record
contains the decision. A record that depends on a model or a training bound is
signed but not replayable from facts alone.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (CertificateError, DecisionModel, DecisionPolicyBase,
                        RELEASE, certify, dsl, rule, verify_certificate)


@dsl
class CertPolicy(DecisionPolicyBase):
    facts = ("amount", "limit")

    assert lambda amount: amount > 0, "amount_positive"

    assert rule("release_low",
                when=lambda p_positive: p_positive < 0.20,
                verdict=RELEASE)


class LowClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [0 for _ in X]

    def predict_proba(self, X):
        return [[0.95, 0.05] for _ in X]


def batch():
    rows = [{"amount": 10.0 + i, "limit": 100.0} for i in range(111)]
    rows += [{"limit": 100.0} for _ in range(9)]   # missing required fact
    return rows


def error_of(fn):
    try:
        fn()
    except CertificateError as ex:
        return str(ex)
    return None


def main() -> None:
    args = parse_args(__doc__)
    key = b"audit-key"
    dm = DecisionModel(LowClf(), CertPolicy, features=("amount", "limit"),
                       regime="off")
    certs = [certify(dm.dispose(row), key=key) for row in batch()]
    replayable = sum(cert["replayable"] for cert in certs)

    sample = certs[0]
    tampered_fact = dict(sample)
    tampered_fact["facts"] = dict(sample["facts"])
    tampered_fact["facts"]["amount"] = 99.0

    flipped = dict(sample)
    flipped["verdict"] = "deny"

    unsigned = certify(dm.dispose(batch()[0]))
    unsigned["facts"] = dict(unsigned["facts"])
    unsigned["facts"]["amount"] = 99.0

    emit(
        "lab05_certificate",
        {
            "n_records": len(certs),
            "replayable": replayable,
            "non_replayable": len(certs) - replayable,
            "all_verify_with_policy": all(
                verify_certificate(c, key=key, policy=CertPolicy)
                for c in certs),
            "tampered_fact_error": error_of(
                lambda: verify_certificate(tampered_fact, key=key,
                                           policy=CertPolicy)),
            "flipped_verdict_error": error_of(
                lambda: verify_certificate(flipped, key=key,
                                           policy=CertPolicy)),
            "wrong_key_error": error_of(
                lambda: verify_certificate(sample, key=b"other",
                                           policy=CertPolicy)),
            "unsigned_tamper_error": error_of(
                lambda: verify_certificate(unsigned, policy=CertPolicy)),
            "signing_is_not_replaying": replayable < len(certs),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
