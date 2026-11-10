"""Lab 8: the audit memo.

The flywheel law governs the direction of a change, not its quality. A
tightening is promoted without approval; a loosening is refused without a
signature and accepted with one. Retraction is additive.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (ApprovalAuthority, ChangeSet, DecisionModel,
                        DecisionPolicyBase, HazardLedger, Journal, LiveSession,
                        RELEASE, dsl, rule)


@dsl
class BaselineRelease(DecisionPolicyBase):
    facts = ("score", "flag")

    assert rule("release_below",
                when=lambda p_positive: p_positive < 0.90,
                verdict=RELEASE)


@dsl
class TightenedRelease(DecisionPolicyBase):
    facts = ("score", "flag")

    assert rule("release_safe_below",
                when=lambda p_positive, flag: p_positive < 0.90 and flag == 0,
                verdict=RELEASE)


@dsl
class LoosenedRelease(DecisionPolicyBase):
    facts = ("score", "flag")

    assert rule("release_safe_below",
                when=lambda p_positive, flag: p_positive < 0.99 and flag == 0,
                verdict=RELEASE)


class ScoreClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def batch():
    rows = []
    rows += [{"score": 0.70, "flag": 1, "row_id": i} for i in range(8)]
    rows += [{"score": 0.20, "flag": 0, "row_id": 8 + i} for i in range(32)]
    rows += [{"score": 0.95, "flag": 0, "row_id": 40 + i} for i in range(8)]
    return rows


def main() -> None:
    args = parse_args(__doc__)
    dm = DecisionModel(ScoreClf(), BaselineRelease,
                       features=("score", "flag", "row_id"), regime="off")
    cal_rows = [{"score": 0.10 + 0.02 * i, "flag": 0, "row_id": -1 - i}
                for i in range(40)]
    dm.calibrate(cal_rows, [0] * 20 + [1] * 20, alpha=0.10)
    ledger = (HazardLedger()
              .add("unsafe_release", lambda row: row["flag"] == 1,
                   "audit:unsafe-release", observable=True))
    auth = ApprovalAuthority(key=b"approval-key")
    journal = Journal(None)
    live = LiveSession(dm, journal=journal, ledger=ledger,
                       approval_verifier=auth.verify)

    for row in batch():
        d = live.decide(row, batch=0)
        live.observe(d.certificate_id, 1 if row["flag"] else 0)

    def oracle(row, y):
        return (row["flag"] == 0, True)

    tightening = ChangeSet(lever="policy", reason="close the hazard release",
                           policy=TightenedRelease.policy(), name="tighten")
    tightening_promotion = live.promote(tightening, oracle=oracle,
                                        ledger=ledger)

    loosening = ChangeSet(lever="policy", reason="recover safe coverage",
                          policy=LoosenedRelease.policy(), name="loosen")
    unsigned = live.promote(loosening, oracle=oracle, ledger=ledger)
    approval = auth.sign(loosening.digest(dm.features), approver="risk-owner")
    signed = live.promote(loosening, oracle=oracle, approval=approval,
                          ledger=ledger)

    retraction = live.retract(policy_digest=BaselineRelease.policy().digest,
                              reason="v1 certificates retracted")
    journal_kinds = sorted({p["kind"] for p in journal.payloads()})

    emit(
        "lab08_flywheel",
        {
            "tightening_promoted": tightening_promotion.approved,
            "tightening_may_loosen": tightening_promotion.evaluation.may_loosen,
            "unsigned_loosening_refused": not unsigned.approved,
            "unsigned_reason": unsigned.reason,
            "signed_loosening_promoted": signed.approved,
            "signed_reason": signed.reason,
            "retraction_count": len(retraction.certificate_ids),
            "retraction_additive": len(retraction.certificate_ids) == 48,
            "journal_kinds": journal_kinds,
            "audit_memo": {
                "lever": "policy",
                "evidence": "held-out facts replay",
                "approver": approval.approver,
                "certificates_affected": len(retraction.certificate_ids),
            },
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
