"""Readiness: the one-shot deployment gate, and its refusal cases."""
import pytest

from mjolnir_ml import (DecisionModel, HazardLedger, ReadinessThresholds,
                        readiness)
from mjolnir_ml.vendors import wire

import policies as P


class StubClf:
    classes_ = [0, 1]

    def __init__(self, p_positive=0.01):
        self.p_positive = p_positive

    def get_params(self, deep=False):
        return {"p_positive": self.p_positive}

    def predict(self, X):
        return [1 if self.p_positive >= 0.5 else 0]

    def predict_proba(self, X):
        return [[1 - self.p_positive, self.p_positive]]


ROWS = [{"amount": 100, "credit_limit": 1000},
        {"amount": 5000, "credit_limit": 1000},
        {"amount": 100, "credit_limit": 1000}]
SAFE = [True, False, True]
DEC = [True, True, True]


def _dm(regime="off"):
    return DecisionModel(StubClf(0.01), P.FraudPolicy,
                         features=("amount", "credit_limit"), regime=regime)


def _th(**kw):
    base = dict(min_precision=0.5, min_recall=0.0, max_abstain_rate=1.0,
                require_calibration=False)
    base.update(kw)
    return ReadinessThresholds(**base)


def test_readiness_approves_and_carries_evidence():
    dm = _dm()
    r = readiness(dm, ROWS, safe=SAFE, decidable=DEC, thresholds=_th(),
                  title="fraud")
    assert r.approved is True
    assert r.schema == "mjolnir.ml.readiness/1"
    assert r.readiness_id.startswith("sha256:")
    assert r.metrics["pair"]["n_claim"] == 3
    assert r.metrics["pair"]["n_overclaim"] == 0
    assert r.digests["policy"] == P.FraudPolicy.policy().digest
    assert r.digests["model"] == dm.model_digest
    assert {g.name for g in r.gates} >= {"precision", "overclaim", "coverage",
                                         "abstain_rate"}
    assert r.evidence["rows"]


def test_readiness_fails_a_gate_and_names_it():
    r = readiness(_dm(), ROWS, safe=SAFE, decidable=DEC,
                  thresholds=_th(min_precision=1.1))
    assert r.approved is False
    failed = [g.name for g in r.gates if not g.passed]
    assert failed == ["precision"]


def test_readiness_refuses_without_calibration_when_required():
    r = readiness(_dm(), ROWS, safe=SAFE, decidable=DEC,
                  thresholds=ReadinessThresholds(min_precision=0.5,
                                                 require_calibration=True))
    assert r.approved is False
    gate = next(g for g in r.gates if g.name == "calibration_present")
    assert gate.passed is False


def test_readiness_is_stateless():
    dm = _dm()
    before = (dm.policy.digest, dm.model_digest, dm.kappa, dm.regime)
    readiness(dm, ROWS, safe=SAFE, decidable=DEC, thresholds=_th())
    after = (dm.policy.digest, dm.model_digest, dm.kappa, dm.regime)
    assert before == after


def test_readiness_id_is_deterministic():
    a = readiness(_dm(), ROWS, safe=SAFE, decidable=DEC, thresholds=_th())
    b = readiness(_dm(), ROWS, safe=SAFE, decidable=DEC, thresholds=_th())
    assert a.readiness_id == b.readiness_id


def test_readiness_hazard_and_signature_gates():
    ledger = (HazardLedger()
              .add("big_amount", lambda r: r["amount"] > 1000, "policy §2"))
    r = readiness(_dm(), ROWS, safe=SAFE, decidable=DEC, ledger=ledger,
                  key=b"k", thresholds=_th(min_hazard_precision=0.99))
    names = {g.name: g for g in r.gates}
    assert "hazard_precision" in names
    assert names["certificate_replay"].passed is True
    assert r.signature.startswith("hmac-sha256:")
    assert r.as_dict()["approved"] is True


def test_readiness_rejects_malformed_requests():
    with pytest.raises(Exception):
        readiness(_dm(), ROWS[:-1], safe=SAFE, decidable=DEC,
                  thresholds=_th())
