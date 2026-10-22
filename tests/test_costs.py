"""Cost-sensitive verdicts: bands, expected cost, and compiled cost policies."""
import pytest

from mjolnir_ml import ASK, DENY, RELEASE, Costs, DecisionModel, cost_policy
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


FEATS = ("amount",)


def _dm(p_positive, policy, regime="off"):
    return DecisionModel(StubClf(p_positive), policy, features=FEATS,
                         regime=regime)


def test_bands_and_optimal_verdicts():
    c = Costs(c_false_release=10.0, c_false_deny=10.0, c_ask=2.0)
    assert c.release_threshold == pytest.approx(0.2)
    assert c.deny_threshold == pytest.approx(0.8)
    assert c.best_verdict(0.05) == RELEASE
    assert c.best_verdict(0.5) == ASK
    assert c.best_verdict(0.95) == DENY


def test_expected_cost_reads_the_asymmetry():
    c = Costs(c_false_release=50.0, c_false_deny=1.0, c_ask=2.0)
    # p=0.5: releasing costs 25, denying costs 0.5, asking 2 -> deny
    assert c.best_verdict(0.5) == DENY
    assert c.expected_cost(RELEASE, 0.5) == pytest.approx(25.0)
    assert c.expected_cost(DENY, 0.5) == pytest.approx(0.5)
    assert c.expected_cost(ASK, 0.5) == pytest.approx(2.0)


def test_no_ask_region_when_asking_is_expensive():
    c = Costs(c_false_release=10.0, c_false_deny=1.0, c_ask=5.0)
    lo, hi = c.thresholds()
    assert lo == hi                        # ask never optimal; crossover only
    assert c.best_verdict(0.01) == RELEASE
    assert c.best_verdict(0.5) == DENY


def test_compiled_cost_policy_bands():
    policy = cost_policy(Costs(10.0, 10.0, 2.0))
    assert policy.digest.startswith("sha256:")
    assert {(s.kind, s.code) for s in policy.statements} == {
        ("rule", "deny_high"), ("rule", "ask_band"),
        ("release", "release_low")}
    # 0.05 -> release, 0.5 -> ask, 0.95 -> deny
    assert _dm(0.05, policy).dispose({"amount": 1}).verdict == RELEASE
    assert _dm(0.5, policy).dispose({"amount": 1}).verdict == ASK
    assert _dm(0.95, policy).dispose({"amount": 1}).verdict == DENY


def test_cost_policy_certified_gate():
    policy = cost_policy(Costs(10.0, 10.0, 2.0), require_certified=True)
    # uncalibrated: `certified` is absent -> fail-closed abstain
    d = _dm(0.05, policy).dispose({"amount": 1})
    assert d.verdict == wire.ABSTAIN_W
    assert d.missing == "certified"


def test_mean_cost_prefers_the_cheaper_batch():
    c = Costs(10.0, 10.0, 2.0)
    ps = [0.01, 0.02, 0.9, 0.95]
    assert c.mean_cost([RELEASE, RELEASE, DENY, DENY], ps) < \
        c.mean_cost([ASK] * 4, ps)
