"""The harness laws: fail-closed, model-never-disposes, only-tighten, replay."""
import pytest

from mjolnir_ml import (BindError, DecisionModel, omega_pair)
from mjolnir_ml.vendors.wire import (ABSTAIN_W as ABSTAIN, DENY_W as DENY,
                                    RELEASE, ASK)

import policies as P


class StubClf:
    """Deterministic binary classifier: P(1) = p_positive."""

    classes_ = [0, 1]

    def __init__(self, p_positive=0.01, fail=False):
        self.p_positive = p_positive
        self.fail = fail

    def get_params(self, deep=False):
        return {"p_positive": self.p_positive}

    def predict(self, X):
        if self.fail:
            raise ValueError("boom")
        return [1 if self.p_positive >= 0.5 else 0]

    def predict_proba(self, X):
        if self.fail:
            raise ValueError("proba boom")
        return [[1 - self.p_positive, self.p_positive]]


class StubReg:
    def __init__(self, value=5.0):
        self.value = value

    def get_params(self, deep=False):
        return {"value": self.value}

    def predict(self, X):
        return [self.value]


FEATURES = ("amount", "credit_limit")
TRAIN = [{"amount": a, "credit_limit": c}
         for a in (10, 100, 1000)
         for c in (50, 500, 5000)]


def _dm(policy=P.FraudPolicy, est=None, regime="off"):
    dm = DecisionModel(est or StubClf(0.01), policy, features=FEATURES,
                       regime=regime)
    if regime == "required":
        dm.fit_regime(TRAIN)
    return dm


# ---------------------------------------------------------------- verdicts --
def test_release_requires_a_firing_release_rule():
    d = _dm().dispose({"amount": 100, "credit_limit": 1000})
    assert d.verdict == RELEASE
    assert d.basis == ("auto_release",)
    assert d.guarantee == "model_backed"
    assert d.prediction["p_positive"] == 0.01


def test_ask_when_rules_escalate():
    d = _dm(est=StubClf(0.5)).dispose({"amount": 100, "credit_limit": 1000})
    assert d.verdict == ASK
    assert set(d.basis) == {"escalate_mid", "escalate_uncertain"}


def test_deny_beats_ask_and_release():
    d = _dm(est=StubClf(0.5)).dispose({"amount": 5000, "credit_limit": 1000})
    assert d.verdict == DENY
    assert "hard_deny" in d.basis


def test_premise_violation_abstains_before_rules():
    # hard_deny would fire on 100 > -200, but the premise gate abstains first
    d = _dm().dispose({"amount": 100, "credit_limit": -100})
    assert d.verdict == ABSTAIN
    assert d.basis == ("limit_positive",)
    assert d.missing is None
    assert "premise limit_positive" in d.say


def test_uncovered_is_abstain_never_release():
    # conf=0.8: no escalate rule (<0.80 strictly), p_positive 0.2 not <0.05
    d = _dm(est=StubClf(0.2)).dispose({"amount": 100, "credit_limit": 1000})
    assert d.verdict == ABSTAIN
    assert d.basis == ("uncovered",)


def test_model_never_disposes():
    """A confident learner still cannot produce a verdict no rule contains."""
    dm = DecisionModel(StubClf(0.99), P.UncoveredPolicy, features=("x",),
                       regime="off")
    d = dm.dispose({"x": 100})
    assert d.verdict == ABSTAIN          # model says 0.99, policy silent
    assert d.basis == ("uncovered",)


# -------------------------------------------------------------- fail closed --
def test_missing_feature_abstains_with_name():
    row = {"amount": 100}
    d = _dm().dispose(row)
    assert d.verdict == ABSTAIN
    assert d.missing == "credit_limit"
    assert d.basis == ("missing:credit_limit",)


def test_non_finite_feature_abstains():
    d = _dm().dispose({"amount": float("nan"), "credit_limit": 1000})
    assert d.verdict == ABSTAIN
    assert d.basis == ("invalid:amount",)


def test_none_feature_abstains():
    d = _dm().dispose({"amount": None, "credit_limit": 1000})
    assert d.verdict == ABSTAIN
    assert d.missing == "amount"


def test_model_exception_abstains():
    d = _dm(est=StubClf(fail=True)).dispose({"amount": 100,
                                             "credit_limit": 1000})
    assert d.verdict == ABSTAIN
    assert d.basis == ("model_error",)
    assert "boom" in d.say


def test_regime_required_but_unfitted_abstains():
    dm = DecisionModel(StubClf(0.01), P.FraudPolicy, features=FEATURES)
    d = dm.dispose({"amount": 100, "credit_limit": 1000})
    assert d.verdict == ABSTAIN
    assert d.basis == ("regime_not_fitted",)


def test_out_of_regime_value_abstains():
    d = _dm(regime="required").dispose({"amount": 10 ** 9,
                                        "credit_limit": 1000})
    assert d.verdict == ABSTAIN
    assert d.basis == ("out_of_regime:amount",)


def test_regime_off_is_explicit():
    # out-of-bounds amount that the policy does not deny -> harness decides
    d = _dm(regime="off").dispose({"amount": 10 ** 9, "credit_limit": 10 ** 12})
    assert d.verdict == RELEASE


def test_unseen_category_is_out_of_regime():
    dm = DecisionModel(StubClf(0.01), P.FraudPolicy,
                       features=("amount", "credit_limit"), regime="required")
    dm.fit_regime([{"amount": 10, "credit_limit": "high"}])
    d = dm.dispose({"amount": 10, "credit_limit": "low"})
    assert d.verdict == ABSTAIN
    assert d.basis == ("out_of_regime:credit_limit",)


# ------------------------------------------------------------------- bind --
def test_bind_rejects_unknown_fact():
    # bind validates names against the fitted schema (the authority)
    dm = DecisionModel(StubClf(0.01), P.NoFactsPolicy, features=("credit",),
                       regime="off")
    assert dm
    with pytest.raises(BindError, match="credit"):
        DecisionModel(StubClf(0.01), P.NoFactsPolicy,
                      features=("amount",), regime="off")


def test_bind_rejects_conf_without_proba():
    class LabelOnly:
        classes_ = [0, 1]

        def get_params(self, deep=False):
            return {}

        def predict(self, X):
            return [0]

    with pytest.raises(BindError, match="conf"):
        DecisionModel(LabelOnly(), P.FraudPolicy, features=FEATURES,
                      regime="off")


def test_bind_rejects_interval_facts_on_classifier():
    with pytest.raises(BindError, match="lo"):
        DecisionModel(StubClf(0.01), P.RegressorPolicy, features=FEATURES,
                      regime="off")


def test_bind_requires_fitted_classifier():
    class Unfitted:
        def get_params(self, deep=False):
            return {}

        def predict_proba(self, X):
            return [[0.5, 0.5]]

        def predict(self, X):
            return [0]

    with pytest.raises(BindError, match="fit it"):
        DecisionModel(Unfitted(), P.FraudPolicy, features=FEATURES,
                      regime="off")


def test_bind_requires_features():
    class Emb:
        def get_params(self, deep=False):
            return {}

        def predict(self, X):
            return [0]

    with pytest.raises(BindError, match="features"):
        DecisionModel(Emb(), P.FraudPolicy, regime="off")


# --------------------------------------------------------------- replay ----
def test_certificate_replay_and_tamper():
    dm = _dm()
    row = {"amount": 100, "credit_limit": 1000}
    d = dm.dispose(row)
    assert dm.replay(d, row)
    assert not dm.replay(d, {"amount": 101, "credit_limit": 1000})


def test_disposition_carries_digests_and_certificate_id():
    dm = _dm()
    row = {"amount": 100, "credit_limit": 1000}
    a, b = dm.dispose(row), dm.dispose(row)
    assert a.policy_digest == P.FraudPolicy.policy().digest
    assert a.model_digest.startswith("sha256:")
    assert a.claim_digest == b.claim_digest
    assert a.certificate_id == b.certificate_id
    assert a.calibration_digest is None and a.alpha is None
    assert a.as_dict()["verdict"] == RELEASE


def test_model_digest_changes_with_the_estimator():
    a = _dm(est=StubClf(0.01)).dispose({"amount": 100,
                                        "credit_limit": 1000})
    b = _dm(est=StubClf(0.30)).dispose({"amount": 100,
                                        "credit_limit": 1000})
    assert a.model_digest != b.model_digest


def test_guarantee_tags():
    certified = _dm().dispose({"amount": 100, "credit_limit": -1})
    assert certified.guarantee == "declared"       # features only
    calibrated = _dm().dispose({"amount": 100, "credit_limit": 1000})
    assert calibrated.guarantee == "model_backed"     # read p_positive/conf


def test_regressor_policy_over_pred():
    # OnlyDenyPolicy needs `amount`; model features are the row keys
    dm = DecisionModel(StubClf(0.01), P.OnlyDenyPolicy, features=("amount",),
                       regime="off")
    assert dm.dispose({"amount": 50}).verdict == ABSTAIN  # uncovered
    assert dm.dispose({"amount": 500}).verdict == DENY


# ------------------------------------------------------------------ pair --
def test_pair_on_dispositions():
    dm = _dm()
    rows = [{"amount": 100, "credit_limit": 1000},
            {"amount": 5000, "credit_limit": 1000},
            {"amount": 100, "credit_limit": 1000}]
    disps = [dm.dispose(r) for r in rows]
    pair = omega_pair(disps, safe=[True, False, True],
                      decidable=[True, True, False])
    assert pair.n_claim == 3
    # release on safe=True -> cert; deny on safe=False -> cert (engine reasons)
    assert pair.n_cert == 2
    assert pair.n_overclaim == 1
    assert pair.prec == pytest.approx(2 / 3)
