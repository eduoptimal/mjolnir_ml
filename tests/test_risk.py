"""The risk track: CRC action thresholds, assurance tags and feasibility."""
import math

import pytest

from mjolnir_ml import (DecisionModel, ReadinessThresholds, certify,
                        crc_threshold, feasibility, fit_risk_index, report,
                        report_data, verify_certificate)
from mjolnir_ml.risk import adverse_event
from mjolnir_ml import ABSTAIN, ASK, DENY, RELEASE

import policies as P


class XReg:
    """prediction = x (no fitting needed)."""

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [row[0] for row in X]


class XCross:
    """p_event(x) = sigmoid(3(x-1)); the event is x > 1."""

    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict_proba(self, X):
        out = []
        for row in X:
            p = 1.0 / (1.0 + math.exp(-3.0 * (row[0] - 1.0)))
            out.append([1.0 - p, p])
        return out


class XC:
    """A binary classifier whose p_positive is x (the crossing event)."""

    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def _xs():
    return [-2.0 + 4.0 * i / 40 for i in range(41)]


def _risk_dm(policy=P.RiskPolicy):
    rows = [{"x": x} for x in _xs()]
    y = [x for x in _xs()]
    dm = DecisionModel(XReg(), policy, features=("x",), regime="off")
    dm.calibrate_risk(rows, y, tau=1.0, estimator=XCross(),
                      alpha_release=0.10, alpha_deny=0.10)
    return dm, rows


# --------------------------------------------------------------- CRC maths --
def test_crc_threshold_low_and_high_sides():
    scores = [0.05 * i for i in range(20)]
    events = [0] * 10 + [1] * 10
    low = crc_threshold(scores, events, alpha=0.10, side="low")
    high = crc_threshold(scores, [1 - e for e in events], alpha=0.10,
                         side="high")
    assert low == pytest.approx(0.45)     # select the ten event-free rows
    assert high == pytest.approx(0.50)    # select the ten event rows


def test_crc_threshold_returns_none_when_no_budget_is_safe():
    assert crc_threshold([0.1, 0.2, 0.3], [1, 1, 1], alpha=0.10) is None


def test_adverse_event_refuses_non_finite():
    with pytest.raises(Exception, match="non-finite"):
        adverse_event([1.0, float("nan")], 0.5)


def test_risk_index_digest_is_content_addressed():
    rows = [{"x": x} for x in _xs()]
    events = adverse_event([x for x in _xs()], 1.0)
    a = fit_risk_index(XCross(), rows, events, features=("x",))
    b = fit_risk_index(XCross(), rows, events, features=("x",))
    assert a.digest == b.digest and a.digest.startswith("sha256:")
    c = fit_risk_index(XCross(), rows, events, features=("x",),
                       alpha_release=0.20)
    assert c.digest != a.digest


# ------------------------------------------------------- decision plumbing --
def test_risk_track_releases_and_denies_without_an_interval():
    dm, _ = _risk_dm()
    assert dm.risk is not None
    low = dm.dispose({"x": -2.0})
    high = dm.dispose({"x": 3.0})
    assert low.verdict == RELEASE
    assert low.assurance == "action_risk_control"
    assert low.prediction["risk_release"] is True
    assert high.verdict == DENY
    assert high.assurance == "action_risk_control"
    assert high.prediction["risk_deny"] is True
    assert low.certificate_id != high.certificate_id


def test_risk_absent_is_missing_not_release():
    dm = DecisionModel(XReg(), P.RiskPolicy, features=("x",), regime="off")
    d = dm.dispose({"x": 0.0})
    assert d.verdict == ABSTAIN
    assert d.missing == "risk_release" or d.missing == "risk_deny"
    assert d.assurance == "action_risk_control"


def test_risk_certificate_pins_the_risk_index():
    dm, _ = _risk_dm()
    d = dm.dispose({"x": 3.0})
    cert = certify(d, key=b"k")
    assert cert["risk_digest"] == dm.risk.digest
    assert cert["assurance"] == "action_risk_control"
    assert verify_certificate(cert, key=b"k", policy=P.RiskPolicy) is True


def test_classifier_selective_risk_uses_the_same_layer():
    xs = [0.01 + 0.98 * i / 40 for i in range(41)]
    rows = [{"x": x} for x in xs]
    y = [1 if x >= 0.5 else 0 for x in xs]
    dm = DecisionModel(XC(), P.RiskPolicy, features=("x",), regime="off")
    dm.calibrate_risk(rows, y, estimator=XC())
    assert dm.dispose({"x": 0.02}).verdict == RELEASE
    assert dm.dispose({"x": 0.98}).verdict == DENY
    assert dm.dispose({"x": 0.02}).assurance == "action_risk_control"


# --------------------------------------------------------------- feasibility --
def _dead_dm():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    dm = DecisionModel(XReg(), P.DeadDenyPolicy, features=("x",),
                       regime="off")
    dm.calibrate(rows, [x + 1.0 for x in xs], alpha=0.10)
    return dm, rows


def test_feasibility_finds_a_dead_coverage_rule():
    dm, rows = _dead_dm()
    f = feasibility(dm, rows, threshold=240.0)
    assert "deny_high" in f.dead_rules
    assert f.coverage_bounds["max_lo"] < 240.0
    assert f.coverage_bounds["release_bound_clears_threshold"] is True


def test_feasibility_sees_a_reachable_risk_rule():
    dm, rows = _risk_dm()
    f = feasibility(dm, [{"x": -2.0}, {"x": 0.0}, {"x": 3.0}], threshold=1.0)
    assert "deny_risk" not in f.dead_rules
    assert "release_risk" not in f.dead_rules
    assert f.risk_bounds["max_p_event"] >= f.risk_bounds["lambda_deny"]


def test_report_shows_feasibility_and_dead_rules():
    dm, rows = _dead_dm()
    text = report(dm, rows, threshold=240.0)
    assert "FEASIBILITY" in text
    assert "DEAD RULES" in text and "deny_high" in text
    data = report_data(dm, rows, threshold=240.0)
    assert data["feasibility"]["dead_rules"] == ["deny_high"]


def test_readiness_can_require_reachable_rules():
    dm, rows = _dead_dm()
    report_obj = None
    from mjolnir_ml import readiness
    report_obj = readiness(
        dm, rows, safe=[True] * len(rows), decidable=[True] * len(rows),
        thresholds=ReadinessThresholds(require_reachable_rules=True),
        threshold=240.0)
    gate = next(g for g in report_obj.gates if g.name == "rules_reachable")
    assert gate.passed is False


def test_joint_thresholds_are_ordered_and_disjoint():
    from mjolnir_ml import crc_thresholds
    scores = [0.05 * i for i in range(20)]
    events = [0] * 10 + [1] * 10
    lam_r, lam_d = crc_thresholds(scores, events, 0.10, 0.10)
    assert lam_r is not None and lam_d is not None
    assert lam_r <= lam_d
    for p in scores:
        assert not (p <= lam_r and p > lam_d)


def test_feasibility_reports_direction_and_negative_slack():
    from mjolnir_ml.feasibility import FeasibilityError
    dm, rows = _dead_dm()
    f = feasibility(dm, rows, threshold=240.0)
    chk = next(c for c in f.checks if c.code == "deny_high")
    assert chk.evidence["direction"] == "lower_gt"
    assert chk.evidence["achieved_bound"] == f.coverage_bounds["max_lo"]
    assert chk.evidence["slack"] < 0
    with pytest.raises(FeasibilityError, match="deny_high"):
        f.assert_reachable()


class XCross2:
    """p_event(x) = sigmoid(3(x-2)); the event is x > 2."""

    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict_proba(self, X):
        out = []
        for row in X:
            p = 1.0 / (1.0 + math.exp(-3.0 * (row[0] - 2.0)))
            out.append([1.0 - p, p])
        return out


def test_dual_track_release_is_tagged_when_both_guarantees_fire():
    xs = [0.2 * i for i in range(20)]
    rows = [{"x": x} for x in xs]
    dm = DecisionModel(XReg(), P.DualPolicy, features=("x",), regime="off")
    dm.calibrate(rows, xs, alpha=0.10)
    dm.calibrate_risk(rows, xs, tau=2.0, estimator=XCross2())
    d = dm.dispose({"x": 0.2})
    assert d.verdict == RELEASE
    assert set(d.basis) == {"release_coverage", "release_risk"}
    assert d.assurance == "dual_track"
