"""κ: conformal calibration — the load-bearing trust index.

A certified claim means: the conformal prediction set is a singleton at the
declared alpha, estimated per region. Unknown region / too little data is
never a silent uncalibrated claim.
"""
import pytest

from mjolnir_ml import CalibrationError, DecisionModel, fit_kappa
from mjolnir_ml.vendors.wire import (ABSTAIN_W as ABSTAIN, DENY_W as DENY,
                                    RELEASE, ASK)

import policies as P


class XClf:
    """P(1) = x, labels are the threshold rule. Deterministic, self-consistent."""

    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


class XReg:
    """prediction = x (add noise through the labels, not the model)."""

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [row[0] for row in X]


FEATS = ("x",)


def _rows(xs, region=None):
    return [{"x": x} | ({"region": region} if region else {}) for x in xs]


def _calib_clf(alpha=0.10, group=None, feats=FEATS, region=None,
               invert=False):
    xs = [0.01 + 0.98 * i / 39 for i in range(40)]
    rows = _rows(xs, region=region)
    y = [1 if x >= 0.5 else 0 for x in xs]
    if invert:                      # a badly-calibrated model: big scores
        y = [1 - v for v in y]
    dm = DecisionModel(XClf(), P.CertifiedPolicy, features=feats, regime="off")
    dm.calibrate(rows, y, alpha=alpha, group=group)
    return dm


def test_kappa_digest_is_content_addressed():
    a = _calib_clf().kappa
    b = _calib_clf().kappa
    assert a.digest == b.digest and a.digest.startswith("sha256:")
    c = _calib_clf(alpha=0.2).kappa
    assert c.digest != a.digest


def test_certified_singleton_releases():
    dm = _calib_clf()
    d = dm.dispose({"x": 0.02})          # P(1)=0.02 -> set {0}, score 0.02
    assert d.verdict == RELEASE
    assert d.basis == ("certified_release",)
    assert d.prediction["certified"] is True
    assert d.prediction["set_size"] == 1
    assert d.alpha == pytest.approx(0.10)
    assert d.calibration_digest == dm.kappa.digest
    assert d.guarantee == "model_backed"


def test_ambiguous_set_is_not_certified():
    dm = _calib_clf(invert=True)         # wide quantile: both labels pass
    d = dm.dispose({"x": 0.60})
    assert d.verdict == ASK
    assert d.basis == ("ambiguous",)
    assert d.prediction["certified"] is False
    assert d.prediction["set_size"] == 2


def test_empty_set_is_not_certified():
    dm = _calib_clf()
    d = dm.dispose({"x": 0.50})          # neither label passes the quantile
    assert d.verdict == ASK
    assert d.prediction["certified"] is False


def test_uncalibrated_required_fact_abstains():
    dm = DecisionModel(XClf(), P.CertifiedPolicy, features=FEATS,
                       regime="off")
    d = dm.dispose({"x": 0.02})
    assert d.verdict == ABSTAIN
    assert d.missing == "certified"
    assert d.calibration_digest is None


def test_unseen_group_is_not_certified():
    xs = [0.01 + 0.98 * i / 39 for i in range(40)]
    rows = [{"x": x, "region": "A"} for x in xs]
    y = [1 if x >= 0.5 else 0 for x in xs]
    dm = DecisionModel(XClf(), P.CertifiedPolicy,
                       features=("x", "region"), regime="off")
    dm.calibrate(rows, y, alpha=0.10, group="region")
    d = dm.dispose({"x": 0.02, "region": "B"})
    assert d.verdict == ASK
    assert d.prediction["certified"] is False
    assert d.prediction["calibrated_group"] == "B"


def test_group_needs_minimum_data():
    dm = DecisionModel(XClf(), P.CertifiedPolicy,
                       features=("x", "region"), regime="off")
    rows = [{"x": 0.1 + 0.01 * i, "region": "A"} for i in range(5)]
    with pytest.raises(CalibrationError, match="conformal floor"):
        dm.calibrate(rows, [0] * 5, alpha=0.10, group="region")


def test_alpha_too_small_raises():
    with pytest.raises(CalibrationError):
        fit_kappa(XClf(), _rows([0.1, 0.9]), [0, 1], alpha=0.001)


def test_calibration_label_outside_classes_raises():
    with pytest.raises(CalibrationError, match="not in estimator classes"):
        fit_kappa(XClf(), _rows([0.1] * 20), [2] * 20, alpha=0.1)


def test_regressor_interval_gates_the_band():
    xs = list(range(1, 21))
    rows = [{"x": x} for x in xs]
    y = [x + 1.0 for x in xs]            # residual |y-pred| = 1
    dm = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                       regime="off")
    dm.calibrate(rows, y, alpha=0.10)
    d = dm.dispose({"x": 5.0})
    assert d.prediction["lo"] == pytest.approx(4.0, abs=0.2)
    assert d.prediction["hi"] == pytest.approx(6.0, abs=0.2)
    assert d.verdict == RELEASE
    assert d.basis == ("safe_band",)
    d2 = dm.dispose({"x": 25.0})
    assert d2.prediction["hi"] > 20
    assert d2.verdict == DENY


def test_regressor_uncalibrated_group_abstains_on_missing_lo():
    xs = list(range(1, 21))
    rows = [{"x": x, "region": "A"} for x in xs]
    dm = DecisionModel(XReg(), P.RegressorPolicy,
                       features=("x", "region"), regime="off")
    dm.calibrate(rows, [x + 1.0 for x in xs], alpha=0.10, group="region")
    d = dm.dispose({"x": 5.0, "region": "B"})
    assert d.verdict == ABSTAIN
    assert d.missing == "hi"  # the engine rule requiring hi runs before releases
