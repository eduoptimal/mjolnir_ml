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


# ------------------------------------------------- local-width calibration --

class Affine:
    """prediction = a*x + b; a stand-in for a fitted scale/quantile model."""

    def __init__(self, a=1.0, b=0.0):
        self.a, self.b = a, b

    def get_params(self, deep=False):
        return {"a": self.a, "b": self.b}

    def predict(self, X):
        return [self.a * row[0] + self.b for row in X]


def test_normalized_interval_uses_the_scale_estimator():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    scale = Affine(1.0, 1.0)                 # sigma(x) = x + 1
    y = [x + 0.5 * (x + 1.0) for x in xs]    # constant normalized score 0.5
    dm = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                       regime="off")
    dm.calibrate(rows, y, alpha=0.10, method="normalized", scale=scale)
    d = dm.dispose({"x": 5.0})
    assert d.prediction["lo"] == pytest.approx(5.0 - 0.5 * 6.0)
    assert d.prediction["hi"] == pytest.approx(5.0 + 0.5 * 6.0)
    assert d.prediction["hi"] - d.prediction["lo"] == pytest.approx(6.0)
    assert d.verdict == RELEASE
    assert dm.kappa.method == "normalized"


def test_normalized_interval_is_narrower_where_the_model_is_quiet():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    dm = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                       regime="off")
    dm.calibrate(rows, [x + 0.5 * (x + 1.0) for x in xs], alpha=0.10,
                 method="normalized", scale=Affine(1.0, 1.0))
    small = dm.dispose({"x": 1.0}).prediction
    large = dm.dispose({"x": 19.0}).prediction
    assert (large["hi"] - large["lo"]) > (small["hi"] - small["lo"])


def test_cqr_interval_uses_both_quantile_estimators():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    y = [x + 1.0 for x in xs]                # score = max(-2, 0) = 0
    dm = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                       regime="off")
    dm.calibrate(rows, y, alpha=0.10, method="cqr",
                 lower=Affine(1.0, -1.0), upper=Affine(1.0, 1.0))
    d = dm.dispose({"x": 5.0})
    assert d.prediction["lo"] == pytest.approx(4.0)
    assert d.prediction["hi"] == pytest.approx(6.0)
    assert d.verdict == RELEASE
    assert dm.kappa.method == "cqr"


def test_width_method_changes_the_calibration_digest():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    y = [x + 1.0 for x in xs]
    split = fit_kappa(XReg(), rows, y, alpha=0.10)
    normalized = fit_kappa(XReg(), rows, y, alpha=0.10,
                           method="normalized", scale=Affine(1.0, 0.0))
    assert split.digest != normalized.digest


def test_auxiliary_methods_refuse_missing_or_mismatched_estimators():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    y = [x + 1.0 for x in xs]
    with pytest.raises(CalibrationError, match="needs a fitted `scale`"):
        fit_kappa(XReg(), rows, y, alpha=0.10, method="normalized")
    with pytest.raises(CalibrationError, match="`lower` and `upper`"):
        fit_kappa(XReg(), rows, y, alpha=0.10, method="cqr",
                  lower=Affine(1.0, 0.0))
    with pytest.raises(CalibrationError, match="regressor width"):
        fit_kappa(XClf(), _rows([0.1] * 20), [0] * 20, alpha=0.10,
                  method="normalized", scale=Affine(1.0, 0.0))
    with pytest.raises(CalibrationError, match="takes no auxiliary"):
        fit_kappa(XReg(), rows, y, alpha=0.10, method="split",
                  scale=Affine(1.0, 0.0))


def test_nonpositive_scale_is_a_calibration_error():
    xs = [float(x) for x in range(1, 21)]
    with pytest.raises(CalibrationError, match="non-positive"):
        fit_kappa(XReg(), [{"x": x} for x in xs], [x + 1.0 for x in xs],
                  alpha=0.10, method="normalized", scale=Affine(0.0, 0.0))


def test_adaptive_normalized_observe_uses_the_same_score():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    dm = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                       regime="off")
    dm.calibrate(rows, [x + 0.5 * (x + 1.0) for x in xs], alpha=0.10,
                 method="normalized", scale=Affine(1.0, 1.0), adaptive=True)
    before = dm.kappa.alpha_at("__all__")
    dm.observe({"x": 5.0}, 100.0)                      # a large miss
    after = dm.kappa.alpha_at("__all__")
    assert after < before                              # widen after a miss


class StepScale:
    """Positive on the calibration rows, zero outside them (a fail-closed probe)."""

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1.0 if row[0] <= 10.0 else 0.0 for row in X]


def test_auxiliary_failure_at_decision_time_omits_the_interval():
    xs = [float(x) for x in range(1, 11)]
    dm = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                       regime="off")
    dm.calibrate([{"x": x} for x in xs], [x + 1.0 for x in xs], alpha=0.10,
                 method="normalized", scale=StepScale())
    d = dm.dispose({"x": 20.0})          # sigma = 0 -> no interval
    assert d.prediction.get("lo") is None
    assert d.verdict == ABSTAIN
    assert d.missing in ("lo", "hi")


# --------------------------------------------------- one-sided (two-track) --

def test_one_sided_split_lifts_the_lower_bound():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    # 18 conservative overpredictions and 2 dangerous underpredictions:
    # the symmetric |residual| quantile is 5, but the lower tail only needs 1.
    y = [x - 1.0 for x in xs]
    y[0] = xs[0] + 5.0
    y[7] = xs[7] + 5.0
    two = fit_kappa(XReg(), rows, y, alpha=0.10)
    one = fit_kappa(XReg(), rows, y, alpha=0.10, one_sided=True)
    d_two = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                          regime="off")
    d_two._kappa = two
    d_one = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                          regime="off")
    d_one._kappa = one
    lo_two = d_two.dispose({"x": 5.0}).prediction["lo"]
    lo_one = d_one.dispose({"x": 5.0}).prediction["lo"]
    assert lo_one > lo_two                 # the lower tail is no longer diluted
    assert one.q_lo("__all__") == pytest.approx(1.0)
    assert one.q_hi("__all__") == pytest.approx(5.0)


def test_one_sided_tails_can_use_different_risk_levels():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    y = [x + (5.0 if i < 2 else -1.0) for i, x in enumerate(xs)]
    k = fit_kappa(XReg(), rows, y, alpha=0.10, one_sided=True,
                  alpha_lower=0.10, alpha_upper=0.40)
    assert k.alpha_lower == pytest.approx(0.10)
    assert k.alpha_upper == pytest.approx(0.40)
    assert k.q_hi("__all__") < fit_kappa(
        XReg(), rows, y, alpha=0.10, one_sided=True).q_hi("__all__")
    assert k.digest != fit_kappa(XReg(), rows, y, alpha=0.10,
                                 one_sided=True).digest


def test_one_sided_cqr_uses_both_quantile_tails():
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    y = [x for x in xs]
    y[0] = xs[0] + 4.0
    y[7] = xs[7] + 4.0
    dm = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                       regime="off")
    dm.calibrate(rows, y, alpha=0.10, method="cqr", one_sided=True,
                 lower=Affine(1.0, -2.0), upper=Affine(1.0, 2.0))
    d = dm.dispose({"x": 5.0})
    assert d.prediction["lo"] == pytest.approx(5.0)
    assert d.prediction["hi"] == pytest.approx(9.0)
    assert dm.kappa.one_sided is True


def test_one_sided_is_regressor_only_and_not_adaptive():
    with pytest.raises(CalibrationError, match="regressor width"):
        fit_kappa(XClf(), _rows([0.1] * 20), [0] * 20, alpha=0.10,
                  one_sided=True)
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    with pytest.raises(CalibrationError, match="not combinable"):
        fit_kappa(XReg(), rows, [x + 1.0 for x in xs], alpha=0.10,
                  one_sided=True, adaptive=True)
    with pytest.raises(CalibrationError, match="need one_sided"):
        fit_kappa(XReg(), rows, [x + 1.0 for x in xs], alpha=0.10,
                  alpha_lower=0.2)


def test_report_marks_one_sided_bounds():
    from mjolnir_ml import report_data
    xs = [float(x) for x in range(1, 21)]
    rows = [{"x": x} for x in xs]
    dm = DecisionModel(XReg(), P.RegressorPolicy, features=FEATS,
                       regime="off")
    dm.calibrate(rows, [x + 1.0 for x in xs], alpha=0.10, one_sided=True)
    data = report_data(dm, rows)
    assert data["contract"]["one_sided"] is True
    assert data["contract"]["alpha_lower"] == pytest.approx(0.10)
    assert data["contract"]["alpha_upper"] == pytest.approx(0.10)
