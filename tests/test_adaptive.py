"""ACI: coverage under drift, the beta_release G4 failure and its repair."""
import random

import pytest

from mjolnir_ml import (CalibrationError, DecisionModel, Kappa, AdaptiveKappa)
from mjolnir_ml.vendors import wire

import policies as P


class XClf:
    """P(1) = x, clean labels are the threshold rule."""

    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def _xs(n=240, offset=0):
    import math
    return [((i + offset) % 40) / 40.0 + 0.0125 for i in range(n)]


def _clean(x):
    return 1 if x >= 0.5 else 0


def _fit(adaptive):
    cal_x = _xs(60, offset=7)
    rows = [{"x": x} for x in cal_x]
    dm = DecisionModel(XClf(), P.CertifiedPolicy, features=("x",),
                       regime="off")
    dm.calibrate(rows, [_clean(x) for x in cal_x], alpha=0.10,
                 adaptive=adaptive, gamma=0.05, window=200)
    return dm


def test_adaptive_kappa_widens_on_error_and_narrows_on_cover():
    dm = _fit(adaptive=True)
    k = dm.kappa
    assert isinstance(k, AdaptiveKappa)
    a0 = k.alpha_at("__all__")
    q0 = k.q("__all__")
    for _ in range(12):
        k.observe("__all__", 1.0)          # large score = miscoverage
    a1, q1 = k.alpha_at("__all__"), k.q("__all__")
    assert a1 < a0                         # alpha_t falls -> set widens
    assert q1 >= q0
    for _ in range(40):
        k.observe("__all__", 0.0)          # perfect coverage -> narrow again
    assert k.alpha_at("__all__") > a1


def test_static_calibration_has_no_feedback():
    dm = _fit(adaptive=False)
    assert isinstance(dm.kappa, Kappa) and not isinstance(dm.kappa, AdaptiveKappa)
    assert dm.observe({"x": 0.2}, 0) is dm      # documented no-op


def test_observe_unknown_group_fails_closed():
    cal_x = _xs(40, offset=3)
    rows = [{"x": x, "region": "A"} for x in cal_x]
    dm = DecisionModel(XClf(), P.CertifiedPolicy, features=("x", "region"),
                       regime="off")
    dm.calibrate(rows, [_clean(x) for x in cal_x], alpha=0.10, group="region",
                 adaptive=True)
    with pytest.raises(CalibrationError, match="never calibrated"):
        dm.observe({"x": 0.2, "region": "B"}, 0)


def _coverage_windows(dm, xs, ys, window=30):
    """Decision-time coverage: was the realized label inside the set?"""
    cover = []
    for x, y in zip(xs, ys):
        d = dm.dispose({"x": x})
        assert d.prediction is not None and "set" in d.prediction
        cover.append(1.0 if y in d.prediction["set"] else 0.0)
        dm.observe({"x": x}, y)
    return [sum(cover[i:i + window]) / window
            for i in range(0, len(cover), window)]


def test_aci_restores_coverage_under_concept_drift():
    xs = _xs(240, offset=1)
    rng = random.Random(0)
    onset = 90
    ys = []
    for i, x in enumerate(xs):
        y = _clean(x)
        if i >= onset and rng.random() < 0.50:
            y = 1 - y                       # concept drift: labels flip
        ys.append(y)

    static = _fit(adaptive=False)
    adaptive = _fit(adaptive=True)
    cov_static = _coverage_windows(static, xs, ys)
    cov_adapt = _coverage_windows(adaptive, xs, ys)

    pre = slice(0, onset // 30)
    post = slice((onset + 60) // 30, None)
    pre_static = sum(cov_static[pre]) / len(cov_static[pre])
    post_static = sum(cov_static[post]) / len(cov_static[post])
    post_adapt = sum(cov_adapt[post]) / len(cov_adapt[post])

    # G4: static coverage degrades under drift
    assert pre_static >= 0.85
    assert post_static < pre_static - 0.10
    # ACI: the adaptive index recovers coverage (at the price of width)
    assert post_adapt > post_static + 0.10
