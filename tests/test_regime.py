"""Multivariate regime floors: calibrated OOD support + certificate pinning."""
import math

import pytest

from mjolnir_ml import (DecisionModel, MultivariateRegimeFloor, RegimeError,
                        certify, report, verify_certificate)
from mjolnir_ml.vendors.wire import ABSTAIN_W as ABSTAIN, ASK

import policies as P


class StubReg:
    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [0.0 for _ in X]


FEATS = ("a", "b")


def _rows(n, scale=1.0):
    return [{"a": scale * math.cos(2 * math.pi * i / n),
             "b": scale * math.sin(2 * math.pi * i / n)} for i in range(n)]


def _dm(alpha_ood=0.05):
    dm = DecisionModel(StubReg(), P.RegimePolicy, features=FEATS)
    dm.fit_regime(_rows(40), method="custom",
                  score_fn=lambda X: [row[0] ** 2 + row[1] ** 2 for row in X],
                  calibration_rows=_rows(200), alpha_ood=alpha_ood)
    return dm


def test_multivariate_floor_abstains_on_true_ood():
    dm = _dm()
    inside = dm.dispose({"a": 0.2, "b": 0.2})
    outside = dm.dispose({"a": 5.0, "b": 5.0})
    assert inside.verdict == ASK
    assert outside.verdict == ABSTAIN
    assert outside.basis == ("out_of_regime:custom",)
    assert outside.prediction["regime_score"] > outside.prediction["regime_tau"]


def test_regime_digest_is_pinned_in_the_certificate():
    dm = _dm()
    d = dm.dispose({"a": 0.2, "b": 0.2})
    assert d.regime_digest == dm.regime_floor.digest
    cert = certify(d, key=b"k")
    assert cert["regime_digest"] == dm.regime_floor.digest
    assert verify_certificate(cert, key=b"k", policy=P.RegimePolicy) is True


def test_regime_floor_requires_calibration_rows():
    dm = DecisionModel(StubReg(), P.RegimePolicy, features=FEATS)
    with pytest.raises(Exception, match="calibration_rows"):
        dm.fit_regime(_rows(20), method="mahalanobis")
    with pytest.raises(Exception, match="custom"):
        dm.fit_regime(_rows(20), method="custom")


def test_regime_floor_refuses_insufficient_calibration():
    dm = DecisionModel(StubReg(), P.RegimePolicy, features=FEATS)
    with pytest.raises(Exception, match="regime calibration needs"):
        dm.fit_regime(_rows(20), method="custom",
                      score_fn=lambda X: [0.0 for _ in X],
                      calibration_rows=_rows(20), alpha_ood=0.01)


def test_report_shows_the_multivariate_floor():
    dm = _dm()
    text = report(dm, [{"a": 0.2, "b": 0.2}])
    assert "method=custom" in text
    assert "alpha_ood=0.05" in text
    assert "tau=" in text


def test_sklearn_mahalanobis_floor_fits():
    pytest.importorskip("sklearn")
    dm = DecisionModel(StubReg(), P.RegimePolicy, features=FEATS)
    train = [{"a": 0.1 * i, "b": 0.03 * (i * i % 17)} for i in range(60)]
    cal = [{"a": 0.1 * i, "b": 0.03 * (i * i % 17)} for i in range(60)]
    dm.fit_regime(train, method="mahalanobis", calibration_rows=cal,
                  alpha_ood=0.05)
    assert dm.regime_floor is not None
    assert dm.regime_floor.method == "mahalanobis"
    assert dm.dispose({"a": 0.2, "b": 0.1}).verdict == ASK
    assert dm.dispose({"a": 100.0, "b": 100.0}).verdict == ABSTAIN
