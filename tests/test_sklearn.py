"""Integration: real sklearn estimators through the harness (the point of it)."""
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification                    # noqa: E402
from sklearn.linear_model import LogisticRegression, LinearRegression  # noqa: E402

from mjolnir_ml import DecisionModel, audit_model                    # noqa: E402
from mjolnir_ml.vendors.wire import (ABSTAIN_W as ABSTAIN,          # noqa: E402
                                    DENY_W as DENY, RELEASE, ASK, WIRE)  # noqa: E402

import policies as P                                                 # noqa: E402

FEATS = ("f0", "f1", "f2", "f3")


def _data(n=400, seed=0):
    X, y = make_classification(n_samples=n, n_features=4, n_informative=3,
                               n_redundant=0, random_state=seed)
    rows = [dict(zip(FEATS, map(float, x))) for x in X]
    return rows, list(y)


def _split(rows, y, cut=260):
    return rows[:cut], y[:cut], rows[cut:], y[cut:]


def test_sklearn_classifier_end_to_end():
    rows, y = _data()
    tr_rows, tr_y, te_rows, te_y = _split(rows, y)
    clf = LogisticRegression(max_iter=1000).fit(
        [[r[f] for f in FEATS] for r in tr_rows], tr_y)
    dm = DecisionModel(clf, P.SklearnPolicy, features=FEATS)
    dm.fit_regime(tr_rows).calibrate(tr_rows, tr_y, alpha=0.10)
    disps = [dm.dispose(r) for r in te_rows]

    assert {d.verdict for d in disps} <= set(WIRE)
    # fail-closed: no verified release without a certified singleton
    for d in disps:
        if d.verdict == RELEASE:
            assert d.prediction["certified"] is True
            assert d.prediction["p_positive"] <= 0.20
        assert d.policy_digest == P.SklearnPolicy.policy().digest
        assert d.calibration_digest == dm.kappa.digest

    # the certificate replays on every held-out row
    for r, d in zip(te_rows[:25], disps[:25]):
        assert dm.replay(d, r)

    report = audit_model(dm, te_rows, safe=[v == 0 for v in te_y],
                         decidable=[True] * len(te_y))
    assert report["pair"]["n"] == len(te_rows)
    assert report["degenerate"]["NEVER"]["prec"] is None
    assert set(report["verdicts"]) == set(WIRE)


def test_sklearn_regressor_end_to_end():
    rows, y = _data()
    yv = [float(v) * 3.0 + 1.0 for v in y]
    tr_rows, tr_y, te_rows, te_y = _split(rows, yv)
    reg = LinearRegression().fit([[r[f] for f in FEATS] for r in tr_rows], tr_y)
    dm = DecisionModel(reg, P.RegressorPolicy, features=FEATS,
                       regime="required")
    dm.fit_regime(tr_rows).calibrate(tr_rows, tr_y, alpha=0.10)
    for r in te_rows[:40]:
        d = dm.dispose(r)
        assert d.verdict in (RELEASE, DENY, ASK, ABSTAIN)
        if d.verdict == RELEASE:
            assert d.prediction["lo"] >= 0.0
            assert d.prediction["hi"] <= 10.0
