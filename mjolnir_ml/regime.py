"""mjolnir_ml.regime — multivariate regime floors with a conformal cutoff.

`fit_regime(rows)` records independent per-feature min/max boxes. That is a
sound *necessary* condition but a poor multivariate support test: correlated
in-distribution points routinely cross one axis while lying well inside the
joint support. This module provides the joint version:

    score(x) = anomaly(x)                 (higher = further from training support)
    tau      = conformal quantile of scores on a held-out calibration set
    in_regime(x) <=> score(x) <= tau      => P(false abstention) <= alpha_ood

Two detectors are built in, imported lazily so the package stays dependency
free at import time:

    mahalanobis       robust center/covariance (MinCovDet) distance
    isolation_forest  path-length anomaly score

A custom score function can be supplied with `from_fn` (or
`DecisionModel.fit_regime(..., method="custom", score_fn=...)`).

The calibrated cutoff, detector configuration and feature list are pinned in
`digest`, which travels with every disposition and certificate like the model
and conformal digests.
"""
from __future__ import annotations

import math

from mjolnir_ml._digest import digest as _digest
from mjolnir_ml._digest import estimator_digest as _estimator_digest


class RegimeError(ValueError):
    """The regime floor cannot be fitted or calibrated (fail closed)."""


def _matrix(rows, features) -> list:
    out = []
    for i, row in enumerate(rows):
        vec = []
        for f in features:
            if f not in row:
                raise RegimeError(f"row {i}: feature {f!r} absent")
            v = row[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise RegimeError(
                    f"row {i}: feature {f!r} is not numeric ({v!r}); "
                    f"multivariate regime needs a numeric feature space")
            v = float(v)
            if not math.isfinite(v):
                raise RegimeError(f"row {i}: feature {f!r} is not finite")
            vec.append(v)
        out.append(vec)
    return out


class MultivariateRegimeFloor:
    """A calibrated multivariate support test for the model's inputs."""

    METHODS = ("mahalanobis", "isolation_forest", "custom")

    def __init__(self, detector, score_fn, method: str, features: tuple,
                 alpha_ood: float, tau: float, n_cal: int,
                 detector_digest: str):
        self.detector = detector
        self._score_fn = score_fn
        self.method = method
        self.features = tuple(features)
        self.alpha_ood = float(alpha_ood)
        self.tau = float(tau)
        self.n_cal = int(n_cal)
        self._digest = _digest({
            "kind": "regime", "method": method,
            "features": list(self.features), "alpha_ood": self.alpha_ood,
            "tau": self.tau, "n_cal": self.n_cal,
            "detector": detector_digest,
        })

    @property
    def digest(self) -> str:
        return self._digest

    def as_dict(self) -> dict:
        return {"method": self.method, "alpha_ood": self.alpha_ood,
                "tau": self.tau, "n_cal": self.n_cal,
                "features": list(self.features), "digest": self.digest}

    def score(self, x) -> float:
        arr = self._score_fn([list(map(float, x))])
        value = float(arr[0])
        if not math.isfinite(value):
            raise RegimeError(f"{self.method} emitted a non-finite score")
        return value

    def check(self, x) -> tuple:
        """(in_regime, reason, score). A score above tau is out of regime."""
        score = self.score(x)
        if score > self.tau:
            return (False,
                    f"{self.method} score={score:.6g} > tau={self.tau:.6g}",
                    score)
        return True, "in_regime", score

    # --------------------------------------------------------------- builders --
    @classmethod
    def from_fn(cls, score_fn, *, method: str, features, rows,
                calibration_rows, alpha_ood: float = 0.02):
        """Build from any callable `score_fn(matrix) -> sequence[float]`."""
        if not callable(score_fn):
            raise RegimeError("score_fn must be callable")
        cal = _matrix(list(calibration_rows), tuple(features))
        return cls._build(None, score_fn, method, tuple(features), cal,
                          alpha_ood, _digest([method, "callable"]))

    @classmethod
    def fit(cls, method: str, rows, features, calibration_rows,
            alpha_ood: float = 0.02, random_state: int = 0):
        """Fit a built-in detector on `rows`, calibrate tau on held-out rows."""
        if method not in ("mahalanobis", "isolation_forest"):
            raise RegimeError(
                f"method must be 'mahalanobis' or 'isolation_forest' "
                f"(or use from_fn for custom), got {method!r}")
        if not (0.0 < float(alpha_ood) < 1.0):
            raise RegimeError(
                f"alpha_ood must be in (0,1), got {alpha_ood!r}")
        train = _matrix(list(rows), tuple(features))
        cal = _matrix(list(calibration_rows), tuple(features))

        if method == "mahalanobis":
            try:
                from sklearn.covariance import MinCovDet
            except Exception as ex:                            # noqa: BLE001
                raise RegimeError(
                    f"mahalanobis regime needs scikit-learn: {ex}")
            detector = MinCovDet(random_state=random_state).fit(train)
            score_fn = lambda X: detector.mahalanobis(X)        # noqa: E731
            state = [repr(getattr(detector, "location_", None))]
            cov = getattr(detector, "covariance_", None)
            if cov is not None:
                try:
                    state.append(repr(cov.tolist()))
                except Exception:                              # noqa: BLE001
                    state.append(repr(cov))
            det_digest = _digest([_estimator_digest(detector), state])
        else:
            try:
                from sklearn.ensemble import IsolationForest
            except Exception as ex:                            # noqa: BLE001
                raise RegimeError(
                    f"isolation_forest regime needs scikit-learn: {ex}")
            detector = IsolationForest(
                n_estimators=200, random_state=random_state,
                n_jobs=-1).fit(train)
            score_fn = lambda X: -detector.score_samples(X)     # noqa: E731
            det_digest = _digest([_estimator_digest(detector), "isolation"])

        return cls._build(detector, score_fn, method, tuple(features), cal,
                          alpha_ood, det_digest)

    @classmethod
    def _build(cls, detector, score_fn, method, features, cal, alpha_ood,
               detector_digest):
        if not cal:
            raise RegimeError("calibration_rows must be non-empty")
        alpha_ood = float(alpha_ood)
        if not (0.0 < alpha_ood < 1.0):
            raise RegimeError(
                f"alpha_ood must be in (0,1), got {alpha_ood!r}")
        scores = []
        for v in score_fn(cal):
            v = float(v)
            if not math.isfinite(v):
                raise RegimeError(f"{method} emitted a non-finite score")
            scores.append(v)
        scores.sort()
        n = len(scores)
        rank = math.ceil((n + 1) * (1.0 - alpha_ood))
        if rank > n:
            raise RegimeError(
                f"regime calibration needs n >= ceil(1/alpha_ood)-1 = "
                f"{math.ceil(1.0 / alpha_ood) - 1}, got n={n}")
        tau = scores[rank - 1]
        return cls(detector, score_fn, method, features, alpha_ood, tau, n,
                   detector_digest)


__all__ = ["RegimeError", "MultivariateRegimeFloor"]
