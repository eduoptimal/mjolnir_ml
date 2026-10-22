"""mjolnir_ml.calibration — κ: the calibrated trust index over a learner.

THE LOAD-BEARING IDEA (COHERENCE.md §5, CERTIFIABILITY.md): what a learner is
allowed to *claim* is not its softmax. It is a finite-sample guarantee about
the learner's own error, estimated per region and pinned by digest. This module
implements split conformal calibration with optional Mondrian (group) splitting:

    classifier  nonconformity  s(x, y) = 1 - p_y(x)
                prediction set  P(x) = { y : s(x, y) <= q(group(x)) }
                certified       |P(x)| == 1          (a guaranteed singleton)

    regressor   nonconformity  s(x) = |y - f(x)|
                interval        [f(x) - q, f(x) + q]   (coverage >= 1 - alpha)

q is the finite-sample conformal quantile ceil((n+1)(1-alpha)) / n. The floor
n >= ceil(1/alpha) - 1 is enforced per group: a region with too little data is
a calibration ERROR, not a silent uncalibrated claim ("unknown is not zero").

WHAT THIS BUYS A HIGH-RISK APPLICATION
--------------------------------------
The policy can say `certified` (a singleton at the declared alpha) instead of
`conf > 0.9` (an uncalibrated number). That is the difference between "the
softmax looked confident" and "at most alpha of these claims are wrong, over
exchangeable data, per region". For a regressor the policy gates on `lo`/`hi`:
an action is releasable only if its entire conformal interval clears the
declared threshold.

NON-CLAIMS: coverage is marginal-per-group and relative to exchangeability.
Under drift, coverage can degrade (the beta_release G4 result); the honest
responses are (a) regroup, (b) recalibrate, (c) adaptive conformal — not a
claim that the guarantee survives distribution shift.
"""
from __future__ import annotations

import hashlib
import json
import math


def _scalar(v):
    return v.item() if hasattr(v, "item") else v


class CalibrationError(ValueError):
    """Calibration refused: too little data, wrong shapes, unfitted estimator."""


def _quantile(sorted_scores: list, n: int, alpha: float) -> float:
    """Split-conformal quantile at level ceil((n+1)(1-alpha))/n."""
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        # alpha too small for n -> the guarantee needs more data; refuse.
        raise CalibrationError(
            f"alpha={alpha} needs n >= ceil(1/alpha)-1; got n={n}")
    return float(sorted_scores[rank - 1])


def _digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


class Kappa:
    """The trust index: per-group conformal quantiles + the data they came from."""

    def __init__(self, kind: str, alpha: float, group: str | None,
                 classes: list | None, per_group: dict, n_total: int):
        self.kind = kind                  # "clf" | "reg"
        self.alpha = float(alpha)
        self.group = group                # feature name or None (pooled)
        self.classes = list(classes) if classes is not None else None
        self.per_group = per_group        # key -> {"q","n","scores"}
        self.n_total = n_total
        self._digest = _digest({
            "kind": kind, "alpha": self.alpha, "group": group,
            "classes": self.classes,
            "groups": {k: {"q": v["q"], "n": v["n"], "scores": v["scores"]}
                       for k, v in per_group.items()}})

    # ------------------------------------------------------------- access --
    @property
    def digest(self) -> str:
        return self._digest

    def group_key(self, row) -> str:
        return str(row.get(self.group, "__unstated__")) if self.group \
            else "__all__"

    def calibrated(self, key: str) -> bool:
        return key in self.per_group

    def q(self, key: str) -> float | None:
        g = self.per_group.get(key)
        return None if g is None else g["q"]

    # ------------------------------------------------------------ predict --
    def prediction_set(self, proba_by_class: dict, key: str) -> list:
        """Conformal prediction set. Unknown group -> ALL classes: the honest
        statement is 'no calibrated set here', and a set of size |classes|
        can never be certified as a singleton."""
        q = self.q(key)
        if q is None:
            return list(self.classes or ())
        return [c for c in self.classes if (1.0 - float(proba_by_class[c])) <= q]

    def interval(self, pred: float, key: str) -> tuple | None:
        """Conformal interval at 1-alpha, or None when the group is uncalibrated
        (the harness then omits lo/hi and the policy abstains on missing)."""
        q = self.q(key)
        if q is None:
            return None
        return (float(pred) - q, float(pred) + q)


class AdaptiveKappa(Kappa):
    """ACI: the per-group risk level moves from realized miscoverage.

    Gibbs & Candes-style update after each observed nonconformity score s:

        err_t = 1{ s_t > q_t }
        alpha_{t+1} = clip( alpha_t + gamma * (alpha0 - err_t),
                            1/(n+2), 1 )

    q_t is recomputed from the score window at level 1 - alpha_t, so the
    prediction set / interval widens after errors and narrows when it is
    over-covering. Static calibration under drift is the G4 failure: coverage
    degrades and a window alone does not restore it. ACI restores coverage by
    trading it for width — the price is reported, not hidden.

    The DIGEST still pins the initial calibration artifact (the data), not the
    moving runtime state; certificates issued before drift remain auditable.
    """

    def __init__(self, kind, alpha, group, classes, per_group, n_total,
                 gamma=0.05, window=400):
        super().__init__(kind, alpha, group, classes, per_group, n_total)
        if not (0.0 < float(gamma) <= 1.0):
            raise CalibrationError(f"gamma must be in (0,1], got {gamma!r}")
        if int(window) < 2:
            raise CalibrationError(f"window must be >= 2, got {window!r}")
        self.gamma = float(gamma)
        self.window = int(window)
        self._state = {k: {"scores": list(v["scores"]), "alpha": self.alpha}
                       for k, v in per_group.items()}

    # ------------------------------------------------------------- dynamic --
    def alpha_at(self, key: str) -> float | None:
        st = self._state.get(key)
        return None if st is None else st["alpha"]

    def q(self, key: str) -> float | None:
        st = self._state.get(key)
        if st is None:
            return None
        scores = sorted(st["scores"])
        n = len(scores)
        rank = math.ceil((n + 1) * (1.0 - st["alpha"]))
        rank = min(max(rank, 1), n)
        return float(scores[rank - 1])

    def observe(self, key: str, score: float) -> None:
        """One realized nonconformity score for the region `key`."""
        st = self._state.get(key)
        if st is None:
            raise CalibrationError(
                f"observe: region {key!r} was never calibrated — unknown is "
                f"not zero; regroup or decline the feedback")
        if not math.isfinite(float(score)):
            raise CalibrationError("observe: non-finite score")
        q = self.q(key)
        err = 1.0 if (q is None or float(score) > q) else 0.0
        st["alpha"] = min(1.0, max(1.0 / (len(st["scores"]) + 2),
                                st["alpha"] + self.gamma * (self.alpha - err)))
        st["scores"].append(float(score))
        if len(st["scores"]) > self.window:
            del st["scores"][0]


def fit_kappa(estimator, rows, y, alpha: float = 0.05,
        group: str | None = None, features=None, adaptive: bool = False,
        gamma: float = 0.05, window: int = 400) -> Kappa:
    """Split-conformal calibration. rows: list[dict] in the model's feature
    schema; y: ground truth per row; features: the estimator's input order
    (defaults to the first row's keys). group: feature name for Mondrian.

    Refuses (CalibrationError) rather than producing an uncalibrated claim:
    wrong lengths, unfitted estimator, labels outside classes_, a group with
    fewer than ceil(1/alpha)-1 samples, or a group key not stated in a row.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise CalibrationError(f"alpha must be in (0,1), got {alpha!r}")
    alpha = float(alpha)
    rows = list(rows)
    y = list(y)
    if len(rows) != len(y) or not rows:
        raise CalibrationError(
            f"need equal non-empty rows/y, got {len(rows)}/{len(y)}")
    if group is not None and not isinstance(group, str):
        raise CalibrationError("group must be a feature name or None")
    if group is not None and any(group not in r for r in rows):
        raise CalibrationError(f"group feature {group!r} absent from some rows")

    has_proba = callable(getattr(estimator, "predict_proba", None))
    kind = "clf" if has_proba else "reg"
    classes = None
    if kind == "clf":
        classes = [_scalar(c) for c in getattr(estimator, "classes_", [])]
        if not classes:
            raise CalibrationError(
                "classifier must be fitted and expose classes_ before "
                "calibration")

    buckets: dict = {}
    feats = tuple(features) if features else tuple(rows[0].keys())
    X_all = [[r[f] for f in feats] for r in rows]
    try:
        if kind == "clf":
            proba = estimator.predict_proba(X_all)
        else:
            pred = estimator.predict(X_all)
    except Exception as ex:                                   # noqa: BLE001
        raise CalibrationError(
            f"estimator failed during calibration: {type(ex).__name__}: {ex}")

    for i, row in enumerate(rows):
        key = str(row[group]) if group else "__all__"
        if kind == "clf":
            label = _scalar(y[i])
            if label not in classes:
                raise CalibrationError(
                    f"row {i}: label {label!r} not in estimator classes_ "
                    f"{classes}")
            p = {c: float(_scalar(proba[i][j])) for j, c in enumerate(classes)}
            s = 1.0 - p[label]
        else:
            s = abs(float(y[i]) - float(_scalar(pred[i])))
        if not math.isfinite(s):
            raise CalibrationError(f"row {i}: non-finite nonconformity score")
        buckets.setdefault(key, []).append(s)

    per_group = {}
    for key, scores in buckets.items():
        scores.sort()
        n = len(scores)
        if n < math.ceil(1.0 / alpha) - 1:
            raise CalibrationError(
                f"group {key!r}: n={n} is below the conformal floor "
                f"ceil(1/alpha)-1={math.ceil(1.0 / alpha) - 1} for "
                f"alpha={alpha}; regroup or collect more calibration data")
        per_group[key] = {"q": _quantile(scores, n, alpha), "n": n,
                          "scores": scores}

    cls = AdaptiveKappa if adaptive else Kappa
    kwargs = {"gamma": gamma, "window": window} if adaptive else {}
    return cls(kind=kind, alpha=alpha, group=group, classes=classes,
               per_group=per_group, n_total=len(rows), **kwargs)


__all__ = ["Kappa", "AdaptiveKappa", "fit_kappa", "CalibrationError"]
