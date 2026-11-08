"""mjolnir_ml.calibration — κ: the calibrated trust index over a learner.

THE LOAD-BEARING IDEA: what a learner is
allowed to *claim* is not its softmax. It is a finite-sample guarantee about
the learner's own error, estimated per region and pinned by digest. This module
implements split conformal calibration with optional Mondrian (group) splitting:

    classifier  nonconformity  s(x, y) = 1 - p_y(x)
                prediction set  P(x) = { y : s(x, y) <= q(group(x)) }
                certified       |P(x)| == 1          (a guaranteed singleton)

    regressor   nonconformity  s(x) = |y - f(x)|
                interval        [f(x) - q, f(x) + q]   (coverage >= 1 - alpha)

Three regressor widths share that guarantee; the method only changes WHERE the
uncertainty is spent (all keep the same finite-sample coverage):

    split        s = |y - f(x)|                 width 2q          (constant)
    normalized   s = |y - f(x)| / sigma(x)      width 2q*sigma(x)  (local)
    cqr          s = max(l(x)-y, y-u(x))        width (u-l)+2q     (local)

`normalized` takes a fitted scale estimator (a residual model); `cqr` takes
fitted lower/upper quantile estimators. The auxiliary estimators are pinned in
the digest. A non-finite or non-positive scale, or an inverted CQR interval,
yields no interval at all (unknown is not zero) — the policy then abstains.

one_sided=True calibrates each tail at its own level instead of paying one
symmetric margin:

    lo = bound_lo - q_lower      P(Y >= lo) >= 1 - alpha_lower
    hi = bound_hi + q_upper      P(Y <= hi) >= 1 - alpha_upper

For quantile models bound_lo/bound_hi are l(x)/u(x); for split/normalized both
are f(x) (times sigma(x)). This is the statistically honest form of a two-track
policy (release reads the upper tail, deny reads the lower), but the pair is a
decision envelope: joint coverage is only >= 1 - alpha_lower - alpha_upper, so
the report labels its width accordingly.

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
Under drift, coverage can degrade (the G4 result); the honest
responses are (a) regroup, (b) recalibrate, (c) adaptive conformal — not a
claim that the guarantee survives distribution shift.
"""
from __future__ import annotations

import math

from mjolnir_ml._digest import digest as _digest
from mjolnir_ml._digest import estimator_digest as _estimator_digest


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




def _predict1(estimator, x) -> float:
    return float(_scalar(estimator.predict([x])[0]))




REGRESSOR_METHODS = ("split", "normalized", "cqr")


def _reg_scores(method: str, aux: dict, x, pred, y) -> tuple:
    """Signed regressor scores: (underprediction, overprediction).

    Both are positive when the realization is on the dangerous side of the
    fitted bound: `lo_score = bound_lo - y` (Y below the lower bound) and
    `hi_score = y - bound_hi` (Y above the upper bound). The two-sided score is
    their max, which is exactly |y - f(x)| for split/normalized and the CQR
    score for quantile models.
    """
    pred = float(pred)
    y = float(y)
    if method == "normalized":
        sigma = _predict1(aux["scale"], x)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise CalibrationError(
                f"scale estimator emitted a non-positive/non-finite width "
                f"({sigma!r}); normalized conformal cannot divide by it")
        return ((pred - y) / sigma, (y - pred) / sigma)
    if method == "cqr":
        lo = _predict1(aux["lower"], x)
        hi = _predict1(aux["upper"], x)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise CalibrationError(
                f"quantile estimators emitted non-finite bounds "
                f"({lo!r}, {hi!r})")
        return (lo - y, y - hi)
    return (pred - y, y - pred)


def _reg_score(method: str, aux: dict, x, pred, y) -> float:
    """Two-sided regressor nonconformity (max of the two signed scores)."""
    lo_s, hi_s = _reg_scores(method, aux, x, pred, y)
    return max(lo_s, hi_s)


class Kappa:
    """The trust index: per-group conformal quantiles + the data they came from."""

    def __init__(self, kind: str, alpha: float, group: str | None,
                 classes: list | None, per_group: dict, n_total: int,
                 method: str = "split", aux: dict | None = None,
                 aux_digest: str | None = None, one_sided: bool = False,
                 alpha_lower: float | None = None,
                 alpha_upper: float | None = None):
        self.kind = kind                  # "clf" | "reg"
        self.alpha = float(alpha)
        self.group = group                # feature name or None (pooled)
        self.classes = list(classes) if classes is not None else None
        self.per_group = per_group        # key -> {"q",...} or {"q_lo","q_hi",...}
        self.n_total = n_total
        self.method = method              # "split" | "normalized" | "cqr"
        self.aux = dict(aux or {})        # scale / lower / upper estimators
        self.one_sided = bool(one_sided)
        self.alpha_lower = float(alpha_lower if alpha_lower is not None
                                 else alpha)
        self.alpha_upper = float(alpha_upper if alpha_upper is not None
                                 else alpha)
        body = {
            "kind": kind, "alpha": self.alpha, "group": group,
            "classes": self.classes, "method": self.method,
            "aux": aux_digest,
            "groups": {k: {kk: vv for kk, vv in v.items()}
                       for k, v in per_group.items()}}
        if self.one_sided:
            body.update(one_sided=True, alpha_lower=self.alpha_lower,
                        alpha_upper=self.alpha_upper)
        self._digest = _digest(body)

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
        return None if g is None else g.get("q")

    def q_lo(self, key: str) -> float | None:
        g = self.per_group.get(key)
        return None if g is None else g.get("q_lo")

    def q_hi(self, key: str) -> float | None:
        g = self.per_group.get(key)
        return None if g is None else g.get("q_hi")

    # ------------------------------------------------------------ predict --
    def prediction_set(self, proba_by_class: dict, key: str) -> list:
        """Conformal prediction set. Unknown group -> ALL classes: the honest
        statement is 'no calibrated set here', and a set of size |classes|
        can never be certified as a singleton."""
        q = self.q(key)
        if q is None:
            return list(self.classes or ())
        return [c for c in self.classes if (1.0 - float(proba_by_class[c])) <= q]

    def interval(self, pred: float, key: str, x=None) -> tuple | None:
        """Conformal bounds, or None when the group is uncalibrated (the
        harness then omits lo/hi and the policy abstains on missing).

        Two-sided: [pred - q, pred + q], a joint 1-alpha interval.
        One-sided: [pred - q_lo, pred + q_hi], where each tail is separately
        calibrated at 1-alpha_lower / 1-alpha_upper. The pair is a decision
        envelope, not a joint 1-alpha interval: P(Y >= lo) >= 1-alpha_lower and
        P(Y <= hi) >= 1-alpha_upper, so a joint conservative bound is
        1-alpha_lower-alpha_upper (union bound).

        Normalized and CQR widths need the input row `x`; when it is absent or
        the auxiliary estimator fails, the honest answer is no bounds."""
        g = self.per_group.get(key)
        if g is None:
            return None
        pred = float(pred)
        if self.method == "normalized":
            if x is None:
                return None
            try:
                sigma = _predict1(self.aux["scale"], x)
            except Exception:                              # noqa: BLE001
                return None
            if not math.isfinite(sigma) or sigma <= 0.0:
                return None
            if self.one_sided:
                return (pred - g["q_lo"] * sigma, pred + g["q_hi"] * sigma)
            return (pred - g["q"] * sigma, pred + g["q"] * sigma)
        if self.method == "cqr":
            if x is None:
                return None
            try:
                lo = _predict1(self.aux["lower"], x)
                hi = _predict1(self.aux["upper"], x)
            except Exception:                              # noqa: BLE001
                return None
            if not (math.isfinite(lo) and math.isfinite(hi)):
                return None
            if self.one_sided:
                lo, hi = lo - g["q_lo"], hi + g["q_hi"]
            else:
                lo, hi = lo - g["q"], hi + g["q"]
            return (lo, hi) if lo <= hi else None
        if self.one_sided:
            return (pred - g["q_lo"], pred + g["q_hi"])
        return (pred - g["q"], pred + g["q"])

    def reg_nonconformity(self, x, pred, y) -> float:
        """The realized score under this index's interval method (ACI feedback)."""
        return _reg_score(self.method, self.aux, x, pred, y)


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
                 gamma=0.05, window=400, method="split", aux=None,
                 aux_digest=None, one_sided=False, alpha_lower=None,
                 alpha_upper=None):
        super().__init__(kind, alpha, group, classes, per_group, n_total,
                         method=method, aux=aux, aux_digest=aux_digest,
                         one_sided=one_sided, alpha_lower=alpha_lower,
                         alpha_upper=alpha_upper)
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
        gamma: float = 0.05, window: int = 400, method: str = "split",
        scale=None, lower=None, upper=None, one_sided: bool = False,
        alpha_lower: float | None = None,
        alpha_upper: float | None = None) -> Kappa:
    """Split-conformal calibration. rows: list[dict] in the model's feature
    schema; y: ground truth per row; features: the estimator's input order
    (defaults to the first row's keys). group: feature name for Mondrian.

    method selects the regressor width: "split" (constant), "normalized"
    (needs a fitted `scale` estimator), or "cqr" (needs fitted `lower`/`upper`
    quantile estimators). All three keep the same finite-sample coverage; the
    auxiliary estimators only spend the width where the error is larger.

    one_sided=True calibrates the lower and upper tails separately:
    `lo = bound_lo - q_lo` guarantees P(Y >= lo) >= 1 - alpha_lower and
    `hi = bound_hi + q_hi` guarantees P(Y <= hi) >= 1 - alpha_upper. The two
    tails may use different risk levels (e.g. a stricter deny tail), but the
    pair is a decision envelope, not a joint 1-alpha interval. ACI and
    one-sided tails are not combinable (adaptive needs one score per update).

    Refuses (CalibrationError) rather than producing an uncalibrated claim:
    wrong lengths, unfitted estimator, labels outside classes_, a group with
    fewer than ceil(1/alpha)-1 samples, a group key not stated in a row, a
    method that does not match the estimator kind, or a missing/invalid
    auxiliary estimator.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise CalibrationError(f"alpha must be in (0,1), got {alpha!r}")
    alpha = float(alpha)
    if method not in REGRESSOR_METHODS:
        raise CalibrationError(
            f"method must be one of {REGRESSOR_METHODS}, got {method!r}")
    one_sided = bool(one_sided)
    if one_sided:
        a_lo = alpha if alpha_lower is None else float(alpha_lower)
        a_hi = alpha if alpha_upper is None else float(alpha_upper)
        for name, a in (("alpha_lower", a_lo), ("alpha_upper", a_hi)):
            if not (0.0 < a < 1.0):
                raise CalibrationError(f"{name} must be in (0,1), got {a!r}")
        if adaptive:
            raise CalibrationError(
                "one-sided bounds and adaptive ACI are not combinable; use "
                "static one-sided or adaptive two-sided")
    elif alpha_lower is not None or alpha_upper is not None:
        raise CalibrationError(
            "alpha_lower/alpha_upper need one_sided=True")
    else:
        a_lo = a_hi = alpha
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
        if method != "split" or scale is not None or lower is not None \
                or upper is not None:
            raise CalibrationError(
                f"method={method!r} is a regressor width; classifiers only "
                f"support method='split' (conformal prediction sets)")
        if one_sided:
            raise CalibrationError(
                "one-sided bounds are a regressor width; classifiers use "
                "conformal prediction sets")

    aux: dict = {}
    if method == "normalized":
        if scale is None:
            raise CalibrationError(
                "method='normalized' needs a fitted `scale` estimator")
        if lower is not None or upper is not None:
            raise CalibrationError(
                "method='normalized' takes `scale`, not quantile estimators")
        if not callable(getattr(scale, "predict", None)):
            raise CalibrationError("scale estimator must expose predict()")
        aux["scale"] = scale
    elif method == "cqr":
        if lower is None or upper is None:
            raise CalibrationError(
                "method='cqr' needs fitted `lower` and `upper` quantile "
                "estimators")
        if scale is not None:
            raise CalibrationError(
                "method='cqr' takes `lower`/`upper`, not a scale estimator")
        for name, est in (("lower", lower), ("upper", upper)):
            if not callable(getattr(est, "predict", None)):
                raise CalibrationError(
                    f"{name} quantile estimator must expose predict()")
        aux["lower"], aux["upper"] = lower, upper
    elif scale is not None or lower is not None or upper is not None:
        raise CalibrationError(
            "method='split' takes no auxiliary estimators; declare "
            "method='normalized' or method='cqr'")
    aux_digest = _digest({k: _estimator_digest(v)
                          for k, v in sorted(aux.items())}) if aux else None

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
            if not math.isfinite(s):
                raise CalibrationError(
                    f"row {i}: non-finite nonconformity score")
            buckets.setdefault(key, []).append(s)
        elif one_sided:
            try:
                lo_s, hi_s = _reg_scores(method, aux, X_all[i],
                                         _scalar(pred[i]), y[i])
            except CalibrationError as ex:
                raise CalibrationError(f"row {i}: {ex}")
            if not (math.isfinite(lo_s) and math.isfinite(hi_s)):
                raise CalibrationError(
                    f"row {i}: non-finite nonconformity score")
            bucket = buckets.setdefault(key, {"lo": [], "hi": []})
            bucket["lo"].append(lo_s)
            bucket["hi"].append(hi_s)
        else:
            try:
                s = _reg_score(method, aux, X_all[i],
                               _scalar(pred[i]), y[i])
            except CalibrationError as ex:
                raise CalibrationError(f"row {i}: {ex}")
            if not math.isfinite(s):
                raise CalibrationError(
                    f"row {i}: non-finite nonconformity score")
            buckets.setdefault(key, []).append(s)

    per_group = {}
    if one_sided:
        floor_lo = math.ceil(1.0 / a_lo) - 1
        floor_hi = math.ceil(1.0 / a_hi) - 1
        for key, bucket in buckets.items():
            lo_scores = sorted(bucket["lo"])
            hi_scores = sorted(bucket["hi"])
            n = len(lo_scores)
            floor = max(floor_lo, floor_hi)
            if n < floor:
                raise CalibrationError(
                    f"group {key!r}: n={n} is below the conformal floor "
                    f"{floor} for alpha_lower={a_lo}, alpha_upper={a_hi}; "
                    f"regroup or collect more calibration data")
            per_group[key] = {
                "q_lo": _quantile(lo_scores, n, a_lo),
                "q_hi": _quantile(hi_scores, n, a_hi),
                "n": n, "scores_lo": lo_scores, "scores_hi": hi_scores}
    else:
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
               per_group=per_group, n_total=len(rows), method=method,
               aux=aux, aux_digest=aux_digest, one_sided=one_sided,
               alpha_lower=a_lo, alpha_upper=a_hi, **kwargs)


__all__ = ["Kappa", "AdaptiveKappa", "fit_kappa", "CalibrationError",
           "REGRESSOR_METHODS"]
