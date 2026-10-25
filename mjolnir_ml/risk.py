"""mjolnir_ml.risk — risk-controlled decisions (the second guarantee).

`calibration.py` answers a COVERAGE question: where will Y lie for this row?
This module answers a DECISION question: if I act on this cohort, is the risk
of that action bounded? The distinction is the whole point:

    coverage   P(Y in C(x)) >= 1 - alpha        per row
    risk       E[loss | action] <= alpha        over the selected cohort

The primitive is the probability of a declared adverse event:

    p_event(x) = P(event | x)

For a regressor, the event is a threshold crossing (`Y > tau`); for a
classifier it is any declared error/danger event (e.g. `y != y_hat`, or
`y == malignant`). The caller supplies a fitted probabilistic estimator for
that event; this module turns it into two calibrated action thresholds by
Conformal Risk Control (CRC, Angelopoulos et al.):

    release_ok(p)  <=>  p <= lambda_release     loss = 1[event]
    deny_ok(p)     <=>  p >= lambda_deny        loss = 1[not event]

Each lambda is the largest/smallest threshold whose corrected empirical risk
on the hold-out rows satisfies `n/(n+1) * risk + 1/(n+1) <= alpha`. The stated
guarantee is on the EXPECTED risk of the selected cohort under exchangeability;
it is not a per-row certainty claim, and the report distinguishes the two.

A threshold of None means the calibration data cannot certify that action at
the requested budget: the action is never taken (fail closed), and the report
shows why. That is the honest answer, not a tuning failure.
"""
from __future__ import annotations

import math

from mjolnir_ml._digest import digest as _digest
from mjolnir_ml._digest import estimator_digest as _estimator_digest


class RiskError(ValueError):
    """Risk calibration refused (malformed data, no probability, too little)."""


def adverse_event(y, threshold: float) -> list:
    """`1[Y > threshold]` — the event a regressor threshold action controls."""
    out = []
    for v in y:
        v = float(v)
        if not math.isfinite(v):
            raise RiskError(f"non-finite target {v!r}")
        out.append(1 if v > float(threshold) else 0)
    return out


def crc_threshold(scores, losses, alpha: float = 0.10,
                  side: str = "low") -> float | None:
    """Conformal Risk Control threshold for a monotone selection rule.

    side="low"  : select {score <= lambda}, loss is the adverse event.
    side="high" : select {score >= lambda}, loss is the adverse event.

    Returns the largest (low) / smallest (high) observed score whose corrected
    empirical risk is <= alpha, or None when no such threshold exists. Ties
    are handled by evaluating whole score groups, so the selected set is
    exactly the set the rule will select at runtime.
    """
    if side not in ("low", "high"):
        raise RiskError(f"side must be 'low' or 'high', got {side!r}")
    if not (0.0 < float(alpha) < 1.0):
        raise RiskError(f"alpha must be in (0,1), got {alpha!r}")
    scores = [float(s) for s in scores]
    losses = [int(v) for v in losses]
    if len(scores) != len(losses) or not scores:
        raise RiskError("scores and losses must be equal-length and non-empty")
    if any(not math.isfinite(s) for s in scores):
        raise RiskError("non-finite score")
    if any(v not in (0, 1) for v in losses):
        raise RiskError("losses must be 0/1")

    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i],
                   reverse=(side == "high"))
    best = None
    prefix_loss = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            prefix_loss += losses[order[k]]
        selected = j + 1
        risk = prefix_loss / selected
        if (n / (n + 1.0)) * risk + 1.0 / (n + 1.0) <= alpha:
            best = scores[order[i]]
        i = j + 1
    return best


def crc_thresholds(scores, events, alpha_release: float = 0.10,
                   alpha_deny: float = 0.10):
    """Jointly choose an ordered pair of action thresholds.

    The deployed rule is disjoint by construction:

        release  {p <= lambda_release}
        deny     {p >  lambda_deny}         with lambda_release <= lambda_deny
        ask      the open middle

    Each threshold is chosen on the same CRC grid; the selected pair must
    satisfy both corrected empirical risk bounds exactly, and the pair with
    the largest acted cohort wins (ties prefer a wider ask gap). When one
    action cannot be certified, that threshold is None and the other action
    may still be used.

    Per-action budgets are marginal guarantees. A joint guarantee over both
    actions needs `alpha_release + alpha_deny <= alpha_total` (union bound);
    the report states the per-action budgets separately.
    """
    scores = [float(s) for s in scores]
    events = [int(e) for e in events]
    n = len(scores)
    if n != len(events) or not scores:
        raise RiskError("scores and events must be equal-length and non-empty")
    if any(not math.isfinite(s) for s in scores):
        raise RiskError("non-finite score")
    if any(e not in (0, 1) for e in events):
        raise RiskError("events must be 0/1")
    for name, a in (("alpha_release", alpha_release),
                    ("alpha_deny", alpha_deny)):
        if not (0.0 < float(a) < 1.0):
            raise RiskError(f"{name} must be in (0,1), got {a!r}")

    uniq = sorted(set(scores))
    order = sorted(range(n), key=lambda i: scores[i])
    prefix = []
    idx = 0
    acc = 0
    for c in uniq:
        while idx < n and scores[order[idx]] <= c:
            acc += events[order[idx]]
            idx += 1
        cnt = idx
        risk = acc / cnt
        prefix.append((c, cnt,
                       (n / (n + 1.0)) * risk + 1.0 / (n + 1.0)))

    suffix = []
    idx = n - 1
    acc = 0
    for c in reversed(uniq):
        while idx >= 0 and scores[order[idx]] > c:
            acc += 1 - events[order[idx]]
            idx -= 1
        cnt = n - 1 - idx
        risk = acc / cnt if cnt else 1.0
        suffix.append((c, cnt,
                       (n / (n + 1.0)) * risk + 1.0 / (n + 1.0)))
    suffix.reverse()

    best = None
    best_i = None
    for j in range(len(uniq)):
        if prefix[j][2] <= alpha_release:
            if best_i is None or prefix[j][1] > prefix[best_i][1]:
                best_i = j
        if suffix[j][2] <= alpha_deny and best_i is not None:
            total = prefix[best_i][1] + suffix[j][1]
            gap = j - best_i
            if (best is None or total > best[2]
                    or (total == best[2] and gap > best[3])):
                best = (best_i, j, total, gap)

    if best is not None:
        return uniq[best[0]], uniq[best[1]]

    # One action may be certifiable while the other is not.
    rel = [i for i in range(len(uniq)) if prefix[i][2] <= alpha_release]
    den = [j for j in range(len(uniq)) if suffix[j][2] <= alpha_deny]
    if rel and den:
        i = max(rel, key=lambda k: (prefix[k][1], -k))
        j = max(den, key=lambda k: (suffix[k][1], k))
        if prefix[i][1] >= suffix[j][1]:
            return uniq[i], None
        return None, uniq[j]
    if rel:
        i = max(rel, key=lambda k: (prefix[k][1], -k))
        return uniq[i], None
    if den:
        j = max(den, key=lambda k: (suffix[k][1], k))
        return None, uniq[j]
    return None, None


class RiskIndex:
    """CRC-calibrated action thresholds over a crossing probability.

    `per_group` maps a region key to its thresholds and the calibration rows
    they came from. A region with too little data is refused at fit time
    ("unknown is not zero"); an uncalibrated region at decision time yields no
    risk facts, so a policy requiring them abstains.
    """

    def __init__(self, estimator, event: str, alpha_release: float,
                 alpha_deny: float, per_group: dict, n_total: int,
                 features: tuple, group: str | None):
        self.estimator = estimator
        self.event = event
        self.alpha_release = float(alpha_release)
        self.alpha_deny = float(alpha_deny)
        self.per_group = per_group
        self.n_total = n_total
        self.features = tuple(features)
        self.group = group
        self._digest = _digest({
            "kind": "risk", "event": event,
            "alpha_release": self.alpha_release,
            "alpha_deny": self.alpha_deny,
            "group": group, "features": list(self.features),
            "estimator": _estimator_digest(estimator),
            "groups": per_group,
        })

    @property
    def digest(self) -> str:
        return self._digest

    def group_key(self, row) -> str:
        return str(row.get(self.group, "__unstated__")) if self.group \
            else "__all__"

    def calibrated(self, key: str) -> bool:
        return key in self.per_group

    def lambda_release(self, key: str) -> float | None:
        g = self.per_group.get(key)
        return None if g is None else g["lambda_release"]

    def lambda_deny(self, key: str) -> float | None:
        g = self.per_group.get(key)
        return None if g is None else g["lambda_deny"]

    def p_event(self, x) -> float:
        """Crossing probability for an input row in `features` order."""
        proba = self.estimator.predict_proba([list(x)])[0]
        p = float(proba[-1])
        if not math.isfinite(p):
            raise RiskError(f"crossing model emitted non-finite p_event {p!r}")
        return p

    def release_ok(self, p: float, key: str) -> bool:
        lam = self.lambda_release(key)
        return lam is not None and float(p) <= lam

    def deny_ok(self, p: float, key: str) -> bool:
        lam = self.lambda_deny(key)
        return lam is not None and float(p) > lam

    def empirical_risk(self, key: str, side: str) -> float | None:
        """Realized risk of the calibrated action over its calibration rows."""
        g = self.per_group.get(key)
        if g is None:
            return None
        lam = g["lambda_release"] if side == "release" else g["lambda_deny"]
        if lam is None:
            return None
        rows = [(s, e) for s, e in zip(g["scores"], g["events"])
                if (s <= lam if side == "release" else s > lam)]
        if not rows:
            return None
        loss = (lambda e: 1 - e) if side == "deny" else (lambda e: e)
        return sum(loss(e) for _, e in rows) / len(rows)


def fit_risk_index(estimator, rows, events, *, alpha_release: float = 0.10,
                   alpha_deny: float = 0.10, features=None,
                   group: str | None = None,
                   event: str = "adverse") -> RiskIndex:
    """Fit CRC thresholds on held-out rows.

    `events` are 0/1 indicators of the adverse event per row (use
    `adverse_event` for a regressor threshold). `estimator` must be fitted and
    expose `predict_proba`; the positive column is taken as p_event(x). The
    estimator must not have been fitted on these rows.
    """
    rows = list(rows)
    events = [int(e) for e in events]
    if len(rows) != len(events) or not rows:
        raise RiskError("need equal non-empty rows/events")
    if any(e not in (0, 1) for e in events):
        raise RiskError("events must be 0/1")
    for name, a in (("alpha_release", alpha_release),
                    ("alpha_deny", alpha_deny)):
        if not (0.0 < float(a) < 1.0):
            raise RiskError(f"{name} must be in (0,1), got {a!r}")
    if not callable(getattr(estimator, "predict_proba", None)):
        raise RiskError("crossing estimator must expose predict_proba()")
    if group is not None and not isinstance(group, str):
        raise RiskError("group must be a feature name or None")
    if group is not None and any(group not in r for r in rows):
        raise RiskError(f"group feature {group!r} absent from some rows")

    feats = tuple(features) if features else tuple(rows[0].keys())
    X = [[r[f] for f in feats] for r in rows]
    try:
        proba = estimator.predict_proba(X)
    except Exception as ex:                                   # noqa: BLE001
        raise RiskError(
            f"crossing estimator failed: {type(ex).__name__}: {ex}")
    scores = []
    for i, p in enumerate(proba):
        v = float(p[-1])
        if not math.isfinite(v):
            raise RiskError(f"row {i}: non-finite p_event {v!r}")
        scores.append(v)

    buckets: dict = {}
    for i, row in enumerate(rows):
        key = str(row[group]) if group else "__all__"
        buckets.setdefault(key, {"scores": [], "events": []})
        buckets[key]["scores"].append(scores[i])
        buckets[key]["events"].append(events[i])

    floor = max(math.ceil(1.0 / float(alpha_release)) - 1,
                math.ceil(1.0 / float(alpha_deny)) - 1)
    per_group = {}
    for key, b in buckets.items():
        n = len(b["scores"])
        if n < floor:
            raise RiskError(
                f"group {key!r}: n={n} is below the CRC floor {floor} for "
                f"alpha_release={alpha_release}, alpha_deny={alpha_deny}; "
                f"regroup or collect more calibration data")
        lam_r, lam_d = crc_thresholds(b["scores"], b["events"],
                                      alpha_release, alpha_deny)
        per_group[key] = {
            "lambda_release": lam_r, "lambda_deny": lam_d, "n": n,
            "scores": b["scores"], "events": b["events"]}

    return RiskIndex(estimator, event, alpha_release, alpha_deny, per_group,
                     len(rows), feats, group)


__all__ = ["RiskError", "RiskIndex", "adverse_event", "crc_threshold",
           "crc_thresholds", "fit_risk_index"]
