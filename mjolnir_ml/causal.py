"""mjolnir_ml.causal — the causal layer: identification strategies as
discoverable capabilities, with do-calculus vocabulary.

The discovery loop (`discovery.py`) admits capabilities when an independent
oracle moves a four-number census. This module gives that loop *causal*
estmands and *identification strategies*, and keeps the same laws:

    claim          a causal estimand with a declared acceptance region
                   (`ATE[Y | do(x=1), do(x=0)] ∈ [0.9, 1.1]`, `a ⟂ b | y`,
                   `ATT[Y | group, time >= 0] ∈ ...`)
    capability     an identification STRATEGY (`backdoor({z})`, `iv(g)`,
                   `did(...)`, `rdd(...)`, `ci(alpha, mde)`, `its(...)`).
                   A strategy is proposed, audited and admitted like a prover
                   rung; it never grades its own claims.
    oracle         the strategy applied to data. It answers CERTIFIED /
                   REFUTED / DECLINED / DECLARED with an interval estimate and
                   the declared assumptions attached.
    twin           a PLACEBO / negative control (placebo treatment, placebo
                   outcome, placebo date, conditional-independence collider).
                   The twin asserts the null, so the gate expects CERTIFIED —
                   an effect where there should be none refuses the candidate.
    frontier       the decline census by identification failure
                   (`unidentified:*`, `positivity_violation`,
                   `assumption_violated:*`, `weak_instrument`, `underpowered`).

WHAT `CERTIFIED` MEANS HERE — and it is deliberately narrower than in the
prover. It means: *under the declared assumptions, the identified interval lies
inside the claim's acceptance region*. It does NOT mean "the effect is true".
The untestable parts (no unmeasured confounding for the declared adjustment
set, exclusion restriction, parallel trends, continuity) ship with the claim as
`assumptions`; the testable parts (positivity, instrument relevance, pre-trends,
covariate continuity, power) gate the verdict. A claim with no declared
assumptions is DECLINED, never certified.

WHY THE INTERVAL, NOT A SCORE. Same reason the prover has four numbers and no
scalar: an estimate without its interval is a point dressed as a fact.
`_verdict` maps the interval onto the same four words the rest of the package
uses — inside the region CERTIFIED, entirely outside REFUTED, straddling it
DECLINED (the ambiguous band).

THE TWO TIERS.
  * Tier 2 (here): cross-sectional identification — backdoor adjustment
    (stratified standardisation or regression adjustment), instrumental
    variables (Wald / 2SLS, relevance tested, exclusion declared), difference-
    in-differences (parallel trends tested on the pre-period), sharp RDD (local
    linear, positivity and covariate continuity checked), and conditional-
    independence tests (partial correlation, declared MDE — bounded
    independence, not absolute).
  * Tier 1 (`event_study`): interrupted time series over the live journal —
    treatment timestamps and outcomes are already recorded, so the flywheel's
    own history becomes an event study. Pre-trend and placebo date are gates;
    the placebo *claim* is the twin.

MARKOV EQUIVALENCE. `CausalSession.markov_class()` reports what the certified
independences pin down (skeleton, colliders) and what stays unidentified
(orientation of non-collider edges). Observational data identify an equivalence
class, never directions; reporting a DAG with confidence scores would be the
scalar mistake in a different costume.

STANDALONE: pure stdlib. No numpy, no sklearn, no scipy.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Iterable

from mjolnir_ml.discovery import (CERTIFIED, DECLINED, REFUTED, Candidate,
                                  DiscoveryError, DiscoverySession)

STRATEGIES = ("none", "backdoor", "iv", "did", "rdd", "ci", "its",
              "paired")
CAUSAL_KINDS = ("ate", "late", "att", "rdd", "intervention_effect")
CLAIM_KINDS = ("association",) + CAUSAL_KINDS + ("independence",)

_NORMAL = statistics.NormalDist()


class CausalError(DiscoveryError):
    """The causal request is malformed (not a failed identification)."""


class _NotIdentified(Exception):
    """Internal: the strategy cannot identify the estimand. Carries a reason
    code that lands in the frontier."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ----------------------------------------------------------------- primitives --
@dataclass(frozen=True)
class Intervention:
    """`do(x=1)` — the intervention operator, as a value."""
    variable: str
    value: object

    def __str__(self) -> str:
        return f"do({self.variable}={self.value})"


def do(variable: str, value=1) -> Intervention:
    """The do-operator: `do('x', 1)` renders as `do(x=1)` in every claim id."""
    if not isinstance(variable, str) or not variable:
        raise CausalError("do() takes a variable name")
    return Intervention(variable, value)


def _fmt_region(criterion: dict) -> str:
    lo = criterion.get("min")
    hi = criterion.get("max")
    if lo == float("-inf") and hi == float("inf"):
        return "∈ ℝ (vacuous)"
    if lo is not None and hi is not None:
        return f"∈ [{lo:g}, {hi:g}]"
    if lo is not None:
        return f"≥ {lo:g}"
    return f"≤ {hi:g}"


# -------------------------------------------------------------------- claims --
@dataclass(frozen=True)
class CausalClaim:
    """A causal (or associational) estimand with an acceptance region.

    kind:
      association        E[Y | X=x1] - E[Y | X=x0]   (no identification claim)
      ate                ATE[Y | do(X=x1), do(X=x0)]
      late               LATE[Y | do(X), instrument] (complier effect)
      att                ATT[Y | group, time >= post_from]
      rdd                LATE[Y | running, cutoff]   (sharp design)
      independence       left ⟂ right | conditions
      intervention_effect  effect of the event at `event_index` on `outcome`
    """
    kind: str
    outcome: str = ""
    treatment: str = ""
    arms: tuple = ()                  # (Intervention, Intervention)
    instrument: str = ""
    group: str = ""
    time: str = ""
    post_from: float = 0.0
    running: str = ""
    cutoff: float | None = None
    left: str = ""
    right: str = ""
    conditions: tuple = ()
    event_index: int | None = None
    criterion: dict = field(default_factory=dict)
    assumptions: tuple = ()
    label: str = ""

    def __post_init__(self):
        if self.kind not in CLAIM_KINDS:
            raise CausalError(f"unknown claim kind {self.kind!r} "
                              f"(one of {list(CLAIM_KINDS)})")
        if not self.criterion or not ({"min", "max"} & set(self.criterion)):
            raise CausalError("a claim declares its acceptance region as "
                              "criterion={'min': ..., 'max': ...}")
        lo, hi = self.criterion.get("min"), self.criterion.get("max")
        if lo is not None and hi is not None and lo > hi:
            raise CausalError("criterion min is greater than max")
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "arms", tuple(self.arms))
        if self.kind != "association" and not self.assumptions:
            raise CausalError(
                f"a {self.kind} claim must DECLARE its assumptions (the "
                f"untestable part is data, and an undeclared one is an "
                f"opinion)")
        if self.kind in ("association", "ate", "late") and len(self.arms) != 2:
            raise CausalError(f"{self.kind} takes two arms: (x1, x0)")
        if self.kind == "late" and not self.instrument:
            raise CausalError("a late claim names its instrument")
        if self.kind == "att" and not (self.group and self.time):
            raise CausalError("an att claim names its group and time columns")
        if self.kind == "rdd" and not (self.running and self.cutoff is not None):
            raise CausalError("an rdd claim names its running variable and cutoff")
        if self.kind == "independence" and not (self.left and self.right):
            raise CausalError("an independence claim names its two variables")
        if self.kind == "intervention_effect" and self.event_index is None:
            raise CausalError("an intervention_effect claim names its event_index")

    @property
    def region(self) -> str:
        return _fmt_region(self.criterion)

    @property
    def id(self) -> str:
        """Canonical, do()-jargon identity. Two claims with the same text are
        the same obligation."""
        k = self.kind
        if k == "association":
            a1, a0 = self.arms
            base = (f"E[{self.outcome}|{a1.variable}={a1.value}] − "
                    f"E[{self.outcome}|{a0.variable}={a0.value}]")
        elif k == "ate":
            a1, a0 = self.arms
            base = f"ATE[{self.outcome}|{a1}, {a0}]"
        elif k == "late":
            a1, a0 = self.arms
            base = f"LATE[{self.outcome}|{a1}, {a0}; {self.instrument}]"
        elif k == "att":
            base = (f"ATT[{self.outcome}|{self.group}, "
                    f"{self.time} ≥ {self.post_from:g}]")
        elif k == "rdd":
            base = f"LATE[{self.outcome}|{self.running}, cutoff {self.cutoff:g}]"
        elif k == "independence":
            cond = (f" | {', '.join(self.conditions)}"
                    if self.conditions else "")
            base = f"{self.left} ⟂ {self.right}{cond}"
        else:
            base = f"IE[{self.outcome} | event {self.event_index}]"
        return f"{base} {self.region}"

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "outcome": self.outcome,
                "treatment": self.treatment,
                "arms": [str(a) for a in self.arms],
                "instrument": self.instrument, "group": self.group,
                "time": self.time, "post_from": self.post_from,
                "running": self.running, "cutoff": self.cutoff,
                "left": self.left, "right": self.right,
                "conditions": list(self.conditions),
                "event_index": self.event_index,
                "criterion": dict(self.criterion),
                "assumptions": list(self.assumptions),
                "label": self.label}


@dataclass(frozen=True)
class Strategy:
    """An identification strategy — the capability the loop proposes.

    `none`     the naive/prior pass: association only; causal claims stay
               unidentified. It is how the registry record is produced.
    `backdoor` standardisation (discrete adjustment set) or regression
               adjustment (continuous). Positivity is checked per cell.
    `iv`       Wald/2SLS. Relevance is TESTED (first-stage t); exclusion is
               DECLARED.
    `did`      two-by-two difference-in-differences. Parallel trends is TESTED
               on the pre-period; no-anticipation is declared.
    `rdd`      sharp local linear regression. Positivity, bandwidth and
               covariate continuity are checked; continuity of potential
               outcomes is declared.
    `paired`   within-unit (matched) contrast: each unit appears under both
               arms, so the estimate is the mean paired difference. This is the
               right design for a mechanistic intervention on a model, where the
               same row is scored under both do() assignments. Full compliance
               is checked (every pair carries both arms).
    `ci`       conditional-independence test (partial correlation). `mde` turns
               "independent" into bounded independence.
    `its`      interrupted time series over a journal series: pre-trend and a
               placebo date gate the estimate.
    """
    kind: str
    adjust: tuple = ()
    instrument: str = ""
    alpha: float = 0.05
    mde: float = 0.10
    pre: int = 6
    post: int = 6
    placebo_offset: int = 6
    bandwidth: float = 0.5
    pair: str = ""
    covariates: tuple = ()
    bins: int = 4
    min_cell: int = 20
    min_stage_t: float = 3.16
    seed: int = 0

    def __post_init__(self):
        if self.kind not in STRATEGIES:
            raise CausalError(f"unknown strategy {self.kind!r} "
                              f"(one of {list(STRATEGIES)})")
        object.__setattr__(self, "adjust", tuple(self.adjust))
        object.__setattr__(self, "covariates", tuple(self.covariates))
        if not (0 < self.alpha < 1):
            raise CausalError("alpha must be in (0, 1)")
        if self.kind == "iv" and not self.instrument:
            raise CausalError("an iv strategy names its instrument")
        if self.kind == "paired" and not self.pair:
            raise CausalError("a paired strategy names its pair column")
        # backdoor(∅) is deliberately allowed: the empty adjustment set is a
        # real (and dangerous) strategy. It is not refused, it is audited —
        # which is how it gets refuted.

    @property
    def id(self) -> str:
        if self.kind == "none":
            return "none"
        if self.kind == "backdoor":
            inner = ", ".join(self.adjust) if self.adjust else "∅"
            return f"backdoor({{{inner}}})"
        if self.kind == "iv":
            return f"iv({self.instrument})"
        if self.kind == "did":
            return "did(parallel-trends)"
        if self.kind == "rdd":
            return f"rdd(bandwidth={self.bandwidth:g})"
        if self.kind == "paired":
            return f"paired({self.pair})"
        if self.kind == "ci":
            return f"ci(alpha={self.alpha:g},mde={self.mde:g})"
        return (f"its(pre={self.pre},post={self.post},"
                f"placebo_offset={self.placebo_offset})")

    def as_dict(self) -> dict:
        return {"kind": self.kind, "id": self.id, "adjust": list(self.adjust),
                "instrument": self.instrument, "alpha": self.alpha,
                "mde": self.mde, "pre": self.pre, "post": self.post,
                "placebo_offset": self.placebo_offset,
                "bandwidth": self.bandwidth, "pair": self.pair,
                "covariates": list(self.covariates),
                "bins": self.bins, "min_cell": self.min_cell,
                "min_stage_t": self.min_stage_t, "seed": self.seed}


# ---------------------------------------------------------------------- data --
class Table:
    """Columnar access over a list of row dicts (the only data shape here)."""

    def __init__(self, rows: Iterable[dict]):
        rows = [dict(r) for r in rows]
        if not rows:
            raise CausalError("a table needs at least one row")
        cols = set(rows[0])
        for i, r in enumerate(rows):
            if set(r) != cols:
                raise CausalError(f"row {i} does not share the table columns")
        self.rows = rows
        self.columns = tuple(rows[0].keys())

    def __len__(self) -> int:
        return len(self.rows)

    def __contains__(self, col) -> bool:
        return col in self.columns

    def column(self, col: str) -> list:
        if col not in self.columns:
            raise CausalError(f"unknown column {col!r} "
                              f"(table has {list(self.columns)})")
        return [r[col] for r in self.rows]

    def numeric(self, col: str) -> list[float]:
        try:
            return [float(v) for v in self.column(col)]
        except (TypeError, ValueError) as ex:
            raise CausalError(f"column {col!r} is not numeric") from ex

    def discrete(self, col: str, max_levels: int = 8) -> bool:
        return len(set(self.column(col))) <= max_levels


@dataclass(frozen=True)
class Series:
    """A time-ordered outcome series (tier 1). `points` are (index, value)."""
    metric: str
    points: tuple

    def __post_init__(self):
        pts = tuple((int(i), float(v)) for i, v in self.points)
        if len(pts) < 4:
            raise CausalError("a series needs at least four points")
        if any(pts[i][0] >= pts[i + 1][0] for i in range(len(pts) - 1)):
            raise CausalError("series indices must be strictly increasing")
        object.__setattr__(self, "points", pts)

    def window(self, start: int, stop: int) -> list[float]:
        return [v for i, v in self.points if start <= i < stop]

    def at(self, index: int) -> float:
        for i, v in self.points:
            if i == index:
                return v
        raise CausalError(f"series has no point at index {index}")


def series_from_journal(journal, *, metric: str = "hazard_release",
                        per: str = "decision") -> Series:
    """Extract a tier-1 series from a real `live.Journal` (same payload shapes
    `LiveSession` writes): hazard releases or absolute error.

    `per="decision"` yields one point per decision; `per="batch"` averages
    within the recorded batch, which is the right unit for an interrupted time
    series (a batch is one draw from the process, a decision is one row of it).
    """
    from mjolnir_ml.vendors import wire

    decisions = journal.payloads("decision")
    outcomes = {p["certificate_id"]: p["y"] for p in journal.payloads("outcome")}
    vals = []
    for i, p in enumerate(decisions):
        d = p["disposition"]
        if metric == "hazard_release":
            val = 1.0 if (d["verdict"] == wire.RELEASE
                          and p.get("hazard_codes")) else 0.0
        elif metric == "abs_error":
            y = outcomes.get(d["certificate_id"])
            pred = (d.get("prediction") or {}).get("pred")
            val = (abs(float(y) - float(pred))
                   if y is not None and pred is not None else None)
        else:
            raise CausalError(f"unknown journal metric {metric!r} "
                              f"(hazard_release | abs_error)")
        if val is not None:
            vals.append((i, val, p.get("batch")))
    if not vals:
        raise CausalError(f"the journal has no {metric!r} points")
    if per == "decision":
        return Series(metric, tuple((i, v) for i, v, _ in vals))
    if per == "batch":
        groups: dict = {}
        for _, v, batch in vals:
            if batch is None:
                raise CausalError("per='batch' needs the decision batches")
            groups.setdefault(batch, []).append(v)
        pts = tuple((int(b), _mean(vs)) for b, vs in sorted(groups.items()))
        return Series(f"{metric}_rate" if metric == "hazard_release" else metric,
                      pts)
    raise CausalError("per must be 'decision' or 'batch'")


# ------------------------------------------------------------------- helpers --
def _mean(xs) -> float:
    return sum(xs) / len(xs)


def _var(xs) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _z(alpha: float) -> float:
    return _NORMAL.inv_cdf(1.0 - alpha / 2.0)


def _corr(xs, ys) -> float:
    mx, my = _mean(xs), _mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        raise _NotIdentified("degenerate_column",
                             "a variable has no variance")
    return sxy / math.sqrt(sxx * syy)


def _invert(matrix):
    n = len(matrix)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-10:
            raise _NotIdentified("design_singular",
                                 "the design matrix is not invertible")
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        m[col] = [v / pv for v in m[col]]
        for r in range(n):
            if r != col and abs(m[r][col]) > 0:
                f = m[r][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [row[n:] for row in m]


def _ols(feats, y, *, se: bool = True):
    """OLS with an intercept supplied by the caller as feats[i][0]. Returns
    (beta, se). Raises _NotIdentified on a singular design."""
    n = len(y)
    k = len(feats[0])
    if n <= k:
        raise _NotIdentified("positivity_violation", "fewer rows than parameters")
    xtx = [[sum(feats[i][a] * feats[i][b] for i in range(n)) for b in range(k)]
           for a in range(k)]
    xty = [sum(feats[i][a] * y[i] for i in range(n)) for a in range(k)]
    inv = _invert(xtx)
    beta = [sum(inv[a][b] * xty[b] for b in range(k)) for a in range(k)]
    if not se:
        return beta, [float("nan")] * k
    rss = sum((y[i] - sum(feats[i][a] * beta[a] for a in range(k))) ** 2
              for i in range(n))
    s2 = rss / max(n - k, 1)
    return beta, [math.sqrt(max(s2 * inv[j][j], 0.0)) for j in range(k)]


def _ols_cov(feats, y):
    """OLS with the full coefficient covariance: (beta, cov). Used by the
    segmented-regression event study, where the effect is a linear combination
    of coefficients and needs var(c'b) = c' (X'X)^-1 c * s^2."""
    n = len(y)
    k = len(feats[0])
    if n <= k:
        raise _NotIdentified("positivity_violation", "fewer rows than parameters")
    xtx = [[sum(feats[i][a] * feats[i][b] for i in range(n)) for b in range(k)]
           for a in range(k)]
    xty = [sum(feats[i][a] * y[i] for i in range(n)) for a in range(k)]
    inv = _invert(xtx)
    beta = [sum(inv[a][b] * xty[b] for b in range(k)) for a in range(k)]
    rss = sum((y[i] - sum(feats[i][a] * beta[a] for a in range(k))) ** 2
              for i in range(n))
    s2 = rss / max(n - k, 1)
    cov = [[s2 * inv[a][b] for b in range(k)] for a in range(k)]
    return beta, cov


def _residuals(values, controls):
    if not controls:
        return list(values)
    feats = [[1.0] + [float(c[i]) for c in controls]
             for i in range(len(values))]
    beta, _ = _ols(feats, [float(v) for v in values], se=False)
    return [float(values[i]) - sum(beta[a] * feats[i][a]
                                   for a in range(len(beta)))
            for i in range(len(values))]


def _boot(fn, n: int, *, seed: int, b: int = 400):
    rng = random.Random(seed)
    stats = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            stats.append(fn(idx))
        except (ZeroDivisionError, _NotIdentified):
            continue
    if len(stats) < max(20, b // 4):
        raise _NotIdentified("positivity_violation",
                             "the bootstrap could not resample the design")
    stats.sort()
    return (stats[int(0.025 * (len(stats) - 1))],
            stats[int(0.975 * (len(stats) - 1))])


def _verdict(lo, hi, criterion) -> tuple[str, str]:
    if lo is None or hi is None:
        return DECLINED, "no_interval"
    mn = criterion.get("min", float("-inf"))
    mx = criterion.get("max", float("inf"))
    if mn == float("-inf") and mx == float("inf"):
        return DECLINED, "vacuous_region"
    if lo >= mn and hi <= mx:
        return CERTIFIED, "certified"
    if hi < mn or lo > mx:
        return REFUTED, "refuted"
    return DECLINED, "ambiguous_ci"


# --------------------------------------------------------------- estimators --
class Estimate:
    __slots__ = ("value", "lo", "hi", "n", "method", "detail")

    def __init__(self, value, lo, hi, n, method, detail=None):
        self.value = value
        self.lo = lo
        self.hi = hi
        self.n = n
        self.method = method
        self.detail = detail or {}

    def as_dict(self) -> dict:
        return {"value": self.value, "lo": self.lo, "hi": self.hi,
                "n": self.n, "method": self.method, "detail": self.detail}

    def __repr__(self):
        if self.value is None:
            return f"Estimate(n={self.n}, {self.method})"
        if self.lo is None or self.hi is None:
            return f"Estimate({self.value:.4g}, n={self.n}, {self.method})"
        return (f"Estimate({self.value:.4g} ∈ [{self.lo:.4g}, {self.hi:.4g}], "
                f"n={self.n}, {self.method})")


def _assoc(table: Table, claim: CausalClaim) -> Estimate:
    a1, a0 = claim.arms
    x = claim.treatment or a1.variable
    if a1.variable != a0.variable:
        raise _NotIdentified("design_mismatch",
                             "the two arms intervene on different variables")
    ys = table.numeric(claim.outcome)
    xs = table.column(x)
    g1 = [v for v, t in zip(ys, xs) if t == a1.value]
    g0 = [v for v, t in zip(ys, xs) if t == a0.value]
    if len(g1) < 2 or len(g0) < 2:
        raise _NotIdentified("positivity_violation",
                             "an arm has fewer than two rows")
    d = _mean(g1) - _mean(g0)
    se = math.sqrt(_var(g1) / len(g1) + _var(g0) / len(g0))
    z = _z(0.05)
    return Estimate(d, d - z * se, d + z * se, len(ys), "difference-in-means")


def _backdoor(table: Table, claim: CausalClaim,
              s: Strategy) -> Estimate:
    a1, a0 = claim.arms
    x, y = claim.treatment, claim.outcome
    if not x:
        raise _NotIdentified("design_mismatch", "ate needs a treatment column")
    for col in s.adjust:
        if col not in table:
            raise _NotIdentified("design_missing_covariate",
                                 f"{col!r} is not in the table")
    ys = table.numeric(y)
    xs = table.column(x)
    outside = sorted({t for t in xs if t not in (a1.value, a0.value)})
    if outside:
        raise _NotIdentified(
            "design_mismatch",
            f"treatment {x!r} takes values {outside!r} outside the claimed "
            f"arms ({a1.value!r}, {a0.value!r}); the binary ATE is undefined")
    if all(table.discrete(col) for col in s.adjust):
        cells = {}
        for i, row in enumerate(table.rows):
            key = tuple(row[c] for c in s.adjust)
            cell = cells.setdefault(key, {"1": [], "0": []})
            if xs[i] == a1.value:
                cell["1"].append(ys[i])
            elif xs[i] == a0.value:
                cell["0"].append(ys[i])
        total = len(ys)
        est = 0.0
        var = 0.0
        used = 0
        for key, cell in cells.items():
            n1, n0 = len(cell["1"]), len(cell["0"])
            if n1 < s.min_cell or n0 < s.min_cell:
                raise _NotIdentified("positivity_violation",
                                     f"cell {key} has {n1}/{n0} rows, below "
                                     f"min_cell={s.min_cell}")
            w = (n1 + n0) / total
            est += w * (_mean(cell["1"]) - _mean(cell["0"]))
            var += (w ** 2) * (_var(cell["1"]) / n1 + _var(cell["0"]) / n0)
            used += n1 + n0
        se = math.sqrt(var)
        z = _z(s.alpha)
        return Estimate(est, est - z * se, est + z * se, used,
                        f"standardisation({s.id})",
                        {"cells": len(cells), "min_cell": s.min_cell})
    # regression adjustment: Y ~ 1 + X + adjust
    feats = [[1.0, 1.0 if xs[i] == a1.value else 0.0]
             + [float(table.rows[i][c]) for c in s.adjust]
             for i in range(len(ys))]
    beta, se = _ols(feats, ys)
    z = _z(s.alpha)
    return Estimate(beta[1], beta[1] - z * se[1], beta[1] + z * se[1],
                    len(ys), f"regression-adjustment({s.id})",
                    {"assumption": "linear adjustment"})


def _iv(table: Table, claim: CausalClaim, s: Strategy) -> Estimate:
    z, x, y = s.instrument, claim.treatment, claim.outcome
    for col in (z, x, y):
        if col not in table:
            raise _NotIdentified("design_missing_covariate",
                                 f"{col!r} is not in the table")
    xs = table.column(x)
    ys = table.numeric(y)
    zs = table.column(z)
    values = sorted(set(zs))
    if len(values) != 2:
        raise _NotIdentified("design_mismatch",
                             "the instrument must be binary here")
    z1, z0 = values[-1], values[0]
    x1 = [float(v) for v, g in zip(xs, zs) if g == z1]
    x0 = [float(v) for v, g in zip(xs, zs) if g == z0]
    if len(x1) < s.min_cell or len(x0) < s.min_cell:
        raise _NotIdentified("positivity_violation", "instrument cells too small")
    first = _mean(x1) - _mean(x0)
    se_first = math.sqrt(_var(x1) / len(x1) + _var(x0) / len(x0))
    t = first / se_first if se_first > 0 else 0.0
    if abs(t) < s.min_stage_t:
        raise _NotIdentified("weak_instrument",
                             f"first-stage t={t:.2f} < {s.min_stage_t}")
    if abs(first) < 1e-9:
        raise _NotIdentified("weak_instrument", "the instrument does not move X")

    def wald(idx):
        y1 = [ys[i] for i in idx if zs[i] == z1]
        y0 = [ys[i] for i in idx if zs[i] == z0]
        xx1 = [float(xs[i]) for i in idx if zs[i] == z1]
        xx0 = [float(xs[i]) for i in idx if zs[i] == z0]
        den = _mean(xx1) - _mean(xx0)
        return (_mean(y1) - _mean(y0)) / den

    est = (_mean([v for v, g in zip(ys, zs) if g == z1])
           - _mean([v for v, g in zip(ys, zs) if g == z0])) / first
    lo, hi = _boot(wald, len(ys), seed=s.seed)
    return Estimate(est, lo, hi, len(ys), f"wald({s.id})",
                    {"first_stage_t": round(t, 3),
                     "assumption": "exclusion restriction (declared)"})


def _did(table: Table, claim: CausalClaim, s: Strategy) -> Estimate:
    g, t, y = claim.group, claim.time, claim.outcome
    for col in (g, t, y):
        if col not in table:
            raise _NotIdentified("design_missing_covariate",
                                 f"{col!r} is not in the table")
    rows = table.rows
    gs = table.column(g)
    ts = table.numeric(t)
    ys = table.numeric(y)
    levels = sorted(set(gs))
    if len(levels) != 2:
        raise _NotIdentified("design_mismatch", "did needs two groups")

    def effect():
        c = {}
        for i in range(len(rows)):
            is_post = ts[i] >= claim.post_from
            c.setdefault((gs[i], is_post), []).append(ys[i])
        cells4 = [c.get((levels[0], False)), c.get((levels[0], True)),
                  c.get((levels[1], False)), c.get((levels[1], True))]
        if any(x is None or len(x) < 2 for x in cells4):
            raise _NotIdentified("positivity_violation",
                                 "a did cell has fewer than two rows")
        ctrl_p, ctrl_q = _mean(cells4[0]), _mean(cells4[1])
        trt_p, trt_q = _mean(cells4[2]), _mean(cells4[3])
        return (trt_q - trt_p) - (ctrl_q - ctrl_p), cells4

    # parallel trends: slope difference over the pre-period
    pre_rows = [i for i in range(len(rows)) if ts[i] < claim.post_from]
    slopes = []
    for lvl in levels:
        idx = [i for i in pre_rows if gs[i] == lvl]
        if len(idx) < 3 or len(set(ts[i] for i in idx)) < 2:
            raise _NotIdentified("positivity_violation",
                                 "not enough pre-periods to test parallel trends")
        beta, se = _ols([[1.0, ts[i]] for i in idx], [ys[i] for i in idx])
        slopes.append((beta[1], se[1]))
    diff = slopes[0][0] - slopes[1][0]
    se_diff = math.sqrt(slopes[0][1] ** 2 + slopes[1][1] ** 2)
    zc = _z(s.alpha)
    if se_diff > 0 and abs(diff) > zc * se_diff:
        raise _NotIdentified("assumption_violated:parallel_trends",
                             f"pre-trend slope difference {diff:.3f} "
                             f"(se {se_diff:.3f})")

    est, cells4 = effect()
    var = sum(_var(c) / len(c) for c in cells4)
    se = math.sqrt(var)
    z = _z(s.alpha)
    return Estimate(est, est - z * se, est + z * se, len(ys),
                    f"did-2x2({s.id})",
                    {"pre_trend_slope_diff": round(diff, 4)})


def _rdd(table: Table, claim: CausalClaim, s: Strategy) -> Estimate:
    r, y = claim.running, claim.outcome
    for col in (r, y):
        if col not in table:
            raise _NotIdentified("design_missing_covariate",
                                 f"{col!r} is not in the table")
    rs = table.numeric(r)
    ys = table.numeric(y)
    c = float(claim.cutoff)
    bw = s.bandwidth
    left = [i for i in range(len(rs)) if c - bw <= rs[i] < c]
    right = [i for i in range(len(rs)) if c <= rs[i] < c + bw]
    if len(left) < s.min_cell or len(right) < s.min_cell:
        raise _NotIdentified("positivity_violation",
                             f"only {len(left)}/{len(right)} rows within "
                             f"the bandwidth {bw:g}")
    fitted = []
    for side in (left, right):
        beta, se = _ols([[1.0, rs[i] - c] for i in side], [ys[i] for i in side])
        fitted.append((beta[0], se[0]))
    est = fitted[1][0] - fitted[0][0]
    se = math.sqrt(fitted[0][1] ** 2 + fitted[1][1] ** 2)
    z = _z(s.alpha)

    for cov in s.covariates:
        cv = table.numeric(cov)
        jumps = []
        for side in (left, right):
            beta, sev = _ols([[1.0, rs[i] - c] for i in side],
                             [cv[i] for i in side])
            jumps.append((beta[0], sev[0]))
        jump = jumps[1][0] - jumps[0][0]
        js = math.sqrt(jumps[0][1] ** 2 + jumps[1][1] ** 2)
        if js > 0 and abs(jump) > z * js:
            raise _NotIdentified(
                "assumption_violated:covariate_discontinuity",
                f"{cov!r} jumps by {jump:.3f} at the cutoff")

    return Estimate(est, est - z * se, est + z * se,
                    len(left) + len(right), f"local-linear({s.id})",
                    {"left": len(left), "right": len(right),
                     "assumption": "continuity of potential outcomes (declared)"})


def _paired(table: Table, claim: CausalClaim, s: Strategy) -> Estimate:
    pair_col, x, y = s.pair, claim.treatment, claim.outcome
    for col in (pair_col, x, y):
        if col not in table:
            raise _NotIdentified("design_missing_covariate",
                                 f"{col!r} is not in the table")
    a1, a0 = claim.arms
    pairs: dict = {}
    for row in table.rows:
        pairs.setdefault(row[pair_col], {})[row[x]] = float(row[y])
    diffs = []
    for key, arms in pairs.items():
        if a1.value not in arms or a0.value not in arms:
            raise _NotIdentified("positivity_violation",
                                 f"pair {key!r} does not carry both arms")
        diffs.append(arms[a1.value] - arms[a0.value])
    if len(diffs) < s.min_cell:
        raise _NotIdentified("positivity_violation",
                             f"{len(diffs)} pairs, below min_cell={s.min_cell}")
    est = _mean(diffs)
    se = math.sqrt(_var(diffs) / len(diffs))
    z = _z(s.alpha)
    return Estimate(est, est - z * se, est + z * se, len(diffs),
                    f"paired-difference({s.id})",
                    {"pairs": len(diffs), "sd_diff": round(math.sqrt(_var(diffs)), 4),
                     "assumption": "full compliance (every pair in both arms)"})


def _independence(table: Table, claim: CausalClaim, s: Strategy):
    xs = table.numeric(claim.left)
    ys = table.numeric(claim.right)
    controls = [table.numeric(c) for c in claim.conditions]
    n = len(xs)
    k = len(controls)
    if n - k - 3 < s.min_cell:
        raise _NotIdentified("positivity_violation",
                             "not enough rows for the conditioning set")
    rx = _residuals(xs, controls)
    ry = _residuals(ys, controls)
    r = _corr(rx, ry)
    zstat = math.atanh(max(min(r, 0.999999), -0.999999)) * math.sqrt(n - k - 3)
    p = 2.0 * (1.0 - _NORMAL.cdf(abs(zstat)))
    detectable = math.tanh((_z(s.alpha) + _NORMAL.inv_cdf(0.8))
                           / math.sqrt(n - k - 3))
    est = Estimate(r, None, None, n, f"partial-correlation({s.id})",
                   {"p_value": round(p, 5),
                    "detectable_r": round(detectable, 4),
                    "mde": s.mde, "conditions": list(claim.conditions)})
    if detectable > s.mde:
        return est, DECLINED, "underpowered"
    return est, (CERTIFIED if p > s.alpha else REFUTED), (
        "independent" if p > s.alpha else "dependent")


def _its(series: Series, claim: CausalClaim, s: Strategy) -> Estimate:
    """Segmented-regression interrupted time series.

        y = b0 + b1*t + b2*D + b3*t*D,    t = index - event, D = 1[t >= 0]

    `b1` is the pre-trend (gated); the effect over the post window is
    `b2 + b3 * mean(t_post)` with `var = c' cov c` — the standard ITS estimator,
    not a bootstrap over a handful of points.
    """
    if series is None:
        raise _NotIdentified("design_missing_series", "no journal series supplied")
    e = int(claim.event_index)
    present = {j for j, _ in series.points}
    idx = [i for i in range(e - s.pre, e + s.post) if i in present]
    if len([i for i in idx if i < e]) < max(3, s.pre // 2) \
            or len([i for i in idx if i >= e]) < max(3, s.post // 2):
        raise _NotIdentified("positivity:insufficient_window",
                             f"pre={len([i for i in idx if i < e])}, "
                             f"post={len([i for i in idx if i >= e])}")

    def fit(points):
        feats = [[1.0, float(i - e), 1.0 if i >= e else 0.0,
                  float(i - e) if i >= e else 0.0] for i in points]
        return _ols_cov(feats, [series.at(i) for i in points])

    beta, cov = fit(idx)
    z = _z(s.alpha)
    slope = beta[1]
    slope_se = math.sqrt(max(cov[1][1], 0.0))
    slope_lo, slope_hi = slope - z * slope_se, slope + z * slope_se
    if slope_lo > 0 or slope_hi < 0:
        raise _NotIdentified("assumption_violated:pre_trend",
                             f"pre-trend slope {slope:.3f} "
                             f"[{slope_lo:.3f}, {slope_hi:.3f}]")

    t_post = [float(i - e) for i in idx if i >= e]
    tbar = _mean(t_post)
    c = (0.0, 0.0, 1.0, tbar)
    effect = beta[2] + beta[3] * tbar
    var = sum(c[a] * c[b] * cov[a][b] for a in range(4) for b in range(4))
    se = math.sqrt(max(var, 0.0))

    # Placebo date: a fake event `placebo_offset` earlier, on pre-event points
    # only. It is REPORTED here and GATED by the twin battery: a placebo is a
    # claim ("no effect at a date with no intervention"), so it goes through
    # the same four-number machinery as any other claim rather than through a
    # hidden check inside the estimator.
    pe = e - s.placebo_offset
    pidx = [i for i in range(pe - s.pre, e) if i in present]
    placebo = None
    if len([i for i in pidx if i < pe]) >= 3 and len([i for i in pidx if i >= pe]) >= 3:
        def pfit(points):
            feats = [[1.0, float(i - pe), 1.0 if i >= pe else 0.0,
                      float(i - pe) if i >= pe else 0.0] for i in points]
            return _ols_cov(feats, [series.at(i) for i in points])
        pb, _ = pfit(pidx)
        ptbar = _mean([float(i - pe) for i in pidx if i >= pe])
        placebo = pb[2] + pb[3] * ptbar

    return Estimate(effect, effect - z * se, effect + z * se, len(idx),
                    f"segmented-its({s.id})",
                    {"pre_slope": round(slope, 4),
                     "pre_slope_ci": [round(slope_lo, 4), round(slope_hi, 4)],
                     "placebo_effect": None if placebo is None
                     else round(placebo, 4),
                     "assumption": "no simultaneous shock (declared)"})


# --------------------------------------------------------------------- oracle --
class StrategyOracle:
    """One strategy, applied to claims. This is the oracle the discovery loop
    evaluates with; it never produces its own claims."""

    def __init__(self, registry: dict, strategy: Strategy, *,
                 table: Table | None, series: Series | None):
        self.strategy = strategy
        self.discovery_id = f"causal:{strategy.id}"
        self._registry = registry
        self._table = table
        self._series = series
        self.estimates: dict = {}
        self.reasons: dict = {}

    def __call__(self, claim_id: str) -> str:
        verdict, reason = self._decide(claim_id)
        self.reasons[claim_id] = reason
        return verdict

    def estimate(self, claim_id: str):
        return self.estimates.get(claim_id)

    def _decide(self, claim_id: str) -> tuple[str, str]:
        claim = self._registry.get(claim_id)
        if claim is None:
            raise CausalError(f"unknown claim {claim_id!r} — the registry is "
                              f"the record; nothing is invented for it")
        s = self.strategy
        if s.kind == "none":
            if claim.kind != "association":
                return DECLINED, "unidentified:no-strategy"
            try:
                est = _assoc(self._table, claim)
            except _NotIdentified as ex:
                return DECLINED, ex.code
            self.estimates[claim_id] = est
            return _verdict(est.lo, est.hi, claim.criterion)

        if not claim.assumptions and claim.kind != "association":
            return DECLINED, "assumptions_undeclared"

        try:
            if claim.kind == "independence":
                if s.kind != "ci":
                    return DECLINED, "strategy_mismatch"
                est, verdict, reason = _independence(self._table, claim, s)
                self.estimates[claim_id] = est
                return verdict, reason
            if claim.kind == "intervention_effect":
                if s.kind != "its":
                    return DECLINED, "strategy_mismatch"
                est = _its(self._series, claim, s)
            elif claim.kind == "ate":
                if s.kind not in ("backdoor", "iv", "paired"):
                    return DECLINED, "strategy_mismatch"
                est = ({"backdoor": _backdoor, "iv": _iv,
                        "paired": _paired}[s.kind](self._table, claim, s))
            elif claim.kind == "late":
                if s.kind != "iv":
                    return DECLINED, "strategy_mismatch"
                est = _iv(self._table, claim, s)
            elif claim.kind == "att":
                if s.kind != "did":
                    return DECLINED, "strategy_mismatch"
                est = _did(self._table, claim, s)
            elif claim.kind == "rdd":
                if s.kind != "rdd":
                    return DECLINED, "strategy_mismatch"
                est = _rdd(self._table, claim, s)
            elif claim.kind == "association":
                est = _assoc(self._table, claim)
            else:
                return DECLINED, "strategy_mismatch"
        except _NotIdentified as ex:
            return DECLINED, ex.code
        self.estimates[claim_id] = est
        return _verdict(est.lo, est.hi, claim.criterion)


class CausalOracle:
    """The claim registry plus the data; hands out one oracle per strategy."""

    def __init__(self, claims: Iterable[CausalClaim], *, table=None,
                 series: Series | None = None):
        self.registry: dict[str, CausalClaim] = {}
        self.register(claims)
        self.table = table if isinstance(table, Table) or table is None \
            else Table(table)
        self.series = series

    def register(self, claims) -> "CausalOracle":
        if isinstance(claims, CausalClaim):
            claims = [claims]
        for c in claims:
            if not isinstance(c, CausalClaim):
                raise CausalError("register takes CausalClaim objects")
            if c.id in self.registry and self.registry[c.id] is not c:
                raise CausalError(f"claim id collision: {c.id!r}")
            self.registry[c.id] = c
        return self

    def strategy(self, strategy: Strategy) -> StrategyOracle:
        if not isinstance(strategy, Strategy):
            raise CausalError("strategy takes a Strategy")
        return StrategyOracle(self.registry, strategy, table=self.table,
                              series=self.series)

    def decide(self, claim_id: str, strategy: Strategy) -> str:
        return self.strategy(strategy)(claim_id)


# -------------------------------------------------------------------- session --
class CausalSession(DiscoverySession):
    """The discovery loop over causal claims.

    The corpus is the registry RECORD: every claim is decided by the prior
    capability (`Strategy("none")`), except claims whose record is supplied —
    a prior study's verdict computed with the strategy that produced it
    (`legacy`), or an explicit verdict (`record`). Causal claims therefore
    start in the frontier as `unidentified`, and a capability that moves them
    is exactly a rung closing a decline class.
    """

    def __init__(self, claims: Iterable[CausalClaim], *, table=None,
                 series: Series | None = None, legacy=None, record=None,
                 approval_verifier=None, journal=None, detectors=None):
        self._claims: dict[str, CausalClaim] = {}
        self._oracle = CausalOracle([], table=table, series=series)
        self._oracle.register(claims)
        self._claims = self._oracle.registry
        prior = self._oracle.strategy(Strategy("none"))
        corpus, reasons = {}, {}
        for cid in self._claims:
            if record and cid in record:
                corpus[cid] = record[cid]
                reasons[cid] = "prior_study"
                continue
            if legacy and cid in legacy:
                strat = legacy[cid]
                corpus[cid] = self._oracle.decide(cid, strat)
                reasons[cid] = f"prior_study:{strat.id}"
                continue
            corpus[cid] = prior(cid)
            reasons[cid] = prior.reasons.get(cid, "unidentified")
        super().__init__(corpus, reasons=reasons, journal=journal,
                         approval_verifier=approval_verifier,
                         detectors=detectors)

    # --------------------------------------------------------------- state --
    @property
    def oracle(self) -> CausalOracle:
        return self._oracle

    @property
    def claims(self) -> dict:
        return dict(self._claims)

    def claim(self, claim_id: str) -> CausalClaim:
        if claim_id not in self._claims:
            raise CausalError(f"unknown claim {claim_id!r}")
        return self._claims[claim_id]

    def strategy(self, strategy: Strategy) -> StrategyOracle:
        return self._oracle.strategy(strategy)

    def register(self, claims, *, corpus: bool = True) -> "CausalSession":
        """Register claims for lookup. `corpus=True` (the public default) also
        puts a new CLAIM under audit: it gets the prior strategy's verdict and
        decline reason, so it can be a candidate target. Candidate batteries
        (controls/tests/twins) pass `corpus=False`: they are looked up and
        decided, but they are evidence about the capability, not part of the
        operational census."""
        claims = list(claims)
        fresh = [c for c in claims if c.id not in self._claims]
        self._oracle.register(claims)
        self._claims = self._oracle.registry
        if corpus and fresh:
            prior = self._oracle.strategy(Strategy("none"))
            if self._reasons is None:
                self._reasons = {}
            for claim in fresh:
                if claim.id not in self._corpus:
                    self._corpus[claim.id] = prior(claim.id)
                    self._reasons[claim.id] = prior.reasons.get(
                        claim.id, "unidentified")
        return self

    def assumptions(self) -> list:
        """The declared tier in force: every assumption behind a claim the
        session currently holds as true."""
        out = set()
        for cid, claim in self._claims.items():
            if self.corpus.get(cid) == CERTIFIED:
                out.update(claim.assumptions)
        return sorted(out)

    def record(self) -> dict:
        """The registry record: every claim's current verdict and the reason
        that produced a decline (the identification frontier class)."""
        return {cid: {"verdict": self.corpus.get(cid),
                      "reason": self._reasons.get(cid)}
                for cid in self._claims}

    def identification_frontier(self) -> dict:
        """The decline census by identification failure — the causal analogue
        of the prover's construct frontier."""
        return dict(self.status()["frontier"])

    # ------------------------------------------------------------ candidates --
    def candidate(self, *, strategy: Strategy, targets, controls=(), twin=(),
                  tests=(), kind: str = "identification",
                  domain: str = "causal", population: int | None = None,
                  proposed_by: str = "causal.analyst") -> Candidate:
        """A capability = one identification strategy. `targets` are the claims
        it must move; `controls` are placebos (they assert the null, so the
        twin expects CERTIFIED); `tests` are further (claim, expected) pairs.

        Every claim must already be in the registry; placebos and tests may be
        registered here so the candidate declares its own battery.
        """
        self.register([c for c in list(controls) + [t[0] for t in tests]
                       if isinstance(c, CausalClaim) and c.id not in self._claims],
                      corpus=False)
        tids = tuple(t.id if isinstance(t, CausalClaim) else t for t in targets)
        if not tids:
            raise CausalError("a candidate needs at least one target claim")
        declined = [self._reasons.get(cid) for cid in tids
                    if self.corpus.get(cid) == DECLINED]
        reason = max(set(declined), key=declined.count) if declined else ""
        twins = tuple((c.id if isinstance(c, CausalClaim) else c, CERTIFIED)
                      for c in controls) + tuple(
            (c.id if isinstance(c, CausalClaim) else c, expected)
            for c, expected in twin)
        tst = tuple(((c.id if isinstance(c, CausalClaim) else c), expected)
                    for c, expected in tests)
        return Candidate(kind=kind, shape=strategy.id, claims=tids,
                         reason=reason, domain=domain, proposed_by=proposed_by,
                         population=population if population is not None
                         else len(tids),
                         tests=tst, twin=twins,
                         evidence={"strategy": strategy.as_dict()})

    # ------------------------------------------------------------ structure --
    def markov_class(self, strategy: Strategy | None = None) -> dict:
        """What the independences identify, and what they do not.

        With a `strategy`, every independence claim is re-decided by it (the
        report is an identification result, not a registry fact). Without one,
        the report reads the corpus record. Observational independences fix the
        skeleton and the colliders; the orientation of non-collider edges stays
        unidentified (an equivalence class, not a DAG). Directions need an
        intervention — `do()`."""
        oracle = self.strategy(strategy) if strategy is not None else None

        def verdict_of(cid):
            return oracle(cid) if oracle is not None else self.corpus.get(cid)

        separated, dependent, colliders = [], [], []
        for cid, claim in self._claims.items():
            if claim.kind != "independence":
                continue
            v = verdict_of(cid)
            if v == CERTIFIED:
                separated.append(claim.id)
            elif v == REFUTED:
                dependent.append(claim.id)
        for cid, claim in self._claims.items():
            if claim.kind != "independence" or verdict_of(cid) != REFUTED:
                continue
            if len(claim.conditions) != 1:
                continue
            base = next((c for c in self._claims.values()
                         if c.kind == "independence"
                         and {c.left, c.right} == {claim.left, claim.right}
                         and not c.conditions), None)
            if base is not None and verdict_of(base.id) == CERTIFIED:
                colliders.append({"at": claim.conditions[0],
                                  "left": claim.left, "right": claim.right})
        return {
            "schema": "mjolnir.ml.causal.markov-class/1",
            "separated": separated,
            "dependent": dependent,
            "colliders": colliders,
            "identified": (["skeleton: non-adjacency from certified "
                            "independences"] +
                           [f"collider at {c['at']}" for c in colliders]),
            "unidentified": ["orientation of non-collider edges",
                             "edges into colliders from outside the triple",
                             "anything not in the measured variables"],
            "note": "observational data identify an equivalence class, never "
                    "a DAG; use do() or a declared asymmetry for direction",
        }


def event_study(series: Series, event_index: int, *, pre: int = 6,
                post: int = 6, alpha: float = 0.05, seed: int = 0,
                placebo_offset: int = 6, placebo_bound: float = 0.10,
                criterion: dict | None = None) -> dict:
    """Interrupted time series at `event_index`: fit the pre-trend, extrapolate
    the counterfactual, and report the post-minus-counterfactual effect with a
    bootstrap interval. The placebo date and the pre-trend are the design's
    negative controls; a failed one is reported, never silently dropped."""
    claim = CausalClaim(
        kind="intervention_effect", outcome=series.metric,
        event_index=event_index,
        criterion=dict(criterion) if criterion is not None
        else {"min": float("-inf"), "max": float("inf")},
        assumptions=("no simultaneous shock",))
    strat = Strategy("its", pre=pre, post=post, alpha=alpha, seed=seed,
                     placebo_offset=placebo_offset)
    oracle = CausalOracle([claim], series=series).strategy(strat)
    verdict = oracle(claim.id)
    est = oracle.estimates.get(claim.id)
    out = {"event_index": event_index, "verdict": verdict,
           "reason": oracle.reasons.get(claim.id)}
    if est is not None:
        out.update(est.as_dict())
        pl = est.detail.get("placebo_effect")
        out["detail"]["placebo_within_bound"] = (
            pl is None or abs(pl) <= placebo_bound)
    else:
        out["detail"] = {"decline": oracle.reasons.get(claim.id)}
    return out


__all__ = ["STRATEGIES", "CAUSAL_KINDS", "CLAIM_KINDS", "CausalError",
           "Intervention", "do", "CausalClaim", "Strategy", "Table", "Series",
           "Estimate", "CausalOracle", "StrategyOracle", "CausalSession",
           "series_from_journal", "event_study"]
