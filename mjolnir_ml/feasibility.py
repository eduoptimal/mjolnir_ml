"""mjolnir_ml.feasibility — are the declared rules reachable at all?

A policy is a contract, and a contract with a rule that can never fire is
defective: the audit sees `deny = 0` but cannot tell "no such case today" from
"this action is impossible for this model". This module answers the second
question over the supplied rows (and their observed fact box):

    observed    how many rows actually fired the statement
    reachable   observed, or reachable somewhere in the fact box spanned by
                the observed model facts (a bound, not a global proof)
    dead        neither

For regressor thresholds the decisive evidence is usually the attainable
bound: a deny rule `lo > tau` is dead when `max(lo) <= tau`; a release rule
`hi <= tau` is dead when `min(hi) > tau`. For the risk track the evidence is
`max(p_event) < lambda_deny` / `min(p_event) > lambda_release`.

Reachability is evaluated by replaying the compiled predicate over the corner
combinations of the observed model-fact ranges. The predicates are pure (the
dsl whitelist), so this is deterministic; it is still a claim about the
supplied sample/fact box, and the report says so.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from mjolnir_ml.model import DecisionModel, Disposition

_NUMERIC_PROBE = ("lo", "hi", "p_event", "pred", "conf", "p_positive",
                  "set_size")
_BOOL_PROBE = ("certified", "risk_release", "risk_deny")


class FeasibilityError(RuntimeError):
    """A declared rule is unreachable (fail-closed deployment gate)."""


@dataclass(frozen=True)
class RuleCheck:
    code: str
    kind: str
    verdict: str | None
    requires: tuple
    assurance: str
    observed: int
    reachable: bool
    evidence: dict

    @property
    def dead(self) -> bool:
        return not self.reachable

    def as_dict(self) -> dict:
        return {"code": self.code, "kind": self.kind, "verdict": self.verdict,
                "requires": list(self.requires),
                "assurance": self.assurance, "observed": self.observed,
                "reachable": self.reachable, "dead": self.dead,
                "evidence": dict(self.evidence)}


@dataclass(frozen=True)
class FeasibilityReport:
    threshold: float | None
    n_rows: int
    checks: tuple
    dead_rules: tuple
    coverage_bounds: dict
    risk_bounds: dict

    @property
    def ok(self) -> bool:
        return not self.dead_rules

    def assert_reachable(self) -> "FeasibilityReport":
        """Fail closed when any declared rule can never fire."""
        if self.dead_rules:
            raise FeasibilityError(
                f"policy contains unreachable rules: {list(self.dead_rules)}; "
                f"adjust the threshold, the model/calibration, or switch to "
                f"the risk track")
        return self

    def as_dict(self) -> dict:
        return {"threshold": self.threshold, "n_rows": self.n_rows,
                "ok": self.ok,
                "dead_rules": list(self.dead_rules),
                "coverage_bounds": dict(self.coverage_bounds),
                "risk_bounds": dict(self.risk_bounds),
                "checks": [c.as_dict() for c in self.checks]}


def _assurance(requires) -> str:
    req = set(requires)
    if req & {"p_event", "risk_release", "risk_deny"}:
        return "action_risk_control"
    if req & {"certified", "set_size", "lo", "hi"}:
        return "pointwise_coverage"
    if req & {"pred", "conf", "p_positive"}:
        return "model_backed"
    return "declared"


def _range(facts, key):
    vals = [float(f[key]) for f in facts
            if key in f and isinstance(f[key], (int, float))]
    return (min(vals), max(vals)) if vals else None


def _candidates(facts, requires) -> list:
    """Corner combinations for the model facts a statement reads."""
    probes = [k for k in requires
              if k in _NUMERIC_PROBE or k in _BOOL_PROBE]
    if not probes:
        return [{}]
    axes = []
    for k in probes:
        if k in _BOOL_PROBE:
            axes.append([(k, False), (k, True)])
            continue
        rng = _range(facts, k)
        if rng is None:
            axes.append([(k, 0.0)])
        elif rng[0] == rng[1]:
            axes.append([(k, rng[0])])
        else:
            axes.append([(k, rng[0]), (k, rng[1])])
    if len(axes) > 6:                     # keep the audit bounded
        return [{}]
    out = []
    for combo in itertools.product(*axes):
        out.append(dict(combo))
    return out


def feasibility_from_dispositions(dm: DecisionModel, disps, *,
                                  threshold: float | None = None,
                                  ) -> FeasibilityReport:
    """Feasibility over already-computed dispositions (no re-disposal)."""
    disps = list(disps)
    facts = [dict(d.facts or {}) for d in disps]
    nonempty = [f for f in facts if f]
    n = len(disps)

    lo_rng = _range(nonempty, "lo")
    hi_rng = _range(nonempty, "hi")
    p_rng = _range(nonempty, "p_event")
    coverage_bounds = {
        "threshold": threshold,
        "min_lo": lo_rng[0] if lo_rng else None,
        "max_lo": lo_rng[1] if lo_rng else None,
        "min_hi": hi_rng[0] if hi_rng else None,
        "max_hi": hi_rng[1] if hi_rng else None,
    }
    if threshold is not None and lo_rng is not None:
        coverage_bounds["deny_bound_clears_threshold"] = \
            lo_rng[1] > float(threshold)
    if threshold is not None and hi_rng is not None:
        coverage_bounds["release_bound_clears_threshold"] = \
            hi_rng[0] <= float(threshold)

    lams_r = [d.prediction.get("lambda_release") for d in disps
              if d.prediction and d.prediction.get("lambda_release")
              is not None]
    lams_d = [d.prediction.get("lambda_deny") for d in disps
              if d.prediction and d.prediction.get("lambda_deny")
              is not None]
    risk_bounds = {
        "event": dm.risk.event if dm.risk is not None else None,
        "min_p_event": p_rng[0] if p_rng else None,
        "max_p_event": p_rng[1] if p_rng else None,
        "lambda_release": min(lams_r) if lams_r else None,
        "lambda_deny": max(lams_d) if lams_d else None,
    }
    if p_rng and lams_d:
        risk_bounds["deny_bound_clears_lambda"] = \
            p_rng[1] >= max(lams_d)
    if p_rng and lams_r:
        risk_bounds["release_bound_clears_lambda"] = \
            p_rng[0] <= min(lams_r)

    checks = []
    dead = []
    for st in dm.policy.statements:
        observed = 0
        possible = False
        for f in nonempty:
            if any(k not in f for k in st.requires):
                continue
            if st.fires(f):
                observed += 1
                possible = True
        if not possible and nonempty:
            for cand in _candidates(nonempty, st.requires):
                for base in nonempty:
                    trial = dict(base)
                    trial.update(cand)
                    if any(k not in trial for k in st.requires):
                        continue
                    if st.fires(trial):
                        possible = True
                        break
                if possible:
                    break
        evidence = {}
        req = set(st.requires)
        if "lo" in req and coverage_bounds["max_lo"] is not None:
            evidence["max_lo"] = coverage_bounds["max_lo"]
        if "hi" in req and coverage_bounds["min_hi"] is not None:
            evidence["min_hi"] = coverage_bounds["min_hi"]
        if "p_event" in req and risk_bounds["max_p_event"] is not None:
            evidence["min_p_event"] = risk_bounds["min_p_event"]
            evidence["max_p_event"] = risk_bounds["max_p_event"]
        if "risk_release" in req:
            evidence["lambda_release"] = risk_bounds["lambda_release"]
        if "risk_deny" in req:
            evidence["lambda_deny"] = risk_bounds["lambda_deny"]
        if threshold is not None:
            evidence["threshold"] = float(threshold)
            if "hi" in req and coverage_bounds["min_hi"] is not None:
                evidence["min_hi"] = coverage_bounds["min_hi"]
                evidence["slack_hi"] = (float(threshold)
                                        - coverage_bounds["min_hi"])
            if "lo" in req and coverage_bounds["max_lo"] is not None:
                evidence["max_lo"] = coverage_bounds["max_lo"]
                evidence["slack_lo"] = (coverage_bounds["max_lo"]
                                        - float(threshold))
            only = req & {"lo", "hi"}
            if len(only) == 1:
                if "hi" in only:
                    evidence["direction"] = "upper_le"
                    evidence["achieved_bound"] = coverage_bounds["min_hi"]
                    evidence["slack"] = evidence.get("slack_hi")
                else:
                    evidence["direction"] = "lower_gt"
                    evidence["achieved_bound"] = coverage_bounds["max_lo"]
                    evidence["slack"] = evidence.get("slack_lo")
        checks.append(RuleCheck(
            code=st.code, kind=st.kind, verdict=st.verdict,
            requires=tuple(st.requires), assurance=_assurance(st.requires),
            observed=observed, reachable=possible, evidence=evidence))
        if not possible and st.kind in ("rule", "release"):
            dead.append(st.code)

    return FeasibilityReport(
        threshold=threshold, n_rows=n, checks=tuple(checks),
        dead_rules=tuple(dead), coverage_bounds=coverage_bounds,
        risk_bounds=risk_bounds)


def feasibility(dm: DecisionModel, rows, *,
                threshold: float | None = None) -> FeasibilityReport:
    """Dispose `rows` and audit whether every declared rule can fire."""
    if not isinstance(dm, DecisionModel):
        raise TypeError("feasibility takes a DecisionModel")
    rows = list(rows)
    if not rows:
        raise ValueError("feasibility needs a non-empty list of rows")
    return feasibility_from_dispositions(
        dm, [dm.dispose(r) for r in rows], threshold=threshold)


__all__ = ["RuleCheck", "FeasibilityReport", "FeasibilityError",
           "feasibility", "feasibility_from_dispositions"]
