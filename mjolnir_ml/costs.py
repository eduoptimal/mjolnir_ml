"""mjolnir_ml.costs — cost-sensitive verdicts (the Chow reject option, priced).

A verdict is not free. For a binary risk score p = P(unsafe | x) and declared
costs

    C_FR   cost of releasing an unsafe action      (false release)
    C_FD   cost of denying a safe action           (false deny)
    C_A    cost of asking a human                  (the reject option)

the expected costs are

    release:  p · C_FR
    deny:     (1 − p) · C_FD
    ask:      C_A

so the optimal bands are

    p <  C_A / C_FR                -> release
    p >  1 − C_A / C_FD            -> deny
    otherwise                      -> ask

(When the ask band is empty, the direct release/deny crossover is
C_FD / (C_FR + C_FD).) `cost_policy` compiles these bands into the same
IR every other policy uses, so a cost model is just another declared policy —
digested, disposable, replayable — and `expected_cost` scores any verdict
batch against the declared prices.

Soundness note: the bands are only as good as p. In a DecisionModel, p should
be a *calibrated* probability (e.g. sklearn CalibratedClassifierCV), and the
optional `require_certified=True` adds the conformal singleton gate on top.
"""
from __future__ import annotations

from dataclasses import dataclass

from mjolnir_ml.policy import Policy
from mjolnir_ml.vendors import wire


@dataclass(frozen=True)
class Costs:
    c_false_release: float = 10.0
    c_false_deny: float = 10.0
    c_ask: float = 2.0

    def __post_init__(self):
        for name in ("c_false_release", "c_false_deny", "c_ask"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v < 0:
                raise ValueError(f"{name} must be a non-negative number, "
                                 f"got {v!r}")
        if self.c_false_release <= 0 or self.c_false_deny <= 0:
            raise ValueError("false-release/false-deny costs must be > 0")

    # ------------------------------------------------------------- bands --
    @property
    def release_threshold(self) -> float:
        """p below which releasing beats asking (clipped to [0,1])."""
        return min(1.0, self.c_ask / self.c_false_release)

    @property
    def deny_threshold(self) -> float:
        """p above which denying beats asking (clipped to [0,1])."""
        return max(0.0, 1.0 - self.c_ask / self.c_false_deny)

    def thresholds(self) -> tuple:
        lo, hi = self.release_threshold, self.deny_threshold
        if lo > hi:                     # asking is never worth its price
            cross = self.c_false_deny / (self.c_false_release +
                                         self.c_false_deny)
            return (cross, cross)
        return (lo, hi)

    # ------------------------------------------------------------ scoring --
    def expected_cost(self, verdict: str, p_unsafe: float) -> float:
        p = float(p_unsafe)
        if verdict == wire.RELEASE:
            return p * self.c_false_release
        if verdict == wire.DENY_W:
            return (1.0 - p) * self.c_false_deny
        if verdict in (wire.ASK, wire.ABSTAIN_W):
            return self.c_ask
        raise ValueError(f"unknown verdict {verdict!r}")

    def best_verdict(self, p_unsafe: float) -> str:
        """Cheapest verdict at p; deterministic tie order release, deny, ask."""
        p = float(p_unsafe)
        options = [(self.expected_cost(wire.RELEASE, p), 0, wire.RELEASE),
                   (self.expected_cost(wire.DENY_W, p), 1, wire.DENY_W),
                   (self.expected_cost(wire.ASK, p), 2, wire.ASK)]
        return min(options)[2]

    def mean_cost(self, verdicts, p_unsafe) -> float:
        if len(verdicts) != len(p_unsafe):
            raise ValueError("verdicts and probabilities differ in length")
        if not verdicts:
            return 0.0
        return sum(self.expected_cost(v, p)
                   for v, p in zip(verdicts, p_unsafe)) / len(verdicts)

    def as_dict(self) -> dict:
        lo, hi = self.thresholds()
        return {"c_false_release": self.c_false_release,
                "c_false_deny": self.c_false_deny, "c_ask": self.c_ask,
                "release_threshold": lo, "deny_threshold": hi}


def _num(x: float) -> str:
    return repr(round(float(x), 12))


def cost_policy(costs: Costs, *, p_fact: str = "p_positive",
               require_certified: bool = False, name: str = "CostPolicy",
               ask_when_uncertified: bool = True) -> Policy:
    """Compile cost bands into the canonical policy IR.

    Statements (same IR as @dsl):
        deny    p > hi
        ask     lo <= p <= hi
        release p < lo   [+ `and certified` when require_certified]
        ask     not certified                       (optional, first)
    With require_certified, an uncalibrated model has no `certified` fact and
    the engine abstains on missing — the fail-closed default, not a release.
    """
    lo, hi = costs.thresholds()
    p = p_fact
    statements = []
    if require_certified and ask_when_uncertified:
        statements.append({"kind": "rule", "code": "await_certification",
                           "requires": ["certified"],
                           "predicate": "not certified", "verdict": "ask"})
    statements.append({"kind": "rule", "code": "deny_high",
                       "requires": [p],
                       "predicate": f"{p} > {_num(hi)}", "verdict": "deny"})
    if lo < hi:
        statements.append({"kind": "rule", "code": "ask_band",
                           "requires": [p],
                           "predicate": f"{p} >= {_num(lo)} and {p} <= "
                                        f"{_num(hi)}", "verdict": "ask"})
    requires = [p] + (["certified"] if require_certified else [])
    predicate = f"{p} < {_num(lo)}"
    if require_certified:
        predicate += " and certified"
    statements.append({"kind": "release", "code": "release_low",
                       "requires": requires, "predicate": predicate,
                       "verdict": "release"})
    ir = {"schema": Policy.SCHEMA, "name": name, "facts": [],
          "statements": statements}
    return Policy(ir)


__all__ = ["Costs", "cost_policy"]
