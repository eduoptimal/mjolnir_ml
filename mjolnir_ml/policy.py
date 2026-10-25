"""mjolnir_ml.policy — the ONE canonical IR: statements, digest, engine compile.

Every frontend (the @dsl class decorator, a future JSON loader) produces THIS
shape and nothing else. The digest is over canonical JSON of the IR — the
table IS its digest, so two environments can prove they dispose by the same
policy, and a mismatch is a hard error rather than a silent drift.

Engine compile is two-phase (see dsl.py): premises and ask/deny become
`vendors.core.verdict.Rule` objects for `evaluate`; release statements are
kept separate and are only consulted when the engine returns ROUTE.
"""
from __future__ import annotations

import hashlib
import json

from mjolnir_ml.vendors import wire
from mjolnir_ml.vendors.core.verdict import (ABSTAIN, DENY, ESCALATE, ROUTE,
                                             UNSTATED, Rule, _Declared,
                                             evaluate)

# Facts the harness supplies from the estimator's output. Reserved: a policy
# may REQUIRE them, may never declare them as input facts. The risk facts are
# supplied by a calibrated RiskIndex (an auxiliary crossing model); when no
# index is fitted they are absent, so a rule requiring them abstains.
MODEL_FACTS = frozenset({"pred", "conf", "p_positive", "certified",
                         "set_size", "lo", "hi",
                         "p_event", "risk_release", "risk_deny"})

_ENGINE_VERDICT = {"ask": ESCALATE, "deny": DENY}


class Statement:
    __slots__ = ("kind", "code", "requires", "predicate", "verdict", "_fn")

    def __init__(self, kind: str, code: str, requires: list,
                 predicate: str, verdict: str | None):
        self.kind = kind                      # premise | rule | release
        self.code = code
        self.requires = tuple(requires)
        self.predicate = predicate
        self.verdict = verdict
        self._fn = _compile(self.requires, predicate)

    def fires(self, facts: dict) -> bool:
        """Evaluate the predicate over ONLY its declared requires.

        Params are bound positionally from the requires tuple; the predicate
        cannot reach anything else (the dsl whitelist guarantees no attribute,
        subscript, call or free name). Missing facts are the caller's job —
        the engine's absent check covers engine rules; release evaluation in
        model.py pre-checks and abstains.
        """
        return bool(self._fn(*(facts[k] for k in self.requires)))

    def as_ir(self) -> dict:
        return {"kind": self.kind, "code": self.code,
                "requires": list(self.requires), "predicate": self.predicate,
                "verdict": self.verdict}

    def __repr__(self):
        return f"Statement({self.kind}, {self.code!r}, requires={self.requires})"


def _compile(requires: tuple, predicate: str):
    """Normalized predicate source -> a pure function. Builtins stripped."""
    params = ", ".join(requires)
    src = f"lambda {params}: {predicate}" if params else f"lambda: {predicate}"
    code = compile(src, "<mjolnir.ml.policy>", "eval")
    return eval(code, {"__builtins__": {}}, {})  # noqa: S307 — whitelisted AST


class Policy:
    SCHEMA = "mjolnir.ml.policy/1"

    def __init__(self, ir: dict):
        if ir.get("schema") != self.SCHEMA:
            raise ValueError(f"unknown policy schema {ir.get('schema')!r}")
        stmts = [Statement(**s) for s in ir["statements"]]
        if not stmts:
            raise ValueError("policy has no statements")
        codes = [s.code for s in stmts]
        if len(codes) != len(set(codes)):
            raise ValueError(f"duplicate statement codes: {codes}")
        self.ir = ir
        self.name = ir["name"]
        self.facts = tuple(ir.get("facts") or ())
        self.statements = tuple(stmts)
        self.premises = tuple(s for s in stmts if s.kind == "premise")
        self.rules = tuple(s for s in stmts if s.kind == "rule")
        self.releases = tuple(s for s in stmts if s.kind == "release")
        self._by_code = {s.code: s for s in stmts}
        self._engine_rules = self.engine_rules()
        self._digest = "sha256:" + hashlib.sha256(
            self._canonical()).hexdigest()

    def _canonical(self) -> bytes:
        return json.dumps(self.ir, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")

    @property
    def digest(self) -> str:
        return self._digest

    def statement(self, code: str) -> Statement:
        return self._by_code[code]

    def uses_model_facts(self, codes) -> bool:
        """Do any of these statements depend on estimator output? (guarantee tag)"""
        return any(set(self._by_code[c].requires) & MODEL_FACTS
                   for c in codes if c in self._by_code)

    def engine_rules(self) -> list[Rule]:
        """Phase 1+2: premises (abstain-on-violation) then ask/deny rules.

        Order is policy: premises first (they are decidability gates), then
        decisive rules in source order; the engine's severity max makes deny
        win over ask when both fire.
        """
        rules: list[Rule] = []
        for s in self.premises:
            rules.append(Rule(s.code, requires=s.requires,
                              when=_negated(s), verdict=ABSTAIN))
        for s in self.rules:
            rules.append(Rule(s.code, requires=s.requires,
                              when=_positive(s),
                              verdict=_ENGINE_VERDICT[s.verdict]))
        return rules

    # ------------------------------------------------------- facts replay --
    def dispose_facts(self, facts: dict) -> tuple:
        """Policy-only disposition over a complete fact table.

        THE third-party audit path: given the facts a certificate publishes,
        re-derive (verdict, basis, missing) WITHOUT the estimator. Used by
        DecisionModel.dispose and by certificate.verify_certificate, so the
        live path and the audit path cannot drift.
        """
        vd = evaluate(facts, self._engine_rules)
        if vd.value == ABSTAIN:
            return (wire.ABSTAIN_W, tuple(vd.basis), vd.missing)
        if vd.value == ROUTE:
            fired = []
            for st in self.releases:
                absent = [k for k in st.requires
                          if facts.get(k, UNSTATED) is UNSTATED]
                if absent:
                    return (wire.ABSTAIN_W, (f"missing:{absent[0]}",),
                            absent[0])
                if st.fires(facts):
                    fired.append(st.code)
            if fired:
                return (wire.RELEASE, tuple(fired), None)
            return (wire.ABSTAIN_W, ("uncovered",), None)
        return (wire.project(vd.value), tuple(vd.basis), None)

    @classmethod
    def load(cls, ir: dict) -> "Policy":
        return cls(ir)

    def __repr__(self):
        return f"Policy({self.name!r}, {len(self.statements)} statements, " \
               f"{self.digest[:22]}...)"


def _positive(s: Statement):
    # Engine hands us a _Declared view; bind params positionally from it.
    return lambda view, _s=s: _s._fn(*(view[k] for k in _s.requires))


def _negated(s: Statement):
    # Premise fires (= abstains) when its lambda is FALSE.
    return lambda view, _s=s: not _s._fn(*(view[k] for k in _s.requires))


__all__ = ["Policy", "Statement", "MODEL_FACTS"]
