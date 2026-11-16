"""Independent, restricted policy-IR evaluator.

No import from mjolnir_ml.model or mjolnir_ml.policy. Predicates are parsed as
restricted expressions and walked explicitly; Python eval is never used.
"""
from __future__ import annotations

import ast

from mjolnir_ml.auditor.canonical import digest

POLICY_SCHEMA = "mjolnir.ml.policy/1"
UNSTATED = None
MODEL_FACTS = frozenset({"pred", "conf", "p_positive", "certified", "set_size",
                         "lo", "hi", "p_event", "risk_release", "risk_deny"})
RISK_FACTS = frozenset({"p_event", "risk_release", "risk_deny"})
COVERAGE_FACTS = frozenset({"certified", "set_size", "lo", "hi"})
RELEASE = "release"
ASK = "ask"
DENY = "deny"
ABSTAIN = "abstain"


class PolicyIRError(ValueError):
    pass


def policy_digest(ir: dict) -> str:
    return digest(ir)


def _validate(node: ast.AST) -> None:
    allowed = (
        ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
        ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.In, ast.NotIn, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow, ast.Name, ast.Load, ast.Constant,
        ast.Tuple,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed):
            raise PolicyIRError(f"disallowed policy expression: {type(child).__name__}")


def _eval(node: ast.AST, env: dict):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise PolicyIRError(f"undeclared predicate name: {node.id}")
        return env[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(x, env) for x in node.elts)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            value = True
            for x in node.values:
                value = _eval(x, env)
                if not value:
                    return value
            return value
        if isinstance(node.op, ast.Or):
            value = False
            for x in node.values:
                value = _eval(x, env)
                if value:
                    return value
            return value
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:
                raise PolicyIRError(f"disallowed comparison: {type(op).__name__}")
            if not ok:
                return False
            left = right
        return True
    raise PolicyIRError(f"disallowed policy expression: {type(node).__name__}")


def _predicate(source: str):
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise PolicyIRError(f"invalid predicate: {source!r}") from exc
    _validate(node)

    def evaluate(env):
        return bool(_eval(node, env))

    return evaluate


def _statements(ir: dict):
    if not isinstance(ir, dict) or ir.get("schema") != POLICY_SCHEMA:
        raise PolicyIRError("unsupported policy schema")
    raw = ir.get("statements")
    if not isinstance(raw, list) or not raw:
        raise PolicyIRError("policy has no statements")
    statements = []
    codes = set()
    for item in raw:
        try:
            kind = item["kind"]
            code = item["code"]
            requires = tuple(item["requires"])
            predicate = item["predicate"]
            verdict = item.get("verdict")
        except Exception as exc:
            raise PolicyIRError("malformed policy statement") from exc
        if kind not in {"premise", "rule", "release"}:
            raise PolicyIRError(f"unknown statement kind: {kind!r}")
        if not isinstance(code, str) or not code:
            raise PolicyIRError("statement code must be a non-empty string")
        if code in codes:
            raise PolicyIRError(f"duplicate statement code: {code}")
        codes.add(code)
        if not all(isinstance(name, str) for name in requires):
            raise PolicyIRError("statement requires must be strings")
        if verdict not in {None, ASK, DENY, RELEASE}:
            raise PolicyIRError(f"unknown statement verdict: {verdict!r}")
        statements.append({"kind": kind, "code": code, "requires": requires,
                           "predicate": predicate, "verdict": verdict,
                           "fn": _predicate(predicate)})
    return statements


def _missing(facts: dict, requires: tuple):
    absent = [name for name in requires if facts.get(name, UNSTATED) is UNSTATED]
    return absent


def _guarantee(statements, basis) -> str:
    by_code = {s["code"]: s for s in statements}
    if any(set(by_code[c]["requires"]) & MODEL_FACTS for c in basis
           if c in by_code):
        return "model_backed"
    return "declared"


def _assurance(statements, verdict, basis, missing) -> str:
    if missing in RISK_FACTS:
        return "action_risk_control"
    if missing in COVERAGE_FACTS:
        return "pointwise_coverage"
    by_code = {s["code"]: s for s in statements}
    reqs = set()
    cov_release = risk_release = False
    for code in basis:
        st = by_code.get(code)
        if st is None:
            continue
        reqs |= set(st["requires"])
        if st["kind"] == "release":
            if set(st["requires"]) & COVERAGE_FACTS:
                cov_release = True
            if set(st["requires"]) & RISK_FACTS:
                risk_release = True
    if verdict == RELEASE and cov_release and risk_release:
        return "dual_track"
    if reqs & RISK_FACTS:
        return "action_risk_control"
    if reqs & COVERAGE_FACTS:
        return "pointwise_coverage"
    if reqs & MODEL_FACTS:
        return "model_backed"
    return "declared"


def dispose_facts(ir: dict, facts: dict) -> tuple:
    statements = _statements(ir)
    premises = [s for s in statements if s["kind"] == "premise"]
    rules = [s for s in statements if s["kind"] == "rule"]
    releases = [s for s in statements if s["kind"] == "release"]

    for statement in premises:
        absent = _missing(facts, statement["requires"])
        if absent:
            return (ABSTAIN, (f"missing:{absent[0]}",), absent[0])
        if not statement["fn"](facts):
            return (ABSTAIN, (statement["code"],), None)

    fired = []
    severity = 0
    for statement in rules:
        absent = _missing(facts, statement["requires"])
        if absent:
            return (ABSTAIN, (f"missing:{absent[0]}",), absent[0])
        if statement["fn"](facts):
            fired.append(statement["code"])
            severity = max(severity, 2 if statement["verdict"] == DENY else 1)
    if fired:
        return (DENY if severity >= 2 else ASK, tuple(fired), None)

    fired = []
    for statement in releases:
        absent = _missing(facts, statement["requires"])
        if absent:
            return (ABSTAIN, (f"missing:{absent[0]}",), absent[0])
        if statement["fn"](facts):
            fired.append(statement["code"])
    if fired:
        return (RELEASE, tuple(fired), None)
    return (ABSTAIN, ("uncovered",), None)


def audit_expectations(ir: dict, facts: dict) -> dict:
    verdict, basis, missing = dispose_facts(ir, facts)
    statements = _statements(ir)
    return {
        "verdict": verdict,
        "basis": basis,
        "missing": missing,
        "guarantee": _guarantee(statements, basis),
        "assurance": _assurance(statements, verdict, basis, missing),
        "policy_digest": digest(ir),
    }
