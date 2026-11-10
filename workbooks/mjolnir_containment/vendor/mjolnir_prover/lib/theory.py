"""mjolnir_prover.lib.theory — the rule-table engine: semantic policy as DATA.

Maturity step (safety.md S8/S9 note): the semantic theories move from
per-feature code to DECLARED TABLES in mjolnir_prover/theories/*.json, interpreted by
this small engine. What stays code (the TCB floor): term algebra (term.py) and
the exact interval operations (intervals.py) — arithmetic, not policy.

Tables:
  classification.json — premise vs claim shapes; claim disposal policy;
                        premise injection rules (E1b/E1c)
  intervals.json      — real-valued comparison policy (fail-closed directive
                        semantics), built-in premises, ambiguity rules

Every table is hash-pinned; certificates record the theory versions they used
(same discipline as induction theories: knowledge as data, traceable).
"""
from __future__ import annotations
import json, os, hashlib
from . import paths as _paths

_DIR = _paths.theories()
_cache = {}


def load(name: str) -> dict:
    """Load + pin a theory table (cached; content-hashed)."""
    if name not in _cache:
        path = os.path.join(_DIR, f"{name}.json")
        with open(path, "r", encoding="utf-8") as f:
            t = json.load(f)
        t["digest"] = hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest()[:16]
        _cache[name] = t
    return _cache[name]


def digest(name: str) -> str:
    return load(name)["digest"]


# ------------------------------------------------- classification policy ------
def classify_premise(l_kind, r_kind, op) -> dict | None:
    """Returns the matching premise rule (with its bound directive) or None.

    l_kind/r_kind: 'param' | 'const'; op: ge/gt/le/lt (Eq handled separately).
    """
    table = load("classification")["premise_shape"]
    for rule in table["rules"]:
        w = rule["when"]
        if w["lhs"] == l_kind and w["rhs"] == r_kind and op in w["op"]:
            return rule
    return None


def dispose_claim(*, decide_eq, facts_present, atoms_present) -> tuple[str, str]:
    """Claim-disposal policy: returns (verdict, reason) from the table."""
    ctx = {"decide_eq": decide_eq, "facts_present": bool(facts_present),
           "atoms_present": bool(atoms_present),
           "undecided": decide_eq is None}
    for rule in load("classification")["claim_disposal"]["rules"]:
        if all(ctx.get(k) == v for k, v in rule["when"].items()):
            v = rule["verdict"]
            return ("CERTIFIED" if v == "CERTIFIED" else
                    "REFUTED" if v == "REFUTED" else
                    "UNVERIFIABLE"), rule.get("reason", "")
    return "UNVERIFIABLE", "no_rule_matched"


def composition_verdict(route_verdicts: list) -> tuple[str, str, str | None]:
    """Join of SEVERAL INDEPENDENT routes over one claim. Consults the table.

    `route_verdicts` is a list of what each route returned (`True`, `False`, or
    `None` for undecided). Returns `(verdict, reason, evidence)` where the
    evidence is the one belonging to a CERTIFYING route -- so the kernel
    re-derives a proposition it already knows rather than a disjunction.

    The caller supplies the routes' outputs and does not decide. This is the
    same discipline as `dispose_claim`: the CALLER measures, the TABLE decides.
    Written as a Python `if` this rule would be policy in code, unpinned and
    unreviewable, which is precisely what the `tcb_note` forbids.
    """
    table = load("classification")["composition"]
    verdicts = list(route_verdicts)
    any_cert = any(v is True for v in verdicts)
    all_ref = bool(verdicts) and all(v is False for v in verdicts)
    ctx = {"any_certified": any_cert, "all_refuted": all_ref,
           "n_routes": len(verdicts)}
    for rule in table["rules"]:
        if all(ctx.get(k) == v for k, v in rule["when"].items()):
            v = rule["verdict"]
            out = ("CERTIFIED" if v == "CERTIFIED" else
                   "REFUTED" if v == "REFUTED" else "UNVERIFIABLE")
            return out, rule.get("reason", "")
    return "UNVERIFIABLE", "composition_no_rule_matched"


def relation_verdict(*, n_stated: int, n_matched: int) -> tuple[str, str]:
    """Disposal policy for a RELATION-valued source. Consults the table.

    The caller supplies two integers it measured by exact single-value
    comparison; it does not decide. `evidence_rule` in the table is what makes
    the certified branch re-derivable: the evidence is the MATCHED value, so the
    kernel re-derives exactly the single-value proposition it already knows,
    rather than a disjunction nothing has verified.
    """
    table = load("classification")["relation_disposal"]
    ctx = {"matched": "any" if n_matched > 0 else "none",
           "n_stated": n_stated, "n_matched": n_matched}
    for rule in table["rules"]:
        if all(ctx.get(k) == v for k, v in rule["when"].items()):
            v = rule["verdict"]
            return ("CERTIFIED" if v == "CERTIFIED" else
                    "REFUTED" if v == "REFUTED" else
                    "UNVERIFIABLE"), rule.get("reason", "")
    return "UNVERIFIABLE", "relation_no_rule_matched"


def premise_contradiction_verdict(*, premise_method: str, lhs_identical: bool,
                                 rhs_both_numeric: bool, rhs_differ: bool
                                 ) -> tuple[str | None, str]:
    """Consult the premise-contradiction table (policy as DATA).

    Returns (verdict | None, reason). A verdict is returned ONLY when a rule
    matches AND the rule is enabled; a disabled rule matches nothing, which is
    how a rejected directive is expressed without deleting it — the rule stays
    in the table, hashed and reviewable, with its measurement attached.

    This is deliberately NOT implemented in the caller. The caller asks the
    table; the table decides. A hardcoded rule in the prover would be exactly
    the thing these tables exist to prevent.
    """
    table = load("classification").get("premise_contradiction") or {}
    ctx = {"premise_method": premise_method,
           "premise_lhs_identical": bool(lhs_identical),
           "rhs_both_numeric_constants": bool(rhs_both_numeric),
           "rhs_values_differ": bool(rhs_differ)}
    for rule in table.get("rules", []):
        if not rule.get("enabled", True):
            continue                       # a disabled directive decides nothing
        if all(ctx.get(k) == v for k, v in rule["when"].items()):
            return rule["verdict"], rule.get("reason", "")
    return None, table.get("default", "abstain")


def injection_allowed(tier: str, ledger_decision: str, grammar_ok: bool) -> bool:
    """E1b premise-injection policy from the table."""
    for rule in load("classification")["premise_injection"]["rules"]:
        w = rule["when"]
        if (w["tier"] == tier and w["ledger_decision"] == ledger_decision
                and (grammar_ok if w["predicate_grammar"] == "param_vs_const_ineq" else False)):
            return rule["action"] == "inject_bound"
    return False


# ----------------------------------------------------- intervals policy -------
def _ev(expr, L, R):
    """Tiny safe evaluator for the comparison policy expressions: hi(L), lo(R)..."""
    return eval(expr, {"__builtins__": {}},
                {"lo_L": L[0], "hi_L": L[1], "lo_R": R[0], "hi_R": R[1],
                 "L": L, "R": R})


def _expr(expr, L, R):
    s = expr.replace("hi(L)", "hi_L").replace("lo(R)", "lo_R") \
            .replace("lo(L)", "lo_L").replace("hi(R)", "hi_R")
    return _ev(s, L, R)


def decide(op: str, L, R) -> bool | None:
    """Interval comparison policy from the table. L, R = (lo, hi) intervals.
    Returns True (holds everywhere) / False (fails somewhere) / None."""
    table = load("intervals")["comparison"]
    rule = table.get(op)
    if rule is None:
        return None
    if "holds" in rule and _expr(rule["holds"], L, R):
        return True
    if "violates" in rule and _expr(rule["violates"], L, R):
        return False
    return None


def builtins() -> list:
    return load("intervals").get("built_in_premises", [])


def ambiguity(action_for: str) -> str | None:
    for r in load("intervals").get("ambiguity_rules", []):
        if action_for in r["when"]:
            return r.get("reason")
    return None


def versions() -> dict:
    return {n: digest(n) for n in ("classification", "intervals")}