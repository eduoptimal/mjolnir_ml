"""mjolnir_ml.dsl — @dsl: compile a policy class's SOURCE into one canonical IR.

AUTHORING (the only runtime-facing surface):

    from mjolnir_ml import dsl, rule, RELEASE, ASK, DENY, DecisionPolicyBase

    @dsl
    class FraudPolicy(DecisionPolicyBase):
        facts = ("amount", "credit_limit")            # optional, compile-tight

        # premises — decidability gates. Violation ABSTAINS, never guesses.
        assert lambda amount: amount > 0, "amount_positive"

        # decisive rules — code, requires (= lambda params), verdict.
        assert rule("hard_deny",
                    when=lambda amount, credit_limit: amount > credit_limit * 2,
                    verdict=DENY)
        assert rule("auto_release",
                    when=lambda p_positive, conf: p_positive < 0.05
                                 and conf >= 0.95,
                    verdict=RELEASE)

        @rule(verdict=ASK)                            # method form: unit-testable
        def escalate_mid(self, conf):
            return conf < 0.80

WHY SOURCE-HARVEST (not runtime introspection)
----------------------------------------------
`assert` statements leave nothing in `cls.__dict__`, and `python -O` deletes
them at compile time. Runtime-side-effect registration inside asserts is
therefore unsound in BOTH directions (invisible, or silently stripped). The
one path that is sound for all of it: read the class's source, parse the AST,
harvest statements as DATA. Consequences, all deliberate:

  * no source (REPL / exec) -> DslCompileError. Fail closed: a policy the
    compiler cannot read cannot be certified. No dict-only fallback — a
    silently dropped premise is the defect class this repo keeps finding.
  * `python -O` is irrelevant: the file text still contains the asserts
    (pinned by tests/test_dsl.py::test_optimized_interpreter_keeps_policy).
  * runtime execution of the class body is irrelevant to the IR; the runtime
    `rule()` below exists only so the class body executes cleanly.

THE WHITELIST (predicates are pure over their params)
-----------------------------------------------------
A predicate may contain only: bool ops (and/or/not), comparisons
(== != < <= > >= in not in), arithmetic (+ - * / // % **), unary +/-,
names bound as parameters, and literals (numbers/strings/tuples).
No calls, no attributes, no subscripting, no free names. This is what makes
a policy REPLAYABLE: no hidden globals, no I/O, no model access inside a
predicate — the engine sees exactly the requires the predicate declared.

TWO-PHASE COMPILE (the engine has no RELEASE verdict)
------------------------------------------------------
`ROUTE` in the frozen core means "no obligation fired", not "release".
So premises compile to engine rules with verdict=ABSTAIN (phase 1), ask/deny
compile to engine rules (phase 2, engine severity max), and release rules are
NOT engine rules at all: they are evaluated only when the engine returns ROUTE,
and only a FIRING release rule produces `release`. ROUTE + no firing release ->
`abstain(uncovered)`, never a default release.

IR (the ONE shape every frontend of mjolnir_ml produces)
---------------------------------------------------------
    {"schema": "mjolnir.ml.policy/1", "name": ..., "facts": [...],
     "statements": [{"kind": "premise"|"rule"|"release", "code": ...,
                     "requires": [...], "predicate": "<normalized expr>",
                     "verdict": "ask"|"deny" | null}]}
Digest = sha256 of the canonical JSON (sorted keys). The digest is over the
IR, never over bytecode or code objects: certificates must survive a Python
patch bump, and co_code does not.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from mjolnir_ml.policy import MODEL_FACTS, Policy

__all__ = ["dsl", "rule", "DecisionPolicyBase", "DslError"]


class DslError(SyntaxError):
    """The policy source cannot be compiled to the IR (fail-closed: no policy)."""


# ------------------------------------------------------------------ runtime --
class _RuleMarker:
    """Truthiness + decorator capability only. The IR comes from the AST."""

    __slots__ = ("kwargs",)

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, fn):
        return fn

    def __bool__(self):
        return True


def rule(code=None, when=None, verdict=None, requires=None):
    """Statement marker. Its ARGUMENTS are read from source, never from here."""
    return _RuleMarker(code=code, when=when, verdict=verdict, requires=requires)


class DecisionPolicyBase:
    """Base for @dsl policy classes. Carries the compiled artifact."""

    facts: tuple = ()
    __policy__: Policy | None = None

    @classmethod
    def policy(cls) -> Policy:
        p = cls.__policy__
        if p is None:
            raise DslError(f"{cls.__name__} was not compiled — apply @dsl")
        return p


# ---------------------------------------------------------------- whitelist --
_ALLOWED_EXPR = (
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow,
    ast.Name, ast.Load, ast.Constant, ast.Tuple,
)

_REJECTED_NAMES = {
    ast.Call: "calls are not allowed in a predicate",
    ast.Attribute: "attribute access is not allowed in a predicate",
    ast.Subscript: "subscripting is not allowed in a predicate",
    ast.IfExp: "conditional expressions are not allowed in a predicate",
    ast.Lambda: "nested lambdas are not allowed in a predicate",
}

_WIRE = {"RELEASE": "release", "ASK": "ask", "DENY": "deny",
         "release": "release", "ask": "ask", "deny": "deny"}
_FORBIDDEN_VERDICTS = {"ABSTAIN": "abstain", "abstain": "abstain",
                       "ROUTE": "route", "route": "route",
                       "ESCALATE": "escalate", "escalate": "escalate"}


def _fail(src_name: str, node: ast.AST, msg: str) -> None:
    line = getattr(node, "lineno", "?")
    raise DslError(f"{src_name}:{line}: {msg}")


def _check_expr(src_name: str, node: ast.expr, params: tuple, ctx: str) -> str:
    """Whitelist-validate an expression; return its normalized (unparsed) form."""
    allowed_names = set(params)
    for n in ast.walk(node):
        if type(n) in _REJECTED_NAMES:
            _fail(src_name, n, _REJECTED_NAMES[type(n)] + f" in {ctx}")
        if isinstance(n, ast.Lambda) and n is not node:
            _fail(src_name, n, f"nested lambda in {ctx}")
        if not isinstance(n, (*_ALLOWED_EXPR, ast.boolop, ast.operator,
                              ast.unaryop, ast.cmpop, type(node))):
            _fail(src_name, n,
                  f"{type(n).__name__} is not allowed in {ctx}")
        if isinstance(n, ast.Name) and n.id not in allowed_names:
            _fail(src_name, n,
                  f"name {n.id!r} in {ctx} is not a declared fact or model "
                  f"fact (allowed: {sorted(allowed_names)})")
    return ast.unparse(node)


def _resolve_verdict(src_name: str, node: ast.expr) -> str:
    if isinstance(node, ast.Name) and node.id in _WIRE:
        return _WIRE[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, str) \
            and node.value in ("release", "ask", "deny"):
        return _WIRE[node.value]
    if (isinstance(node, ast.Name) and node.id in _FORBIDDEN_VERDICTS) or \
            (isinstance(node, ast.Constant) and node.value in _FORBIDDEN_VERDICTS):
        word = node.id if isinstance(node, ast.Name) else node.value
        _fail(src_name, node,
              f"verdict {word} cannot be produced by a rule: abstain is "
              f"the harness's vocabulary (write a premise), route/escalate "
              f"are engine-internal (use ask)")
    _fail(src_name, node, "verdict must be RELEASE, ASK or DENY")


def _lambda_parts(src_name: str, lam: ast.Lambda, ctx: str):
    if lam.args.vararg or lam.args.kwarg or lam.args.kwonlyargs \
            or lam.args.defaults or lam.args.kw_defaults:
        _fail(src_name, lam, f"{ctx}: lambda must be plain positional params")
    params = tuple(a.arg for a in lam.args.posonlyargs + lam.args.args)
    if not params:
        _fail(src_name, lam, f"{ctx}: predicate needs at least one param "
                             f"(the facts it reads)")
    pred = _check_expr(src_name, lam.body, params, ctx)
    return params, pred


def _parse_rule_call(src_name: str, call: ast.Call, default_code: str | None,
                     ctx: str, predicate: str | None = None,
                     params: tuple | None = None):
    if not isinstance(call.func, ast.Name) or call.func.id != "rule":
        _fail(src_name, call, f"{ctx}: expected a rule(...) call")
    code = default_code
    pos = call.args
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    unknown = set(kw) - {"code", "when", "verdict", "requires"}
    if unknown or len(pos) > 1:
        _fail(src_name, call,
              f"{ctx}: rule takes (code, when=..., verdict=..., requires=...)")
    if pos:
        if not isinstance(pos[0], ast.Constant) or \
                not isinstance(pos[0].value, str):
            _fail(src_name, call, f"{ctx}: rule code must be a string literal")
        if code is not None and code != pos[0].value:
            _fail(src_name, call,
                  f"{ctx}: code given twice ({code!r} vs {pos[0].value!r})")
        code = pos[0].value
    if "code" in kw:
        if not isinstance(kw["code"], ast.Constant) or \
                not isinstance(kw["code"].value, str):
            _fail(src_name, call, f"{ctx}: rule code must be a string literal")
        code = kw["code"].value
    if code is None:
        _fail(src_name, call,
              f"{ctx}: rule needs a stable code (tests assert on it)")
    if not code.isidentifier():
        _fail(src_name, call, f"{ctx}: code {code!r} must be an identifier")
    if "verdict" not in kw:
        _fail(src_name, call, f"{ctx}: rule needs an explicit verdict")
    verdict = _resolve_verdict(src_name, kw["verdict"])
    if "when" in kw:
        if not isinstance(kw["when"], ast.Lambda):
            _fail(src_name, call, f"{ctx}: when= must be a lambda")
        params, pred = _lambda_parts(src_name, kw["when"], f"rule {code!r}")
    elif predicate is not None and params is not None:
        pred = predicate                       # method form: body already read
    else:
        _fail(src_name, call, f"{ctx}: rule needs when=<lambda>")
    if "requires" in kw:
        r = kw["requires"]
        if not isinstance(r, (ast.Tuple, ast.List)) or not all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in r.elts):
            _fail(src_name, call, f"{ctx}: requires must be a tuple of strings")
        declared = tuple(e.value for e in r.elts)
        extra = set(declared) - set(params)
        if extra:
            _fail(src_name, call,
                  f"{ctx}: requires names {sorted(extra)} are not lambda params")
        params = declared
    kind = "release" if verdict == "release" else "rule"
    return {"kind": kind, "code": code, "requires": list(params),
            "predicate": pred, "verdict": verdict}


def _parse_assert(src_name: str, node: ast.Assert) -> dict:
    test = node.test
    if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) \
            and test.func.id == "rule":
        return _parse_rule_call(src_name, test, None, "assert rule(...)")
    if isinstance(test, ast.Lambda):
        msg = node.msg
        if not isinstance(msg, ast.Constant) or not isinstance(msg.value, str):
            _fail(src_name, node,
                  'a premise assert needs a code: assert lambda ..., "code"')
        code = msg.value
        if not code.isidentifier():
            _fail(src_name, node, f"premise code {code!r} must be an identifier")
        params, pred = _lambda_parts(src_name, test, f"premise {code!r}")
        return {"kind": "premise", "code": code, "requires": list(params),
                "predicate": pred, "verdict": None}
    _fail(src_name, node,
          "assert must hold a premise lambda or a rule(...) call")


def _parse_method(src_name: str, node: ast.FunctionDef) -> dict | None:
    deco = [d for d in node.decorator_list
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
            and d.func.id == "rule"]
    other = [d for d in node.decorator_list if d not in deco]
    if not deco:
        if node.name.startswith("_"):
            return None
        _fail(src_name, node,
              f"method {node.name!r}: policy methods carry @rule(...) — "
              f"helpers must be underscore-private (predicates cannot call "
              f"them anyway)")
    if len(deco) > 1 or other:
        _fail(src_name, node,
              f"method {node.name!r}: exactly one @rule decorator, no others")
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Return) \
            or node.body[0].value is None:
        _fail(src_name, node,
              f"method {node.name!r}: body must be a single return expression")
    args = node.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
        _fail(src_name, node, f"method {node.name!r}: plain params only")
    names = [a.arg for a in args.posonlyargs + args.args]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    if not names:
        _fail(src_name, node,
              f"method {node.name!r}: predicate needs at least one fact param")
    pred = _check_expr(src_name, node.body[0].value, tuple(names),
                       f"rule {node.name!r}")
    return _parse_rule_call(src_name, deco[0], node.name,
                            f"@rule method {node.name!r}", predicate=pred,
                            params=tuple(names))


def _parse_facts(src_name: str, node: ast.Assign | ast.AnnAssign) -> tuple | None:
    """`facts = (...)` -> tuple; any other target -> None (not a facts decl)."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if not (len(targets) == 1 and isinstance(targets[0], ast.Name)
            and targets[0].id == "facts"):
        return None
    val = node.value
    if not isinstance(val, (ast.Tuple, ast.List)) or not all(
            isinstance(e, ast.Constant) and isinstance(e.value, str)
            for e in val.elts):
        _fail(src_name, node, "facts must be a tuple of string literals")
    names = tuple(e.value for e in val.elts)
    clash = set(names) & MODEL_FACTS
    if clash:
        _fail(src_name, node,
              f"facts may not declare model facts {sorted(clash)} — "
              f"{sorted(MODEL_FACTS)} are reserved for the harness")
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        _fail(src_name, node, f"duplicate facts {sorted(dupes)}")
    return names


# ------------------------------------------------------------------ harvest --
def _ipython_source(cls) -> str | None:
    """Interactive fallback: find the cell that defined this class.

    Notebook/REPL classes have no file for `inspect.getsource`, but IPython
    keeps every cell's source in `In`. We parse those cells and take the most
    recent one that actually contains this ClassDef — never a guess, and never
    a class whose source is absent (fail-closed remains)."""
    try:
        from IPython import get_ipython
    except Exception:                                      # noqa: BLE001
        return None
    ip = get_ipython() if callable(get_ipython) else None
    if ip is None:
        return None
    target = f"class {cls.__name__}"
    for cell in reversed(list(ip.user_ns.get("In") or [])):
        if not cell or target not in cell:
            continue
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                return cell
    return None


def _class_def(cls) -> ast.ClassDef:
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError) as ex:
        src = _ipython_source(cls)
        if src is None:
            raise DslError(
                f"{cls.__name__}: policy source unavailable ({ex}). @dsl "
                f"compiles from source — define policies in a module file or "
                f"an IPython/Jupyter cell (source is read, never executed "
                f"for the IR)") from None
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError as ex:
        raise DslError(f"{cls.__name__}: cannot parse policy source: {ex}") \
            from None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return node
    raise DslError(f"{cls.__name__}: no class definition found in source")


def compile_class(cls) -> Policy:
    """Source -> IR -> Policy. The single harvest path (asserts, assigns, methods)."""
    cd = _class_def(cls)
    src_name = getattr(cls, "__module__", "?")
    # Only an explicit declaration validates names at compile time; without
    # one, bind validates against the estimator schema (the authority).
    facts: tuple | None = (tuple(cls.__dict__["facts"])
                           if "facts" in cls.__dict__ else None)
    statements: list[dict] = []
    seen: dict[str, str] = {}

    for node in cd.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue                                            # docstring
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            declared = _parse_facts(src_name, node)
            if declared is not None:
                if facts is not None and tuple(facts) != declared:
                    _fail(src_name, node, "facts mismatch between AST and class")
                facts = declared
                continue
            # assignment form: auto_release = rule("code", when=..., verdict=...)
            value = node.value
            if isinstance(value, ast.Call):
                stmt = _parse_rule_call(src_name, value, None, "assignment")
                statements.append(stmt)
                continue
            _fail(src_name, node,
                  "only facts=... and rule(...) assignments are assignable")
        if isinstance(node, ast.Assert):
            statements.append(_parse_assert(src_name, node))
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmt = _parse_method(src_name, node)
            if stmt is not None:
                statements.append(stmt)
            continue
        if isinstance(node, ast.Pass):
            continue
        _fail(src_name, node,
              f"{type(node).__name__} is not a policy statement")

    if not statements:
        raise DslError(f"{cls.__name__}: no statements — a policy that says "
                       f"nothing cannot dispose")

    for s in statements:
        if s["code"] in seen:
            raise DslError(f"{cls.__name__}: duplicate code {s['code']!r} "
                           f"(kind {seen[s['code']]} and {s['kind']})")
        seen[s["code"]] = s["kind"]
        if facts is not None:
            unknown = set(s["requires"]) - set(facts) - MODEL_FACTS
            if unknown:
                raise DslError(
                    f"{cls.__name__}: rule {s['code']!r} requires "
                    f"{sorted(unknown)}, which are neither declared facts "
                    f"{list(facts)} nor model facts {sorted(MODEL_FACTS)}")

    ir = {"schema": Policy.SCHEMA, "name": cls.__name__,
          "facts": list(facts or ()), "statements": statements}
    policy = Policy(ir)
    cls.__policy__ = policy
    return policy


def dsl(cls):
    """Class decorator: harvest source, compile IR, attach Policy."""
    if not isinstance(cls, type):
        raise DslError("@dsl applies to a class")
    compile_class(cls)
    return cls
