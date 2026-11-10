"""mjolnir_prover.lib.effects — E6: heap-effect contracts (gate side, deterministic).

An EFFECT CONTRACT for a function declares what it does to the outside world:
  pure        : no state change; returns a value (calls to known-pure functions
                are allowed — assume-guarantee transitivity)
  mutates:<t> : modifies the caller-visible target t (parameter or attribute)
  opaque:<n>  : calls into n functions we hold no contract for (network, IO,
                ML frameworks) — runtime-monitor class

The extractor is DETERMINISTIC (AST analysis): it computes the ground truth
the verifier checks proposals against. The FROZEN ENCODER proposes contracts
(see effects_rank in p23); the gate disposes. Separation preserved.
"""
from __future__ import annotations
import ast

# Names that are pure BY CONSTRUCTION: evaluating them cannot change any
# caller-visible state, so holding no contract for them is sound.
#
# `print` and `input` are deliberately ABSENT: both have observable effects
# (stdout / stdin) and so are opaque. Opaque is the fail-closed direction.
PURE_CALLABLES = {
    # builtins: constructors, comparisons, iteration, arithmetic
    "len", "min", "max", "abs", "sum", "sorted", "reversed", "range",
    "enumerate", "zip", "isinstance", "issubclass", "int", "float", "str",
    "bool", "round", "complex", "bytes", "bytearray", "list", "tuple",
    "set", "dict", "frozenset", "all", "any", "map", "filter", "divmod",
    "pow", "repr", "hash", "chr", "ord", "bin", "hex", "oct", "format",
    "slice", "iter", "next", "getattr", "hasattr", "callable", "vars",
    # exception construction mutates nothing observable in the caller
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
    "AttributeError", "NotImplementedError", "ZeroDivisionError",
    "StopIteration", "Exception", "AssertionError", "OverflowError",
    "RecursionError", "LookupError", "ArithmeticError", "NameError",
    "UnboundLocalError", "FileNotFoundError", "PermissionError",
    # pure stdlib functions, observed in this corpus
    "reduce", "cmp_to_key", "partial", "wraps", "lru_cache", "cache",
    "deepcopy", "chain", "product", "combinations", "permutations",
    "groupby", "count", "gcd", "lcm", "factorial", "isqrt",
}

# Attribute calls rooted at one of these module names are pure functions of
# their arguments. Anything else reached through an attribute is opaque --
# we hold no contract for it, which is the sound direction.
PURE_MODULES = {
    "math", "cmath", "itertools", "functools", "operator", "typing",
    "string", "heapq", "bisect", "decimal", "fractions", "statistics",
    "collections", "dataclasses", "enum", "abc", "numbers", "copy",
    "reprlib", "keyword",
    # `sys` is deliberately ABSENT: sys.stdout.write(...) is an effect, and
    # whitelisting the module would skip it. Fail closed.
}


def _targets(node):
    """Names written through: assignment targets, subscript/attr roots, aug."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            out.add(("name", n.target.id))
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(("name", t.id))
                elif isinstance(t, ast.Subscript):
                    r = t.value
                    while isinstance(r, ast.Subscript):
                        r = r.value
                    if isinstance(r, ast.Name):
                        out.add(("container", r.id))
                elif isinstance(t, ast.Attribute):
                    r = t.value
                    while isinstance(r, ast.Attribute):
                        r = r.value
                    if isinstance(r, ast.Name):
                        out.add(("attr", r.id))
    return out


def _bare_calls(node, self_name=None):
    """Statement-level calls (possible external effects), minus recursion."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            f = n.value.func
            if isinstance(f, ast.Name):
                if f.id != self_name:
                    out.add(f.id)
            elif isinstance(f, ast.Attribute):
                root = f.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    out.add(f"{root.id}.{f.attr}")
    return out


def _deep_calls(node, self_name=None):
    """All called names anywhere in the body (for transitive checks)."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id != self_name:
                out.add(n.func.id)
    return out


def extract_effects(fn: ast.FunctionDef, pure_known=None) -> dict:
    """Deterministic effect extraction for one function.

    pure_known: set of function names already verified pure (transitivity).
    Returns contract: {status: pure|mutating|opaque, mutates: [...],
                       opaque_calls: [...]}.
    """
    pure_known = set(pure_known or ())
    self_name = fn.name
    params = {p.arg for p in fn.args.args}

    # containers/attrs mutated through a parameter root
    mutates = []
    opaque = []
    for n in ast.walk(fn):
        # in-place container mutation through a parameter root
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Subscript):
                    r = t.value
                    while isinstance(r, ast.Subscript):
                        r = r.value
                    if isinstance(r, ast.Name) and r.id in params:
                        mutates.append(f"{r.id}[*]")
        elif isinstance(n, ast.AugAssign):
            if isinstance(n.target, ast.Name) and n.target.id in params:
                mutates.append(n.target.id)
            elif isinstance(n.target, ast.Subscript):
                r = n.target.value
                while isinstance(r, ast.Subscript):
                    r = r.value
                if isinstance(r, ast.Name) and r.id in params:
                    mutates.append(f"{r.id}[*]")
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
            r = n.value
            while isinstance(r, ast.Attribute):
                r = r.value
            if isinstance(r, ast.Name) and r.id in params:
                mutates.append(f"{r.id}.*")
        # OPAQUE CALLS -- DEEP. Every Call node anywhere in the body, not just
        # bare expression statements.
        #
        # DEFECT FIXED (2026-10-03). This scan used to be `isinstance(n,
        # ast.Expr) and isinstance(n.value, ast.Call)`, so a call in a return
        # or assignment position was invisible. `_deep_calls` above implements
        # the correct scan and was never called. Consequence: `intro_sort.sort`
        # was certified `pure` while delegating to intro_sort / heap_sort /
        # insertion_sort, and 14.5% of top-level functions in TheAlgorithmsPython
        # were certified pure while calling a SAME-MODULE function. That is a
        # false certificate in the disposer -- silent, not a fail-closed FAULT.
        # (Verified case: `intro_sort.sort` calls `intro_sort` in a RETURN
        # position, so the old scan saw nothing. It does not itself call
        # heap_sort/insertion_sort -- those are one level down, inside
        # intro_sort.)
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                if f.id != self_name and f.id not in PURE_CALLABLES \
                        and f.id not in pure_known:
                    opaque.append(f.id)
            elif isinstance(f, ast.Attribute):
                root = f.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    if root.id in PURE_MODULES:
                        continue                       # pure module function
                    if root.id in params:
                        mutates.append(f"{root.id}.{f.attr}")
                    elif root.id != self_name:
                        opaque.append(f"{root.id}.{f.attr}")
    # arguments mutated via in-place methods: xs.sort(), xs.append(...)
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            root = n.func.value
            if isinstance(root, ast.Name) and root.id in params \
                    and n.func.attr in ("sort", "append", "extend", "insert",
                                        "remove", "pop", "clear", "update",
                                        "add", "discard", "reverse"):
                mutates.append(f"{root.id}.{n.func.attr}")
    mutates = sorted(set(mutates))
    opaque = sorted(set(opaque))
    status = "pure" if not mutates and not opaque else \
        "mutating" if mutates else "opaque"
    return dict(status=status, mutates=mutates, opaque_calls=opaque,
                params=sorted(params))

# --------------------------------------------------------------------------
# Gate-artifact invariants. A statistical gate's bar must be POSSIBLE and its
# verdict must FOLLOW from its own numbers; otherwise a wrong input (measured
# once: a majority computed over distinct combinations instead of rows gave
# 0.005 and flipped a NO-GO triage to `enabled`) can silently switch a
# component on. These checks are what the self-check runs.
# --------------------------------------------------------------------------
def validate_gate(block: dict, n_classes: int, eps: float = 1e-9):
    """-> (ok, reasons). Pure arithmetic; no I/O, no model."""
    bad = []
    if not isinstance(block, dict) or not block:
        return False, ["gate block missing"]
    n, acc, maj = block.get("n"), block.get("accuracy"), block.get("majority")
    if not isinstance(n, int) or n <= 0:
        bad.append(f"n must be a positive int (got {n!r})")
    if not isinstance(acc, (int, float)) or not (0.0 <= acc <= 1.0):
        bad.append(f"accuracy must be in [0,1] (got {acc!r})")
    if not isinstance(maj, (int, float)):
        bad.append(f"majority must be a number (got {maj!r})")
    else:
        floor = 1.0 / max(1, n_classes)
        if maj < floor - eps:
            bad.append(f"impossible bar: majority {maj:.4f} < 1/{n_classes} = {floor:.4f}")
        if not (0.0 <= maj <= 1.0):
            bad.append(f"majority must be in [0,1] (got {maj!r})")
    if isinstance(acc, (int, float)) and isinstance(maj, (int, float)):
        want = "GO" if acc > maj else "NO-GO"
        if block.get("verdict") != want:
            bad.append(f"verdict {block.get('verdict')!r} != recomputed {want!r}")
        d = block.get("delta")
        if isinstance(d, (int, float)) and abs(d - (acc - maj)) > 1e-3:
            bad.append(f"delta {d!r} != accuracy-majority {acc - maj:.4f}")
    return (not bad), bad


def validate_artifact(cats: dict, n_classes: int | None = None):
    """-> (ok, reasons) for a category artifact carrying a gate block."""
    if not isinstance(cats, dict):
        return False, ["artifact must be a mapping"]
    cls = cats.get("categories") or {}
    n_classes = n_classes or max(1, len(cls))
    enc = cats.get("encoder") or {}
    tri = enc.get("triage") or {}
    ok, bad = validate_gate(tri, n_classes)
    status = enc.get("triage_status")
    if tri.get("verdict") == "GO" and status != "enabled":
        bad.append(f"GO verdict but triage_status={status!r}")
    if tri.get("verdict") and tri.get("verdict") != "GO" and status != "disabled":
        bad.append(f"non-GO verdict but triage_status={status!r}")
    if status == "enabled" and not tri.get("evidence"):
        bad.append("triage enabled without re-derivable evidence")
    if not cats.get("class_map"):
        bad.append("class_map missing (producer/gate vocabularies must be mapped)")
    if cats.get("status") == "active" and not cats.get("assignment"):
        bad.append("status=active without an assignment declaration")
    return (not bad), bad
