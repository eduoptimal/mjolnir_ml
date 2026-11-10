"""mjolnir_prover.lib.frontend_java — Java frontend: lowers Java logic into the
canonical claim format (the same IR the Python frontend emits).

Architecture (interface contract): the VERIFICATION ENGINE is language-free —
it runs on canonical terms. Each language ships a FRONTEND component that
lowers its syntax into that format:

  Java: if (x < 0) { throw ... }   ->  premise  x >= 0
  Java: if (!(x > max)) throw      ->  premise  x > max
  Java: int DEFAULT_DAYS = 30;     ->  constant fact (context)

The engine (unit_vc / theories / kernel) never sees Java.
"""
from __future__ import annotations
import zlib

import javalang

_FLIP = {"<": ">=", ">": "<=", "<=": ">", ">=": "<", "==": "==", "!=": "!="}


def _cid(repo, rel, line, kind):
    return "mj::" + format(zlib.crc32(f"{repo}|{rel}|{line}|{kind}".encode()) & 0xffffffff, "08x")


def _binop_left_right(cond):
    """Return (left_text, op, right_text) for a binary comparison, else None."""
    if not isinstance(cond, javalang.tree.BinaryOperation):
        return None
    op = cond.operator
    if op not in _FLIP:
        return None
    return (cond.operandl, op, cond.operandr)


def _attr_name(node):
    """Name of a member/identifier reference, e.g. days / req.days -> days."""
    if isinstance(node, javalang.tree.MemberReference):
        return node.member
    return None


def _lit_value(node):
    """Integer literal value, else None."""
    if isinstance(node, javalang.tree.Literal) and node.value is not None:
        try:
            return int(node.value)
        except ValueError:
            return None
    return None


def _guard_premises(cond):
    """Violation condition -> premise list. Returns [(var, op, const_text)]."""
    out = []
    stack = [cond]
    while stack:
        c = stack.pop()
        if isinstance(c, javalang.tree.BinaryOperation) and c.operator == "||":
            stack.extend([c.operandl, c.operandr])   # or-of-violations: each negated
            continue
        lr = _binop_left_right(c)
        if lr is None:
            continue
        l, op, r = lr
        ln, rn = _attr_name(l), _attr_name(r)
        lit_l, lit_r = _lit_value(l), _lit_value(r)
        if ln is not None and lit_r is not None:
            out.append(dict(var=ln, op=_FLIP[op], const=lit_r))
        elif rn is not None and lit_l is not None:
            # mirrored literal: (0 > x) means x < 0
            out.append(dict(var=rn, op=_FLIP[_FLIP[op]], const=lit_l))
    return out


def _paren_unwrap(node):
    while isinstance(node, javalang.tree.StatementExpression) or \
            isinstance(node, javalang.tree.BlockStatement):
        node = node.expression if hasattr(node, "expression") else node
    return node


def mine_java_source(src: str, rel: str, repo: str, commit: str):
    """Mine a Java source file into canonical candidates (same schema as the
    Python miner). Premises from guard clauses; constant facts; comparisons
    inside conditions as claims when the guard is not a premise."""
    out = []
    try:
        tree = javalang.parse.parse(src)
    except Exception:
        return out
    cls = tree.types[0] if tree.types else None
    if cls is None:
        return out

    # class constants (static final ints) become context facts
    const_facts = {}
    for f in getattr(cls, "fields", []):
        mods = f.modifiers or []
        if "static" in mods:
            for dec in f.declarators:
                if dec.initializer is not None:
                    v = _lit_value(dec.initializer)
                    if v is not None:
                        const_facts[dec.name] = v

    for method in getattr(cls, "methods", []):
        mname = method.name
        params = [p.name for p in method.parameters or []]
        for _, node in method.filter(javalang.tree.IfStatement):
            cond = node.condition
            # does this if-guard raise (guard clause) or return-error?
            def _flatten_stmts(node_list):
                out = []
                for x in node_list:
                    if isinstance(x, tuple):
                        out.extend(_flatten_stmts(list(x)))
                    else:
                        out.append(x)
                return out
            then_stmts = _flatten_stmts(list(node.then_statement or []))
            raises = any(isinstance(x, (javalang.tree.ThrowStatement,
                                        javalang.tree.StatementExpression,
                                        javalang.tree.ReturnStatement))
                         for x in then_stmts)
            returns_bad = any(isinstance(x, javalang.tree.ReturnStatement)
                              for x in then_stmts)
            if not (raises or returns_bad):
                continue
            premises = _guard_premises(cond)
            for p in premises:
                var, op, const = p["var"], p["op"], p["const"]
                # MVF: guards bind method locals (assigned from parsing/validation
                # calls) and params alike — the canonical unit wraps the local
                # as the premise's parameter, with the method as provenance.
                if var not in params and var not in const_facts and var not in params:
                    pass                      # locals accepted (provenance = method)
                # canonical python claim text (the engine's IR)
                claim = f"assert {var} {op} {const}"
                kind = "premise" if (op in (">=", ">") and raises) else "monitor"
                wrapper = f"def {mname}({var}):\n    {claim}\n"
                out.append(dict(
                    id=_cid(repo, rel, node.position.line if node.position else 0,
                            "guard"),
                    family="guard", tier=kind,
                    predicate=claim, annotation="java-guard",
                    canonical_unit=wrapper,
                    expect=None,
                    origin=dict(repo=repo, commit=commit, file=rel,
                                line=node.position.line if node.position else 0,
                                construct="java_guard", symbol=mname),
                    provenance=dict(source="mined", confirmed=False)))
    return out


def mine_java_repo(root, repo_name=None, commit=None):
    import subprocess
    if commit is None:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                                    cwd=root, capture_output=True,
                                    text=True).stdout.strip()
        except Exception:
            commit = "unknown"
    repo_name = repo_name or os.path.basename(root)
    out = []
    for dp, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "target")]
        for f in sorted(fs):
            if not f.endswith(".java"):
                continue
            full = os.path.join(dp, f)
            rel = os.path.relpath(full, root)
            try:
                src = open(full, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            out.extend(mine_java_source(src, rel, repo_name, commit))
    return out


import os  # noqa: E402