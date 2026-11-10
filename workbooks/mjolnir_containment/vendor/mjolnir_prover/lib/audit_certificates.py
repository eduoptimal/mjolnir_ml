"""mjolnir_prover.lib.audit_certificates — does a CERTIFIED claim actually have a model?

    python -m mjolnir_prover.lib.audit_certificates [repo ...]

WHAT THIS IS FOR
-----------------
Every claim this kernel certified has been re-derived by an independent tracer and
an arbiter, and agreed on. All of that is INTERNAL: it checks that the proof
follows from the stated premises. Neither step asks whether the premises and the
conclusion are *satisfied by an assignment* — and a unit whose conclusion was
absorbed into its premise list, or whose premises are mutually unsatisfiable, would
agree with itself perfectly and be certified while saying nothing.

A countermodel discharges that question directly: an assignment satisfying every
premise and falsifying the conclusion is a PROOF OF FALSEHOOD. If one exists the
certificate is unsound, and no amount of internal agreement makes it right.

THE DIRECTION IS ONE-WAY AND THAT IS THE POINT
----------------------------------------------
    countermodel found      -> the certificate is FALSE. Certain.
    no countermodel found   -> INCONCLUSIVE. A bounded grid cannot prove anything.

A semi-decision procedure that only reports certainty is the right shape for an
audit: it can create embarrassment and never creates false comfort. The grid is
deliberately narrow (rationals over a small range), which is exactly why a miss is
worth nothing.

INDEPENDENT BY CONSTRUCTION
---------------------------
It reads the asserts with `ast`, takes the LAST assert as the conclusion and the
others as premises, and evaluates with its own interpreter. It does not call the
kernel's parser, its lowering, or its obligations — if it did, it would inherit
whatever mistake the kernel made and agree with it, which is the failure mode that
produced the third and fourth instrument defects of this project.

The one-assert case is excluded on purpose: a unit with a single assert is a
DECLARED PREMISE under the readings this project uses (see `frag_free_ge0`), and
searching a countermodel against an assumption is a category error, not a defect.
"""
from __future__ import annotations

import ast
import os
import sys
from collections import Counter
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mjolnir_prover.lib import certify, mine  # noqa: E402

REPOS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "mjolnir", "storage", "data", "repos")


class Unknown(Exception):
    """This model cannot decide this expression -- not a failure, not a verdict."""


# --------------------------------------------------------------- the interpreter

def _eval(node, env):
    """Evaluate an assert expression. Raises Unknown rather than guessing.

    Every construct not handled raises: a silent `True` here would be a
    countermodel generator that certifies everything, which is worse than no
    audit at all.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, (int, float)):
            return F(node.value)
        raise Unknown("constant")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise Unknown("name")
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.Not):
            return not v
        raise Unknown("unary")
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if b == 0:
                raise Unknown("div0")
            return a / b
        if isinstance(node.op, ast.FloorDiv):
            if b == 0:
                raise Unknown("div0")
            return a // b
        if isinstance(node.op, ast.Mod):
            if b == 0:
                raise Unknown("mod0")
            return a % b
        if isinstance(node.op, ast.Pow):
            if isinstance(b, F) and b.denominator != 1:
                if a < 0:
                    raise Unknown("fract_pow")
                return F(float(a) ** float(b))
            if b < 0 and a == 0:
                raise Unknown("pow0")
            if isinstance(b, F) and abs(b) > 64:
                raise Unknown("big_exp")
            return a ** int(b)
        raise Unknown("binop")
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.Gt):
                ok = left > right
            else:
                raise Unknown("cmp")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(vals)
        return any(vals)
    raise Unknown(type(node).__name__)


def _names(exprs):
    out = set()
    for e in exprs:
        for n in ast.walk(e):
            if isinstance(n, ast.Name):
                out.add(n.id)
    return out


def split_unit(code: str):
    """-> (premise ASTs, conclusion AST | None, reason)

    The LAST assert is the conclusion — the same rule the kernel's lowering uses,
    read independently here. One assert means the unit is a declared premise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None, "syntax"
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    asserts.sort(key=lambda n: n.lineno)
    if not asserts:
        return None, None, "no_assert"
    if len(asserts) == 1:
        return None, None, "declared_premise_only"
    return [a.test for a in asserts[:-1]], asserts[-1].test, "ok"


def countermodel(premises, conclusion, *, values=None):
    """-> {var: Fraction} falsifying `conclusion` while satisfying `premises`,
    or None. None means INCONCLUSIVE, never "verified"."""
    vars_ = sorted(_names(list(premises) + [conclusion]))
    if not vars_:
        return None                      # ground claim: evaluator says True/False
    if len(vars_) > 6:
        return None                      # grid would be dishonest at this width
    vals = values or [F(i, 2) for i in range(-8, 9)]      # -4..4 in halves
    if len(vars_) > 4:
        vals = [F(i) for i in range(-3, 4)]               # integers only

    def fits(env):
        for p in premises:
            try:
                if _eval(p, env) is not True:
                    return False
            except Unknown:
                return False             # cannot check -> not this model
        return True

    def falsifies(env):
        try:
            return _eval(conclusion, env) is False
        except Unknown:
            return None                  # unknown, not false

    # iterate the cartesian product with explicit depth
    import itertools
    for combo in itertools.product(vals, repeat=len(vars_)):
        env = dict(zip(vars_, combo))
        if not fits(env):
            continue
        f = falsifies(env)
        if f is True:
            return env
        if f is None:
            continue
    return None


def function_units(path: str, limit: int = 400):
    """-> [candidate] for every FUNCTION with a premise and a conclusion.

    THE MINER NEVER PRODUCES THESE, WHICH IS WHY THE AUDIT NEEDED THEM.

    Measuring the 2,060 certified claims first: 2,049 carry ONE assert, 11 carry
    none, and **0 carry two or more**. A single assert is a declared premise under
    this project's readings, so the headline "2,060 certificates" was a count of
    assumed statements, not proven obligations. The countermodel audit had nothing
    to convict because there was nothing to convict *of*.

    Function-level units restore the object: every assert but the last is a
    premise, the last is the conclusion, and only then is "certified" a claim
    about implication rather than about an assumption.
    """
    out = []
    for dp, _dirs, files in os.walk(path):
        if len(out) >= limit:
            return out
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            if len(out) >= limit:
                return out
            fp = os.path.join(dp, fn)
            try:
                src = open(fp, encoding="utf-8", errors="replace").read()
                tree = ast.parse(src)
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
                if len(asserts) < 2:
                    continue
                # The unit is the WHOLE function, sliced verbatim. The first
                # version reconstructed it line by line and re-indented by hand,
                # which produced 101 units that did not parse -- an extraction bug
                # showing up as 101 "unprovable" obligations. Source-preserving
                # extraction is the only defensible way to build a unit: any
                # repair the auditor performs is a repair the prover then judges.
                code = ast.get_source_segment(src, node) or ""
                if not code.strip():
                    continue
                out.append(dict(
                    id="fu::" + _crc(fp + str(node.lineno)),
                    family="vc", tier="static", predicate=node.name,
                    canonical_unit=code, origin={"file": fp, "line": node.lineno},
                    provenance={"source": "function_units"},
                ))
    return out


def _crc(s: str) -> str:
    import zlib
    return zlib.crc32(s.encode()).to_bytes(4, "big").hex()


def audit_records(records):
    """-> (stats, findings). `records` are miner candidates."""
    stats, findings = Counter(), []
    for c in records:
        if (c.get("provenance") or {}).get("expected") == "REFUTED":
            continue                     # already expected false; not this audit
        code = c.get("canonical_unit") or ""
        premises, conclusion, why = split_unit(code)
        if why != "ok":
            stats[f"skip:{why}"] += 1
            continue
        stats["audited"] += 1
        model = countermodel(premises, conclusion)
        if model is None:
            stats["inconclusive"] += 1
            continue
        stats["COUNTERMODEL"] += 1
        origin = c.get("origin") or {}
        findings.append({
            "file": origin.get("file"), "line": origin.get("line"),
            "predicate": c.get("predicate"),
            "model": {k: str(v) for k, v in model.items()},
        })
    return stats, findings


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    repos = argv or ["pyaes", "jwcrypto", "mpmath", "rotnov-pycc"]
    if len(repos) == 1 and repos[0].endswith(".py"):
        repos = ["mpmath"]
    total, all_findings = Counter(), []
    for repo in repos:
        path = repo if os.path.isabs(repo) else os.path.join(REPOS, repo)
        if not os.path.isdir(path):
            print(f"{repo}: absent", flush=True)
            continue
        cands = function_units(path)
        total[f"units_extracted"] += len(cands)
        certified = []
        for c in cands:
            try:
                r = certify.certify_claim(c)
            except Exception:  # noqa: BLE001
                continue
            total[f"unit_status:{r['status']}"] += 1
            if r["status"] == "CERTIFIED":
                certified.append(c)
        stats, findings = audit_records(certified)
        total.update(stats)
        all_findings.extend([dict(f, repo=repo) for f in findings])
        print(f"{repo:16s} certified={len(certified):5d} "
              f"audited={stats['audited']:5d} "
              f"inconclusive={stats['inconclusive']:5d} "
              f"COUNTERMODEL={stats['COUNTERMODEL']}", flush=True)

    print("\n" + "=" * 72)
    for k in sorted(total):
        print(f"  {k:32s} {total[k]:5d}")
    if all_findings:
        print(f"\nUNSOUND CERTIFICATES ({len(all_findings)}):")
        for f in all_findings[:25]:
            print(f"  {f['repo']}/{f['file']}:{f['line']}  {f['predicate'][:44]}")
            print(f"      model {f['model']}")
    else:
        print("\nno countermodel found in any audited certificate")
        print("NOTE: inconclusive is NOT verified. A bounded grid can only")
        print("      convict, never acquit.")
    return 1 if all_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())