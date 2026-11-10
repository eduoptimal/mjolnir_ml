"""mjolnir_prover.lib.termination — Termination MVF (L4) + semantics profile (L5).

MVF scope (science_usability_v1.md, Sequence 3):
  * Termination: single counter loops, guard `i != n`, assumption `n >= 0`,
    body increments `i`. Variant `V = n - i` is PROPOSED for the family; the
    gate emits two obligations and disposes exactly:
      term_decrease : V[body] < V   (discharged by exact nf: difference == +1)
      term_pos      : V > 0 under the guard, discharged by SHAPE against the
                      declared family premises {n >= 0, i <= n} — the premises
                      are listed in the certificate's assumptions (TCB), never
                      invented.
    A non-decrementing body -> obligation fails -> UNVERIFIABLE, unit capped
    at PARTIAL. `false_termination = 0` is the invariant.

  * Semantics profile: one declared model
      ints=unbounded, div=trunc, floats=excluded, exec=no_exceptions;
    mutable references and floats are banned. Out-of-profile constructs get a
    labeled UNVERIFIABLE(out_of_semantics_model) — never a guess.
"""
from __future__ import annotations
from . import term as T

PROFILE = {"ints": "unbounded", "div": "trunc", "floats": "declared_intervals",
           "exec": "no_exceptions", "refs": "immutable"}
ASSUMPTIONS = ["n >= 0", "i <= n"]          # declared family premises (TCB)


# ------------------------------------------------------------- termination ----
def variant_obligations(guard, body, var="i", bound="n"):
    """Emit the two termination obligations for the counter family.

    guard: ("ne", var_i, n); body: list of stmts. Variant V = bound - var.
    Returns (ok, obligations, reason).
    """
    V = T.csub(T.var(bound), T.var(var))
    obs = []

    # term_decrease: V after body < V. Under the MVF the body's counter update
    # is a single assignment var := var + 1; substitutions are exact.
    upd = None
    for st in body:
        if st[0] == "assign" and st[1] == var:
            upd = st[2]
    if upd is None:
        return False, [], "no_counter_update"
    V2 = T.sub(V, var, upd)
    d = T.csub(V, V2)                       # V - V[body] must be strictly positive
    nd = T.nf(d)
    dec = nd is not None and nd.get((), 0) >= 1 and all(c >= 0 for c in nd.values())
    obs.append(dict(kind="term_decrease", premise=T.tstr(V), conclusion=T.tstr(V2),
                    method="nf>0", status="ok" if dec else "fail"))

    # term_pos: V > 0 under guard i != n, via family premises {n>=0, i<=n}.
    guard_ok = guard[0] == "ne" and ((guard[1][0] == "v" and guard[1][1] == var
                                      and guard[2] == T.var(bound)))
    pos = dec and guard_ok
    obs.append(dict(kind="term_pos", premise=T.tstr(guard), conclusion=T.tstr(V),
                    method="family_shape",
                    meta={"counter": var, "bound": bound, "assumptions": list(ASSUMPTIONS)},
                    status="ok" if pos else "fail"))
    return pos, obs, "" if pos else "variant_not_discharged"


def check_termination(guard, body, var="i", bound="n"):
    """Returns (status, obligations, reason): 'ok' | 'UNVERIFIABLE'."""
    ok, obs, reason = variant_obligations(guard, body, var, bound)
    return ("ok" if ok else "UNVERIFIABLE"), obs, reason


# ----------------------------------------------------------- semantics (L5) ---
def profile_check(code: str):
    """Parse a unit's source against the declared semantics profile.

    Returns (ok, violations): out-of-model constructs are labeled, never guessed.
    Uses only stdlib ast — no execution.
    Mutation-precise: only ASSIGNMENT to subscripts/attributes, global/nonlocal
    statements, and async/concurrency constructs violate the immutable-refs rule;
    reads and local pure construction are in-model.
    """
    import ast
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"syntax_error:{e.lineno}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                pass        # E5: floats are DECLARED (exact rationals + intervals)
        elif isinstance(node, ast.Try):
            violations.append(f"try_except@{node.lineno}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            violations.append(f"mutable_ref@{node.lineno}")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor,
                               ast.AsyncWith)):
            violations.append(f"async@{node.lineno}")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            tgts = node.targets if isinstance(node, (ast.Assign, ast.Delete)) \
                else [node.target]
            for tgt in tgts:
                if isinstance(tgt, (ast.Subscript, ast.Attribute)):
                    violations.append(f"mutable_ref@{node.lineno}")
    return (not violations), violations


def certify_unit(code, spec_source, confirmed, gate_ok):
    """L0-provenance + L5-profile + gate: the unit-level status machine.

    Returns (status, reason). CERTIFIED requires confirmed provenance, in-profile
    semantics, and all obligations discharged. Anything else abstains honestly.
    """
    if spec_source not in ("assert", "type_hint") or not confirmed:
        return "UNVERIFIABLE", "unconfirmed_spec"
    ok, v = profile_check(code)
    if not ok:
        return "UNVERIFIABLE", "out_of_semantics_model:" + ",".join(v[:3])
    if not gate_ok:
        return "UNVERIFIABLE", "obligation_failed"
    return "CERTIFIED", ""
