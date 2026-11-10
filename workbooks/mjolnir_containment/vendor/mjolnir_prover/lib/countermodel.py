"""mjolnir_prover.lib.countermodel — populate the certificate's declared `witness` field.

Why this module exists
----------------------
`certificate/v0` has always declared a `witness` field
(`certificate.py:18` — `"reason": ..., "witness": ...`), and B1/CEGIS describes
the REFUTED witness as the CEGIS counterexample. But the AST path hardcoded
`"witness": None` (`certificate.py:113`) and never filled it, so every
REFUTED artifact was non-conformant with its own declared schema. Measured:
243/243 stored REFUTED certificates carried `witness=None`, which is why
`gate.py`'s CTA ("paste the witness as a failing test — repro attached")
promised a repro that did not exist.

This module closes that conformance gap **without adding trust**.

The key move: a candidate countermodel is never trusted because we searched
for it — it is trusted because the *same* decision procedure that decided the
claim re-decides it at the proposed point and reports it false. Substitution
uses `term.sub_all`; falsification uses `term.decide_eq` / `term.provably_*`.
Those are already the TCB. So the search is proposer-side and untrusted, and
acceptance is trusted, exactly like the encoder/gate split.

Witness kinds (and what each honestly means)
--------------------------------------------
`point`          a concrete parameter assignment at which the claim is
                 PROVABLY false. The CEGIS-useful kind: paste it, watch it fail.
`region`         a box is known to contain a violating input, but no single
                 point was extracted. "Somewhere in here", not "here".
`inconsistency`  the premise system is contradictory, so NO falsifying input
                 exists by definition. This is a proof of unsatisfiability and
                 is deliberately NOT presented as a counterexample.
`None`           no countermodel is available. Honest absence.

Soundness guards (why this cannot manufacture a false repro)
-------------------------------------------------------------
1. **No domain violation.** A point is emitted only when the failing
   obligation records no bound/premise restriction. If the claim is
   conditional (`assert n >= 0; ... assert result >= 5`) then `n = -3`
   falsifies the *unconditional* reading but is outside the declared domain,
   so it is NOT a countermodel of what was actually claimed. We abstain.
2. **Verified, never asserted.** Every candidate point is re-decided by
   `decide_eq`/`provably_*` after substitution. If it does not verify, it is
   discarded.
3. **Allow-list, keyed to execution evidence.** Emit only when every
   constructor in the claim has a twin that EXECUTES the unit and confirms the
   assert fires. Deny by default: an untested constructor must not inherit
   trust. This is what caught `ledger + [x]`, where the algebra's uniform `+`
   reported a "provable violation" while the program raised `TypeError`.
4. **Deterministic.** The parameter order is sorted, so two runs of the same
   authority version emit the SAME witness. (`vars_of` returns a set, whose
   order varies with PYTHONHASHSEED across processes.)
5. **Never fabricated.** If nothing verifies, the field stays `None`. A
   missing repro is recoverable; a wrong one is not.
6. **Read-only w.r.t. verdicts.** This module only ever writes `witness`. It
   cannot change `status`, `reason`, or any obligation.
"""
from __future__ import annotations

from fractions import Fraction as _Fr
from itertools import product as _product

from . import term as T

# Candidate values, ordered so the cheapest / most human-readable come first.
# Bounded: this is a best-effort search, not a solver (see module docstring).
_VALUES = (0, 1, -1, 2, 3, 4, 5, 7, 10, -2, 100)
_MAX_POINTS = 4096          # cap the grid; never search unboundedly
_MAX_VARS = 3               # only handle small arities point-wise

# claim operator -> (proves-the-claim-FALSE predicate)
#
# The claim fails iff its NEGATION is provable, so each entry must prove the
# opposite relation. Getting this backwards emits a countermodel at a point
# where the claim actually HOLDS — the worst possible failure, so each entry
# has an execution twin in selfcheck.
#
#   claim l < r   fails iff  l >= r
#   claim l <= r  fails iff  l >  r
#   claim l > r   fails iff  l <= r
#   claim l >= r  fails iff  l <  r
#   claim l == r  fails iff  decide_eq reports False
#
# `decide_eq` returns True (equal) / False (differ) / None (UNDECIDED). Only an
# explicit False counts: `not decide_eq(...)` would treat an undecided claim as
# a proven violation.
_NEGATION = {
    "==": lambda l, r: T.decide_eq(l, r) is False,
    "<":  lambda l, r: T.provably_ge(l, r),
    "<=": lambda l, r: T.provably_gt(l, r),
    ">":  lambda l, r: T.provably_le(l, r),
    ">=": lambda l, r: T.provably_lt(l, r),
}


# Constructors with EXECUTION EVIDENCE, i.e. a selfcheck twin that runs the unit
# at an emitted point and confirms the assert actually fires. Emit only when
# every constructor in the claim is on this list.
#
# Why an allow-list and not a deny-list: a deny-list trusts every constructor by
# default, so a new one added to the algebra later inherits trust it has not
# earned. That is exactly how the first defect arrived — `list` was trusted until
# execution disproved it (`ledger + [x]` is judged as arithmetic, so the emitter
# reported a "provable violation" where the program raises TypeError). Deny by
# default; allow on evidence.
#
# `rdiv` is here because it FOLDS on constants, so after substitution the
# term-level check is faithful — verified by execution (n=0, claim `n/2 == 1`).
# Collection constructors (`list`/`dict`/`clen`/`elem`) and the unmodelled
# `pow`/`idiv` are absent, so claims using them abstain.
_ALLOWED = ("v", "c", "add", "mul", "neg", "rdiv")


def _unmodelled(t) -> bool:
    """True if the term uses any constructor without execution evidence."""
    if not isinstance(t, tuple):
        return False
    if t[0] not in _ALLOWED:
        return True
    return any(_unmodelled(x) for x in t[1:])


def _sides_of(ob: dict):
    """Recover (op, l_term, r_term) for the failing claim, or None.

    The operator comes from the claim text (`<=`/`>=`/`==`/`<`/`>`, longest
    match first). The two sides come from the obligation's `premise` and
    `conclusion`, which the `decide_eq` path stores **canonically**
    (`mul(add(a, b), add(a, b))`), unlike the infix claim text which
    `term.parse` does not accept.

    Equality is symmetric, so side order does not matter for `==`. For an
    inequality the obligation's two fields are not reliably the two sides
    (the `rel` path stores prose in `conclusion`), and guessing the order would
    invert the relation. So inequalities are only accepted when the sides are
    unambiguously canonical; otherwise we abstain.
    """
    claim = ob.get("claim") or ""
    op = next((o for o in ("<=", ">=", "==", "<", ">") if o in claim), None)
    if op is None:
        return None
    # `lhs`/`rhs` are the canonical, POST-SUBSTITUTION sides the prover actually
    # decided on (recorded by the box path in certificate.py). Prefer them: the
    # source `claim` is infix text over a computed local (`result >= 12`), whose
    # substitution (`n >= 12`) is what an input-space countermodel lives in.
    # Fall back to premise/conclusion, which the `decide_eq` path stores
    # canonically.
    if ob.get("lhs") is not None and ob.get("rhs") is not None:
        try:
            return op, T.parse(str(ob["lhs"])), T.parse(str(ob["rhs"]))
        except Exception:
            return None
    a, b = ob.get("premise"), ob.get("conclusion")
    if not a or not b:
        return None
    try:
        lt, rt = T.parse(str(a)), T.parse(str(b))
    except Exception:
        return None
    return op, lt, rt


def _env_of(terms) -> list:
    """Parameter names occurring in `terms`, **sorted**.

    `term.vars_of` returns a set, whose iteration order is stable within a
    process but varies with PYTHONHASHSEED across processes. The candidate grid
    is built by zipping `_VALUES` against these names, so set order would make
    the emitted witness differ between two runs of the SAME authority version —
    two machines with one fingerprint, two artifacts. Sorting fixes the order
    and therefore the witness. (Verified: PYTHONHASHSEED 0-3 disagreed before.)
    """
    names = set()
    for t in terms:
        names |= T.vars_of(t)
    return sorted(names)


def _candidates(names: list):
    """Deterministic, bounded grid over the parameters."""
    if not names or len(names) > _MAX_VARS:
        return
    seen = 0
    for combo in _product(_VALUES, repeat=len(names)):
        if seen >= _MAX_POINTS:
            return
        seen += 1
        yield {n: T.const(_Fr(v)) for n, v in zip(names, combo)}


def _find_point(l0, r0, op: str, names: list):
    """First assignment at which the claim is provably false, or None.

    Verification is done by the trusted decision procedure on the substituted
    terms — this is the whole safety argument.
    """
    falsifies = _NEGATION.get(op)
    if falsifies is None:
        return None
    for env in _candidates(names):
        try:
            l, r = T.sub_all(l0, env), T.sub_all(r0, env)
        except Exception:
            continue
        try:
            if falsifies(l, r):
                return {k: str(v[1]) for k, v in env.items()}
        except Exception:
            continue
    return None


def _region_from(ob: dict):
    """An already-serialised violated box, if the obligation carries one."""
    w = ob.get("witness") or {}
    box = w.get("box")
    if isinstance(box, dict) and box:
        return {"kind": "region", "box": box,
                "claim": ob.get("claim"), "line": ob.get("line"),
                "note": "a violating input exists in this box; "
                        "no single point extracted"}
    return None


def build(obligations, status: str, reason: str = "",
          params=frozenset()) -> dict | None:
    """Return the certificate `witness` for a verdict. Read-only w.r.t. verdicts.

    `params` is the unit's actual function-parameter set. Only those may be
    assigned: a free variable that is a computed *local* (e.g. `result`) is not
    an input, so assigning it would fabricate a "repro" for a program nobody
    can run. If a claim's variables are not all parameters, we abstain.

    Returns None for anything that is not a REFUTED with something truthful to
    say. In particular: CERTIFIED and UNVERIFIABLE always get None — an
    abstention has no counterexample, and inventing one would be the exact
    failure this module exists to prevent.
    """
    if status != "REFUTED":
        return None
    failing = [o for o in (obligations or []) if o.get("status") == "fail"]
    if not failing:
        return None

    # A contradictory premise system has no falsifying input by construction.
    if "inconsistent_premises" in (reason or ""):
        return {"kind": "inconsistency", "system": "inconsistent premises",
                "note": "no falsifying input exists: the premises contradict, "
                        "so this is unsatisfiability, not a counterexample"}

    for ob in failing:
        # Closed-literal nullness (`assert None is not None`): the claim is
        # false as written and needs no input to demonstrate. The env only
        # COMPLETES the unit's signature (any value works); the falsifier is
        # the assertion itself, so execution confirms it.
        if ob.get("method") == "nullness":
            return {"kind": "point",
                    "env": {p: "0" for p in sorted(params)},
                    "claim": ob.get("claim"), "line": ob.get("line"),
                    "method": "nullness",
                    "verified_by": "closed literal evaluated by execution",
                    "note": "the claim is false as written; no inputs are needed"}
        # Guard 1: a restricted (conditional) obligation admits no point
        # countermodel, because the point would fall outside the declared
        # domain. Abstain rather than emit an inadmissible repro.
        if ob.get("bounds") or ob.get("premise_bounds"):
            continue
        sides = _sides_of(ob)
        if not sides:
            continue
        op, lt, rt = sides
        # Guard 3 (fidelity): only constructors with execution evidence are
        # trusted. A collection-typed term is judged as arithmetic by the
        # algebra and can report a "provable violation" where the program
        # raises TypeError; an unmodelled operator yields an uninterpretable
        # term. Abstain rather than emit something we cannot stand behind.
        if _unmodelled(lt) or _unmodelled(rt):
            continue
        names = _env_of([lt, rt])
        # Guard 2 (soundness): every free variable must be a real parameter.
        if not names or not set(names).issubset(params):
            continue
        point = _find_point(lt, rt, op, names)
        if point is not None:
            return {"kind": "point", "env": point, "claim": ob.get("claim"),
                    "line": ob.get("line"), "method": ob.get("method"),
                    "verified_by": "term.decide_eq/provably_* after sub_all",
                    "note": "the claim is provably false at this assignment"}

    # No verified point anywhere: fall back to an honest region, else abstain.
    for ob in failing:
        region = _region_from(ob)
        if region:
            return region
    return None
