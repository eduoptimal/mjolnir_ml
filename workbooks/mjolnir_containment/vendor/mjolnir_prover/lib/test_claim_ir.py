"""mjolnir_prover.lib.test_claim_ir -- regressions for the claim IR disposal.

    python -m mjolnir_prover.lib.test_claim_ir

THE DEFECT THIS PINS
--------------------
`_mod_discharge` exists to discharge `expr % m OP const` by residue constancy.
Its second branch swapped the operands to bring a `mod` node to the left:

    if l[0] == "c" and _int(l[1]) is not None:
        return _mod_discharge(r, l, cong)

When BOTH sides were integer constants the swap was a no-op, so a closed claim
`eq(const(300), const(300))` recursed until the interpreter stack died:
`RecursionError` on a perfectly well-formed claim through the public
`dispose_claims` API. A frontend that folds a theorem to constants -- or any
lowering of a closed goal, which is exactly what Contractual Lowering produces --
hits it. The guard requires the swap to make progress; both-constant claims fall
through to the exact constant comparison, so `300 == 300` CERTIFIES and a true
mismatch (`3000 == 300`) REFUTES soundly instead of crashing.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mjolnir_prover.lib import claim_ir, term as T  # noqa: E402

FAILS = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name}: {exc}")
        FAILS.append(name)


def _closed(a, b):
    return [claim_ir.eq(T.const(a), T.const(b), line=1, is_premise=False,
                        source=f"{a} == {b}")]


def test_constant_eq_certifies_not_recurses():
    unit, results, _ = claim_ir.dispose_claims(_closed(300, 300))
    assert unit == "CERTIFIED", (unit, results)


def test_constant_mismatch_refutes_soundly():
    unit, results, _ = claim_ir.dispose_claims(_closed(3000, 300))
    assert unit == "REFUTED", (unit, results)


def test_zero_eq_and_mismatch():
    assert claim_ir.dispose_claims(_closed(0, 0))[0] == "CERTIFIED"
    assert claim_ir.dispose_claims(_closed(0, 3))[0] == "REFUTED"


def test_string_constants_certify_only():
    same = [claim_ir.eq(T.const("DROP(a)"), T.const("DROP(a)"), line=1,
                        is_premise=False, source="s")]
    assert claim_ir.dispose_claims(same)[0] == "CERTIFIED"
    diff = [claim_ir.eq(T.const("DROP(a)"), T.const("DROP(b)"), line=1,
                        is_premise=False, source="s")]
    # Inert constants have exact equality, but the polynomial path cannot turn a
    # string mismatch into a refutation: abstain, never guess.
    assert claim_ir.dispose_claims(diff)[0] == "UNVERIFIABLE"


def test_mod_discharge_still_works():
    # n%3 == 2 as a premise, (3n-7)%3 == 2 as a conclusion: the rule the branch
    # above protects. Regression cover for the guard, not just the crash.
    claims = [
        claim_ir.eq(T.cmod(T.var("n"), T.const(3)), T.const(2), line=1,
                    is_premise=True, source="n%3 = 2"),
        claim_ir.eq(T.cmod(T.csub(T.cmul(T.const(3), T.var("n")), T.const(7)),
                           T.const(3)), T.const(2), line=2, is_premise=False,
                    source="(3n-7)%3 = 2"),
    ]
    unit, results, _ = claim_ir.dispose_claims(claims)
    assert unit == "CERTIFIED", (unit, results)


def main():
    print("mjolnir_prover.lib claim-IR disposal regressions")
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            check(n, f)
    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
