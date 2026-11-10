"""mjolnir_prover.lib.compose — Suite B, composition over grounded facts.

A single-hop claim is a document fact. A composed claim needs two of them:

    TOTAL = 4200        fact        RATE = 30        fact
    assert 4200 == TOTAL                          <- single hop
    assert 1260 == TOTAL * RATE // 100            <- two hops

MEASURED (`tactics/suite_b.py`): 40/40 discrimination on both shapes, and the
decisive property — a chain over a document that DISAGREES **refutes at the
broken hop** rather than accepting the composed total. Composition does not paper
over a bad input.

SCOPE, and it is deliberate: INTRA-function only. Cross-function composition is
genuinely blocked — it needs `module_source` exposed in `certify_unit`, and today
`assert total() == 4200` is UNVERIFIABLE because each function is seen alone.
That is recorded as UNVERIFIABLE rather than worked around.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout


def dispose_single(quantity: str, asserted, stated) -> str:
    """One grounded fact. The SOURCE states `stated`; the CLAIM asserts
    `asserted`. They agree -> CERTIFIED; they do not -> REFUTED."""
    from . import certify
    src = (f"def f():\n"
           f"    {quantity} = {stated}\n"
           f"    assert {quantity} == {asserted}\n")
    with redirect_stdout(io.StringIO()):
        cert, _, _ = certify.certify_unit("f", src, 1, "compose.py")
    return cert["status"]


def dispose_relation(quantity: str, asserted, stated_list) -> tuple[str, str | None]:
    """A RELATION-valued source: one referent, SEVERAL stated values.

    Returns `(verdict, evidence)`. The policy is `theory.relation_verdict`; this
    function only measures -- it counts how many stated values the claim matches
    and reports which one matched. The matched value becomes the evidence, so
    the certified branch is re-derivable by the kernel as the single-value
    proposition it already verifies.

    A claim matching none is NOT refuted. Refuting over a relation is a
    disjunction of inequalities (`q != v1 and q != v2`), no rung decides it, and
    a refutation nothing can re-derive is exactly the false-refutation failure
    this project exists to not have. It abstains, and says why.
    """
    from . import theory
    matched = None
    n = 0
    for stated in stated_list:
        if dispose_single(quantity, asserted, stated) == "CERTIFIED":
            n += 1
            if matched is None:
                matched = stated
    verdict, _reason = theory.relation_verdict(n_stated=len(stated_list),
                                               n_matched=n)
    if verdict == "CERTIFIED":
        return verdict, str(matched)
    return verdict, None


def dispose_composed(a_q, a_v, a_stated, b_q, b_v, b_stated) -> str:
    """Two grounded facts composed through a derived value.

    Every intermediate is bound in the same body, so no `module_source` is
    needed and no value is invented.
    """
    from . import certify
    prod = int(a_stated) * int(b_stated) // 100
    src = (f"def f():\n"
           f"    {a_q} = {a_stated}\n"
           f"    {b_q} = {b_stated}\n"
           f"    assert {a_q} * {b_q} // 100 == {int(a_v) * int(b_v) // 100}\n"
           f"    assert {a_q} + {a_q} * {b_q} // 100 == "
           f"{int(a_v) + int(a_v) * int(b_v) // 100}\n")
    with redirect_stdout(io.StringIO()):
        cert, _, _ = certify.certify_unit("f", src, 1, "compose.py")
    return cert["status"]


def cross_function_status() -> str:
    """The blocked case, reported rather than worked around."""
    from . import certify
    src = "def total():\n    return 4200\n\ndef f():\n    assert total() == 4200\n"
    with redirect_stdout(io.StringIO()):
        cert, _, _ = certify.certify_unit("f", src, 1, "compose.py")
    return cert["status"]
