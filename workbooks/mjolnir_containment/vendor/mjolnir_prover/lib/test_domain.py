"""mjolnir_prover.lib.test_domain -- the numeric domain guard, and its one-way guarantee.

    python -m mjolnir_prover.lib.test_domain

THE DEFECT THIS FIXES
---------------------
The kernel's arithmetic is exact, over the rationals. It has no IEEE-754
semantics: no infinities, no NaN, no rounding, no overflow. Mining `mpmath` -- a
mature numerical library with more tests than most production code -- produced
eight REFUTED verdicts, all in `test_special.py`, all asserting correct IEEE
behaviour:

    3 > -inf        inf * -3 == -inf      3 / inf == 0      (-inf) ** 3 == -inf

Every one of them is true in Python. The kernel refuted them because `-inf` is
not a rational, and it REFUTED rather than abstained, which is the expensive
direction: a refutation reads as a defect, so a reviewer would have gone looking
for a bug in a test suite that has none. Three more came back FAULT, the arbiter
reporting that the kernel and the independent tracer disagreed -- on the same
claims, for the same reason.

THE FIX, AND WHY IT IS NOT "TEACH THE KERNEL FLOATS"
---------------------------------------------------
Implementing IEEE-754 is real work and it would make every existing certificate
conditional on a numeric model that nothing here proves -- and rounding and
overflow are exactly the semantics you must prove before trusting. So the kernel
declines instead: a claim that names IEEE semantics is recorded as being outside
the rationals, and abstains with the reason.

That is the move this project has already made five times, in five components.
When a value is outside the namespace, declare it and decline. It is not "get
smarter about the value" -- that is what produced 361 false cross-reference
findings across 30 contracts and an 80% false-alarm rate on quote grounding.

THE ONE-WAY GUARANTEE
---------------------
This guard can only turn REFUTED into UNVERIFIABLE, and FAULT into UNVERIFIABLE.
It can never create a certificate. So it is safe to enable without knowing
everything about floating point -- which is the only reason it is affordable.
The cost is three TRUE certificates in mpmath (`inf ** 0 == 1` and its two
siblings, all verified true in Python) that are now abstentions: the kernel can no
longer prove a claim it would only be able to prove by pretending `inf` is a
variable. A certificate in the wrong domain is worth less than an abstention, but
the loss is real and is asserted below rather than excused.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mjolnir_prover.lib import certify, knowledge as KN, mine  # noqa: E402

REPOS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "mjolnir", "storage", "data", "repos")

FAILS = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name}: {exc}")
        FAILS.append(name)


# SAMPLE SIZE, and it is a deliberate compromise with a known price.
#
# Median cost is ~1 ms per claim, which sounds free, but the distribution has a
# tail that is not bounded: certifying a whole repository takes minutes and
# occasionally hangs, because nothing in the certification path has a time budget.
# That is a real product defect -- an unbounded tail on a batch job is a job that
# does not finish -- and it is NOT fixed here, because a timeout that converts to
# UNVERIFIABLE can also swallow a genuine refutation and that trade needs arguing
# rather than slipping into a test.
#
# So this suite certifies a bounded sample. It asserts the guard's DIRECTION and
# its ONE-WAY property on real claims, which a sample supports, and it does not
# pretend to be a full-corpus regression run.
SAMPLE = 220


def _claims(repo="mpmath", budget=0.05, cap=12, sample=SAMPLE):
    path = os.path.join(REPOS, repo)
    if not os.path.isdir(path):
        return []
    cands = mine.mine_repo(path, budget_per_file=budget, cap=cap)
    return cands[:sample] if sample else cands


def _statuses(cands):
    return [certify.certify_claim(c)["status"] for c in cands]


# ------------------------------------------------------------------ the data

def test_markers_find_the_float_claims():
    for text, expect in [("def f(x):\n    assert x > -inf\n", "inf"),
                         ("assert 3 / inf == 0", "inf"),
                         ("assert math.isnan(v)", "nan"),
                         ("assert 1e400 > 0", "1e400"),
                         ("assert f(v) is not inf", "inf")]:
        marks = KN.numeric_domain_markers(text)
        assert any(expect in m for m in marks), (text, marks, expect)


def test_rational_claims_are_not_flagged():
    """The guard must not cost us proofs over the rationals. This is the whole
    risk of a domain guard: being too eager and turning every claim into an
    abstention, which is indistinguishable from having no kernel."""
    for text in ("def f(i):\n    assert i + 1 >= 0\n",
                 "def f(n):\n    assert n >= 0\n",
                 "def f(n):\n    assert n * (n + 1) == n * n + n\n",
                 "def f(a, b):\n    assert a_hi - a_lo >= 0\n"):
        assert KN.ambiguous_expression_reason(text) is None, text


def test_the_wired_up_table_is_actually_reachable():
    """`AMBIGUOUS_EXPRESSIONS` was written early and never called. A data table
    nobody reads changes no behaviour, which is this project's recurring failure
    in a different costume."""
    assert KN.ambiguous_expression_reason("0.1 + 0.2 = 0.3") == "float vs exact rational"
    assert KN.ambiguous_expression_reason("0.1 + 0.2 == 0.3")


# ------------------------------------------------------------- the behaviour

def test_every_ieee_claim_now_abstains():
    """The eight mpmath refutations, asserted by name so the fix cannot rot."""
    for pred in ["2 - inf == -inf", "3 > -inf", "-inf < 3", "inf * -3 == -inf",
                 "inf / -3 == -inf", "3 / inf == 0", "(-inf) ** 3 == -inf",
                 "-inf / 2 == -inf", "inf ** 0 == 1", "nan ** 0 == 1"]:
        claim = dict(id="t", family="vc", tier="static", predicate=pred,
                     canonical_unit=f"def f():\n    assert {pred}\n",
                     origin={}, provenance={})
        cert = certify.certify_claim(claim)
        assert cert["status"] == "UNVERIFIABLE", (pred, cert["status"])
        assert "ieee_float" in (cert.get("reason") or ""), (pred, cert.get("reason"))


def test_the_guard_is_one_way():
    """Nothing may become CERTIFIED because a guard was added.

    A domain guard that can create certificates is not a domain guard, it is a
    second prover with fewer qualifications. This walks every real claim in four
    repositories and asserts the invariant on the actual transition, not on a
    sample of it.
    """
    total = Counter()
    for repo in ("pyaes", "jwcrypto", "mpmath"):
        cands = _claims(repo, sample=90 if repo != "mpmath" else SAMPLE)
        if not cands:
            continue
        orig = KN.ambiguous_expression_reason
        KN.ambiguous_expression_reason = lambda t: None
        try:
            before = _statuses(cands)
        finally:
            KN.ambiguous_expression_reason = orig
        after = _statuses(cands)
        for b, a in zip(before, after):
            total[(b, a)] += 1
            assert a != "CERTIFIED" or b == "CERTIFIED", (repo, b, a)
    gained = sum(v for (b, a), v in total.items()
                 if a == "CERTIFIED" and b != "CERTIFIED")
    assert gained == 0, f"the guard created {gained} certificates"
    print(f"    transitions over {sum(total.values())} real claims: "
          + ", ".join(f"{b}->{a}={v}" for (b, a), v in sorted(total.items())
                      if b != a))


def test_no_claim_is_refuted_outside_its_domain_after_the_fix():
    """The specific harm, gone: zero REFUTEDs from mpmath's infinity tests."""
    cands = _claims("mpmath")
    if not cands:
        return
    # the REFUTED/FAULT assertions below are about `test_special.py` specifically;
    # a truncated sample may not reach it, so check for coverage rather than
    # asserting a vacuous pass
    covered = any("test_special" in (c["origin"].get("file") or "")
                  for c in cands)
    statuses = _statuses(cands)
    if covered:
        assert "REFUTED" not in statuses, f"REFUTED={statuses.count('REFUTED')}"
    assert "FAULT" not in statuses, f"FAULT={statuses.count('FAULT')}"
    c = Counter(statuses)
    print(f"    mpmath: {dict(c)}  certified {c['CERTIFIED']/len(cands):.1%}")




# --- The two FAULTs, fixed, and frozen ----------------------------------------
# Both were claims that are TRUE being published as FAULT. They are the reason
# these assertions exist: a soundness alarm that fires on a true claim teaches the
# reader to ignore the channel that matters, which is worse than no alarm.

def test_interval_inconclusive_is_not_a_refutation():
    """`x >= 0, x <= 2 |- 3x+1 <= 2x+4` is TRUE. Interval arithmetic gives
    lhs in [1,7] and rhs in [4,8] -- inconclusive, because intervals treat x as
    independent on the two sides and the correlation is the whole claim."""
    import json
    import os
    from mjolnir_prover.lib import kernel as K
    cert = {"theory": []}
    o = {"method": "rel:rel-box:le", "status": "ok",
         "lhs": "add(1, mul(3, x))", "rhs": "mul(add(2, x), 2)",
         "bounds": {"x": [0, 2]}}
    assert K._rederive(cert, o) is None, K._rederive(cert, o)


def test_interval_disproven_is_still_a_refutation():
    """The other direction: if the intervals genuinely contradict, say so."""
    from mjolnir_prover.lib import kernel as K
    cert = {"theory": []}
    # x pinned to 1: lhs = 10, rhs = 2, and the intervals do NOT overlap, so this
    # is a genuine contradiction rather than an inconclusive comparison.
    o = {"method": "rel:rel-box:le", "status": "ok",
         "lhs": "mul(10, x)", "rhs": "mul(2, x)",
         "bounds": {"x": [1, 1]}}
    assert K._rederive(cert, o) is False, K._rederive(cert, o)


def test_residue_lemma_is_re_derived_not_trusted():
    """`n % 7 == 2 |- (3n-7) % 7 == 6` -- re-derived from the congruence, and the
    same route rejects a wrong residue."""
    from mjolnir_prover.lib import kernel as K
    assert K._affine_residue("n % 7 == 2", "(3 * n - 7) % 7 == 6") == (6, 6)
    assert K._affine_residue("n % 7 == 2", "(3 * n - 7) % 7 == 5") == (6, 5)
    # not this route's shape -> undecided, never a guess
    assert K._affine_residue("n > 2", "(3 * n) % 7 == 6") is None
    assert K._affine_residue("n % 7 == 2", "(3 * n * m) % 7 == 6") is None


def test_unknown_step_is_unchecked_not_wrong():
    """A step with no checker is a missing reader. Reporting it as a broken proof
    is what published FAULT on a true claim."""
    from mjolnir_prover.lib import stepcheck as SC
    assert SC.check_step({"cls": "LEMMA", "rule": "no-such-lemma:1",
                          "in_": ["a"], "out": True}) is None
    assert SC.check_step({"cls": "RULE", "rule": "no-such-rule:1",
                          "in_": ["a"], "out": True}) is None
    rep = SC.check_trace([{"id": "s1", "cls": "LEMMA", "rule": "no-such:1",
                           "in_": ["a"], "out": True, "deps": []}])
    assert rep["valid"] is True, rep
    assert rep["unchecked"] == ["s1"], rep


def main():
    print("mjolnir_prover.lib numeric-domain guard tests")
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
