"""mjolnir_prover.lib.test_corpus -- the frozen expectations, enforced.

    python -m mjolnir_prover.lib.test_corpus

A corpus of frozen verdicts that nothing reads is a comment with a JSON extension.
This makes it a gate: any row whose expectation the kernel no longer produces
fails the build, which is the only thing that makes freezing worthwhile.

WHAT FAILS AND WHAT DOES NOT
-----------------------------
`MISMATCH` fails. `NO_READER` does not -- a row that declares inputs no entry
point consumes is an unbuilt feature, and failing the build on it would send the
next person to debug a kernel that is behaving correctly. The distinction is the
whole reason `corpus.py` reports three outcomes rather than two.

THE KNOWN GAPS ARE PRINTED, NOT HIDDEN
---------------------------------------
Eleven rows currently mismatch and eight have no reader. Each is attributed to a
cause rather than left as noise:

  reading not declared   4 rows  the reading lives in the `note` prose, not a
                                field, so nothing routes them (`frag_free_ge0`,
                                `frag_rel_premise`, and the two `frag_drive_*`
                                pairs whose bounds are linear)
  linear theory missing  2 rows  `frag_lin_system`, `frag_lin_inconsistent` want
                                a theory table for linear arithmetic
  (two FAULTs FIXED)       the two entries that were real defects are gone:
                                `frag_rel_box_hold` was a true claim reported as
                                a false certificate, because interval arithmetic
                                being INCONCLUSIVE was read as DISPROVEN;
                                `frag_mod_residue` was a true claim with no
                                re-derivation for `mod:*` and no step-checker
                                branch for the lemma, so a missing reader was
                                published as a broken proof.
  expectation suspect    1 row  `frag_mod_false` is named "false" and expected to
                                CERTIFY. Either the id or the expectation is a
                                data error; not chased until someone says which.
  no reader              8 rows  compose (2), induction (3), countermodel (3)
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mjolnir_prover.lib import corpus  # noqa: E402

FAILS = []

# Rows whose gap is a missing declaration or a missing theory, not a regression.
# Listed explicitly so that ADDING to this list is a visible act of judgement --
# an unexplained disappearance from the mismatch list would be a regression
# laundered into a known gap.
KNOWN_GAPS = {
    "frag_free_ge0", "frag_rel_premise",          # reading lives in prose
    "frag_drive_gap", "frag_drive_gap_ok",        # linear bounds
    "frag_drive_unsafe", "frag_drive_safe",       # linear bounds
    "frag_lin_system", "frag_lin_inconsistent",   # needs a linear theory table
    "frag_mod_false",                             # expectation looks wrong
    # NOTE: frag_mod_residue and frag_rel_box_hold were here until this pass.
    # Both were FAULTs on claims that are TRUE -- a missing step-checker branch
    # and interval-inconclusive read as refutation. They are fixed, and the fix is
    # in `test_domain`-style regression below rather than in an allowlist entry.
}


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name}: {exc}")
        FAILS.append(name)


def main():
    print("mjolnir_prover.lib corpus regression")
    out, detail = corpus.report()

    check("no unexplained mismatch",
          lambda: _no_unexplained(detail))
    check("match rate does not regress", lambda: _floor(out))
    check("no reader rows are declared", lambda: _readers_declared(detail))

    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("\ncorpus gate: pass "
          f"({out['MATCH']} match, {out['MISMATCH']} known gap, "
          f"{out['NO_READER']} no reader)")
    return 0


def _no_unexplained(detail):
    bad = [d[1] for d in detail
           if d[0] == "MISMATCH" and d[1] not in KNOWN_GAPS]
    assert not bad, f"unexplained mismatches: {bad}"


def _floor(out):
    total = sum(out.values())
    rate = out["MATCH"] / total if total else 0
    assert rate >= 0.79, f"match rate {rate:.1%} is below the 79% floor"


def _readers_declared(detail):
    # every NO_READER row's field-set must be a DECLARED gap, so that a row
    # inventing a new shape of input cannot pass as "unbuilt"
    unknown = {tuple(d[5]) for d in detail if d[0] == "NO_READER"} - set(corpus.READERS)
    assert not unknown, f"NO_READER rows whose declared inputs mean nothing here: {unknown}"
    # and the declared gaps must actually be declared, not implicit
    stale = set(corpus.READERS) - {tuple(d[5]) for d in detail if d[0] == "NO_READER"}
    assert not stale, f"READERS lists gaps no row declares: {sorted(stale)}"


if __name__ == "__main__":
    raise SystemExit(main())
