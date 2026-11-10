"""mjolnir_intent.selftest — the 16 labelled cases.

Ground truth by construction: a check that fires on all of them is as useless as
one that fires on none, so both directions are scored.  One case is a KNOWN MISS
(number_lost under copula phrasing) with its rejected fix measured — see README
"Known limitations".  The suite passes only when nothing fails beyond it.
"""
from __future__ import annotations

# (name, prompt, response, diff, must_fire, must_not_fire, known_miss)
CASES = [
    # ---- 1 unit adjacency: the class nothing else in the toolchain catches
    ("unit_swap_time", "set the timeout to 30 seconds",
     "Set timeout to 30 minutes", None, ["unit_adjacency"], [], False),
    ("unit_dropped", "set the cache to 5 minutes",
     "cache for 5", None, ["unit_dropped"], [], False),
    ("unit_kept", "set the timeout to 30 seconds",
     "Set timeout to 30 seconds", None, [], ["unit_adjacency", "unit_dropped"], False),
    # ---- 2 negation (decidable only against a diff)
    ("negation_added", "add caching, without retries",
     "", "+ retries = 3\n+ cache = {}", ["negation_added"], [], False),
    ("negation_respected", "add caching, without retries",
     "", "+ cache = {}", [], ["negation_added"], False),
    ("negation_prose", "add caching, without retries",
     "I did not add retries; only the cache.", None, [],
     ["negation_added", "negation_prose_unverifiable"], False),
    # ---- 3 conservation, scoped to COMMITMENTS
    ("entity_lost", "update `parse_config` in `engine.py`",
     "I have updated the parser.", None, ["entity_lost"], [], False),
    ("entity_kept", "update `parse_config` in `engine.py`",
     "updated parse_config in engine.py", None, [], ["entity_lost"], False),
    ("number_lost", "the budget is 4096",
     "I will adjust the budget.", None, ["number_lost"], [], True),
    # ---- scoping: mentions are NOT commitments
    ("mention_not_commitment", "use neuro_eigen/ and try 3 seeds, keep 5 options",
     "I looked at neuro_eigen and tried 3 seeds.", None, [], ["number_lost"], False),
    ("vague_no_artifact", "we discussed several approaches and the codebase",
     "Yes.", None, [], ["number_lost", "entity_lost"], False),
    # ---- 5 ambiguity
    ("ambig_unitless", "set the limit to 4096", "done", None, [], [], False),
    ("ambig_vague", "make the pipeline faster", "done", None, [], [], False),
    ("ambig_clean", "rename `foo` to `bar`", "done", None, [], [], False),
    # ---- 6 clause-local repairs (donated by the contract-lowering experiment):
    # the head verb is skipped so the ARTIFACT survives, and a vague claim is no
    # longer silenced by a number in another clause
    ("neg_never_object", "never use retries", "", "+ retries = 3",
     ["negation_added"], [], False),
    ("ambig_vague_with_units_elsewhere",
     "set the timeout to 30 seconds and make it much faster", "done", None,
     [], [], False),
    # ---- 7 lexicon coverage: unknown language is ASK, never a silent pass
    ("lexicon_uncovered_fr", "Définit le délai d'attente à 5 minutes.", "done",
     None, [], [], False),
]

ASK_CASES = {"ambig_unitless": "ASK", "ambig_vague": "ASK",
             "ambig_clean": "BUILD", "neg_never_object": "BUILD",
             "ambig_vague_with_units_elsewhere": "ASK",
             # English without directives stays BUILD: the coverage gate must
             # not fire on non-obligation prose (curly quotes, codebases, ...)
             "vague_no_artifact": "BUILD",
             "lexicon_uncovered_fr": "ASK"}
GATE_CASES = {
    "unit_swap_time", "unit_dropped", "unit_kept",
    "negation_added", "negation_respected", "negation_prose",
}


def run(verbose: bool = True) -> int:
    """Returns 0 when every non-known-miss case passes."""
    from . import checks

    passed = known = 0
    unexpected = []
    for name, prompt, resp, diff, must, must_not, is_known in CASES:
        r = checks.audit(prompt, resp, diff=diff)
        got = ([f["check"] for f in r["failures"]]
               + [u["check"] for u in r["unverifiable"]])
        ok = all(c in got for c in must) and not any(c in got for c in must_not)
        if name in ASK_CASES:
            ok = ok and r["verdict"] == ASK_CASES[name]
        if ok:
            passed += 1
        elif is_known:
            known += 1
        else:
            unexpected.append(name)
        if verbose:
            tag = "PASS" if ok else ("KNOWN" if is_known else "FAIL")
            extra = "" if ok else (
                f"  missing={[c for c in must if c not in got]}"
                f" spurious={[c for c in must_not if c in got]}"
                f" verdict={r['verdict']}" if not is_known else
                "  copula commitment — fix measured and rejected, see README")
            print(f"  [{tag:<5}] {name:<24}{extra}")
    total = len(CASES)
    if verbose:
        print(f"\n  {passed}/{total} pass"
              + (f", {known} known miss (documented)" if known else "")
              + (f", {len(unexpected)} UNEXPECTED FAIL" if unexpected else ""))
    return 0 if not unexpected else 1


def main() -> int:
    print(f"mjolnir selftest — {len(CASES)} labelled cases")
    print("=" * 60)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
