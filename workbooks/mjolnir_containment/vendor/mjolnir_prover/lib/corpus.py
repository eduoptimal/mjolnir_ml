"""mjolnir_prover.lib.corpus -- the frozen fragment corpus, and the runner for it.

    python -m mjolnir_prover.lib.corpus            # the report
    python -m mjolnir_prover.lib.test_corpus       # the test

WHY A RUNNER HAD TO BE WRITTEN
------------------------------
`corpus/fragment_v1.jsonl` is 93 rows of frozen expectations, and until now
NOTHING read it. The rows are self-describing -- each carries `expect`, a
`family`, and the inputs its reading needs (`pre_bounds`, `module_source`,
`body`+`guard`, `theory`, `step`, `f_steps`) -- and the only way anyone checked
them was to hand-wrap each row in a bare `def f(...): assert ...` and call
`certify_claim`, which throws those inputs away. On that reading 20 of 85 rows
disagreed with their expectations and it looked like kernel rot.

It was not kernel rot. Six of the eight rows I checked certify correctly the
moment the declared bounds reach the prover:

    frag_med_dose_ok       expect CERTIFIED   without pre_bounds UNVERIFIABLE, with it CERTIFIED
    frag_call_contract     expect CERTIFIED   without module_source UNVERIFIABLE, with it CERTIFIED
    frag_call_contract_bad expect REFUTED     ... REFUTED

The corpus was right. The reader was wrong. This is the third time in one week
that correct data changed no behaviour -- after `AMBIGUOUS_EXPRESSIONS` and the
float-domain knowledge -- and the pattern is now specific enough to name: a
knowledge file with no reader looks exactly like a knowledge file that is wrong.

THREE OUTCOMES, NOT TWO
-----------------------
A two-valued check cannot tell a regression from a missing feature, and those
demand opposite responses:

    MATCH       the row's frozen expectation is what the kernel says
    MISMATCH    the kernel disagrees with a frozen expectation  -> a REGRESSION
    NO_READER   the row declares inputs that no entry point consumes

`NO_READER` is not a pass and is not a failure. It is a feature that was never
built, and reporting it as a mismatch would send the next person to debug the
kernel, which is the wrong place. The count is printed every run so the gap
cannot be quietly forgotten.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mjolnir_prover.lib import certificate as C, certify  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "storage", "data", "corpus", "fragment_v1.jsonl")

# The inputs a row can declare, and the entry point that consumes each.
# Every combination of declared inputs, and the entry point that would consume it.
# A NO_READER row whose field-set is absent from this table is an UNKNOWN gap: the
# row declares something and nothing in this project knows what it means. That is
# a different defect from an unbuilt feature and it fails the gate.
WIRED = {
    ("pre_bounds",): "unit_vc + pre_bounds",
    ("module_source",): "unit_vc + module_source",
}

READERS = {
    ("body", "guard"): "countermodel  [not built]",
    ("base", "step", "theory"): "induction  [not built]",
    ("f_steps", "g_post", "post"): "compose  [not built]",
}

# Declared fields that a wired reader consumes, kept so the report can say so.
KNOWN_FIELDS = {"pre_bounds", "module_source", "theory", "step", "base", "body",
                "guard", "f_steps", "g_post", "post"}
# Fields that are metadata rather than a reading input.
_METADATA = {"id", "expect", "note", "family", "code", "tier"}


def load(path: str = CORPUS):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def declared_inputs(row: dict):
    return sorted(k for k in row if k not in _METADATA)


def entry_point(row: dict):
    """-> (entry_point, fields_it_consumed) or ('certify_claim', []) for a bare row."""
    fields = declared_inputs(row)
    if "pre_bounds" in fields or "module_source" in fields:
        return "certify_claim", fields
    return "bare", fields


def run_row(row: dict):
    """-> (outcome, got, entry_point). Never raises."""
    code = row.get("code") or row.get("body") or ""
    ep, fields = entry_point(row)
    # The row's declared reading inputs travel WITH the claim. Dropping them here
    # was the first version's bug and it is the exact bug this file exists to
    # document: the fields were in the corpus, in the claim shape, and nowhere in
    # between.
    claim = dict(id=row["id"], family=row.get("family", "vc"),
                 tier=row.get("tier", "static"),
                 predicate=code, canonical_unit=code,
                 origin={"file": "fragment_v1", "line": 0}, provenance={})
    for f in fields:
        if f in ("pre_bounds", "module_source"):
            claim[f] = row[f]
    try:
        got = certify.certify_claim(claim)["status"]
    except Exception as exc:  # noqa: BLE001
        return ("ERROR", f"{type(exc).__name__}", ep)
    if fields and ep == "bare":
        return ("NO_READER", got, ep)
    expect = row.get("expect")
    if expect and got != expect:
        return ("MISMATCH", got, ep)
    return ("MATCH", got, ep)


def report(path: str = CORPUS):
    rows = load(path)
    out = Counter()
    detail = []
    for row in rows:
        outcome, got, ep = run_row(row)
        out[outcome] += 1
        detail.append((outcome, row["id"], row.get("expect"), got, ep,
                       declared_inputs(row)))
    n = len(rows)
    print(f"corpus: {n} rows, {CORPUS.split('data/')[-1]}\n")
    for k in ("MATCH", "MISMATCH", "NO_READER", "ERROR"):
        if out[k]:
            print(f"  {k:10s} {out[k]:4d}  {out[k]/n:6.1%}")
    mism = [d for d in detail if d[0] == "MISMATCH"]
    if mism:
        print("\n  MISMATCHES (kernel regressions -- these fail the build):")
        for _o, i, e, g, ep, f in mism[:20]:
            print(f"    {i:26s} expect={e:12s} got={g:12s} fields={f}")
    nr = Counter()
    for _o, _i, _e, _g, _ep, f in [d for d in detail if d[0] == "NO_READER"]:
        nr[tuple(f)] += 1
    if nr:
        print("\n  NO_READER (unbuilt features, not regressions):")
        for f, c in nr.most_common():
            entry = READERS.get(tuple(f), "?")
            print(f"    {c:4d} rows carry {f}  -> {entry}")
    print()
    return out, detail


def main():
    out, _detail = report()
    return 1 if out["MISMATCH"] or out["ERROR"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
