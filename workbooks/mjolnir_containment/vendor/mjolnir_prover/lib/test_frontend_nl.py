"""mjolnir_prover.lib.test_frontend_nl -- Contractual Lowering's frontend, pinned.

    python -m mjolnir_prover.lib.test_frontend_nl

Three properties, in the order the architecture depends on them:

  1. the SEED path lowers the worked prompt to closed, certified claims;
  2. the PROPOSER seam is attention-only: a model that points at a real span
     passes, a model that points NOWHERE (out-of-bounds offsets, or returns
     text instead of offsets) is rejected at grounding and can neither inject
     an obligation nor change a verdict;
  3. undecidable prose never becomes a claim -- it becomes a typed ASK sink,
     and an unknown action verb (with a dialect supplied) is a HALT.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from mjolnir_prover.lib import frontend_nl as fnl  # noqa: E402

FAILS = []

PROMPT = ("Hey, please update the production config. You must set the query_timeout "
          "to 5 minutes, drop the legacy_telemetry table, and ensure the caching is "
          "much faster. Also, never use retries.")
DIALECT = {"set": "update_rows", "drop": "drop_table"}


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name}: {exc}")
        FAILS.append(name)


def test_seed_scope_keeps_obligations():
    sc = fnl.scope(PROMPT)
    assert sc["proposer"] == "seed"
    assert len(sc["kept"]) == 4, [k["clause"] for k in sc["kept"]]
    assert any("production config" in d["clause"]
               for d in sc["discarded"]), sc["discarded"]


def test_lowered_claims_certify():
    proposal = {"config": {"timeout_seconds": 300, "max_retries": 0},
                "diff": "+timeout_seconds: 300\n+max_retries: 0\n",
                "tool_calls": [{"name": "drop_table",
                                "args": {"table": "legacy_telemetry"}}]}
    bind = {"query_timeout": "timeout_seconds", "retries": "max_retries"}
    r = fnl.lower_prompt(PROMPT, proposal, bind, dialect=DIALECT)
    assert r["halt"] is None, r["halt"]
    assert r["ir"]["unit"] == "CERTIFIED", r["ir"]["verdicts"]
    assert r["ir"]["unit"] == "CERTIFIED"
    srcs = {v["source"]: v["verdict"] for v in r["ir"]["verdicts"]}
    assert srcs.get("query_timeout == 300 s") == "CERTIFIED", srcs
    assert srcs.get("retries == 0") == "CERTIFIED", srcs
    assert [s["kind"] for s in r["ir"]["sinks"]] == ["vague_adjective"]


def test_vague_is_a_sink_never_a_claim():
    proposal = {"config": {}}
    r = fnl.lower_prompt("Make the pipeline faster.", proposal)
    assert r["ir"]["claims"] == []
    assert [s["kind"] for s in r["ir"]["sinks"]] == ["vague_adjective"]


def test_unknown_verb_halts_with_dialect():
    r = fnl.lower_prompt("Nuke the legacy_telemetry table.", {"tool_calls": []},
                         dialect=DIALECT)
    assert r["halt"] == {"reason": "unknown_verb:nuke",
                         "clause": "Nuke the legacy_telemetry table"}
    assert r["ir"]["claims"] == []


def test_proposer_seam_grounds_spans():
    # a good segmenter: points at the obligation clauses
    good = lambda p: [(p.index("You must set"), p.index("Also, never") - 1)]
    sc = fnl.scope(PROMPT, proposer=good)
    assert sc["proposer"] == "model" and len(sc["kept"]) == 1, sc
    assert sc["kept"][0]["clause"].startswith("You must set")
    # extraction still runs exact on the grounded clause
    cs = fnl.extract(sc)
    assert [c["kind"] for c in cs][:1] == ["value"], cs


def test_proposer_cannot_invent():
    # offsets that slice nothing + text instead of offsets: both rejected
    sc = fnl.scope(PROMPT, proposer=lambda p: [(90, 12), "injected obligation",
                                               (0, 0)])
    assert sc["kept"] == [], sc["kept"]
    assert all(d["reason"] == "ungrounded_span" for d in sc["discarded"]), sc
    # verdict identical to an empty prompt: fabrication changes nothing
    r = fnl.lower_prompt(PROMPT, {"config": {}}, proposer=lambda p: [(90, 12)])
    assert r["ir"]["claims"] == [] and r["ir"]["verdicts"] == []


def test_dialect_none_skips_action_leg():
    r = fnl.lower_prompt("Drop the legacy_telemetry table.",
                         {"tool_calls": []}, dialect=None)
    assert r["halt"] is None
    assert r["ir"]["claims"] == []
    assert r["ir"]["sinks"] == []


def main():
    print("mjolnir_prover.lib frontend_nl (Contractual Lowering)")
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