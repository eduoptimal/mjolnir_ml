"""Tests for mjolnir_decide.batch — state splitting + joint-AND aggregation.

Offline: the stub model stands in for the checkpoint (selftest convention),
so these run everywhere the package imports.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mjolnir_decide.batch import answer_batch, split_state  # noqa: E402
from mjolnir_decide.contract import build_response, parse_request  # noqa: E402
from mjolnir_decide.selftest import EXAMPLE  # noqa: E402

TWO_CALL_STATE = (
    "Task: quarterly cleanup.\n"
    'Proposed tool call: delete_rows(table="customers", '
    'where="last_login < 2023-01-01")\n'
    'Proposed tool call: refund(order_id="W1", amount=12.00)\n'
    "Context: no backup was taken today.")


class _Stub:
    """Scores by marker: the delete call is unsafe, the refund safe."""

    def __init__(self):
        self.calls = []

    def explain(self, state):
        self.calls.append(state)
        unsafe = "delete_rows" in state
        return {"noul": 0.02 if unsafe else 0.97,
                "in_regime": "gibberish" not in state,
                "flow_score": -80.0 if unsafe else -1.0}

    def input_tokens(self, state):
        return 100

    def provenance(self):
        from mjolnir_decide.thresholds import declaration
        return {"checkpoint": {"name": "stub.pt", "digest": "sha256:stub"},
                "base_model": "stub", "head": "flow", "floor": declaration(),
                "blind_spot": {"prose_hazards": "unread",
                               "behaviour": "fail-closed"}}


def _req(state):
    return {"model": EXAMPLE["model"], "state": state,
            "questions": EXAMPLE["questions"]}


def test_split_state_single_is_identity():
    s = EXAMPLE["state"]
    assert split_state(s) == [s]


def test_split_state_two_calls_keeps_task_and_context():
    states = split_state(TWO_CALL_STATE)
    assert len(states) == 2
    for s in states:
        assert s.startswith("Task: quarterly cleanup.")
        assert "no backup was taken today." in s
    assert "delete_rows" in states[0] and "refund" not in states[0]
    assert "refund" in states[1] and "delete_rows" not in states[1]


def test_single_call_certifies_the_delete():
    m = _Stub()
    resp, exs = answer_batch(m, parse_request(_req(EXAMPLE["state"])),
                             build_response, m.provenance)
    assert len(exs) == 1
    assert "batch" not in resp["mjolnir"]
    cert = resp["mjolnir"]["certification"]
    assert cert["noul_head"] == 0.02                     # the stub's estimate, kept
    assert resp["answers"]["safe_to_run"]["noul"] == 0   # narrowed by the rule
    assert [v["rule"] for v in cert["rule_vetoes"]] == ["destructive_without_backup"]


def test_batch_is_min_not_product():
    m = _Stub()
    resp, exs = answer_batch(m, parse_request(_req(TWO_CALL_STATE)),
                             build_response, m.provenance)
    assert len(exs) == 2
    steps = resp["mjolnir"]["batch"]["steps"]
    assert [s["noul"] for s in steps] == [0, 0.97]   # step 0: delete + no backup -> veto
    assert [s["noul_head"] for s in steps] == [0.02, 0.97]
    assert resp["mjolnir"]["batch"]["n_steps"] == 2
    cert = resp["mjolnir"]["certification"]
    assert cert["noul_head"] == 0.02                 # min of the raw heads, not 0.0194
    assert resp["answers"]["safe_to_run"]["noul"] == 0   # min of the certified, never partial
    assert cert["rule_vetoes"][0]["step"] == 0


def test_batch_regime_is_joint():
    class _OOD(_Stub):
        def explain(self, state):
            ex = super().explain(state)
            ex["in_regime"] = "delete_rows" not in state   # one step OOD
            return ex
    m = _OOD()
    resp, _ = answer_batch(m, parse_request(_req(TWO_CALL_STATE)),
                           build_response, m.provenance)
    assert resp["answers"]["safe_to_run"]["noul"] == 0.0
    assert resp["mjolnir"]["in_regime"] is False


def test_batch_is_deterministic():
    m1, m2 = _Stub(), _Stub()
    a, _ = answer_batch(m1, parse_request(_req(TWO_CALL_STATE)),
                        build_response, m1.provenance)
    b, _ = answer_batch(m2, parse_request(_req(TWO_CALL_STATE)),
                        build_response, m2.provenance)
    assert a["answers"] == b["answers"] and a["mjolnir"]["batch"] == b["mjolnir"]["batch"]
