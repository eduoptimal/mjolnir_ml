"""Tests for mjolnir_common.batch — the joint-AND law, pinned by examples."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mjolnir_common.batch import (ALLOW, ASK, DENY, min_noul, split_plan,
                                  worst_case)


def test_worst_case_is_joint_and():
    assert worst_case([ALLOW, ALLOW]) == ALLOW
    assert worst_case([ALLOW, ASK]) == ASK
    assert worst_case([ASK, DENY]) == DENY
    assert worst_case([DENY, ALLOW, ASK]) == DENY
    assert worst_case([ASK]) == ASK


def test_worst_case_order_is_independent():
    assert worst_case([ALLOW, ASK, DENY]) == worst_case([DENY, ALLOW, ASK])


def test_empty_plan_raises():
    for fn, args in ((worst_case, ([],)), (min_noul, ([], []))):
        try:
            fn(*args)
        except ValueError:
            continue
        raise AssertionError(f"{fn.__name__} accepted an empty plan")


def test_min_noul_is_min_not_product():
    # 0.9 * 0.9 = 0.81 (product, assumes independence); min = 0.9.
    # Sequential steps are not independent -> the law returns the min.
    assert min_noul([0.9, 0.9], [True, True]) == 0.9
    assert min_noul([1.0, 0.04], [True, True]) == 0.04


def test_one_step_out_of_regime_zeroes_the_batch():
    assert min_noul([1.0, 1.0, 1.0], [True, False, True]) == 0.0


def test_ragged_plan_raises():
    try:
        min_noul([0.5], [True, True])
    except ValueError:
        return
    raise AssertionError("ragged plan accepted")


# ------------------------------------------------------------------ splitter --
def _is_step(seg: str) -> bool:
    return any(seg.startswith(v) for v in ("cancel ", "refund ", "transfer "))


def test_split_single_has_no_connective():
    out = split_plan("cancel order W1", _is_step)
    assert out == {"kind": "single", "steps": ["cancel order W1"]}


def test_split_multi_on_connectives():
    out = split_plan("cancel order W1 and refund payment P2", _is_step)
    assert out["kind"] == "multi"
    assert out["steps"] == ["cancel order W1", "refund payment P2"]
    for sep in ("cancel order W1 then refund payment P2",
                "cancel order W1, then refund payment P2",
                "cancel order W1 and then refund payment P2",
                "cancel order W1; refund payment P2",
                "cancel order W1, and refund payment P2"):
        assert split_plan(sep, _is_step)["kind"] == "multi", sep


def test_split_ambiguous_when_a_segment_is_not_a_step():
    # "and W2" is likely an argument continuation, not a second instruction.
    out = split_plan("cancel order W1 and W2", _is_step)
    assert out["kind"] == "ambiguous"
    assert out["steps"] == ["cancel order W1", "W2"]


def test_split_empty_text_is_ambiguous():
    assert split_plan("   ", _is_step)["kind"] == "ambiguous"
    assert split_plan("", _is_step)["kind"] == "ambiguous"


def test_split_is_deterministic():
    t = "cancel order W1 and refund payment P2 and transfer funds to X"
    a, b = split_plan(t, _is_step), split_plan(t, _is_step)
    assert a == b and a["kind"] == "multi" and len(a["steps"]) == 3
