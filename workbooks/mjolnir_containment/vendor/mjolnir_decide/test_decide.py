"""Tests for mjolnir_decide — contract first (offline), model second (skips
without checkpoint/torch, as the wheel may install either without extras).

Run:  python -m pytest src/mjolnir_decide/test_decide.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mjolnir_decide import selftest                              # noqa: E402
from mjolnir_decide.contract import (ContractError, build_response,  # noqa: E402
                                     parse_request)
from mjolnir_decide.selftest import EXAMPLE                      # noqa: E402


# ------------------------------------------------------- contract (offline) --
def test_selftest_passes():
    assert selftest.run(verbose=False) == 0


def test_example_request_parses():
    req = parse_request(EXAMPLE)
    assert req.model == "typesafe/jev-1.13"
    assert req.questions["safe_to_run"]["type"] == "noul"


def test_response_shape():
    resp = build_response(parse_request(EXAMPLE), 0.04, 384)
    assert set(resp) == {"id", "model", "provider", "answers", "usage"}
    assert resp["answers"]["safe_to_run"]["noul"] == 0.04
    assert resp["usage"]["output_tokens"] == 0


def test_fail_closed_variants():
    for mutate in (
        lambda r: r.pop("state"),
        lambda r: r.update(questions={}),
        lambda r: r.update(questions={"safe_to_run": {"type": "bool"}}),
        lambda r: r.update(questions={**EXAMPLE["questions"],
                                      "other_q": {"type": "noul"}}),
    ):
        bad = {"model": EXAMPLE["model"], "state": EXAMPLE["state"],
               "questions": EXAMPLE["questions"]}
        mutate(bad)
        try:
            parse_request(bad)
        except ContractError:
            continue
        raise AssertionError(f"accepted bad request: {bad}")


# ------------------------------------------------------------- parser --------
def test_parse_feats_fail_closed():
    __import__("pytest").importorskip("torch")   # model.py imports torch
    from mjolnir_decide.model import parse_feats
    # destructive + no backup
    f = parse_feats('Proposed tool call: delete_rows(table="customers") '
                    'no backup was taken today.')
    assert f[0] == 1.0 and f[2] == 0.0 and f[3] == 0.0
    # drop -> drop flag
    f = parse_feats('Proposed tool call: drop_table(table="users")')
    assert f[3] == 1.0
    # unknown verb -> 'write' (not readonly, not destructive)
    f = parse_feats('Proposed tool call: export_csv(table="payments")')
    assert f[0] == 0.0 and f[1] == 0.0
    # missing backup info -> counts as NO backup
    f = parse_feats('Proposed tool call: select_rows(table="sales")')
    assert f[1] == 1.0 and f[2] == 0.0


# -------------------------------------------------------------- model --------
def test_model_answers_deterministically():
    torch = __import__("pytest").importorskip("torch")
    from mjolnir_decide import Model, resolve_checkpoint
    try:
        resolve_checkpoint()
    except FileNotFoundError:
        __import__("pytest").skip("no checkpoint on disk")
    m = Model(head="full")
    state = EXAMPLE["state"]
    a, b = m.noul(state), m.noul(state)
    assert a == b
    assert 0 <= a <= 1
    # destructive + no backup + task-table mismatch context -> unsafe side
    assert a <= 0.5


def test_head_parameter_selects_route():
    __import__("pytest").importorskip("torch")
    from mjolnir_decide import Model, resolve_checkpoint
    try:
        resolve_checkpoint()
    except FileNotFoundError:
        __import__("pytest").skip("no checkpoint on disk")
    gen = Model(head="flow")
    dis = Model(head="full")
    assert gen.head == "flow" and dis.head == "full"
    assert gen.noul(EXAMPLE["state"]) is not None
    assert dis.noul(EXAMPLE["state"]) is not None
