"""Tests for mjolnir_guard — contract + pipeline offline (stub reader);
real-weights smoke skips without mlx + checkpoint.

Run:  python -m pytest src/mjolnir_guard/test_guard.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mjolnir_guard import selftest                                   # noqa: E402
from mjolnir_guard.contract import build_response, parse_request     # noqa: E402
from mjolnir_guard.dialect import extract_action, to_tau             # noqa: E402
from mjolnir_guard.guard import Guard                                # noqa: E402
from mjolnir_guard.selftest import EXAMPLE                           # noqa: E402


def test_selftest_passes():
    assert selftest.run(verbose=False) == 0


def test_example_request_roundtrip():
    resp = build_response(parse_request(EXAMPLE), 0, 30,
                          decision={"stage": "abstained-by-authority"})
    assert resp["answers"]["safe_to_run"]["noul"] == 0
    assert resp["decision"]["stage"] == "abstained-by-authority"


def test_dialect_is_data():
    assert to_tau(extract_action(EXAMPLE["state"])) == "sql_delete_rows"
    assert to_tau({"name": "purge_everything", "args": {}}) is None


def test_sql_delete_pipeline():
    """The SQL surface is judged on its own facts: scope, backup, row count --
    the reasons converge with decide's safety policy, never `missing:money`."""
    g = Guard(reader=None)
    # canonical: scoped + state says no backup + 48,210 rows -> blocked, both codes
    noul, d = g.answer(EXAMPLE["state"])
    assert (noul, d["stage"]) == (0, "blocked-by-authority")
    assert d["authority"]["reasons"] == ["destructive_without_backup",
                                         "fanout_3_on_sql_delete"]
    # drop_table rides the same fact machinery
    noul, d = g.answer('Proposed tool call: drop_table(table="logs")\n'
                       'Context: no backup was taken; 9,000 rows.')
    assert (noul, d["stage"]) == (0, "blocked-by-authority")
    assert d["authority"]["reasons"] == ["destructive_without_backup",
                                         "fanout_3_on_sql_delete"]
    # small + scoped + backup stated -> genuinely routable (reader off)
    noul, d = g.answer('Task: fix one row.\n'
                       'Proposed tool call: delete_rows(table="customers", '
                       'where="id=7")\n'
                       'Context: 1 row matches. A backup was taken this morning.')
    assert (noul, d["stage"]) == (1, "released-authority-only")
    # backup never stated -> abstain names THE FACT, not money
    noul, d = g.answer('Proposed tool call: delete_rows(table="customers", '
                       'where="id=7")\nContext: 2 rows match.')
    assert (noul, d["stage"]) == (0, "abstained-by-authority")
    assert d["authority"]["reasons"] == ["missing:backup"]


def test_pipeline_stages():
    class Stub:
        def generate(self, prompt):
            return "VERDICT: ROUTE", {}
    assert Guard(reader=None).answer(EXAMPLE["state"])[0] == 0
    noul, d = Guard(reader=Stub()).answer(
        'Proposed tool call: select_rows(table="sales")')
    assert (noul, d["stage"]) == (1, "released")


def test_real_reader_smoke():
    pytest = __import__("pytest")
    pytest.importorskip("mlx_lm")
    from mjolnir_guard.reader import resolve_reader_path
    try:
        resolve_reader_path()
    except FileNotFoundError:
        pytest.skip("no reader checkpoint on disk")
    g = Guard(reader="on")
    noul, d = g.answer(EXAMPLE["state"])          # delete_rows, no backup
    assert noul in (0, 1) and d["stage"] in (
        "released", "vetoed-by-reader", "blocked-by-authority",
        "abstained-by-authority", "reader-error")
