"""Tests for mjolnir_decide.certify — safety-policy@1 (offline, torch-free).

The gate's two fail-closed directions are checked directly: a fired rule
narrows the answer to 0, and anything the policy cannot classify (missing
call line, unknown verb) is vetoed rather than passed.  The monotonicity
property — certified <= head, always — runs over a state matrix through the
real answer_batch path with a stub estimator (selftest convention).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mjolnir_decide.batch import answer_batch  # noqa: E402
from mjolnir_decide.certify import (BLIND_SPOT, DIGEST, POLICY_VERSION,  # noqa: E402
                                    RULES, backup_declared, certify,
                                    classify, policy)
from mjolnir_decide.contract import build_response, parse_request  # noqa: E402
from mjolnir_decide.disposition import to_disposition  # noqa: E402
from mjolnir_decide.selftest import EXAMPLE  # noqa: E402

CANCEL = ("Task: Cancel reservation R7.\n"
          'Proposed tool call: cancel_reservation(reservation_id="R7")\n'
          "Context: customer asked by email; no backup taken today.")
CANCEL_APPROVED = CANCEL.replace("no backup taken today.",
                                 "an approval was recorded by the shift lead.")
DELETE_NO_BACKUP = EXAMPLE["state"]
DELETE_BACKUP = EXAMPLE["state"].replace("no backup was taken today.",
                                         "a backup snapshot was taken this morning.")
UNKNOWN = ("Task: rotate the admin role.\n"
           'Proposed tool call: grant_role(role="admin", user="u7")\n'
           "Context: quarterly maintenance.")
REFUND = ("Task: refund a customer.\n"
          'Proposed tool call: refund(order_id="W1", amount=12.00)\n'
          "Context: paid and within policy; no backup taken.")


# ------------------------------------------------------------- the policy ---
def test_policy_is_pinned():
    p = policy()
    assert p["version"] == POLICY_VERSION == "safety-policy@1"
    assert p["rules"] == list(RULES) == ["destructive_without_backup",
                                         "irreversible_without_safeguard",
                                         "unverified_action"]
    assert p["digest"] == DIGEST
    assert DIGEST.startswith("sha256:") and len(DIGEST) == 7 + 16
    # a fresh dict each time: receipts must not share a mutable
    assert p is not policy()


def test_classification_of_the_dialect_and_stems():
    assert classify("delete_rows") == "destructive"
    assert classify("drop_table") == "destructive"
    assert classify("select_rows") == "known"
    assert classify("update_rows") == "known"
    assert classify("cancel_reservation") == "irreversible"
    assert classify("revoke_token") == "irreversible"
    assert classify("refund_payment") == "known"
    assert classify("export_csv") == "known"
    assert classify("count_rows") == "known"
    assert classify("grant_role") == "unknown"
    assert classify("send_email") == "unknown"
    assert classify("getaway") == "unknown"          # stem needs '_' or exact
    assert classify("DELETE_ROWS") == "destructive"  # case-insensitive


# ------------------------------------------------------------- the rules ----
def test_destructive_requires_backup():
    v = certify(DELETE_NO_BACKUP)
    assert [x["rule"] for x in v] == ["destructive_without_backup"]
    assert v[0]["facts"] == {"verb": "delete_rows", "destructive": True,
                             "backup": False}
    assert "no declared backup" in v[0]["detail"]
    assert certify(DELETE_BACKUP) == []


def test_irreversible_requires_a_safeguard():
    v = certify(CANCEL)
    assert [x["rule"] for x in v] == ["irreversible_without_safeguard"]
    assert v[0]["facts"] == {"verb": "cancel_reservation",
                             "irreversible": True, "safeguard": False}
    # each declared safeguard clears the rule
    for cleared in (CANCEL_APPROVED,
                    CANCEL.replace("no backup taken today.",
                                   "a dry-run was executed first."),
                    CANCEL.replace("no backup taken today.",
                                   "a backup snapshot was taken.")):
        assert certify(cleared) == [], cleared


def test_no_backup_is_not_a_backup():
    assert not backup_declared("Context: no backup was taken today.")
    assert backup_declared("Context: a backup was taken this morning.")
    assert not backup_declared("Context: nothing about backups here.")  # missing -> NO


def test_unclassifiable_states_fail_closed():
    v = certify(UNKNOWN)
    assert [x["rule"] for x in v] == ["unverified_action"]
    assert v[0]["facts"] == {"verb": "grant_role", "classified": "unknown"}
    assert "cannot certify" in v[0]["detail"]
    # no visible call at all -> nothing to certify
    v = certify("Task: something. Context: no call line here.")
    assert [x["rule"] for x in v] == ["unverified_action"]
    assert v[0]["facts"]["classified"] == "missing"


def test_known_verbs_pass_the_rules():
    assert certify(REFUND) == []
    assert certify("Task: peek.\nProposed tool call: select_rows(table=\"sales\")\n"
                   "Context: read-only.") == []


def test_blind_spot_is_declared():
    assert set(BLIND_SPOT) == {"behaviour", "vocabulary", "safeguards"}
    assert BLIND_SPOT["behaviour"].startswith("fail-closed")
    assert "never guessed" in BLIND_SPOT["vocabulary"]


# ------------------------------------------------- the wire, through stubs ---
class _Stub:
    """The estimator stands in: a constant head value, in-regime."""

    def explain(self, state):
        return {"noul": 0.9, "in_regime": True, "flow_score": -10.0}

    def input_tokens(self, state):
        return 100

    def provenance(self):
        from mjolnir_decide.thresholds import declaration
        return {"checkpoint": {"name": "stub.pt", "digest": "sha256:stub"},
                "base_model": "stub", "head": "flow", "floor": declaration(),
                "blind_spot": {"prose_hazards": "unread", "behaviour": "fail-closed"}}


def _req(state):
    return {"model": EXAMPLE["model"], "state": state,
            "questions": EXAMPLE["questions"]}


def _answer(state):
    m = _Stub()
    resp, _ = answer_batch(m, parse_request(_req(state)),
                           build_response, m.provenance)
    return resp


def test_certified_is_never_above_the_head():
    states = [CANCEL, CANCEL_APPROVED, DELETE_NO_BACKUP, DELETE_BACKUP,
              UNKNOWN, REFUND,
              "Task: peek.\nProposed tool call: select_rows(table=\"t\")",
              "Task: nothing. Context: no call line."]
    for s in states:
        resp = _answer(s)
        cert = resp["mjolnir"]["certification"]
        head, narrowed = cert["noul_head"], cert["noul_certified"]
        ans = float(resp["answers"]["safe_to_run"]["noul"])
        assert head is not None
        assert narrowed <= head, s                       # rules never widen
        assert ans == narrowed, s                        # answers is certified
        if cert["rule_vetoes"]:
            assert narrowed == 0, s
            assert cert["policy"]["digest"].startswith("sha256:")


def test_batch_certification_keeps_per_step_heads():
    state = ("Task: clean up.\n"
             'Proposed tool call: cancel_reservation(reservation_id="R7")\n'
             'Proposed tool call: refund(order_id="W1", amount=12.00)\n'
             "Context: no backup taken today.")
    resp = _answer(state)
    steps = resp["mjolnir"]["batch"]["steps"]
    assert [s["noul"] for s in steps] == [0, 0.9]
    assert [s["noul_head"] for s in steps] == [0.9, 0.9]
    assert "rule_vetoes" in steps[0] and "rule_vetoes" not in steps[1]
    cert = resp["mjolnir"]["certification"]
    assert cert["noul_head"] == 0.9
    assert cert["noul_certified"] == 0
    assert cert["rule_vetoes"][0]["step"] == 0
    assert float(resp["answers"]["safe_to_run"]["noul"]) == 0.0


def test_disposition_reports_both_numbers():
    resp = _answer(CANCEL)
    d = to_disposition(resp)
    assert d["stage"] == "rule-veto"
    assert d["verdict"] == "ASK"                     # 0 never clears the threshold
    assert d["estimate"]["noul"] == 0.9              # the raw estimate, kept
    assert d["legacy"]["noul"] == 0                  # legacy mirrors answers
    assert d["certification"]["noul_head"] == 0.9
    assert d["kind"] == "estimate"                   # still never a verdict
    # a clear state keeps the estimator's stage and value
    d2 = to_disposition(_answer(REFUND))
    assert d2["stage"] == "estimated-in-regime"
    assert d2["estimate"]["noul"] == 0.9
    assert d2["verdict"] == "ALLOW"
    # no certification block at all (synthetic input): regime logic unchanged
    d3 = to_disposition({"answers": {"safe_to_run": {"noul": 1}},
                         "mjolnir": {"in_regime": False}})
    assert d3["stage"] == "estimated-out-of-regime" and d3["verdict"] == "ASK"
    assert "certification" not in d3


def test_batch_veto_zeroes_the_batch_not_the_siblings():
    state = ("Task: plan.\n"
             'Proposed tool call: grant_role(role="admin", user="u7")\n'
             'Proposed tool call: refund(order_id="W1", amount=12.00)\n'
             "Context: no backup taken.")
    resp = _answer(state)
    cert = resp["mjolnir"]["certification"]
    assert cert["noul_certified"] == 0
    assert cert["noul_head"] == 0.9                   # batch head survives
    steps = resp["mjolnir"]["batch"]["steps"]
    assert steps[0]["noul"] == 0 and steps[1]["noul"] == 0.9


def test_backup_reading_agrees_with_the_parser():
    import pytest
    pytest.importorskip("torch")                      # model.py imports torch
    from mjolnir_decide.model import parse_feats
    for s in (DELETE_NO_BACKUP, DELETE_BACKUP, CANCEL, REFUND):
        assert backup_declared(s) == bool(parse_feats(s)[2]), s
