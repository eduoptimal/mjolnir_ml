"""Tests for the Tier-1 policy language (mjolnir_prover.lib.policy).

The spec is locked (see the module docstring): ONE section, FOUR keys, one
expression grammar; fail-closed with line numbers; dispositions come from the
single authority — never from the compiler. These tests pin the accept set,
the reject set, the measured dispositions (including the and-chained rule that
found the emitter/kernel FAULT, fixed 2026-10-07), order-independence of `is`
conjunctions, and dotted-scope units (no Attribute nodes — outside the
fragment, measured)."""
from __future__ import annotations

import pytest

from mjolnir_prover.lib import certify, policy

GOOD = """
version: 1
claims:
  - name: TI_CASH.AMOUNT
    is: "!= null and >= 0"
  - name: TI_CASH.OTHER
    is: ">= 0 and != null"
  - name: currency_amt
    is: "money(scale=2, digits=10)"
  - name: description
    is: "varchar(255)"
  - name: credit_rule
    when: "grace_days >= 0"
    then: "grace_days + 30 >= 30"
  - name: chain_rule
    when:
      - "low >= 0"
      - "high >= low"
    then: "high >= 0"
  - name: status
    is: "enum('ACTIVE', 'CLOSED')"
  - name: total
    is: "== sum(detail.amounts)"
"""


def _disp(c):
    return certify.certify_claim(c)


def test_compile_shapes_and_measured_dispositions():
    cands = policy.compile_text(GOOD, source="claims.yaml", repo="t")
    assert len(cands) == 8, [c["predicate"] for c in cands]
    by_pred = {c["predicate"]: c for c in cands}

    # both conjunction orders certify (unit order is normalized; measured)
    for p in ("TI_CASH.AMOUNT != null and TI_CASH.AMOUNT >= 0",
              "TI_CASH.OTHER >= 0 and TI_CASH.OTHER != null"):
        r = _disp(by_pred[p])
        assert r["status"] == "CERTIFIED", (p, r["status"], r.get("reason"))

    # definitions declare
    for p in ("range currency_amt |v| < 10^8", "range description len <= 255"):
        assert _disp(by_pred[p])["status"] == "CERTIFIED", p

    # rules prove — including the and-chained rule (the FAULT that was fixed)
    for p in ("grace_days + 30 >= 30", "high >= 0"):
        r = _disp(by_pred[p])
        assert r["status"] == "CERTIFIED", (p, r["status"])
        assert not (r.get("kernel") or {}).get("false_certs"), p

    # sum / enum route to the observation venue (derived, never authored)
    for p in ("total == sum(detail.amounts)",
              "enum status in {'ACTIVE','CLOSED'}"):
        r = _disp(by_pred[p])
        assert r["status"] == "UNVERIFIABLE", p
        assert r.get("reason") == "runtime_observation", (p, r.get("reason"))


def test_dotted_subjects_alias_in_the_unit_only():
    cands = policy.compile_text(
        'claims:\n  - name: TI_CASH.AMOUNT\n    is: ">= 0"\n',
        source="d.yaml", repo="t")
    c = cands[0]
    assert c["predicate"].startswith("TI_CASH.AMOUNT")   # display truth
    assert "TI_CASH_AMOUNT" in c["canonical_unit"]       # valid identifier
    assert "." not in c["canonical_unit"]                # no Attribute nodes
    assert _disp(c)["status"] == "CERTIFIED"


def test_ids_are_deterministic():
    a = policy.compile_text(GOOD, source="claims.yaml", repo="t")
    b = policy.compile_text(GOOD, source="claims.yaml", repo="t")
    assert [c["id"] for c in a] == [c["id"] for c in b]


@pytest.mark.parametrize("label,text", [
    ("unknown key", 'claims:\n  - name: x\n    check: ">0"\n'),
    ("subject repeated", 'claims:\n  - name: a\n    is: ">= a"\n'),
    ("or", 'claims:\n  - name: a\n    is: "> 0 or < 5"\n'),
    ("not", 'claims:\n  - name: a\n    is: "not >= 0"\n'),
    ("inexact decimal", 'claims:\n  - name: a\n    is: ">= 0.1"\n'),
    ("parentheses", 'claims:\n  - name: a\n    is: "> (1 + 2)"\n'),
    ("then without when", 'claims:\n  - name: a\n    then: "1 == 1"\n'),
    ("when without conditions",
     'claims:\n  - name: a\n    when:\n    then: "1 == 1"\n'),
    ("unquoted predicate", 'claims:\n  - name: a\n    is: >= 0\n'),
    ("no claims root", 'version: 1\n'),
    ("is mixed with rule",
     'claims:\n  - name: a\n    is: ">= 0"\n    when: "b > 1"\n'
     '    then: "c > 1"\n'),
    ("subject in is", 'claims:\n  - name: amount\n    is: "amount >= 0"\n'),
    ("bad money args", 'claims:\n  - name: a\n    is: "money(scale=2)"\n'),
])
def test_rejects_fail_closed_with_a_line(label, text):
    with pytest.raises(policy.PolicyError) as e:
        policy.compile_text(text, source="x.yaml", repo="t")
    assert str(e.value), label


def test_exact_decimals_are_allowed():
    c = policy.compile_text(
        'claims:\n  - name: a\n    is: ">= 0.5"\n',
        source="f.yaml", repo="t")[0]
    assert _disp(c)["status"] == "CERTIFIED"
