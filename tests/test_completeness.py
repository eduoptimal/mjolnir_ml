"""External hazard ledger and completeness measurement."""
import pytest

from mjolnir_ml import (DecisionModel, HazardLedger, measure_completeness)
from mjolnir_ml.policy import Policy
from mjolnir_ml.vendors import wire


class StubClf:
    """Releases on a low risk probability; the policy decides."""

    classes_ = [0, 1]

    def __init__(self, p_positive=0.01):
        self.p_positive = p_positive

    def get_params(self, deep=False):
        return {"p_positive": self.p_positive}

    def predict(self, X):
        return [1 if self.p_positive >= 0.5 else 0]

    def predict_proba(self, X):
        return [[1 - self.p_positive, self.p_positive]]


FEATS = ("x", "hazard_flag")


def _policy(with_deny: bool) -> Policy:
    statements = [
        {"kind": "premise", "code": "x_stated", "requires": ["x"],
         "predicate": "x >= 0", "verdict": None},
    ]
    if with_deny:
        statements.append({
            "kind": "rule", "code": "deny_hazard", "requires": ["hazard_flag"],
            "predicate": "hazard_flag > 0.5", "verdict": "deny"})
    statements.append({
        "kind": "release", "code": "release_ok", "requires": ["x"],
        "predicate": "x < 100", "verdict": "release"})
    return Policy({"schema": Policy.SCHEMA, "name": "HazardPolicy",
                   "facts": [], "statements": statements})


def _rows():
    rows = [{"x": float(i), "hazard_flag": 0.0} for i in range(20)]
    for i in (5, 10, 15):
        rows[i]["hazard_flag"] = 1.0
    return rows


def _dm(with_deny=True):
    return DecisionModel(StubClf(0.01), _policy(with_deny),
                         features=FEATS, regime="off")


def _ledger():
    return (HazardLedger()
            .add("out_of_spec", lambda r: r["hazard_flag"] > 0.5,
                 "spec sheet §4.1")
            .add("hidden_bit", lambda r: r.get("secret", 0) == 1,
                 "incident report #77", observable=False))


def test_ledger_is_union_only_and_digested():
    led = _ledger()
    d1 = led.digest
    assert d1.startswith("sha256:")
    with pytest.raises(ValueError, match="already in the ledger"):
        led.add("out_of_spec", lambda r: True, "duplicate")
    with pytest.raises(ValueError, match="refutation"):
        led.retire("out_of_spec", "  ")
    led.retire("out_of_spec", "spec §4.1 withdrawn in rev C")
    assert led.digest != d1
    assert "out_of_spec" in led.as_dict()["retired"]
    assert led.hits({"hazard_flag": 1.0, "x": 0}) == []


def test_hazard_predicate_errors_are_not_swallowed():
    led = HazardLedger().add("boom", lambda r: r["missing_key"] > 0, "test")
    with pytest.raises(RuntimeError, match="boom"):
        led.hits({"x": 1})


def test_full_policy_catches_and_gap_policy_does_not():
    rows = _rows()
    led = _ledger()

    full = measure_completeness([_dm(True).dispose(r) for r in rows], rows, led)
    gap = measure_completeness([_dm(False).dispose(r) for r in rows], rows, led)

    assert full["recall"] == 1.0
    assert full["precision"] == 1.0
    assert full["n_false_certificate"] == 0

    # removing the declared hazard rule is measurable: false certificates
    assert gap["recall"] == 0.0
    assert gap["n_false_certificate"] == 3
    assert gap["precision"] == pytest.approx(17 / 20)

    # hidden hazards are outside the representation: ceiling < 1, recall
    # within it (no policy over these inputs can see the hidden bit)
    assert full["ceiling"] == pytest.approx(3 / 3)  # no hidden rows here
    assert full["within_ceiling"]


def test_hidden_hazard_hits_the_chow_ceiling():
    rows = _rows()
    for i in (7, 14):
        rows[i]["secret"] = 1
    led = _ledger()
    rep = measure_completeness([_dm(True).dispose(r) for r in rows], rows, led)

    # 3 observable hazard rows + 2 hidden-only rows
    assert rep["n_hazard"] == 5
    assert rep["n_caught"] == 3                 # hidden rows are released
    assert rep["recall"] == pytest.approx(3 / 5)
    assert rep["ceiling"] == pytest.approx(3 / 5)
    assert rep["within_ceiling"]
    assert rep["recall_observable"] == 1.0
    assert rep["recall_hidden_only"] == 0.0
    assert rep["n_hidden_only"] == 2
    assert rep["per_hazard"]["hidden_bit"]["released"] == 2


def test_completeness_carries_ledger_identity_and_wilson_intervals():
    rows = _rows()
    rep = measure_completeness([_dm(True).dispose(r) for r in rows], rows, _ledger())
    assert rep["ledger_digest"] == _ledger().digest
    assert rep["ledger_version"] == "hazard-ledger@1"
    lo, hi = rep["recall_ci"]
    assert 0.0 <= lo <= rep["recall"] <= hi <= 1.0
    assert rep["coverage"] == pytest.approx(17 / 20)  # 3 hazard rows denied
