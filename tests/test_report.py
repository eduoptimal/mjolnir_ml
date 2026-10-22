"""The comprehensive report: one artifact for problem, decisions and hazards."""
import pytest

from mjolnir_ml import (DecisionModel, HazardLedger, report, report_data)
from mjolnir_ml.vendors import wire

import policies as P


class StubClf:
    classes_ = [0, 1]

    def __init__(self, p_positive=0.01):
        self.p_positive = p_positive

    def get_params(self, deep=False):
        return {"p_positive": self.p_positive}

    def predict(self, X):
        return [1 if self.p_positive >= 0.5 else 0]

    def predict_proba(self, X):
        return [[1 - self.p_positive, self.p_positive]]


FEATS = ("amount", "credit_limit")
ROWS = [{"amount": 100, "credit_limit": 1000},
        {"amount": 5000, "credit_limit": 1000},
        {"amount": 100, "credit_limit": 1000}]


def _dm(p_positive=0.01):
    return DecisionModel(StubClf(p_positive), P.FraudPolicy, features=FEATS,
                         regime="off")


def test_report_sections_without_oracle_or_ledger():
    text = report(_dm(), ROWS, title="Fraud review")
    assert "Fraud review" in text
    assert "CONTRACT (the problem)" in text
    assert "policy statements" in text.lower() or "POLICY" in text
    assert "DECISIONS" in text
    assert "SAMPLE DECISIONS" in text
    assert "OMEGA PAIR" not in text
    assert "HAZARDS" not in text
    assert "CERTIFICATES" not in text


def test_report_includes_pair_and_degenerate_controls():
    text = report(_dm(), ROWS, safe=[True, False, True],
                  decidable=[True, True, False])
    assert "OMEGA PAIR" in text
    assert "degenerate: ALWAYS" in text
    assert "precision" in text and "recall" in text


def test_report_includes_hazard_table():
    ledger = (HazardLedger()
              .add("big_amount", lambda r: r["amount"] > 1000, "policy §2"))
    text = report(_dm(), ROWS, ledger=ledger)
    assert "HAZARDS (external ledger)" in text
    assert "big_amount" in text
    assert "observable recall" in text


def test_report_includes_certificate_census():
    text = report(_dm(), ROWS, key=b"k")
    assert "CERTIFICATES" in text
    assert "signed & verified 3/3" in text


def test_report_data_is_structured_and_consistent():
    dm = _dm()
    data = report_data(dm, ROWS, safe=[True, False, True],
                       decidable=[True, True, False], sample=2)
    assert data["schema"] == "mjolnir.ml.report/1"
    assert data["decisions"]["n"] == 3
    assert data["decisions"]["verdicts"][wire.RELEASE] == 2
    assert data["pair"]["pair"]["n_claim"] == 3
    assert [r["index"] for r in data["rows"]] == [0, 1, 2]
    assert data["rows"][0]["verdict"] == wire.RELEASE
    assert data["contract"]["policy_digest"] == P.FraudPolicy.policy().digest
    assert data["contract"]["n_features"] == 2
    assert len(data["contract"]["statements"]) == 6


def test_report_abstention_causes_are_named():
    dm = DecisionModel(StubClf(0.01), P.UncoveredPolicy, features=("x",),
                       regime="off")
    text = report(dm, [{"x": 1}])
    assert "uncovered (no rule fires)" in text


def test_report_data_has_no_model_object_leak():
    data = report_data(_dm(), ROWS)
    assert isinstance(data, dict)
    import json
    json.dumps(data)          # JSON-able by construction


def test_notebooks_are_valid_and_have_no_error_outputs():
    """The shipped notebooks must parse and contain no failed cells."""
    import json
    from pathlib import Path
    here = Path(__file__).resolve().parents[1]
    nbs = sorted((here / "examples").glob("*.ipynb"))
    assert len(nbs) >= 2
    for path in nbs:
        doc = json.loads(path.read_text())
        assert doc["nbformat"] == 4
        assert doc["metadata"]["kernelspec"]["language"] == "python"
        for cell in doc["cells"]:
            if cell["cell_type"] != "code":
                continue
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", \
                    f"{path.name}: failed cell output {out.get('ename')}"
