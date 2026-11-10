"""Smoke tests for the mjolnir_contract (Volume I) workbook."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = sorted((ROOT / "labs").glob("lab*.py"))


def run_lab(name: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(ROOT / "labs" / name), "--replay"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, (name, result.returncode, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["lab"] == name.removesuffix(".py")
    assert payload["replay"] is True
    return payload


def test_ten_labs_exist() -> None:
    assert len(LABS) == 10


def test_every_lab_emits_json() -> None:
    for lab in LABS:
        run_lab(lab.name)


def test_totality_routes_corrupted_rows_to_abstain() -> None:
    p = run_lab("lab01_totality.py")
    assert p["ungated_share_above_0.80"] > 0.30
    assert p["corrupted_abstention_rate"] == 1.0


def test_four_verdicts_name_every_basis() -> None:
    p = run_lab("lab02_verdicts.py")
    assert p["two_verdict_abstentions"] == 0
    assert p["two_verdict_asks"] == 0
    assert p["every_row_labelled"] is True
    assert p["missing_fact_row"]["missing"] == "score"


def test_regime_reports_all_injected_rows() -> None:
    p = run_lab("lab03_regime.py")
    assert p["all_injected_abstain"] is True
    assert p["box_vs_quadratic"]["disagree"] is True


def test_uncalibrated_contract_names_missing_fact() -> None:
    p = run_lab("lab04_facts.py")
    assert p["uncalibrated"]["abstains_with_named_missing_fact"] is True
    assert p["stated_fact"]["fabricated_certified"] is False


def test_compiler_digests_and_refusals() -> None:
    p = run_lab("lab05_compile.py")
    assert p["digest_stable_on_recompile"] is True
    assert p["digest_changes_on_semantic_edit"] is True
    assert p["unknown_fact_error"]
    assert p["duplicate_code_error"]
    assert p["dead_rule_detected"] is True


def test_coverage_track_and_vacuous_cell() -> None:
    p = run_lab("lab06_coverage.py")
    assert 0.0 <= p["large_calibration"]["coverage"] <= 1.0
    assert p["zero_claim_cell"]["precision"] is None
    assert p["zero_claim_cell"]["vacuous"] is True


def test_mondrian_and_cqr() -> None:
    p = run_lab("lab07_groups.py")
    assert p["undersampled_group"]["refused"] is True
    assert p["cqr_width_varies"] is True


def test_two_risk_tracks() -> None:
    p = run_lab("lab08_risk.py")
    assert p["risk_track"]["thresholds_disjoint"] is True
    assert p["reported_separately"] is True


def test_cost_bands_and_ask_monotonicity() -> None:
    p = run_lab("lab09_costs.py")
    assert p["analytic_bands"]["release_threshold"] == 0.2
    assert p["analytic_bands"]["deny_threshold"] == 0.8
    assert p["ask_volume_monotone_nonincreasing"] is True


def test_capstone_certificate_replays() -> None:
    p = run_lab("lab10_capstone.py")
    assert p["verified"] is True
    assert p["replayable"] is True
    assert "signature" in p["tamper_check"]
