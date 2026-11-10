"""Smoke tests for the mjolnir_governance (Volume II) workbook."""

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


def test_drift_flip() -> None:
    p = run_lab("lab01_drift.py")
    assert p["static_coverage_collapses"] is True
    assert p["adaptive_falls_less"] is True
    assert p["adaptive_set_size_rises"] is True
    assert p["terminal_alpha_below_start"] is True


def test_degenerate_controls() -> None:
    p = run_lab("lab02_pair.py")
    assert p["never_precision_is_vacuous"] is True
    assert p["scalar_ranks_never_as_safe"] is True


def test_head_to_head() -> None:
    p = run_lab("lab03_bench.py")
    assert p["sign_held"] is True
    assert p["pair_beats_scalar"] is True


def test_completeness_ceiling() -> None:
    p = run_lab("lab04_completeness.py")
    assert p["gap_costs_recall"] is True
    assert p["hidden_at_ceiling"] is True


def test_tampering_cases() -> None:
    p = run_lab("lab05_certificate.py")
    assert p["replayable"] == 111
    assert "signature" in p["tampered_fact_error"]
    assert "signature" in p["flipped_verdict_error"]
    assert "signature" in p["wrong_key_error"]


def test_readiness_gate() -> None:
    p = run_lab("lab06_readiness.py")
    assert p["strict_approved"] is False
    assert p["blocking_gate"] is not None
    assert p["id_changes_with_threshold"] is True


def test_live_hazard_containment() -> None:
    p = run_lab("lab07_live.py")
    assert p["first_alarm_batch"] == p["containment_batch"]
    assert p["journal_verifies"] is True
    assert p["contained"] is True


def test_flywheel_direction_and_approval() -> None:
    p = run_lab("lab08_flywheel.py")
    assert p["tightening_promoted"] is True
    assert p["unsigned_loosening_refused"] is True
    assert p["signed_loosening_promoted"] is True
    assert p["retraction_count"] == 48


def test_discovery_census_and_refusals() -> None:
    p = run_lab("lab09_discovery.py")
    assert p["census_has_four_counts"] is True
    assert p["admission_approved"] is True
    assert p["refused_admission"] is True
    assert p["self_certification_alarm"]
    assert p["self_certification_refused"] is True


def test_three_causal_cases() -> None:
    p = run_lab("lab10_causal.py")
    assert p["three_cases_reproduce"] is True
    assert p["etiology"]["declined"] is True
    assert p["over_adjusted"]["refuted"] is True
    assert p["paired"]["placebo_is_zero"] is True
