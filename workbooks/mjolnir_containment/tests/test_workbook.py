"""Smoke tests for the mjolnir_containment workbook."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = sorted((ROOT / "labs").glob("lab*.py"))


def test_twelve_labs_exist() -> None:
    assert len(LABS) == 12


def test_every_lab_emits_json() -> None:
    for lab in LABS:
        expected = 1 if lab.name == "lab08_fidelity.py" else 0
        result = subprocess.run(
            [sys.executable, str(lab), "--replay"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == expected, (lab.name, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        assert payload["lab"] == lab.stem
        assert payload["replay"] is True


def test_lab08_blocks_the_unit_swap() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "labs" / "lab08_fidelity.py"), "--replay"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["gate_passes"] is False
    assert "unit changed" in payload["gate_reason"]


def test_vendor_packages_are_present() -> None:
    names = {
        "mjolnir_guard",
        "mjolnir_decide",
        "mjolnir_prover",
        "mjolnir_rlvf",
        "mjolnir_intent",
        "mjolnir_common",
        "mjolnir_containment",
    }
    present = {path.name for path in (ROOT / "vendor").iterdir() if path.is_dir()}
    assert names <= present


def test_book_containment_snippet_runs() -> None:
    code = """
from mjolnir_containment import Guard, Authority, typed_verdict
authority = Authority.pinned("containment")
guard = Guard(authority=authority)
assert guard.admits({"on_fragment": True}, {"action": "read"})
assert not guard.admits({"on_fragment": False}, {"action": "delete"})
assert authority.dispose({"on_fragment": True}, {"action": "read"}) is typed_verdict.CERTIFIED
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "vendor")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_book_spy_snippet_runs() -> None:
    code = """
from mjolnir_containment import release, DeniedFixtures, Spy, ReaderRevisions
for state, proposal, basis in DeniedFixtures.all():
    spy = Spy.attach(release)
    stage, reason = release(state, proposal)
    assert stage != "released"
    assert spy.release_calls == 0
    assert reason == basis
for state, proposal in DeniedFixtures.all_pairs():
    for reader in ReaderRevisions.suite():
        spy = Spy.attach(release)
        stage, _ = release(state, proposal, reader=reader)
        assert stage != "released"
        assert spy.release_calls == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "vendor")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_containment_facade_does_not_import_mjolnir_ml() -> None:
    code = """
import sys
import mjolnir_containment
assert "mjolnir_ml" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "vendor")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_volume_integrations_are_lazy() -> None:
    code = """
import sys
import mjolnir_containment.integrations as integrations
assert "mjolnir_ml" not in sys.modules
assert callable(integrations.volume_one_certify)
assert callable(integrations.volume_one_verify)
assert callable(integrations.volume_two_hazard_ledger)
assert callable(integrations.volume_two_readiness)
assert callable(integrations.volume_two_omega_pair)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "vendor")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
