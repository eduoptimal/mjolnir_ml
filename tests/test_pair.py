"""The Ω-certifiability pair and the degenerate table."""
import pytest

from mjolnir_ml import (audit, degenerate_controls, omega_pair, wilson)
from mjolnir_ml.vendors.wire import (ABSTAIN_W as ABSTAIN, DENY_W as DENY,
                                    RELEASE, ASK)


def test_pair_exact_numbers():
    verdicts = [RELEASE, RELEASE, DENY, ABSTAIN,
                ASK]
    safe = [True, False, False, True, True]
    dec = [True, True, True, True, True]
    p = omega_pair(verdicts, safe, dec)
    assert (p.n_claim, p.n_cert, p.n_A, p.n) == (3, 2, 5, 5)
    assert p.prec == pytest.approx(2 / 3)
    assert p.rec == pytest.approx(2 / 5)
    assert p.n_missed == 2 and p.n_overclaim == 0


def test_vacuous_precision_is_none_never_one():
    p = omega_pair([ABSTAIN] * 4, [True, False, True, False],
                   [True] * 4)
    assert p.prec is None                # ⊥
    assert p.rec == 0.0
    assert p.n_claim == 0


def test_claim_on_undecidable_is_overclaim_not_certified():
    p = omega_pair([RELEASE, DENY], [True, False], [False, True])
    assert p.n_overclaim == 1
    assert p.n_cert == 1
    assert p.n_A == 1


def test_ask_is_not_a_claim():
    p = omega_pair([ASK, ASK], [True, False], [True, True])
    assert p.n_claim == 0 and p.prec is None
    assert p.n_missed == 2


def test_degenerate_controls_table():
    safe = [True, False, True, False]
    dec = [True, True, True, True]
    d = degenerate_controls(safe, dec)
    assert d["ALWAYS"]["n_claim"] == 4
    assert d["ALWAYS"]["prec"] == pytest.approx(0.5)
    assert d["NEVER"]["prec"] is None
    assert d["NEVER"]["rec"] == 0.0


def test_audit_report_shape():
    verdicts = [RELEASE, DENY, ABSTAIN]
    report = audit(verdicts, [True, False, True], [True, True, True])
    assert report["pair"]["prec"] == pytest.approx(1.0)
    assert report["verdicts"][ABSTAIN] == 1
    lo, hi = report["prec_ci"]
    assert 0.0 <= lo <= report["pair"]["prec"] <= hi <= 1.0


def test_wilson_bounds():
    lo, hi = wilson(2, 3)
    assert 0.0 <= lo < 2 / 3 < hi <= 1.0
    assert wilson(0, 0) == (None, None)
