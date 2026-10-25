"""Signed certificates: structure, HMAC, and estimator-free replay."""
import pytest

from mjolnir_ml import (CertificateError, DecisionModel, certify,
                        verify_certificate)
from mjolnir_ml.policy import Policy
from mjolnir_ml.vendors import wire

import policies as P


class StubClf:
    classes_ = [0, 1]

    def __init__(self, p_positive=0.01):
        self.p_positive = p_positive

    def get_params(self, deep=False):
        return {"p_positive": self.p_positive}

    def predict(self, X):
        return [1 if self.p_positive >= 0.5 else 0] * len(X)

    def predict_proba(self, X):
        return [[1 - self.p_positive, self.p_positive]] * len(X)


FEATS = ("amount", "credit_limit")
ROW = {"amount": 100, "credit_limit": 1000}
KEY = b"audit-key"


def _dm(regime="off"):
    return DecisionModel(StubClf(0.01), P.FraudPolicy, features=FEATS,
                         regime=regime)


def test_signed_certificate_verifies_with_policy_only():
    d = _dm().dispose(ROW)
    cert = certify(d, key=KEY)
    assert cert["schema"] == "mjolnir.ml.certificate/1"
    assert cert["signature"].startswith("hmac-sha256:")
    assert cert["replayable"] is True
    # third party: no estimator, only the key and the declared policy
    assert verify_certificate(cert, key=KEY, policy=P.FraudPolicy) is True


def test_signature_required_when_present():
    cert = certify(_dm().dispose(ROW), key=KEY)
    with pytest.raises(CertificateError, match="no key"):
        verify_certificate(cert, policy=P.FraudPolicy)


def test_wrong_key_rejected():
    cert = certify(_dm().dispose(ROW), key=KEY)
    with pytest.raises(CertificateError, match="signature"):
        verify_certificate(cert, key=b"other", policy=P.FraudPolicy)


def test_tampered_verdict_breaks_signature():
    cert = certify(_dm().dispose(ROW), key=KEY)
    cert["verdict"] = "deny"   # a different, valid wire verdict
    with pytest.raises(CertificateError, match="signature"):
        verify_certificate(cert, key=KEY, policy=P.FraudPolicy)


def test_tampered_facts_break_signature_and_replay():
    d = _dm().dispose(ROW)
    signed = certify(d, key=KEY)
    signed["facts"]["amount"] = 5000        # would flip the decision
    with pytest.raises(CertificateError, match="signature"):
        verify_certificate(signed, key=KEY, policy=P.FraudPolicy)

    unsigned = certify(d)                   # unsigned: replay still catches it
    unsigned["facts"]["amount"] = 5000
    with pytest.raises(CertificateError, match="replay disagrees"):
        verify_certificate(unsigned, policy=P.FraudPolicy)


def test_wrong_policy_digest_rejected():
    cert = certify(_dm().dispose(ROW), key=KEY)
    with pytest.raises(CertificateError, match="policy digest"):
        verify_certificate(cert, key=KEY, policy=P.CertifiedPolicy)


def test_unsigned_requires_signature_flag():
    cert = certify(_dm().dispose(ROW))
    assert cert["signature"] is None
    assert verify_certificate(cert, policy=P.FraudPolicy) is True
    with pytest.raises(CertificateError, match="unsigned"):
        verify_certificate(cert, require_signature=True)


def test_harness_attested_stage_is_not_replayable():
    dm = _dm(regime="required")             # never fitted -> regime abstain
    d = dm.dispose(ROW)
    cert = certify(d, key=KEY)
    assert d.verdict == wire.ABSTAIN_W
    assert cert["replayable"] is False
    # attested certificates still verify structurally and by signature
    assert verify_certificate(cert, key=KEY, policy=P.FraudPolicy) is True


def test_policy_ir_roundtrips_as_audit_input():
    d = _dm().dispose(ROW)
    cert = certify(d, key=KEY)
    ir = P.FraudPolicy.policy().ir
    assert verify_certificate(cert, key=KEY, policy=Policy(ir)) is True


def test_deny_and_ask_certificates_replay():
    for p_positive in (0.5, 0.2):
        d = DecisionModel(StubClf(p_positive), P.FraudPolicy, features=FEATS,
                          regime="off").dispose(ROW)
        cert = certify(d, key=KEY)
        assert cert["replayable"] is True
        assert verify_certificate(cert, key=KEY, policy=P.FraudPolicy)


# ------------------------------------------- class-general replay (both) ----
class StubReg:
    """Fixed-prediction regressor: kappa supplies the interval, not the model."""

    def __init__(self, pred=5.0):
        self.pred = pred

    def get_params(self, deep=False):
        return {"pred": self.pred}

    def predict(self, X):
        return [self.pred] * len(X)


FEATS_R = ("x",)
ROW_R = {"x": 1.0}


def test_regressor_interval_certificate_replays():
    """lo/hi conformal facts flow through the SAME fact-table replay path."""
    dm = DecisionModel(StubReg(5.0), P.RegressorPolicy, features=FEATS_R,
                       regime="off")
    dm.calibrate([{"x": i * 0.1} for i in range(40)],
                 [i * 0.1 for i in range(40)], alpha=0.10)
    d = dm.dispose(ROW_R)
    assert {"lo", "hi"} <= set(d.facts)          # interval facts recorded
    cert = certify(d, key=KEY)
    assert cert["replayable"] is True
    assert verify_certificate(cert, key=KEY, policy=P.RegressorPolicy,
                              row=ROW_R) is True
    # unsigned fact tamper on the interval -> replay re-derives the verdict
    bad = dict(certify(d), facts=dict(cert["facts"], hi=999.0))
    with pytest.raises(CertificateError, match="replay disagrees"):
        verify_certificate(bad, policy=P.RegressorPolicy)


def test_classifier_conformal_certificate_replays():
    """certified/set_size conformal facts replay without the forest."""
    dm = DecisionModel(StubClf(0.05), P.CertifiedPolicy, features=FEATS,
                       regime="off")
    dm.calibrate([{"amount": 1.0, "credit_limit": 10.0}] * 40,
                 [0] * 40, alpha=0.10)
    d = dm.dispose(ROW)
    assert {"certified", "set_size"} <= set(d.facts)   # set facts recorded
    cert = certify(d, key=KEY)
    assert cert["replayable"] is True
    assert verify_certificate(cert, key=KEY, policy=P.CertifiedPolicy,
                              row=ROW) is True
    bad = dict(certify(d), facts=dict(cert["facts"], certified=False))
    with pytest.raises(CertificateError, match="replay disagrees"):
        verify_certificate(bad, policy=P.CertifiedPolicy)


def test_certificate_id_binding_check():
    """The id is recomputed from the pinned parts — no rewritten ids."""
    cert = certify(_dm().dispose(ROW))                    # unsigned
    forged = dict(cert, certificate_id="sha256:" + "0" * 64)
    with pytest.raises(CertificateError, match="certificate_id"):
        verify_certificate(forged, policy=P.FraudPolicy)


def test_row_integrity_check():
    d = _dm().dispose(ROW)
    cert = certify(d, key=KEY)
    assert verify_certificate(cert, key=KEY, policy=P.FraudPolicy,
                              row=ROW) is True
    with pytest.raises(CertificateError, match="input tamper"):
        verify_certificate(cert, key=KEY, policy=P.FraudPolicy,
                           row=dict(ROW, amount=9999))
    with pytest.raises(CertificateError, match="row must be a mapping"):
        verify_certificate(cert, key=KEY, policy=P.FraudPolicy, row=[1, 2])


def test_guarantee_and_assurance_are_policy_derived():
    """A signer-less relabel fails even after the id is RECOMPUTED (the id
    recipe is public) — guarantee/assurance must match the policy."""
    from mjolnir_ml.certificate import _binding_id
    cert = certify(_dm().dispose(ROW))                    # unsigned

    def _rebind(c):
        return dict(c, certificate_id=_binding_id(c))

    other_g = ("declared" if cert["guarantee"] == "model_backed"
               else "model_backed")
    with pytest.raises(CertificateError, match="guarantee disagrees"):
        verify_certificate(_rebind(dict(cert, guarantee=other_g)),
                           policy=P.FraudPolicy)
    with pytest.raises(CertificateError, match="assurance disagrees"):
        verify_certificate(_rebind(dict(cert, assurance="dual_track")),
                           policy=P.FraudPolicy)
    # signed: the HMAC catches the relabel before any derivation does
    signed = certify(_dm().dispose(ROW), key=KEY)
    with pytest.raises(CertificateError, match="signature"):
        verify_certificate(_rebind(dict(signed, guarantee=other_g)),
                           key=KEY, policy=P.FraudPolicy)


def test_row_check_also_covers_regressor_decisions():
    dm = DecisionModel(StubReg(5.0), P.RegressorPolicy, features=FEATS_R,
                       regime="off")
    dm.calibrate([{"x": i * 0.1} for i in range(40)],
                 [i * 0.1 for i in range(40)], alpha=0.10)
    cert = certify(dm.dispose(ROW_R), key=KEY)
    with pytest.raises(CertificateError, match="input tamper"):
        verify_certificate(cert, key=KEY, policy=P.RegressorPolicy,
                           row={"x": 42.0})
