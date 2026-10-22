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
        return [1 if self.p_positive >= 0.5 else 0]

    def predict_proba(self, X):
        return [[1 - self.p_positive, self.p_positive]]


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
