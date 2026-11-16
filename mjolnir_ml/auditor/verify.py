"""Independent audit verification and replay."""
from __future__ import annotations

from mjolnir_ml.auditor import crypto
from mjolnir_ml.auditor.bundle import (AUDIT_SCHEMA, BUNDLE_SCHEMA,
                                       REPLAY_SPEC, load_bundle, load_trust)
from mjolnir_ml.auditor.canonical import canonical_bytes, digest, digest_parts
from mjolnir_ml.auditor.policy_ir import audit_expectations, policy_digest
from mjolnir_ml.auditor.report import AuditReport

WIRE = {"release", "ask", "deny", "abstain"}


def _binding_id(cert: dict) -> str:
    return digest_parts(
        cert["claim_digest"],
        cert["policy_digest"],
        cert["model_digest"],
        cert.get("calibration_digest") or "-",
        cert.get("risk_digest") or "-",
        cert.get("regime_digest") or "-",
        cert["verdict"],
        ",".join(cert["basis"]),
        cert["guarantee"],
    )


def _report(valid, replayable, checks, cert=None, errors=(), spec=None,
            key_id=None):
    cert = cert or {}
    return AuditReport(
        schema=AUDIT_SCHEMA,
        valid=valid,
        replayable=replayable,
        checks=checks,
        verdict=cert.get("verdict"),
        basis=tuple(cert.get("basis") or ()),
        missing=cert.get("missing"),
        errors=tuple(errors),
        spec_version=spec,
        key_id=key_id,
        certificate_id=cert.get("certificate_id"),
    )


def verify_bundle(bundle, trust, *, require_signature=True):
    checks = {"structure": False, "signature": False, "binding": False,
              "replay": False}
    bundle = load_bundle(bundle)
    trust = load_trust(trust)
    if not isinstance(bundle, dict):
        return _report(False, False, checks, errors=("malformed bundle",))
    if bundle.get("bundle_schema") != BUNDLE_SCHEMA:
        return _report(False, False, checks,
                       errors=(f"unsupported bundle schema: "
                               f"{bundle.get('bundle_schema')!r}",))
    spec = bundle.get("spec_version")
    if spec != REPLAY_SPEC:
        return _report(False, False, checks,
                       errors=(f"unsupported spec version: {spec!r}",),
                       spec=spec)
    cert = bundle.get("certificate")
    if not isinstance(cert, dict):
        return _report(False, False, checks, errors=("malformed certificate",),
                       spec=spec)
    if cert.get("verdict") not in WIRE:
        return _report(False, False, checks, cert,
                       errors=("invalid certificate verdict",), spec=spec)
    if not isinstance(cert.get("basis"), list):
        return _report(False, False, checks, cert,
                       errors=("malformed certificate basis",), spec=spec)
    checks["structure"] = True

    signature = bundle.get("signature")
    if signature is not None or require_signature:
        key_id = bundle.get("key_id")
        key_text = trust.get(key_id)
        if key_text is None:
            return _report(False, False, checks, cert,
                           errors=("unknown trust key",), spec=spec,
                           key_id=key_id)
        try:
            public = crypto.public_from_text(key_text)
            body = {k: v for k, v in bundle.items() if k != "signature"}
            crypto.verify(public, signature, canonical_bytes(body))
            checks["signature"] = True
        except Exception as exc:
            return _report(False, False, checks, cert,
                           errors=(f"signature failure: {exc}",), spec=spec,
                           key_id=key_id)

    policy_ir = bundle.get("policy_ir")
    facts = bundle.get("facts")
    row = bundle.get("row")
    try:
        if not isinstance(policy_ir, dict) or not isinstance(facts, dict):
            raise ValueError("policy_ir and facts are required")
        if digest(policy_ir) != cert["policy_digest"]:
            raise ValueError("policy digest mismatch")
        if bundle.get("row_hash") != cert["claim_digest"]:
            raise ValueError("row hash mismatch")
        if digest(row) != cert["claim_digest"]:
            raise ValueError("row does not hash to claim_digest")
        if _binding_id(cert) != cert["certificate_id"]:
            raise ValueError("certificate binding mismatch")
        checks["binding"] = True
    except Exception as exc:
        checks["binding"] = False
        return _report(False, False, checks, cert,
                       errors=(f"binding failure: {exc}",), spec=spec,
                       key_id=bundle.get("key_id"))

    replayable = bool(cert.get("replayable"))
    if not replayable:
        checks["replay"] = True
        return _report(True, False, checks, cert, spec=spec,
                       key_id=bundle.get("key_id"))

    try:
        expected = audit_expectations(policy_ir, facts)
        if (expected["verdict"], tuple(expected["basis"]), expected["missing"]) != (
                cert["verdict"], tuple(cert["basis"]), cert.get("missing")):
            raise ValueError("replay disagrees with certificate")
        if expected["guarantee"] != cert.get("guarantee"):
            raise ValueError("guarantee disagrees with policy")
        if cert.get("assurance") not in (None, expected["assurance"]):
            raise ValueError("assurance disagrees with policy")
        checks["replay"] = True
    except Exception as exc:
        return _report(False, True, checks, cert,
                       errors=(f"replay failure: {exc}",), spec=spec,
                       key_id=bundle.get("key_id"))
    return _report(True, True, checks, cert, spec=spec,
                   key_id=bundle.get("key_id"))
