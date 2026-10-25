"""mjolnir_ml.certificate — signed, third-party-replayable decision proofs.

A certificate is the product of this package: one dict that an auditor can
check WITHOUT the learner, the training data, or this repository's live stack.

    cert = certify(disposition, key=b"audit-key")      # HMAC-SHA256
    verify_certificate(cert, key=b"audit-key", policy=Policy)   # True

Three checks, each fail-closed:

  1. structure   schema, verdict vocabulary, sha256-shaped digests, id
  2. signature   HMAC over the canonical certificate minus the signature
                 (`hmac.compare_digest`; a present signature requires a key)
  3. binding     certificate_id is RECOMPUTED from the pinned digests +
                 verdict/basis/guarantee (the model._pack recipe), and an
                 optional `row` is hashed against claim_digest — so the id
                 cannot be rewritten and the published facts cannot be
                 swapped for another row's
  4. replay      for decisions the policy made over a complete fact table, the
                 verdict/basis/missing are RE-DERIVED from the published facts
                 with Policy.dispose_facts — no estimator involved — and the
                 guarantee/assurance tags are re-derived from the policy, so
                 an unsigned certificate cannot relabel them either.

The replay path is estimator-class-agnostic by construction: live dispose
calls the SAME Policy.dispose_facts the auditor calls, over the same fact
table (regressor lo/hi, classifier conf/p_positive/certified/set_size,
risk p_event/risk_release/risk_deny — whatever the policy requires).

What is attested rather than replayed: harness-stage abstentions that depend
on the model or the training bounds (model_error, invalid_prediction,
out_of_regime*, regime_not_fitted) and missing/invalid fact detections. Those
carry `replayable=False`; the signature and digests still bind them, but a
third party cannot recompute them from facts alone. The certificate says so
instead of pretending otherwise.

Digest pinning (all four, so drift is a hard mismatch, not a discovery):
    claim_digest        canonical JSON of the input row
    policy_digest       canonical JSON of the policy IR
    model_digest        estimator class + params + fitted state repr
    calibration_digest  conformal score table + alpha + group
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Mapping

from mjolnir_ml.model import (Disposition, _assurance, _claim_digest,
                              _digest_parts)
from mjolnir_ml.policy import Policy
from mjolnir_ml.vendors import wire

CERT_SCHEMA = "mjolnir.ml.certificate/1"
SIG_SCHEME = "hmac-sha256"

# Reasons whose proof depends on the learner/training bounds (attested).
_ATTESTED_PREFIXES = ("out_of_regime", "regime_not_fitted", "model_error",
                      "invalid_prediction", "missing:", "invalid:")


class CertificateError(ValueError):
    """The certificate does not verify (fail-closed: never a warning)."""


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=repr, ensure_ascii=False).encode("utf-8",
                                                               "surrogateescape")


def _keyb(key) -> bytes:
    return key.encode("utf-8") if isinstance(key, str) else bytes(key)


def _signature(cert: dict, key) -> str:
    body = {k: v for k, v in cert.items() if k != "signature"}
    mac = hmac.new(_keyb(key), _canonical(body), hashlib.sha256)
    return f"{SIG_SCHEME}:{mac.hexdigest()}"


def _binding_id(cert: dict) -> str:
    """Recompute the certificate id from its pinned parts.

    The recipe is DecisionModel._pack's, imported so the live path and the
    audit path cannot drift: claim | policy | model | calibration | risk |
    regime | verdict | basis | guarantee."""
    return _digest_parts(cert["claim_digest"], cert["policy_digest"],
                         cert["model_digest"],
                         cert.get("calibration_digest") or "-",
                         cert.get("risk_digest") or "-",
                         cert.get("regime_digest") or "-",
                         cert["verdict"], ",".join(cert["basis"]),
                         cert["guarantee"])


def _replayable(disp: Disposition) -> bool:
    if not disp.facts:
        return False
    return not str(disp.reason).startswith(_ATTESTED_PREFIXES)


def certify(disp: Disposition, *, key=None) -> dict:
    """Disposition -> certificate dict. `key` signs it (omit for unsigned)."""
    if not isinstance(disp, Disposition):
        raise CertificateError("certify() takes a Disposition")
    cert = {
        "schema": CERT_SCHEMA,
        "certificate_id": disp.certificate_id,
        "verdict": disp.verdict,
        "basis": list(disp.basis),
        "missing": disp.missing,
        "reason": disp.reason,
        "guarantee": disp.guarantee,
        "assurance": disp.assurance,
        "facts": dict(disp.facts),
        "prediction": disp.prediction,
        "claim_digest": disp.claim_digest,
        "policy_digest": disp.policy_digest,
        "model_digest": disp.model_digest,
        "calibration_digest": disp.calibration_digest,
        "risk_digest": disp.risk_digest,
        "regime_digest": disp.regime_digest,
        "alpha": disp.alpha,
        "replayable": _replayable(disp),
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cert["signature"] = _signature(cert, key) if key is not None else None
    return cert


def verify_certificate(cert: dict, *, key=None, policy=None,
                       require_signature: bool = False,
                       row: Mapping | None = None) -> bool:
    """Verify structure, signature, binding and (when possible) replay.

    `row`, when given, is the stated input the decision was made on: its
    canonical hash must equal the certificate's claim_digest (input-integrity
    check, both classes — the recipe is the live path's). Raises
    CertificateError with the first failing check; returns True otherwise."""
    if not isinstance(cert, dict):
        raise CertificateError("certificate must be a dict")
    if cert.get("schema") != CERT_SCHEMA:
        raise CertificateError(f"unknown certificate schema "
                               f"{cert.get('schema')!r}")
    if cert.get("verdict") not in wire.WIRE:
        raise CertificateError(f"verdict {cert.get('verdict')!r} is not in "
                               f"{sorted(wire.WIRE)}")
    if not isinstance(cert.get("basis"), list) or not all(
            isinstance(b, str) for b in cert["basis"]):
        raise CertificateError("basis must be a list of strings")
    for field in ("certificate_id", "claim_digest", "policy_digest",
                  "model_digest", "guarantee"):
        if not isinstance(cert.get(field), str) or not cert[field]:
            raise CertificateError(f"missing {field}")
    if not isinstance(cert.get("facts"), dict):
        raise CertificateError("missing facts table")
    if row is not None and not isinstance(row, Mapping):
        raise CertificateError("row must be a mapping of stated facts")
    assurance = cert.get("assurance", "declared")
    if assurance not in ("declared", "model_backed", "pointwise_coverage",
                         "action_risk_control", "dual_track"):
        raise CertificateError(f"unknown assurance {assurance!r}")
    if cert.get("risk_digest") is not None and not isinstance(
            cert["risk_digest"], str):
        raise CertificateError("risk_digest must be a string or null")
    if cert.get("regime_digest") is not None and not isinstance(
            cert["regime_digest"], str):
        raise CertificateError("regime_digest must be a string or null")

    sig = cert.get("signature")
    if sig is not None:
        if key is None:
            raise CertificateError(
                "certificate is signed but no key was supplied (fail-closed)")
        if not isinstance(sig, str) or not sig.startswith(SIG_SCHEME + ":"):
            raise CertificateError(f"unknown signature scheme {sig!r}")
        expect = _signature(cert, key)
        if not hmac.compare_digest(sig, expect):
            raise CertificateError("signature does not match the certificate")
    elif require_signature:
        raise CertificateError("certificate is unsigned")

    # Binding: the id is a function of the pinned parts — recompute it so an
    # unsigned certificate (or a re-signed one) cannot carry a rewritten id.
    if _binding_id(cert) != cert["certificate_id"]:
        raise CertificateError(
            "certificate_id does not match the pinned digests "
            "(binding check failed)")

    # Input integrity: the auditor supplies the row; its hash must be the
    # claim_digest the certificate pins.
    if row is not None and _claim_digest(row) != cert["claim_digest"]:
        raise CertificateError(
            "input tamper: the supplied row does not hash to the "
            "certificate's claim_digest")

    if policy is not None:
        if isinstance(policy, type):
            policy = policy.policy()
        if isinstance(policy, dict):
            policy = Policy(policy)
        if not isinstance(policy, Policy):
            raise CertificateError("policy must be a Policy, IR or @dsl class")
        if cert["policy_digest"] != policy.digest:
            raise CertificateError(
                "policy digest mismatch: the certificate was not produced by "
                "this policy")
        # Guarantee and assurance are policy-derived, not signer-asserted:
        # recompute both from the deciding statements (same recipes as
        # DecisionModel._pack — estimator-class-agnostic).
        want_guarantee = ("model_backed"
                          if policy.uses_model_facts(tuple(cert["basis"]))
                          else "declared")
        if cert["guarantee"] != want_guarantee:
            raise CertificateError(
                f"guarantee disagrees with the policy: the certificate says "
                f"{cert['guarantee']!r}, the deciding statements require "
                f"{want_guarantee!r}")
        want_assurance = _assurance(policy, cert["verdict"],
                                    tuple(cert["basis"]), cert.get("missing"))
        if cert.get("assurance", "declared") != want_assurance:
            raise CertificateError(
                f"assurance disagrees with the policy: the certificate says "
                f"{cert.get('assurance', 'declared')!r}, the deciding "
                f"statements require {want_assurance!r}")
        if cert.get("replayable"):
            v, b, m = policy.dispose_facts(cert["facts"])
            if (v, tuple(b), m) != (cert["verdict"], tuple(cert["basis"]),
                                    cert.get("missing")):
                raise CertificateError(
                    f"replay disagrees: certificate says "
                    f"({cert['verdict']}, {cert['basis']}, {cert.get('missing')}"
                    f"), policy over the published facts says ({v}, {list(b)}, "
                    f"{m})")
    return True


__all__ = ["certify", "verify_certificate", "CertificateError", "CERT_SCHEMA"]
