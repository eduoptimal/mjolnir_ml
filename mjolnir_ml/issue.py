"""Issuer bundle construction. Model imports are local to this path."""
from __future__ import annotations

import time

from mjolnir_ml.auditor.bundle import BUNDLE_SCHEMA, REPLAY_SPEC
from mjolnir_ml.auditor.canonical import canonical_bytes, digest, row_digest
from mjolnir_ml.signing import key_id, sign


def issue_bundle(disposition, policy, row, signer, *, issuer="mjolnir_ml"):
    from mjolnir_ml.certificate import certify
    from mjolnir_ml.policy import Policy

    if isinstance(policy, type):
        policy = policy.policy()
    if isinstance(policy, dict):
        policy = Policy(policy)
    if not isinstance(policy, Policy):
        raise TypeError("policy must be a Policy, IR dict, or @dsl class")

    certificate = certify(disposition)
    facts = dict(disposition.facts)
    bundle = {
        "bundle_schema": BUNDLE_SCHEMA,
        "spec_version": REPLAY_SPEC,
        "certificate": certificate,
        "policy_ir": policy.ir,
        "facts": facts,
        "row": dict(row),
        "row_hash": row_digest(row),
        "issuer": issuer,
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "key_id": key_id(signer.public_key()),
    }
    body = {k: v for k, v in bundle.items() if k != "signature"}
    bundle["signature"] = sign(signer, canonical_bytes(body))
    if certificate["claim_digest"] != bundle["row_hash"]:
        raise ValueError("row hash does not match certificate claim_digest")
    if digest(policy.ir) != certificate["policy_digest"]:
        raise ValueError("policy digest does not match certificate")
    return bundle
