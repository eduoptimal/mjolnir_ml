"""Auditor-side Ed25519 verification. No issuer imports."""
from __future__ import annotations

import base64
import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return "ed25519:" + base64.urlsafe_b64encode(
        hashlib.sha256(raw).digest()).decode().rstrip("=")


def public_from_text(text: str) -> Ed25519PublicKey:
    raw = base64.b64decode(text)
    return Ed25519PublicKey.from_public_bytes(raw)


def verify(public_key: Ed25519PublicKey, signature: str, data: bytes) -> None:
    if not isinstance(signature, str) or not signature.startswith("ed25519:"):
        raise ValueError("unsupported signature scheme")
    raw = base64.b64decode(signature.split(":", 1)[1])
    public_key.verify(raw, data)
