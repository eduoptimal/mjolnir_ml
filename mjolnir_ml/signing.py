"""Ed25519 signing for audit bundles. Issuer-only, no model dependency."""
from __future__ import annotations

import base64
import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def public_bytes(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_bytes(public_key)
    return "ed25519:" + base64.urlsafe_b64encode(
        hashlib.sha256(raw).digest()).decode().rstrip("=")


def public_key_b64(public_key: Ed25519PublicKey) -> str:
    return base64.b64encode(public_bytes(public_key)).decode()


def private_from_bytes(raw: bytes) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(raw)


def sign(private_key: Ed25519PrivateKey, data: bytes) -> str:
    return "ed25519:" + base64.b64encode(
        private_key.sign(data)).decode()


def verify(public_key: Ed25519PublicKey, signature: str, data: bytes) -> None:
    if not isinstance(signature, str) or not signature.startswith("ed25519:"):
        raise ValueError("unsupported signature scheme")
    raw = base64.b64decode(signature.split(":", 1)[1])
    public_key.verify(raw, data)
