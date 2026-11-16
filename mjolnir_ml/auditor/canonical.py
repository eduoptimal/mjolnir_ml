"""Canonical JSON and digest helpers for independent audit replay."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=repr, ensure_ascii=False).encode(
                          "utf-8", "surrogateescape")


def digest(obj) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(obj)).hexdigest()


def digest_parts(*parts) -> str:
    blob = "|".join(p if isinstance(p, str) else repr(p) for p in parts)
    return "sha256:" + hashlib.sha256(
        blob.encode("utf-8", "surrogateescape")).hexdigest()


def row_digest(row: Mapping) -> str:
    return digest(dict(row))
