"""Audit bundle loading and trust-store parsing."""
from __future__ import annotations

import json
from pathlib import Path

BUNDLE_SCHEMA = "warrantledger.bundle/1"
REPLAY_SPEC = "warrantledger.replay/1"
AUDIT_SCHEMA = "warrantledger.audit/1"


def load_bundle(value):
    if isinstance(value, dict):
        return dict(value)
    path = Path(value)
    return json.loads(path.read_text())


def load_trust(value):
    if isinstance(value, dict):
        data = dict(value)
    else:
        data = json.loads(Path(value).read_text())
    keys = data.get("keys", data)
    if not isinstance(keys, dict):
        raise ValueError("trust store must contain a keys mapping")
    return {str(k): str(v) for k, v in keys.items()}
