"""WarrantLedger facade: issuer bundle creation and auditor replay."""
from __future__ import annotations

from pathlib import Path


class WarrantLedger:
    def __init__(self, bundle: dict, trust=None):
        self.bundle = dict(bundle)
        self.trust = trust

    @classmethod
    def issue(cls, disposition, policy, row, signer, **kwargs):
        from mjolnir_ml.issue import issue_bundle
        return cls(issue_bundle(disposition, policy, row, signer, **kwargs))

    @classmethod
    def from_bundle(cls, path, trust=None):
        from mjolnir_ml.auditor.bundle import load_bundle
        return cls(load_bundle(path), trust=trust)

    def write(self, path):
        import json
        Path(path).write_text(json.dumps(self.bundle, indent=2, sort_keys=True))
        return Path(path)

    def audit(self, trust=None):
        from mjolnir_ml.auditor.verify import verify_bundle
        return verify_bundle(self.bundle, trust if trust is not None else self.trust)
