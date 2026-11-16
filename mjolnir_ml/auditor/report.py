"""Auditor-facing report object."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuditReport:
    schema: str
    valid: bool
    replayable: bool
    checks: dict
    verdict: str | None
    basis: tuple[str, ...] = ()
    missing: str | None = None
    errors: tuple[str, ...] = ()
    spec_version: str | None = None
    key_id: str | None = None
    certificate_id: str | None = None

    @property
    def exit_code(self) -> int:
        if self.valid:
            return 0
        if any("unsupported" in e for e in self.errors):
            return 3
        if any("signature" in e or "trust" in e for e in self.errors):
            return 4
        if any("malformed" in e or "io" in e.lower() for e in self.errors):
            return 5
        return 2

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "valid": self.valid,
            "replayable": self.replayable,
            "checks": dict(self.checks),
            "verdict": self.verdict,
            "basis": list(self.basis),
            "missing": self.missing,
            "errors": list(self.errors),
            "spec_version": self.spec_version,
            "key_id": self.key_id,
            "certificate_id": self.certificate_id,
        }
