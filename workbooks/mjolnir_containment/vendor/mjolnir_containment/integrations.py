"""Explicit adapters to the Volume I and Volume II engines.

The containment workbook does not import ``mjolnir_ml`` at package import
time. Callers that need a Volume I or Volume II primitive cross this module
explicitly, so the boundary is visible in the call graph.
"""

from __future__ import annotations

from typing import Any


def volume_one_certify(*args: Any, **kwargs: Any) -> Any:
    """Call the Volume I certificate constructor."""
    from mjolnir_ml import certify

    return certify(*args, **kwargs)


def volume_one_verify(certificate: Any) -> Any:
    """Verify a Volume I certificate."""
    from mjolnir_ml import verify_certificate

    return verify_certificate(certificate)


def volume_two_hazard_ledger(*args: Any, **kwargs: Any) -> Any:
    """Construct the Volume II hazard ledger."""
    from mjolnir_ml import HazardLedger

    return HazardLedger(*args, **kwargs)


def volume_two_readiness(*args: Any, **kwargs: Any) -> Any:
    """Evaluate the Volume II readiness gate."""
    from mjolnir_ml import readiness

    return readiness(*args, **kwargs)


def volume_two_omega_pair(*args: Any, **kwargs: Any) -> Any:
    """Compute the Volume II Omega pair."""
    from mjolnir_ml import omega_pair

    return omega_pair(*args, **kwargs)


__all__ = [
    "volume_one_certify",
    "volume_one_verify",
    "volume_two_hazard_ledger",
    "volume_two_readiness",
    "volume_two_omega_pair",
]
