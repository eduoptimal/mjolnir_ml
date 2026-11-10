"""Declared thresholds must name the quantity they were calibrated on.

A threshold compared against a different quantity is meaningless — and silently
so. Measured three times in one day: an impossible majority bar (counted over
distinct combinations), a class vocabulary compared across producers
(`mutating` vs `mutates`), and an OOD floor calibrated on a per-dimension flow
density but compared against the total joint log-likelihood (so it fired on
every input and the service answered a constant 0).

    validate_threshold({"quantity": "flow_score_per_dim", "value": -2000,
                        "calibrated_on": {"in_domain": {...}, "ood": {...}}},
                       quantity_used="flow_score_per_dim")
"""
from __future__ import annotations


def validate_threshold(block: dict, quantity_used: str | None = None):
    """-> (ok, reasons). Pure arithmetic; no I/O, no model."""
    bad: list[str] = []
    if not isinstance(block, dict) or not block:
        return False, ["threshold block missing"]
    q, v, cal = block.get("quantity"), block.get("value"), block.get("calibrated_on")
    if not q:
        bad.append("threshold without a declared quantity")
    if quantity_used is not None and q and q != quantity_used:
        bad.append(f"threshold calibrated on {q!r} but compared to {quantity_used!r}")
    if v is not None:
        if not isinstance(v, (int, float)):
            bad.append(f"value must be numeric or null (got {v!r})")
        if not isinstance(cal, dict) or not cal.get("in_domain") or not cal.get("ood"):
            bad.append("threshold without calibration tails (in_domain/ood)")
        else:
            ood, dom = cal["ood"].get("mean"), cal["in_domain"].get("mean")
            if isinstance(ood, (int, float)) and isinstance(dom, (int, float)):
                if not ood < v < dom:
                    bad.append(f"impossible threshold: {v} not between "
                               f"ood {ood} and in_domain {dom}")
    return (not bad), bad
