"""Declared thresholds, WITH the quantity each was calibrated on.

A threshold is only meaningful against the quantity it is compared to. Measured
today: the per-dimension flow log-density (`flow_score`) is ≈ −76…−800
in-domain and ≈ −10,800…−89,439 for OOD/gibberish, so −2000 sits between the
tails (source: out_ood_smollm_flow.json, AUROC 0.949 vs a 0.9 bar). The code
previously compared −2000 against the TOTAL joint log-likelihood (−14k…−1.4M),
so the floor fired on every input and the service returned a constant 0 —
the same class of error as a threshold compared to the wrong quantity anywhere.
"""
from __future__ import annotations

OOD_FLOOR_DEFAULT = -2000.0
OOD_FLOOR_QUANTITY = "flow_score_per_dim"
OOD_FLOOR_CALIBRATION = {
    "in_domain": {"mean": -76.33, "n": 2000, "source": "out_ood_smollm_flow.json"},
    "ood": {"mean": -89439.01, "n": 1200, "kinds": "800 mismatch + 400 gibberish"},
    "auroc_id_vs_ood": 0.9491, "bar": 0.9, "verdict": "PASS",
}


def declaration(value: float | None = OOD_FLOOR_DEFAULT) -> dict:
    """The threshold block, in the shape the meta-rule validates."""
    return {"quantity": OOD_FLOOR_QUANTITY, "value": value,
            "calibrated_on": OOD_FLOOR_CALIBRATION}


def is_possible(value: float | None = OOD_FLOOR_DEFAULT,
                cal: dict = OOD_FLOOR_CALIBRATION) -> bool:
    """A threshold must lie BETWEEN the measured tails of its quantity."""
    if value is None:
        return True                      # explicitly disabled
    return cal["ood"]["mean"] < value < cal["in_domain"]["mean"]
