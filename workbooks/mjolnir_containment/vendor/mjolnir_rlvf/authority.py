"""rlvf.authority — the authority fingerprint.

The verdicts define the labels, so every run records the fingerprint of the
authority that produced it. Any change to the authority invalidates prior runs,
datasets, and adapters keyed to the old fingerprint; downstream stages refuse to
consume stale data. This is the anti-propagation mechanism.
"""
from __future__ import annotations

import hashlib
import os


def fingerprint() -> str:
    """The authority fingerprint — single source in mjolnir.lib.authority."""
    from mjolnir_prover.lib.authority import fingerprint as _fp
    return _fp()


def benchmark_fingerprint() -> str:
    """Content hash of the benchmark task set (tasks change = labels change)."""
    from . import config
    d = os.path.join(config.BENCH, "tasks")
    h = hashlib.sha256()
    for name in sorted(os.listdir(d)):
        with open(os.path.join(d, name), "rb") as f:
            h.update(name.encode())
            h.update(f.read())
    return h.hexdigest()
