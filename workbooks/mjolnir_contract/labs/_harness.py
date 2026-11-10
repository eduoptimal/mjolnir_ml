"""Shared helpers for the mjolnir_contract (Volume I) workbook labs.

The workbook lives inside the ``mjolnir_ml`` checkout, so the harness puts the
repository root on ``sys.path`` when the engine has not been installed. A lab
therefore runs from a clean checkout with ``python labs/lab01_totality.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="replay the pinned deterministic fixture instead of sampling",
    )
    return parser.parse_args()


def emit(name: str, payload: dict[str, Any], replay: bool) -> None:
    record = {"lab": name, "replay": replay}
    record.update(payload)
    print(json.dumps(record, indent=2, sort_keys=True, default=str))


def out_of_regime(basis) -> bool:
    return any(str(b).startswith("out_of_regime:") for b in basis)


def normalize_basis(basis) -> list[str]:
    """Map the engine's fine-grained basis to the book's vocabulary."""
    out = []
    for b in basis:
        out.append("OUT_OF_REGIME_INPUT" if str(b).startswith("out_of_regime:")
                   else b)
    return out
