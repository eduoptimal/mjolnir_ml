"""Shared helpers for the mjolnir_containment workbook labs."""

from __future__ import annotations

import argparse
import json
from typing import Any


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
    print(json.dumps(record, indent=2, sort_keys=True))
