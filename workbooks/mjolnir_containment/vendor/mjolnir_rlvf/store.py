"""rlvf.store — JSON Lines / JSON persistence confined to rlvf/data/."""
from __future__ import annotations

import json
import os
import time
import uuid

from . import config


def new_run_id(tag: str = "run") -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{tag}-{stamp}-{uuid.uuid4().hex[:6]}"


def run_dir(run_id: str, create: bool = False) -> str:
    path = os.path.join(config.DATA, "runs", run_id)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _makedirs(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_jsonl(path: str, rows) -> None:
    _makedirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: str, row) -> None:
    _makedirs(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, obj) -> None:
    _makedirs(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)
