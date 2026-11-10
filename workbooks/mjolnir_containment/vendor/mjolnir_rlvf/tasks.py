"""rlvf.tasks — contract-conditioned prompt suites.

Suite A: decidable contracts (a reference implementation certifies under the
authority). Suite C: out-of-fragment contracts whose correct behaviour is an
explicit ABSTAIN. The oracle label is authored for the MVP and validated offline
by `rlvf probe`; it is replaced by an empirical oracle once the miner/proposer
can establish decidability directly.
"""
from __future__ import annotations

import hashlib
import os

from . import config

PROMPT_TEMPLATE = """You are a coding assistant. Implement the function below.

Supported: integer and exact-rational arithmetic (+ - * // %), straight-line
assignments, and assert statements. Unsupported: method calls, mutation,
subscripts, strings, floats.

TASK: {task}
FUNCTION: {fn}({signature})
Compute the answer into a variable named `result`, then include exactly: assert {contract}

Output ONLY the Python function (no prose). If the task needs an unsupported
feature, output ABSTAIN: <short reason>."""


def suite_path(name: str) -> str:
    return os.path.join(config.SUITES, f"suite{name.upper()}.jsonl")


def load_suite(name: str):
    from . import store
    return store.read_jsonl(suite_path(name))


def load_all():
    tasks = []
    for name in ("A", "C"):
        for task in load_suite(name):
            task.setdefault("suite", name)
            tasks.append(task)
    return tasks


def by_id():
    return {t["id"]: t for t in load_all()}


def render(task: dict) -> str:
    return PROMPT_TEMPLATE.format(
        task=task["task"],
        fn=task["fn"],
        signature=task.get("signature", ""),
        contract=task["contract"],
    )


def content_hash(task: dict) -> str:
    payload = f"{task['id']}|{task['suite']}|{task['contract']}|{task['task']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frozen_split(seed: int = 0, eval_frac: float = 0.34):
    """Deterministic, content-derived train/eval partition (no random state).

    Eval tasks are never used for data generation, and the manifest freezes
    their content hashes so the eval set cannot drift between runs.
    """
    parts = {"train": [], "eval": []}
    cutoff = int(eval_frac * 1000)
    for t in load_all():
        h = int(hashlib.sha256(f"{seed}:{t['id']}".encode()).hexdigest(), 16)
        parts["eval" if (h % 1000) < cutoff else "train"].append(t)
    return parts


def split_path(name: str) -> str:
    return os.path.join(config.SPLITS, f"{name}.jsonl")


def ids_in_split(name: str):
    from . import store
    return {row["id"] for row in store.read_jsonl(split_path(name))}


def manifest(seed: int = 0, eval_frac: float = 0.34) -> dict:
    parts = frozen_split(seed, eval_frac)
    eval_ids = {t["id"] for t in parts["eval"]}
    return {"seed": seed, "eval_frac": eval_frac,
            "train": len(parts["train"]), "eval": len(parts["eval"]),
            "hashes": {t["id"]: content_hash(t) for t in load_all()},
            "split": {t["id"]: ("eval" if t["id"] in eval_ids else "train")
                      for t in load_all()}}
