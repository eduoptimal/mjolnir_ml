"""rlvf.pairs — typed preference pairs from a rollout run (M1 scaffolding).

Pair rules follow rlvf.md §7. The primary pair teaches the mandate:
`ABSTAIN` on an undecidable task is CHOSEN over an overclaim on the same task.
"""
from __future__ import annotations

import os

from . import store, tasks

PRIMARY = "abstain_vs_overclaim"


def _response(row: dict) -> str:
    if row["outcome"] == "abstain":
        return f"ABSTAIN: {row.get('reason') or 'out of fragment'}"
    return row.get("code") or ""


def _first(rows, outcome):
    return next((r for r in rows if r["outcome"] == outcome), None)


MAX_PER_TYPE = 8


def build(rows) -> list:
    groups = {}
    for row in rows:
        groups.setdefault(row["task_id"], []).append(row)
    pairs = []
    for task_id, group in groups.items():
        oracle = group[0]["oracle"]
        # The rendered task prompt. dpo_train.py builds its chat template from
        # `ex["prompt"]`, so pairs must carry it to be DPO-consumable. (It was
        # missing here: `rlvf pairs` output was never directly consumable by
        # dpo_train — the 1.5b run used a separately-built pairs dir.)
        task = tasks.by_id().get(task_id)
        prompt = tasks.render(task) if task is not None else ""
        cert = [r for r in group if r["outcome"] == "certified"]
        refuted = [r for r in group if r["outcome"] == "refuted"]
        unver = [r for r in group if r["outcome"] == "unverifiable"]
        abst = [r for r in group if r["outcome"] == "abstain"]

        def emit(ptype, chosen, rejected):
            n = 0
            for c in chosen:
                for rj in rejected:
                    pairs.append({"pair_type": ptype, "task_id": task_id,
                                  "prompt": prompt,
                                  "chosen": _response(c),
                                  "rejected": _response(rj)})
                    n += 1
                    if n >= MAX_PER_TYPE:
                        return

        if oracle == "undecidable" and abst and (refuted or unver):
            emit(PRIMARY, abst, refuted + unver)
        if cert:
            if refuted:
                emit("certified_vs_refuted", cert, refuted)
            if unver:
                emit("certified_vs_unverifiable", cert, unver)
            if abst:
                emit("certified_vs_abstain", cert, abst)
    return pairs


def write_pairs(run_id: str, valid_frac: float = 0.1):
    """Write DPO-consumable preference pairs for a rollout run.

    Writes BOTH:
      - `pairs.jsonl`                       all pairs (the reference artifact)
      - `pairs/train.jsonl`, `pairs/valid.jsonl`
                                        the layout `dpo_train.py` loads
                                        (`load_dataset("json", data_files=...)`
                                        with a data DIRECTORY)

    Schema per record: {prompt, chosen, rejected, pair_type, task_id} — `prompt`
    is required by dpo_train.py, which applies the chat template to it.

    Previously this function wrote a single file with NO `prompt` field, so its
    output was never consumable by dpo_train.py; the 1.5b run only worked
    because it used a separately-built pairs directory. This makes the
    producer/consumer contract explicit in one place.
    """
    run_dir = store.run_dir(run_id)
    rows = store.read_jsonl(os.path.join(run_dir, "rollouts.jsonl"))
    pairs = build(rows)
    out = os.path.join(run_dir, "pairs.jsonl")
    store.write_jsonl(out, pairs)

    split_dir = os.path.join(run_dir, "pairs")
    os.makedirs(split_dir, exist_ok=True)
    cut = max(1, int(len(pairs) * (1.0 - valid_frac)))
    train, valid = pairs[:cut], pairs[cut:]
    if not valid:                       # a tiny run must still yield a valid set
        train, valid = pairs[:-1], pairs[-1:]
    store.write_jsonl(os.path.join(split_dir, "train.jsonl"), train)
    store.write_jsonl(os.path.join(split_dir, "valid.jsonl"), valid)
    return out, pairs
