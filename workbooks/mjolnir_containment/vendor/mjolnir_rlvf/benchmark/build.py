"""rlvf.benchmark.build — validate generated candidates through the authority and
write the released benchmark (tasks, oracle ledger, frozen splits, manifest).

A candidate is KEPT only if its oracle label holds under the authority:
Suite A reference must CERTIFY; Suite C naive attempt must NOT certify. The
authority's source fingerprint is recorded, because the verdicts define the
labels — a benchmark run must pin the authority version.
"""
from __future__ import annotations

import hashlib
import os

from mjolnir_prover.lib import certify

from .. import config, store
from . import generator

AUTH_FILES = ("certify.py", "certificate.py", "kernel.py", "stepcheck.py")

# Machine-checked regression controls. A false claim over a computed local must
# REFUTE (not FAULT); true claims must CERTIFY; premises must stay premises.
NEGATIVE_CONTROLS = (
    ("ctl_false_eq_linear",
     "def ctl_false_eq_linear(n):\n    x = 2 * n\n    assert x == 3 * n\n",
     "REFUTED"),
    ("ctl_false_eq_quad",
     "def ctl_false_eq_quad(n):\n    r = n + 1\n    assert r == (n + 1) * (n + 1)\n",
     "REFUTED"),
    ("ctl_true_eq_quad",
     "def ctl_true_eq_quad(n):\n    r = n * n + 2 * n + 1\n    assert r == (n + 1) * (n + 1)\n",
     "CERTIFIED"),
    ("ctl_premise_chain",
     "def ctl_premise_chain(a, b):\n    assert a + b == 10\n    assert a - b == 2\n"
     "    assert a == 6\n",
     "CERTIFIED"),
)


def authority_fingerprint() -> str:
    from ..authority import fingerprint
    return fingerprint()


def content_hash(task: dict) -> str:
    payload = "|".join(str(task[k]) for k in
                       ("id", "suite", "contract", "task"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _split(tasks: list, seed: int = 0, eval_frac: float = 0.34):
    parts = {"train": [], "eval": []}
    cutoff = int(eval_frac * 1000)
    for t in tasks:
        h = int(hashlib.sha256(f"{seed}:{t['id']}".encode()).hexdigest(), 16)
        parts["eval" if (h % 1000) < cutoff else "train"].append(t)
    return parts


def _run_controls() -> list:
    """Every control must hold, and none may FAULT. Returns the observed list."""
    out = []
    for name, code, expected in NEGATIVE_CONTROLS:
        cert, _p, _v = certify.certify_unit(name, code)
        status = cert.get("status")
        ok = status == expected
        out.append({"name": name, "expected": expected, "observed": status,
                    "ok": ok})
        if not ok:
            raise RuntimeError(
                f"negative control {name!r} expected {expected}, got {status} "
                f"({cert.get('reason')})")
    return out


def build() -> dict:
    kept, oracle, dropped = [], [], []
    for task, ref in generator.candidates():
        cert, _p, _v = certify.certify_unit(task["fn"], ref)
        status = cert.get("status")
        want_cert = task["oracle"] == "decidable"
        ok = (status == "CERTIFIED") if want_cert else (status != "CERTIFIED")
        if ok:
            kept.append(task)
            oracle.append({"id": task["id"], "oracle": task["oracle"],
                           "reference": ref, "observed_status": status})
        else:
            dropped.append({"id": task["id"], "observed_status": status})

    tasks_dir = os.path.join(config.BENCH, "tasks")
    for suite in ("A", "C"):
        rows = [t for t in kept if t["suite"] == suite]
        store.write_jsonl(os.path.join(tasks_dir, f"suite{suite}.jsonl"), rows)
    store.write_jsonl(os.path.join(config.BENCH, "oracle.jsonl"), oracle)

    parts = _split(kept)
    store.write_jsonl(os.path.join(config.BENCH, "splits", "train.jsonl"),
                      [{"id": t["id"]} for t in parts["train"]])
    store.write_jsonl(os.path.join(config.BENCH, "splits", "eval.jsonl"),
                      [{"id": t["id"]} for t in parts["eval"]])

    manifest = {
        "benchmark": "MJ-Eval", "version": "0.1.0",
        "authority_fingerprint": authority_fingerprint(),
        "counts": {"total": len(kept),
                   "A": sum(1 for t in kept if t["suite"] == "A"),
                   "C": sum(1 for t in kept if t["suite"] == "C"),
                   "dropped": len(dropped)},
        "splits": {"train": len(parts["train"]), "eval": len(parts["eval"])},
        "task_hashes": {t["id"]: content_hash(t) for t in kept},
        "dropped": dropped,
        "negative_controls": _run_controls(),
    }
    store.write_json(os.path.join(config.BENCH, "manifest.json"), manifest)
    return manifest
