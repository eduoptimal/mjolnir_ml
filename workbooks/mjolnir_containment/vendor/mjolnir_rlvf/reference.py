"""rlvf.reference — authored reference solutions used to validate the oracle.

`rlvf probe` disposes these through the authority offline (no LLM):
  - Suite A references must come back CERTIFIED (the contract is decidable);
  - Suite C references must come back NOT certified (out of fragment / undecidable),
    so the correct proposer behaviour there is ABSTAIN.
"""
from __future__ import annotations

import os

from . import config, tasks

# Suite A: a straight-line implementation that satisfies the contract.
_A_SOLUTIONS = {
    "A-add": "def add(a, b):\n    result = a + b\n    assert result == a + b\n",
    "A-double": "def double(n):\n    result = n * 2\n    assert result == 2 * n\n",
    "A-triple-plus-one": "def triple_plus_one(n):\n    result = 3 * n + 1\n"
                         "    assert result == 3 * n + 1\n",
    "A-square": "def square(n):\n    result = n * n\n    assert result == n * n\n",
    "A-rect-area": "def rect_area(w, h):\n    result = w * h\n"
                   "    assert result == w * h\n",
    "A-rect-perimeter": "def rect_perimeter(w, h):\n    result = 2 * w + 2 * h\n"
                        "    assert result == 2 * (w + h)\n",
    "A-distribute": "def distribute(a, b, c):\n    result = a * (b + c)\n"
                    "    assert result == a * b + a * c\n",
    "A-affine": "def affine(n):\n    result = 5 * n - 3\n"
                "    assert result == 5 * n - 3\n",
    "A-cancel": "def cancel(x, y):\n    result = x + y - y\n"
                "    assert result == x\n",
    "A-sub-identity": "def sub_identity(a, b):\n    result = a + b - b\n"
                      "    assert result == a\n",
    "A-diff-squares": "def diff_squares(a, b):\n    result = a * a - b * b\n"
                      "    assert result == (a + b) * (a - b)\n",
    "A-binomial": "def binomial(n):\n    result = n * n + 2 * n + 1\n"
                  "    assert result == (n + 1) * (n + 1)\n",
    "A-loop-sum": "def loop_sum(n):\n    result = 0\n    for i in range(n):\n"
                  "        result += i\n    assert result == n * (n - 1) // 2\n",
    "A-loop-const": "def loop_const(n):\n    result = 0\n    for i in range(n):\n"
                    "        result += 5\n    assert result == 5 * n\n",
    "A-loop-count": "def loop_count(n):\n    result = 0\n    for i in range(n):\n"
                    "        result += 1\n    assert result == n\n",
    "A-loop-sq": "def loop_sq(n):\n    result = 0\n    for i in range(n):\n"
                 "        result += i * i\n"
                 "    assert result == n * (n - 1) * (2 * n - 1) // 6\n",
    "A-loop-affine": "def loop_affine(n):\n    result = 0\n    for i in range(n):\n"
                     "        result += 3 * i + 1\n"
                     "    assert result == 3 * (n * (n - 1) // 2) + n\n",
    "A-prem-nonneg": "def prem_nonneg(n):\n    assert n >= 0\n    result = n + 1\n"
                     "    assert result > 0\n",
    "A-prem-guard": "def prem_guard(n):\n    if n < 0:\n        raise ValueError\n"
                    "    result = 2 * n\n    assert result >= 0\n",
    "A-prem-strict": "def prem_strict(n):\n    assert n > 0\n    result = n - 1\n"
                     "    assert result >= 0\n",
    "A-bound-le": "def bound_le(x):\n    assert x >= 0\n    assert x <= 10\n"
                  "    result = x - 5\n    assert result <= 5\n",
    "A-bound-ge": "def bound_ge(x):\n    assert x >= 1\n    result = x + 1\n"
                  "    assert result >= 2\n",
    "A-sys-sum": "def sys_sum(a, b):\n    assert a + b == 10\n"
                 "    assert a - b == 2\n    assert a == 6\n",
    "A-sys-scale": "def sys_scale(a, b):\n    assert 2 * a == 8\n"
                   "    assert a + b == 7\n    assert b == 3\n",
    "A-combo-loop-prem": "def combo_loop_prem(n):\n    assert n >= 0\n"
                         "    result = 0\n    for i in range(n):\n"
                         "        result += i\n"
                         "    assert result == n * (n - 1) // 2\n",
    "A-combo-loop-bound": "def combo_loop_bound(n):\n    assert n <= 100\n"
                          "    result = 0\n    for i in range(n):\n"
                          "        result += 1\n    assert result <= 100\n",
}


def solution(task: dict):
    """The authored candidate for a task (A: real solution; C: the naive claim
    that must NOT certify)."""
    if task["suite"] == "A":
        return _A_SOLUTIONS.get(task["id"])
    return (f"def {task['fn']}({task.get('signature', '')}):\n"
            f"    assert {task['contract']}\n")


def probe_rows():
    """Prefer the benchmark's oracle ledger; fall back to authored solutions."""
    from . import store
    ledger = os.path.join(config.BENCH, "oracle.jsonl")
    if os.path.exists(ledger):
        tm = tasks.by_id()
        out = []
        for row in store.read_jsonl(ledger):
            task = tm.get(row["id"])
            if task is not None:
                out.append({"task": task, "code": row["reference"]})
        if out:
            return out
    return [{"task": task, "code": solution(task)} for task in tasks.load_all()]
