"""rlvf.rollout — generate K candidates per task, dispose through the authority.

The authority (mjolnir.lib.certify) is the ONLY verdict writer. This module
consumes its certificates; it never imports a checker and never calls the prover
directly. A candidate is classified from the authority's status, whether the
required contract was actually encoded, and — for behaviour purposes — whether
the model chose to abstain.
"""
from __future__ import annotations

import ast
import os
import re

from mjolnir_prover.lib import certify

from . import authority, config, proposer, reward, store, tasks


def _norm(node) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _contract_node(contract: str):
    return ast.parse(contract, mode="eval").body


def required_assert_present(code: str, contract: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    want = _norm(_contract_node(contract))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and _norm(node.test) == want:
            return True
    return False


def _emitted_premises(code: str, params: set | None = None) -> dict:
    """Emitted premise bounds: {var: [lower, upper]} from asserts of the shape
    `assert v >= c` / `assert v <= c` / `assert lo <= v <= hi`.

    Only PARAMETER variables count as premises — the contract's own predicate
    (over `result` and other locals) is a postcondition, not a premise, and must
    be excluded or every certification is flagged as invention."""
    if params is None:
        params = set()
    out: dict = {}
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Assert):
            continue
        txt = ast.unparse(node.test)
        for m in re.finditer(r"(\b[a-zA-Z_]\w*)\s*(?:>=|≤)\s*(-?\d+)", txt):
            var, c = m.group(1), int(m.group(2))
            if var in params:
                out.setdefault(var, [None, None])[0] = c
        for m in re.finditer(r"(\b[a-zA-Z_]\w*)\s*(?:<=|≥)\s*(-?\d+)", txt):
            var, c = m.group(1), int(m.group(2))
            if var in params:
                out.setdefault(var, [None, None])[1] = c
    return out


def _premise_fidelity(code: str, task: dict) -> bool:
    """P3: emitted premises must be entailed by the task's declared premises.
    A stricter emitted bound is PREMISE INVENTION — an overclaim the oracle
    cannot see (the artifact-level certificate is sound; the task-level
    obligation is violated)."""
    declared = task.get("declared_premise")
    if not declared:
        return True
    dec: dict = {}
    for m in re.finditer(r"([a-zA-Z_]\w*)\s*(?:>=|≤)\s*(-?\d+)", declared):
        dec.setdefault(m.group(1), [None, None])[0] = int(m.group(2))
    for m in re.finditer(r"([a-zA-Z_]\w*)\s*(?:<=|≥)\s*(-?\d+)", declared):
        dec.setdefault(m.group(1), [None, None])[1] = int(m.group(2))
    fn = task.get("fn")
    params: set = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn:
                params = {a.arg for a in node.args.args}
                break
    except SyntaxError:
        return True
    emitted = _emitted_premises(code, params)
    for var, (lo, hi) in emitted.items():
        d = dec.get(var)
        if d is None:
            return False                     # premise on a var with none declared
        # STRICTER emitted bound = invention (demands more than declared):
        if lo is not None and d[0] is not None and lo > d[0]:
            return False                     # stronger lower bound
        if hi is not None and d[1] is not None and hi < d[1]:
            return False                     # tighter upper bound
    return True


def classify(text: str, task: dict) -> dict:
    """THE single classification entry point.

    Parses the model output, disposes it through the authority, demotes a
    vacuous certificate, and applies premise fidelity.

    There used to be TWO functions here and one omitted the fidelity step.
    Three callers picked the incomplete one — `redispose`, `e4_refine` and
    `cegis` — so a model that silently narrowed the declared domain was
    recorded as `certified` instead of `premise_invention`: the P-series
    fidelity failure, masked by tooling. `contract`/`oracle`/`name` are all
    fields of `task`, so the loose-arg form was redundant as well as unsafe.
    One entry point makes the mistake impossible rather than merely fixed.
    """
    contract, name = task["contract"], task["fn"]
    parsed = proposer.parse(text)
    if parsed["kind"] == "error":
        return {"outcome": "error", "code": None, "cert_status": None,
                "reason": parsed.get("reason"), "cert": None}
    if parsed["kind"] == "abstain":
        return {"outcome": "abstain", "code": None, "cert_status": None,
                "reason": parsed["reason"], "cert": None}
    code = parsed["code"]
    if not required_assert_present(code, contract):
        return {"outcome": "no_claim", "code": code, "cert_status": None,
                "reason": "required_contract_not_encoded", "cert": None}
    try:
        cert, _prof_ok, _viol = certify.certify_unit(name, code)
    except Exception as e:  # disposal must never crash the rollout
        return {"outcome": "error", "code": code, "cert_status": None,
                "reason": f"disposal_error:{type(e).__name__}", "cert": None}
    status = cert.get("status")
    mapping = {"CERTIFIED": "certified", "REFUTED": "refuted",
               "UNVERIFIABLE": "unverifiable", "FAULT": "fault"}
    if status not in mapping:
        # fail-closed: an unknown verdict is a defect, never a soft outcome
        return {"outcome": "fault", "code": code, "cert_status": status,
                "reason": f"unknown_cert_status:{status}", "cert": cert}
    outcome = mapping[status]
    if outcome == "certified" and not reward.is_non_vacuous(cert):
        outcome = "unverifiable"
    reason = cert.get("reason", "")
    if outcome == "certified" and not _premise_fidelity(code or "", task):
        outcome = "premise_invention"
        reason = "emitted_stricter_premise_than_declared"
    # H-R2': a claim naming a quantity the source does not contain is
    # FABRICATION, not "unverifiable". The two are different acts and were
    # scoring identically, so the reward could not punish inventing a metric.
    # Read from the authority's reason -- never re-derived, never guessed.
    if outcome in ("unverifiable", "certified", "refuted") and reward.is_fabrication(cert):
        outcome = "fabricated"
        reason = reason or "quantity_not_in_source"
    return {"outcome": outcome, "code": code, "cert_status": status,
            "reason": reason, "cert": cert}


def run_task(task: dict, model: str, k: int, backend=None, adapter=None):
    prompt = tasks.render(task)
    rows = []
    for i in range(k):
        text = proposer.generate(model, prompt, config.TEMPERATURE,
                                 config.SEED + i, backend=backend,
                                 adapter=adapter)
        cls = classify(text, task)
        cert = cls["cert"] or {}
        rows.append({
            "task_id": task["id"], "suite": task["suite"],
            "rung": task.get("rung"), "oracle": task["oracle"],
            "model": model, "sample": i,
            "outcome": cls["outcome"], "cert_status": cls["cert_status"],
            "reason": cls["reason"],
            "reward_typed": reward.typed_reward(cert.get("reading"),
                                                cls["cert_status"]),
            "reward_calibration": reward.calibration_reward(cls["outcome"],
                                                            task["oracle"]),
            "code": cls["code"],
            "cert": cls["cert"] if cls["cert"] else None,
            "raw": text if cls["outcome"] == "error" else None,
        })
    return rows


def rollout(tasks_list, model: str, k: int, run_id: str, backend=None,
            adapter=None):
    run_dir = store.run_dir(run_id, create=True)
    out = os.path.join(run_dir, "rollouts.jsonl")
    meta = {"run_id": run_id, "model": model, "k": k,
            "backend": backend or config.BACKEND, "adapter": adapter,
            "temperature": config.TEMPERATURE, "seed": config.SEED,
            "authority_fingerprint": authority.fingerprint(),
            "benchmark_fingerprint": authority.benchmark_fingerprint(),
            "tasks": len(tasks_list)}
    store.write_json(os.path.join(run_dir, "meta.json"), meta)
    total = 0
    faults = 0
    for task in tasks_list:
        task_rows = run_task(task, model, k, backend, adapter)
        total += len(task_rows)
        faults += sum(1 for r in task_rows if r["outcome"] == "fault")
        for row in task_rows:
            store.append_jsonl(out, row)
        print(f"[{task['id']}] " + ",".join(r["outcome"] for r in task_rows),
              flush=True)
    store.write_json(os.path.join(run_dir, "summary.json"),
                     {"meta": meta, "rows": total, "faults": faults,
                      "failed": faults > 0})
    if faults:
        print(f"STOP-THE-LINE: {faults} FAULT(s) — prover/checker disagreement. "
              f"This is a mjolnir defect, not model behaviour. Run failed.",
              flush=True)
    return out
