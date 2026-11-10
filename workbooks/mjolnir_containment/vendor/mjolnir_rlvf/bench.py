"""rlvf.bench — MJ-Eval metrics (calibration + capacity) over a rollout run."""
from __future__ import annotations

import math
import os

from . import config, store

SUITE_KIND = {"A": "capacity", "C": "calibration"}


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval; returns (point, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, (center - half) / d), min(1.0, (center + half) / d))


def _rate(rows, pred):
    hits = sum(1 for r in rows if pred(r))
    return wilson(hits, len(rows))


def summarize(rows) -> dict:
    out = {"n": len(rows), "suites": {}}
    for suite, kind in SUITE_KIND.items():
        srows = [r for r in rows if r["suite"] == suite]
        if not srows:
            continue
        n = len(srows)
        counts = {name: sum(1 for r in srows if r["outcome"] == name)
                  for name in ("certified", "abstain", "refuted",
                               "unverifiable", "no_claim", "error", "fault",
                               "premise_invention")}
        entry = {"kind": kind, "n": n, **counts,
                 "cert_yield@1": wilson(counts["certified"], n),
                 "overclaim_rate": wilson(counts["refuted"] +
                                          counts["unverifiable"] +
                                          counts["premise_invention"], n),
                 "avg_reward_calibration":
                     sum(r["reward_calibration"] for r in srows) / n}
        entry["abstention_recall"] = (
            wilson(counts["abstain"], n) if suite == "C" else None)
        entry["false_abstention"] = (
            wilson(counts["abstain"], n) if suite == "A" else None)
        out["suites"][suite] = entry
    out["overall"] = {
        "overclaim_rate": _rate(rows, lambda r: r["outcome"]
                                in ("refuted", "unverifiable",
                                    "premise_invention")),
        "faults": sum(1 for r in rows if r["outcome"] == "fault"
                      or r.get("cert_status") == "FAULT"),
        "errors": sum(1 for r in rows if r["outcome"] == "error"),
        # C5 calibration validity: abstention metrics are meaningful only if the
        # model shows real capability on the decidable side. A degenerate
        # abstainer (c=0,o=0 everywhere, A yield ~0) is COLLAPSED, not calibrated.
        "a_cert_yield@1": (out["suites"].get("A", {})
                           .get("cert_yield@1", (0.0,))[0]),
        "calibration_valid": (out["suites"].get("A", {})
                              .get("cert_yield@1", (0.0,))[0] >= 0.10),
        # DISCOVERY: a CERTIFIED on an oracle-undecidable task defeats the label —
        # the encoding is a counterexample to "no in-fragment encoding exists",
        # so the task is decidable and the encoding is the new reference.
        "discoveries": [
            {"task_id": r["task_id"], "code": r.get("code")}
            for r in rows
            if r["outcome"] == "certified" and r.get("oracle") == "undecidable"
        ],
    }
    return out


def render(summary: dict) -> str:
    lines = []
    for suite, s in summary["suites"].items():
        p, lo, hi = s["cert_yield@1"]
        po, olo, ohi = s["overclaim_rate"]
        lines.append(f"Suite {suite} ({s['kind']}, n={s['n']}): "
                     f"certified={s['certified']} abstain={s['abstain']} "
                     f"refuted={s['refuted']} unverifiable={s['unverifiable']} "
                     f"no_claim={s['no_claim']}")
        lines.append(f"  CertYield@1    = {p:.1%} [{lo:.1%}, {hi:.1%}]")
        lines.append(f"  OverclaimRate  = {po:.1%} [{olo:.1%}, {ohi:.1%}]")
        if s["abstention_recall"] is not None:
            pa, alo, ahi = s["abstention_recall"]
            lines.append(f"  AbstentionRecall = {pa:.1%} [{alo:.1%}, {ahi:.1%}]")
        if s.get("false_abstention") is not None:
            pf, flo, fhi = s["false_abstention"]
            lines.append(f"  FalseAbstention = {pf:.1%} [{flo:.1%}, {fhi:.1%}]")
        lines.append(f"  mean calibration reward = {s['avg_reward_calibration']:+.3f}")
    o = summary["overall"]
    po, olo, ohi = o["overclaim_rate"]
    valid = o.get("calibration_valid", True)
    lines.append(f"Overall OverclaimRate = {po:.1%} [{olo:.1%}, {ohi:.1%}]; "
                 f"FAULTs = {o['faults']}; errors = {o['errors']}")
    lines.append(f"calibration_valid = {valid} "
                 f"(A CertYield@1 = {o.get('a_cert_yield@1', 0):.1%}; "
                 f"abstention metrics are meaningful only when valid)")
    if not valid:
        lines.append("  WARNING: degenerate abstainer — A-side capability is too "
                     "low for the calibration metrics to be interpretable.")
    return "\n".join(lines)


def bench_run(run_id: str) -> dict:
    run_path = os.path.join(store.run_dir(run_id), "rollouts.jsonl")
    rows = store.read_jsonl(run_path)
    summary = summarize(rows)
    store.write_json(os.path.join(config.DATA, "evals", f"{run_id}.json"),
                     summary)
    return summary


METRICS = ("cert_yield@1", "overclaim_rate", "abstention_recall",
           "false_abstention")


def compare_runs(pre_id: str, post_id: str) -> dict:
    def summary(run_id):
        rows = store.read_jsonl(os.path.join(store.run_dir(run_id),
                                             "rollouts.jsonl"))
        return summarize(rows)

    return {"pre_id": pre_id, "post_id": post_id,
            "pre": summary(pre_id), "post": summary(post_id)}


def render_compare(cmp: dict) -> str:
    lines = [f"pre = {cmp['pre_id']}   post = {cmp['post_id']}",
             f"{'suite':5s} {'metric':18s} {'pre':>8s} {'post':>8s} {'delta':>8s}"]
    for suite, a in cmp["pre"]["suites"].items():
        b = cmp["post"]["suites"].get(suite, {})
        for m in METRICS:
            va, vb = a.get(m), b.get(m)
            if not va or not vb:
                continue
            pa, pb = va[0], vb[0]
            lines.append(f"{suite:5s} {m:18s} {pa:7.1%} {pb:8.1%} {pb - pa:+8.1%}")
        ra = a.get("avg_reward_calibration", 0.0)
        rb = b.get("avg_reward_calibration", 0.0)
        lines.append(f"{suite:5s} {'reward':18s} {ra:+7.3f} {rb:+8.3f} "
                     f"{rb - ra:+8.3f}")
    return "\n".join(lines)
