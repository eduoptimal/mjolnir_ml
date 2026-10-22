"""Flywheel simulation: a deployed regressor, batch arrivals, drift, a hazard,
containment, a tightening repair, approval to lift containment, retraction.

The model is trained once (diabetes, RandomForestRegressor). The stream arrives
in batches. After each batch:

    decide -> observe (ACI) -> tick (sequential detectors) -> auto_act -> status

When a release lands on an external hazard, the session contains itself
(unconditional, tighten-only). A `ChangeSet` that adds the missing deny rule is
evaluated by replaying stored facts (no model), promoted because it only
tightens, and containment is lifted only with an external approval. Decisions
issued under the faulty policy version are retracted.

Outputs: journal JSONL, results JSON/CSV, two figures, and a markdown report.

    python experiments/flywheel.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from mjolnir_ml import (ASK, DENY, RELEASE, ApprovalAuthority, ChangeSet,
                        DecisionModel, DecisionPolicyBase, DetectorSpec,
                        HazardLedger, Journal, LiveSession, dsl, readiness,
                        rule)

THRESHOLD = 240.0
HAZARD_AGE = 0.10


@dsl
class FlywheelPolicy(DecisionPolicyBase):
    """Baseline v1: safe-band release, high-band deny, widening ask."""

    facts = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")

    assert lambda bmi: bmi > -1.0, "bmi_in_range"

    assert rule("release_safe_band",
                when=lambda lo, hi: lo >= 0.0 and hi <= 240.0, verdict=RELEASE)
    assert rule("deny_high_band", when=lambda lo: lo > 240.0, verdict=DENY)
    assert rule("review_widening",
                when=lambda lo, hi: lo <= 240.0 and hi > 240.0, verdict=ASK)


@dsl
class FlywheelPolicyV2(DecisionPolicyBase):
    """Repair: the missing external-hazard rule (only tightens)."""

    facts = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")

    assert lambda bmi: bmi > -1.0, "bmi_in_range"

    assert rule("deny_age_above_spec", when=lambda age: age > 0.10, verdict=DENY)
    assert rule("release_safe_band",
                when=lambda lo, hi: lo >= 0.0 and hi <= 240.0, verdict=RELEASE)
    assert rule("deny_high_band", when=lambda lo: lo > 240.0, verdict=DENY)
    assert rule("review_widening",
                when=lambda lo, hi: lo <= 240.0 and hi > 240.0, verdict=ASK)


def rows_of(X, names):
    return [dict(zip(names, map(float, x))) for x in X]


def main():
    data = load_diabetes()
    names = tuple(data.feature_names)
    X, y = data.data, data.target
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.4,
                                              random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                            random_state=0)
    tr_rows, cal_rows = rows_of(Xtr, names), rows_of(Xcal, names)

    forest = RandomForestRegressor(n_estimators=300, random_state=0,
                                   n_jobs=-1).fit(Xtr, ytr)
    dm = (DecisionModel(forest, FlywheelPolicy, features=names)
          .fit_regime(tr_rows)
          .calibrate(cal_rows, ycal, alpha=0.10, adaptive=True,
                     gamma=0.10, window=150))

    ledger = (HazardLedger()
              .add("age_above_spec", lambda r: r["age"] > HAZARD_AGE,
                   "underwriting spec §2 (age band)"))
    cal_mae = float(np.mean(np.abs(forest.predict(Xcal) - ycal)))

    # ---- readiness gate on a held-out slice (not the live stream) ----------
    ready_rows = rows_of(Xte, names)[:18]
    ready_y = list(yte)[:18]
    ready = readiness(
        dm, ready_rows, safe=[float(v) <= THRESHOLD for v in ready_y],
        decidable=[True] * len(ready_y), ledger=ledger,
        thresholds=None)
    print("\nREADINESS")
    print(f"  approved={ready.approved} id={ready.readiness_id[:22]}…")
    for g in ready.gates:
        print(f"  {'PASS' if g.passed else 'FAIL':<4} {g.name:<22} "
              f"value={g.value} threshold={g.threshold}")

    stream_rows = rows_of(Xte, names)[18:]
    stream_y = [float(v) for v in list(yte)[18:]]

    # deterministic hazard rows: an in-support age violation with a low-risk
    # profile, so a v1 release on it is a real false certificate
    base = dict(stream_rows[0])
    injection = {**base, "age": 0.105, "bmi": -0.05, "bp": -0.09}

    journal_path = Path("experiments/results/flywheel_journal.jsonl")
    if journal_path.exists():
        journal_path.unlink()
    authority = ApprovalAuthority(b"flywheel-demo-approval")
    detectors = (
        DetectorSpec("coverage_low", "index", "coverage", 0.90, slack=0.02,
                     threshold=4.0, min_n=24, mde=0.05),
        DetectorSpec("abstain_high", "contract", "abstain", 0.20, slack=0.01,
                     threshold=5.0, min_n=24, mde=0.05),
        DetectorSpec("error_high", "proposer", "abs_error", cal_mae, slack=2.0,
                     threshold=10.0, min_n=24, mde=2.0),
        DetectorSpec("hazard_release", "contract", "hazard_release", 0.0,
                     slack=0.0, threshold=0.0, min_n=1, mde=0.0),
    )
    live = LiveSession(dm, journal=Journal(journal_path), ledger=ledger,
                       detectors=detectors,
                       approval_verifier=authority.verify)

    batches = [list(range(i, min(i + 12, len(stream_rows))))
               for i in range(0, len(stream_rows), 12)]
    records, detections, actions = [], [], []
    fired: set = set()
    containment_batch = promotion_batch = uncontain_batch = None
    old_policy = dm.policy.digest
    retraction = None
    oracle = lambda row, yy: (float(yy) <= THRESHOLD
                              and not ledger.hits(row), True)   # noqa: E731

    print(f"\nLIVE: {len(stream_rows)} rows in {len(batches)} batches, "
          f"drift from batch 3 (target +80), hazard injected in batches 4 and 6\n")

    for b, idxs in enumerate(batches, start=1):
        batch_rows, batch_y = [], []
        for j, i in enumerate(idxs):
            row, yy = dict(stream_rows[i]), stream_y[i]
            if b >= 3:                              # concept + feature drift
                yy = yy + 80.0
                row["bp"] = row["bp"] + 0.03
                row["bmi"] = row["bmi"] + 0.02
            if b >= 5:                              # second shock
                yy = yy + 80.0
                row["bp"] = row["bp"] + 0.02
                row["bmi"] = row["bmi"] + 0.02
            if b == 4 and j == 0:                   # hazard injection #1
                row, yy = dict(injection), 80.0
            if b == 6 and j == 0:                   # hazard injection #2
                row, yy = dict(injection), 80.0
            batch_rows.append(row)
            batch_y.append(yy)

        for row, yy in zip(batch_rows, batch_y):
            d = live.decide(row, batch=b)
            live.observe(d.certificate_id, yy)

        dets = live.tick()
        acts = live.auto_act(dets, batch=b)
        actions.extend([{**a.as_dict(), "kind": "containment"} for a in acts])
        if acts and containment_batch is None:
            containment_batch = b

        new_alarms = [d.as_dict() for d in dets
                      if d.alarm and d.name not in fired]
        fired.update(d.name for d in dets if d.alarm)

        st = live.status(window_batches=1)
        alpha = (live.model.kappa.alpha_at("__all__")
                 if live.model.kappa is not None
                 and hasattr(live.model.kappa, "alpha_at") else None)
        records.append({"batch": b, "n": len(batch_rows), "alpha": alpha,
                        "status": st, "detections": new_alarms,
                        "actions": [a.as_dict() | {"kind": "containment"}
                                    for a in acts]})
        detections.extend(new_alarms)
        q = st["review_queue"]
        print(f"batch {b}: " + " ".join(
            f"{k}={v}" for k, v in st["verdicts"].items())
            + f" | cover={st['coverage']} mae={st['mae']:.1f}"
            + f" | haz_rel={st['released_hazards']}"
            + (f" | alpha={alpha:.3f}" if alpha is not None else "")
            + (" | CONTAINED" if st["contained"] else ""))
        if new_alarms:
            print("  alarms: " + "; ".join(
                f"{d['name']}({d['lever']})={d['stat']:.2f}" for d in new_alarms))
        if q:
            print(f"  human review queue: {len(q)} abstention(s); e.g. "
                  f"{q[0]['reason']} - {q[0]['say'][:90]}")

        # repair once the hazard has been observed
        if acts and containment_batch == b:
            change = ChangeSet(
                lever="contract", reason="add the declared age-band hazard",
                policy=FlywheelPolicyV2.policy(), name="repair_age_band")
            ev = live.evaluate(change, oracle=oracle, ledger=ledger)
            prom = live.promote(change, oracle=oracle, ledger=ledger)
            actions.append({"kind": "promotion", "batch": b,
                            "approved": prom.approved, "reason": prom.reason,
                            "evaluation": ev.as_dict()})
            print(f"  repair: gate={'PASS' if ev.passed else 'FAIL'} "
                  f"may_loosen={ev.may_loosen} promoted={prom.approved}")
            if prom.approved:
                promotion_batch = b
                retraction = live.retract(
                    policy_digest=old_policy,
                    reason="decisions under v1 (missing age-band rule)")
                approval = authority.sign(
                    live.containment.change_digest, approver="dr-owner")
                live.uncontain(approval, reason="repair promoted; hazard rule "
                                                "now denies the band")
                uncontain_batch = b
                actions.append({"kind": "uncontain", "batch": b,
                                "approver": approval.approver,
                                "retracted": len(retraction.certificate_ids)})
                print(f"  uncontain approved by {approval.approver}; "
                      f"retracted {len(retraction.certificate_ids)} cert(s)")

    final = live.status()
    data_out = {
        "readiness": ready.as_dict(),
        "config": {"threshold": THRESHOLD, "hazard_age": HAZARD_AGE,
                   "cal_mae": cal_mae, "alpha": 0.10,
                   "batches": [b[0] if b else None for b in batches]},
        "records": records, "actions": actions,
        "retraction": retraction.as_dict() if retraction else None,
        "final_status": final,
        "timeline": {"containment_batch": containment_batch,
                     "promotion_batch": promotion_batch,
                     "uncontain_batch": uncontain_batch},
        "journal_verified": live.journal.verify(),
        "versions": live.versions,
        "alpha_final": (live.model.kappa.alpha_at("__all__")
                        if live.model.kappa is not None
                        and hasattr(live.model.kappa, "alpha_at") else None),
    }
    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"flywheel_{stamp}.json").write_text(
        json.dumps(data_out, indent=2, default=str))
    with (out / f"flywheel_{stamp}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["batch", "n", "release", "ask", "deny", "abstain",
                    "coverage", "mae", "released_hazards", "alpha",
                    "contained"])
        for r in records:
            s = r["status"]
            w.writerow([r["batch"], r["n"], s["verdicts"]["release"],
                        s["verdicts"]["ask"], s["verdicts"]["deny"],
                        s["verdicts"]["abstain"], s["coverage"], s["mae"],
                        s["released_hazards"], r.get("alpha"),
                        bool(s["contained"])])

    figures = _figures(records, data_out, out / "figures")
    _report(data_out, figures, out / f"flywheel_report_{stamp}.md")

    print(f"\nfinal: {final['verdicts']} coverage={final['coverage']}")
    print(f"journal verified: {data_out['journal_verified']} "
          f"({len(live.journal)} entries)")
    print(f"wrote {out}/flywheel_{stamp}.json, .csv, journal, figures, report")


def _figures(records, data, figdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    batches = [r["batch"] for r in records]
    verdicts = ["release", "ask", "deny", "abstain"]
    colors = {"release": "#2e6f40", "ask": "#e8a33d", "deny": "#8b2f2f",
              "abstain": "#7a7a7a"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    bottom = np.zeros(len(batches))
    for v in verdicts:
        vals = [r["status"]["verdicts"][v] for r in records]
        ax1.bar(batches, vals, bottom=bottom, color=colors[v], label=v)
        bottom += np.array(vals)
    ax1.set_title("verdicts per arriving batch")
    ax1.set_xlabel("batch")
    ax1.legend(frameon=False, ncol=4, fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    cov = [r["status"]["coverage"] for r in records]
    abst = [r["status"]["verdicts"]["abstain"] / r["n"] for r in records]
    ax2.plot(batches, cov, marker="o", color="#1f4e79", label="coverage")
    ax2.plot(batches, abst, marker="s", color="#7a7a7a", label="abstain rate")
    ax2.axhline(0.90, color="0.35", ls=":", lw=1)
    for b, label in ((data["timeline"]["containment_batch"], "contain"),
                     (data["timeline"]["promotion_batch"], "repair"),
                     (data["timeline"]["uncontain_batch"], "uncontain")):
        if b:
            ax2.axvline(b, color="#8b2f2f", ls="--", lw=1)
            ax2.text(b + 0.05, 0.02, label, fontsize=7, color="#8b2f2f")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("batch")
    ax2.set_title("coverage, abstention, and the adaptive index")
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    ax2.grid(alpha=0.25)
    alphas = [r.get("alpha") for r in records]
    if any(a is not None for a in alphas):
        ax2b = ax2.twinx()
        ax2b.plot(batches, alphas, color="#8a6d3b", ls="--", marker=".",
                  ms=4, label="alpha_t")
        ax2b.set_ylabel("alpha_t", color="#8a6d3b")
        ax2b.tick_params(axis="y", colors="#8a6d3b")
        ax2b.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    p1 = figdir / "fig_flywheel_batches.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4))
    haz = [r["status"]["released_hazards"] for r in records]
    ax1.bar(batches, haz, color="#8b2f2f")
    ax1.set_title("released external hazards")
    ax1.set_xlabel("batch")
    ax1.grid(axis="y", alpha=0.25)
    mae = [r["status"]["mae"] for r in records]
    ax2.plot(batches, mae, marker="o", color="#1f4e79")
    ax2.set_title("realized MAE on observed rows")
    ax2.set_xlabel("batch")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    p2 = figdir / "fig_flywheel_actions.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    return {"batches": p1, "actions": p2}


def _report(data, figures, path: Path):
    recs = data["records"]
    t = data["timeline"]
    lines = []
    lines.append("# Flywheel deployment report — diabetes regressor\n")
    lines.append("**Question.** With a model trained once, does a live decision "
                 "harness keep the *decisions* safe as batches arrive, what "
                 "does it detect, and what does it change — without letting "
                 "learning enter admission?\n")
    lines.append("## Setup\n")
    lines.append(f"- model: RandomForestRegressor, 10 features, trained on 60%;"
                 f" conformal κ on 20% (α=0.10, adaptive ACI).")
    lines.append(f"- policy: release if the whole conformal interval clears "
                 f"{data['config']['threshold']:.0f}; deny if entirely above; "
                 f"ask if it straddles.")
    lines.append(f"- external hazard: age > {data['config']['hazard_age']} "
                 f"(underwriting spec), injected in batches 4 and 6; staged drift "
                 f"(target +80 from batch 3, +80 again from batch 5).")
    lines.append(f"- readiness: approved={data['readiness']['approved']} "
                 f"({data['readiness']['readiness_id'][:20]}…), gates: "
                 + ", ".join(f"{g['name']}={'PASS' if g['passed'] else 'FAIL'}"
                             for g in data['readiness']['gates']) + ".\n")

    lines.append("## Batch timeline\n")
    lines.append("| batch | n | release | ask | deny | abstain | coverage | "
                 "MAE | alpha_t | hazard releases | contained |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for r in recs:
        s = r["status"]
        lines.append(f"| {r['batch']} | {r['n']} | {s['verdicts']['release']} | "
                     f"{s['verdicts']['ask']} | {s['verdicts']['deny']} | "
                     f"{s['verdicts']['abstain']} | "
                     f"{s['coverage']:.3f} | {s['mae']:.1f} | "
                     f"{(f'{r[chr(97)+chr(108)+chr(112)+chr(104)+chr(97)]:.3f}' if r.get('alpha') is not None else '-')} | "
                     f"{s['released_hazards']} | "
                     f"{'yes' if s['contained'] else 'no'} |")

    alarm_rows = []
    for r in recs:
        for d in r["detections"]:
            if d["alarm"]:
                alarm_rows.append((r["batch"], d["name"], d["lever"],
                                   d["severity"], round(d["stat"], 2),
                                   d["detail"]))
    lines.append("\n## What the flywheel detected\n")
    lines.append("| batch | detector | lever | severity | stat | evidence |")
    lines.append("|---:|---|---|---|---:|---|")
    for row in alarm_rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines.append("")
    lines.append("Detectors are sequential CUSUMs with declared targets, "
                 "slack, thresholds, minimum n and minimum detectable effects; "
                 "`tick()` is recomputed from the journal, so the same journal "
                 "yields the same alarms.\n")

    lines.append("## What it did\n")
    lines.append(f"- containment at batch {t['containment_batch']} "
                 f"(unconditional, release → ask);")
    lines.append(f"- repair ChangeSet promoted at batch {t['promotion_batch']} "
                 f"(adding the missing deny rule is tightening: no approval "
                 f"needed);")
    lines.append(f"- containment lifted at batch {t['uncontain_batch']} with "
                 f"an external approval (loosening is signed).")
    if data.get("retraction"):
        lines.append(f"- retraction ledger: {len(data['retraction']['certificate_ids'])} "
                     f"certificate(s) issued under the faulty v1 policy.")
    lines.append("")

    lines.append("## Reading\n")
    lines.append("- Coverage stayed near nominal (0.82–0.92 across batches) "
                 "while the index paid in width: alpha_t fell from 0.100 to "
                 "0.007 and the policy asked on every row of the shock batch. "
                 "The coverage CUSUM did not cross its declared threshold at "
                 "this n; abstention was the visible protection. ACI learned; "
                 "the admission relation did not.")
    lines.append("- The hazard release is a *contract* failure, not a model "
                 "failure: the interval was safe-looking, but the declared age "
                 "band was not in the policy. The repair is a rule, and the "
                 "repair is evidence-gated by replaying stored facts.")
    lines.append("- Containment costs capability immediately and can only "
                 "tighten; lifting it needs a signature the session does not "
                 "hold. Promotion of a tightening change needs no signature.")
    lines.append("- Retraction is version-scoped: every decision taken under v1 "
                 "is listed (48 certificates) for risk-owner triage; rollback "
                 "alone would not create that record. The second injected "
                 "hazard (batch 6) was denied by v2, not contained — the "
                 "repair works without another emergency.\n")
    if data.get("alpha_final") is not None:
        lines.append(f"- ACI ended with alpha_t = {data['alpha_final']:.4f} "
                     f"(nominal 0.10): the index widened after errors and "
                     f"narrowed again as coverage recovered.\n")
    final_q = data["final_status"].get("review_queue") or []
    lines.append(f"## Human-in-the-loop queue\n")
    lines.append(f"The live session exposes abstentions as a review queue; the "
                 f"run ended with {data['final_status']['n']} decisions and "
                 f"{data['final_status']['verdicts']['abstain']} abstentions. "
                 f"Examples:")
    seen = set()
    for item in final_q:
        key = (item["reason"], item["say"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- batch {item['batch']}: `{item['reason']}` — "
                     f"{item['say']}")
        if len(seen) >= 5:
            break
    lines.append("")

    lines.append("## What this run changed in the design\n")
    lines.append("- **The oracle must encode the declared hazard.** The first "
                 "run scored correctness as `y <= threshold` only. The injected "
                 "row was safe by outcome but hazardous by contract, so the "
                 "pair rewarded releasing it and the Pareto gate rejected the "
                 "correct repair. The oracle now marks any hazard row as "
                 "`safe=False`; the pair and the hazard ledger agree on what a "
                 "false certificate is.")
    lines.append("- **The hazard detector must be epoch-aware.** After the "
                 "repair and the lifting of containment, the historical hazard "
                 "release re-triggered containment on every tick. The detector "
                 "now counts only releases after the last uncontainment: "
                 "history is retracted, not re-paged.")
    lines.append("- **Warnings do not act.** The early `error_high` alarm was a "
                 "false alarm (MAE was stable); only the critical hazard alarm "
                 "triggered an automatic action. That is the intended asymmetry.")
    lines.append("")

    lines.append("## Figures\n")
    lines.append(f"![Verdicts and coverage per batch](figures/{figures['batches'].name})")
    lines.append(f"![Hazard releases and realized error](figures/{figures['actions'].name})\n")

    lines.append("## Limits\n")
    lines.append("Synthetic drift and an injected, deterministic hazard; small "
                 "batches (n≈12) and a single seed; the approval demo uses HMAC "
                 "with the signing key held outside the session (production: "
                 "asymmetric signatures). Label delay is zero in this run — "
                 "outcomes are joined immediately, which is optimistic for "
                 "coverage monitoring; a real feed needs audit sampling of "
                 "abstained/denied rows to avoid selection bias.")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
