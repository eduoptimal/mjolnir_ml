"""Does a better model make better DECISIONS? A two-axis learning curve.

The standard learning curve plots a model metric (MAE, accuracy). This
experiment asks the question a decision system is judged on: as the learner
improves, do the *decisions* improve — certified precision, decision recall,
and hazard completeness — and at what coverage?

Two axes, two datasets, seed-paired:

    data      train fraction of the training pool: 15% … 100%
    capacity  RandomForest complexity ladder:
                poor   =  10 trees, depth 3,  leaf 20
                medium = 100 trees, depth 10, leaf 5
                rich   = 300 trees, unlimited depth, leaf 1

Protocol (rigorous bits):

  * the conformal calibration split is FIXED across the sweep — more data goes
    into the learner, never into kappa, so coverage stays comparable;
  * 3 seeds per cell; reported values are means with 95% CIs;
  * every cell reports BOTH axes: decision (precision/recall/claim rate) and
    hazards (precision/recall/coverage vs an external ledger), never a scalar;
  * the model metric (MAE / accuracy) is carried alongside so "better model"
    is visible, not assumed;
  * a paired test at full data asks whether rich − poor exceeds seed noise;
  * Spearman between model quality and decision recall is reported per dataset.

Datasets: diabetes (regression, threshold 240) and wine (multiclass, class 0
approved). All features are used; policies are the shipped examples.

    python experiments/model_ladder.py [--seeds 3] [--quick]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes, load_wine
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.model_selection import train_test_split

from mjolnir_ml import (ASK, DENY, RELEASE, DecisionModel, DecisionPolicyBase,
                        HazardLedger, audit, dsl, measure_completeness, omega_pair,
                        rule)

# --------------------------------------------------------------- capacity --
CAPACITIES = {
    "poor": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 20},
    "medium": {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5},
    "rich": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
}
SIZES = (0.15, 0.30, 0.50, 0.75, 1.00)
ALPHA = 0.10
REGRESSION_THRESHOLD = 240.0


# --------------------------------------------------------------- policies --
@dsl
class ProgressionPolicy(DecisionPolicyBase):
    facts = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")

    assert lambda bmi: bmi > -1.0, "bmi_in_range"

    assert rule("release_safe_band",
                when=lambda lo, hi: lo >= 0.0 and hi <= 240.0, verdict=RELEASE)
    assert rule("deny_high_band", when=lambda lo: lo > 240.0, verdict=DENY)
    assert rule("review_widening",
                when=lambda lo, hi: lo <= 240.0 and hi > 240.0, verdict=ASK)


@dsl
class WinePolicy(DecisionPolicyBase):
    assert rule("deny_high_alcohol",
                when=lambda alcohol: alcohol > 14.5, verdict=DENY)
    assert rule("deny_off_profile",
                when=lambda pred, conf: pred == 2 and conf >= 0.80, verdict=DENY)
    assert rule("release_approved",
                when=lambda certified, pred: certified and pred == 0,
                verdict=RELEASE)
    assert rule("review_unsettled",
                when=lambda certified: not certified, verdict=ASK)


# ---------------------------------------------------------------- helpers --
def rows_of(X, names):
    return [dict(zip(names, map(float, x))) for x in X]


def _regression_hazards():
    return (HazardLedger()
            .add("extreme_bmi", lambda r: r["bmi"] > 0.10,
                 "clinical spec §2 (BMI)")
            .add("high_bp", lambda r: r["bp"] > 0.14,
                 "clinical spec §3 (blood pressure)"))


def _wine_hazards():
    return (HazardLedger()
            .add("high_alcohol", lambda r: r["alcohol"] > 13.5,
                 "product spec §1 (alcohol by volume)"))


def _rows_digest(rows):
    import hashlib
    import json
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                      default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _certified_share(disps):
    vals = [(d.prediction or {}).get("certified") for d in disps]
    vals = [v for v in vals if v is not None]
    return sum(1 for v in vals if v is True) / len(vals) if vals else None


def _spearman(xs, ys):
    """Rank correlation with average ranks for ties; None if degenerate."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(list(xs)), ranks(list(ys))
    n = len(rx)
    if n < 3:
        return None
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def _mean_ci(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, 1.96 * sd / math.sqrt(len(vals))


# ------------------------------------------------------------------ sweeps --
def sweep_regression(sizes, capacities, seeds, alpha=ALPHA, log=print):
    d = load_diabetes()
    names = tuple(d.feature_names)
    ledger = _regression_hazards()
    records = []
    for seed in seeds:
        Xtr, Xrest, ytr, yrest = train_test_split(
            d.data, d.target, test_size=0.4, random_state=seed)
        Xcal, Xte, ycal, yte = train_test_split(
            Xrest, yrest, test_size=0.5, random_state=seed)
        tr, cal, te = rows_of(Xtr, names), rows_of(Xcal, names), rows_of(Xte, names)
        te_X = Xte
        safe = [float(v) <= REGRESSION_THRESHOLD for v in yte]
        decidable = [True] * len(yte)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(Xtr))
        cal_digest = _rows_digest(cal)

        for cap, params in capacities.items():
            for frac in sizes:
                k = max(20, int(round(len(Xtr) * frac)))
                idx = order[:k]
                est = RandomForestRegressor(random_state=seed, n_jobs=-1,
                                            **params).fit(Xtr[idx], ytr[idx])
                dm = (DecisionModel(est, ProgressionPolicy, features=names)
                      .fit_regime(rows_of(Xtr[idx], names))
                      .calibrate(cal, ycal, alpha=alpha))
                disps = [dm.dispose(r) for r in te]
                pair = omega_pair(disps, safe, decidable)
                haz = measure_completeness(disps, te, ledger)
                pred = est.predict(te_X)
                mae = float(np.mean(np.abs(pred - yte)))
                records.append({
                    "dataset": "diabetes", "kind": "regression",
                    "seed": seed, "capacity": cap, "size": frac,
                    "n_train": int(k), "model_metric": mae,
                    "prec": pair.prec, "rec": pair.rec,
                    "claim_rate": pair.n_claim / len(disps),
                    "haz_prec": haz["precision"], "haz_rec": haz["recall"],
                    "haz_coverage": haz["coverage"],
                    "certified": _certified_share(disps),
                    "n_test": len(disps),
                    "policy_digest": dm.policy.digest,
                    "model_digest": dm.model_digest,
                    "calibration_digest": dm.kappa.digest,
                    "cal_rows_digest": cal_digest,
                    "n_cal": len(cal),
                })
        log(f"  regression seed {seed}: {len(capacities) * len(sizes)} cells")
    return records


def sweep_multiclass(sizes, capacities, seeds, alpha=ALPHA, log=print):
    d = load_wine()
    names = tuple(n.replace("/", "_") for n in d.feature_names)
    ledger = _wine_hazards()
    records = []
    for seed in seeds:
        Xtr, Xrest, ytr, yrest = train_test_split(
            d.data, d.target, test_size=0.4, stratify=d.target,
            random_state=seed)
        Xcal, Xte, ycal, yte = train_test_split(
            Xrest, yrest, test_size=0.5, stratify=yrest, random_state=seed)
        tr, cal, te = rows_of(Xtr, names), rows_of(Xcal, names), rows_of(Xte, names)
        te_X = Xte
        safe = [int(v) == 0 for v in yte]
        decidable = [True] * len(yte)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(Xtr))
        cal_digest = _rows_digest(cal)

        for cap, params in capacities.items():
            for frac in sizes:
                k = max(20, int(round(len(Xtr) * frac)))
                idx = order[:k]
                est = RandomForestClassifier(random_state=seed, n_jobs=-1,
                                             **params).fit(Xtr[idx], ytr[idx])
                dm = (DecisionModel(est, WinePolicy, features=names)
                      .fit_regime(rows_of(Xtr[idx], names))
                      .calibrate(cal, ycal, alpha=alpha))
                disps = [dm.dispose(r) for r in te]
                pair = omega_pair(disps, safe, decidable)
                haz = measure_completeness(disps, te, ledger)
                acc = float(np.mean(est.predict(te_X) == yte))
                records.append({
                    "dataset": "wine", "kind": "multiclass",
                    "seed": seed, "capacity": cap, "size": frac,
                    "n_train": int(k), "model_metric": acc,
                    "prec": pair.prec, "rec": pair.rec,
                    "claim_rate": pair.n_claim / len(disps),
                    "haz_prec": haz["precision"], "haz_rec": haz["recall"],
                    "haz_coverage": haz["coverage"],
                    "certified": _certified_share(disps),
                    "n_test": len(disps),
                    "policy_digest": dm.policy.digest,
                    "model_digest": dm.model_digest,
                    "calibration_digest": dm.kappa.digest,
                    "cal_rows_digest": cal_digest,
                    "n_cal": len(cal),
                })
        log(f"  multiclass seed {seed}: {len(capacities) * len(sizes)} cells")
    return records


# -------------------------------------------------------------- aggregate --
METRICS = ("model_metric", "prec", "rec", "claim_rate", "haz_prec", "haz_rec",
           "haz_coverage", "certified")


def aggregate(records):
    cells = {}
    for r in records:
        key = (r["dataset"], r["capacity"], r["size"])
        cells.setdefault(key, []).append(r)
    out = []
    for (dataset, cap, frac), rs in sorted(cells.items()):
        row = {"dataset": dataset, "capacity": cap, "size": frac,
               "n_train": rs[0]["n_train"], "n_seeds": len(rs)}
        for m in METRICS:
            mean, ci = _mean_ci([r[m] for r in rs])
            row[m] = mean
            row[f"{m}_ci"] = ci
        out.append(row)
    return out


def paired_effect(records, capacity_a="poor", capacity_b="rich", size=1.0):
    """rich − poor at full data, paired by seed. Returns per metric."""
    def by_seed(cap):
        return {r["seed"]: r for r in records
                if r["capacity"] == cap and r["size"] == size}
    a, b = by_seed(capacity_a), by_seed(capacity_b)
    seeds = sorted(set(a) & set(b))
    out = {}
    for m in ("model_metric", "prec", "rec", "haz_rec", "claim_rate",
              "haz_prec"):
        diffs = []
        for s in seeds:
            if a[s][m] is None or b[s][m] is None:
                continue
            diffs.append(b[s][m] - a[s][m])
        mean, ci = _mean_ci(diffs)
        out[m] = {"mean_diff": mean, "ci95": ci, "n_pairs": len(diffs),
                  "excludes_zero": (len(diffs) >= 2 and mean is not None
                                    and ci is not None
                                    and (mean - ci > 0 or mean + ci < 0))}
    return out


def decision_vs_model(records):
    usable = [r for r in records
              if r["prec"] is not None and r["model_metric"] is not None]
    rec = [r["rec"] for r in usable]
    # lower MAE / higher accuracy = better; flip MAE so "quality" increases
    quality = [(-r["model_metric"] if r["kind"] == "regression"
                else r["model_metric"]) for r in usable]
    return {"spearman_model_quality_vs_recall": _spearman(quality, rec),
            "n_cells": len(usable)}


# ------------------------------------------------------------------ printing --
def _fmt(v, nd=3):
    if v is None:
        return "  -  "
    if isinstance(v, float) and abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.{nd}f}"


def print_table(agg, dataset, capacity_order):
    small = "MAE" if dataset == "diabetes" else "acc"
    hdr = (f"{'capacity':<9}{'size':>6}{'n_train':>8}{small:>9}"
           f"{'prec':>8}{'rec':>8}{'claims':>8}"
           f"{'haz_prec':>9}{'haz_rec':>8}{'cert':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in [x for x in agg if x["dataset"] == dataset]:
        print(f"{r['capacity']:<9}{r['size']:>6.2f}{r['n_train']:>8}"
              f"{_fmt(r['model_metric']):>9}{_fmt(r['prec']):>8}"
              f"{_fmt(r['rec']):>8}{_fmt(r['claim_rate']):>8}"
              f"{_fmt(r['haz_prec']):>9}{_fmt(r['haz_rec']):>8}"
              f"{_fmt(r['certified']):>7}")


def plot(agg, dataset, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [r for r in agg if r["dataset"] == dataset]
    caps = [c for c in CAPACITIES if any(r["capacity"] == c for r in data)]
    sizes = sorted({r["size"] for r in data})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{dataset}: learner → decisions (mean, {len(data) and data[0]['n_seeds']} seeds)")
    panels = [
        ("rec", "decision recall", axes[0, 0]),
        ("prec", "decision precision", axes[0, 1]),
        ("haz_rec", "hazard recall (external ledger)", axes[1, 0]),
        ("claim_rate", "claim rate (coverage of decisions)", axes[1, 1]),
    ]
    for key, label, ax in panels:
        for cap in caps:
            ys, cis = [], []
            for frac in sizes:
                row = next(r for r in data
                           if r["capacity"] == cap and r["size"] == frac)
                ys.append(float("nan") if row[key] is None else row[key])
                cis.append(0.0 if row.get(f"{key}_ci") is None
                           else row[f"{key}_ci"])
            ax.errorbar(sizes, ys, yerr=cis, marker="o", capsize=3, label=cap)
        ax.set_title(label)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        if key in ("haz_rec", "claim_rate"):
            ax.set_xlabel("training fraction")
    axes[0, 0].set_ylabel("value")
    axes[1, 0].set_ylabel("value")
    axes[0, 0].legend(title="capacity")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="2 sizes, 2 capacities, 1 seed")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    if args.quick:
        sizes, caps, seeds = (0.4, 1.0), ("poor", "rich"), (0,)
    else:
        sizes, caps, seeds = SIZES, tuple(CAPACITIES), tuple(range(args.seeds))

    cap_subset = {k: CAPACITIES[k] for k in caps}
    t0 = time.perf_counter()
    print(f"\nmodel ladder — capacities={list(cap_subset)} sizes={list(sizes)} "
          f"seeds={list(seeds)} alpha={ALPHA}\n")
    print("training (regression)…")
    rec_reg = sweep_regression(sizes, cap_subset, seeds)
    print("training (multiclass)…")
    rec_clf = sweep_multiclass(sizes, cap_subset, seeds)
    records = rec_reg + rec_clf
    agg = aggregate(records)

    for dataset in ("diabetes", "wine"):
        print(f"\n{'=' * 78}\n{dataset}\n{'=' * 78}")
        print_table(agg, dataset, caps)
        ds_records = [r for r in records if r["dataset"] == dataset]
        eff = paired_effect(ds_records, "poor", "rich", size=max(sizes))
        print("\npaired rich − poor at full data (95% CI over seeds):")
        for m, v in eff.items():
            print(f"  {m:<12} {_fmt(v['mean_diff']):>9} "
                  f"± {_fmt(v['ci95']):<8} "
                  f"{'significant' if v['excludes_zero'] else 'not separable'}")
        dm = decision_vs_model(ds_records)
        print(f"  Spearman(model quality, decision recall) = "
              f"{_fmt(dm['spearman_model_quality_vs_recall'])} "
              f"over {dm['n_cells']} cells")
        fig = plot(agg, dataset,
                   Path(args.out) / "figures" / f"model_ladder_{dataset}.png")
        print(f"  curve → {fig}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"model_ladder_{stamp}.json"
    path.write_text(json.dumps({"alpha": ALPHA, "sizes": list(sizes),
                                "capacities": cap_subset,
                                "seeds": list(seeds),
                                "records": records, "aggregate": agg,
                                "paired_full": {
                                    d: paired_effect(
                                        [r for r in records if r["dataset"] == d],
                                        "poor", "rich", size=max(sizes))
                                    for d in ("diabetes", "wine")},
                                "decision_vs_model": {
                                    d: decision_vs_model(
                                        [r for r in records if r["dataset"] == d])
                                    for d in ("diabetes", "wine")}},
                               indent=2, default=str))
    csv_path = out / f"model_ladder_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)
    print(f"\nwrote {path}\nwrote {csv_path} "
          f"({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()