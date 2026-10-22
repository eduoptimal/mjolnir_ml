"""ACI under concept drift on real data (beta_release G4, repaired).

Static conformal coverage is a promise about exchangeable data. This experiment
induces concept drift mid-stream (labels flip at 45% after row 80) and reports
decision-time coverage for a static κ and an adaptive κ on the same model, same
calibration split, same rows. ACI moves alpha_t from realized miscoverage.

    python experiments/drift.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mjolnir_ml import (ASK, DENY, RELEASE, DecisionModel,
                        DecisionPolicyBase, dsl, rule)


@dsl
class DriftPolicy(DecisionPolicyBase):
    assert rule("deny_risky",
                when=lambda p_positive, conf: p_positive >= 0.80 and conf >= 0.80,
                verdict=DENY)
    assert rule("release_benign",
                when=lambda certified, p_positive: certified and p_positive < 0.50,
                verdict=RELEASE)
    assert rule("review_unsettled", when=lambda certified: not certified,
                verdict=ASK)


def coverage_windows(dm, rows, ys, window=25, observe=True):
    cover, sizes = [], []
    for row, y in zip(rows, ys):
        d = dm.dispose(row)
        pset = (d.prediction or {}).get("set") or []
        cover.append(1.0 if y in pset else 0.0)
        sizes.append(len(pset))
        if observe:
            dm.observe(row, y)
    cov = [sum(cover[i:i + window]) / window
           for i in range(0, len(cover), window)]
    sz = [sum(sizes[i:i + window]) / window
          for i in range(0, len(sizes), window)]
    return cov, sz


def main():
    d = load_breast_cancer()
    names = tuple(n.replace(" ", "_") for n in d.feature_names)
    X, y = d.data, d.target
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.5,
                                              stratify=y, random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                            stratify=yrest, random_state=0)
    est = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    rows = lambda A: [dict(zip(names, map(float, x))) for x in A]  # noqa: E731

    onset, n = 70, 240
    rng = random.Random(0)
    stream_rows = rows(Xte)[:n] if len(Xte) >= n else rows(Xte)
    stream_y = []
    for i, v in enumerate(list(yte)[:len(stream_rows)]):
        yy = int(v)
        if i >= onset and rng.random() < 0.45:
            yy = 1 - yy                       # concept drift
        stream_y.append(yy)

    static = (DecisionModel(est, DriftPolicy, features=names)
              .fit_regime(rows(Xtr)).calibrate(rows(Xcal), ycal, alpha=0.10))
    adaptive = (DecisionModel(est, DriftPolicy, features=names)
                .fit_regime(rows(Xtr))
                .calibrate(rows(Xcal), ycal, alpha=0.10, adaptive=True,
                           gamma=0.10, window=120))

    cov_s, sz_s = coverage_windows(static, stream_rows, stream_y, observe=False)
    cov_a, sz_a = coverage_windows(adaptive, stream_rows, stream_y)

    pre = slice(0, 2)                       # first two windows: clean data
    post = slice(max(0, len(cov_s) - 2), None)   # last two: fully drifted
    r = {
        "dataset": "breast_cancer", "alpha": 0.10, "onset": onset,
        "n_stream": len(stream_rows), "window": 25,
        "static": {"coverage_pre": sum(cov_s[pre]) / len(cov_s[pre]),
                   "coverage_post": sum(cov_s[post]) / len(cov_s[post]),
                   "mean_set_pre": sum(sz_s[pre]) / len(sz_s[pre]),
                   "mean_set_post": sum(sz_s[post]) / len(sz_s[post]),
                   "windows": cov_s},
        "adaptive": {"coverage_pre": sum(cov_a[pre]) / len(cov_a[pre]),
                     "coverage_post": sum(cov_a[post]) / len(cov_a[post]),
                     "mean_set_pre": sum(sz_a[pre]) / len(sz_a[pre]),
                     "mean_set_post": sum(sz_a[post]) / len(sz_a[post]),
                     "alpha_t": adaptive.kappa.alpha_at("__all__"),
                     "windows": cov_a},
    }
    print(f"\nACI under concept drift — breast_cancer, alpha=0.10, "
          f"onset at row {onset}, flip p=0.45, {len(stream_rows)} rows\n")
    hdr = f"{'arm':<10}{'cov pre':>9}{'cov post':>10}{'set pre':>9}{'set post':>10}"
    print(hdr)
    print("-" * len(hdr))
    for arm in ("static", "adaptive"):
        v = r[arm]
        print(f"{arm:<10}{v['coverage_pre']:>9.3f}{v['coverage_post']:>10.3f}"
              f"{v['mean_set_pre']:>9.2f}{v['mean_set_post']:>10.2f}")
    print(f"\nadaptive alpha_t at end: {r['adaptive']['alpha_t']:.4f} "
          f"(started at 0.1000)")
    print("\nwindow coverage (25 rows each):")
    print("  static  :", " ".join(f"{v:.2f}" for v in cov_s))
    print("  adaptive:", " ".join(f"{v:.2f}" for v in cov_a))

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"drift_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(r, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
