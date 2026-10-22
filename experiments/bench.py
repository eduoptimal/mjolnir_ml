"""Real-data bench: the harness against the raw learner, on three decision shapes.

    binary        breast_cancer   malignant = risky; safe = benign
    multiclass    wine            approved class 0; classes 1,2 off-profile
    regression    diabetes        safe = progression <= 150 (decision threshold)

Each dataset runs through three learners (linear, RandomForest, HistGradient-
Boosting). Two decision models are scored against the same oracle:

    ungated   the fitted learner as a TOTAL decision model: it releases when
              its point rule says safe and denies otherwise — it never
              abstains, so every claim it makes on an unseen row is a claim.
    harnessed the same fitted weights behind DecisionModel: regime floor,
              conformal κ at alpha=0.10, and the declared @dsl policy.

The metric is the Omega pair (precision, recall) with Wilson intervals plus
the degenerate controls (ALWAYS/NEVER). This is the companion paper's
protocol, instantiated for learners.

Run:  python experiments/bench.py [--alpha 0.10] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mjolnir_ml import (ASK, DENY, RELEASE, DecisionModel,
                        DecisionPolicyBase, audit, degenerate_controls, dsl,
                        omega_pair, rule, wilson)


# ---------------------------------------------------------------- policies --
@dsl
class BinaryPolicy(DecisionPolicyBase):
    """Release only a certified benign prediction; deny a confident risky one."""

    assert rule("deny_risky",
                when=lambda p_positive, conf: p_positive >= 0.80 and conf >= 0.80,
                verdict=DENY)
    assert rule("release_certified_benign",
                when=lambda certified, p_positive: certified and p_positive < 0.50,
                verdict=RELEASE)
    assert rule("review_unsettled", when=lambda certified: not certified,
                verdict=ASK)


@dsl
class MulticlassPolicy(DecisionPolicyBase):
    """Class 0 is the approved outcome; class 2 is the throwaway."""

    assert rule("deny_off_profile",
                when=lambda pred, conf: pred == 2 and conf >= 0.80,
                verdict=DENY)
    assert rule("release_approved",
                when=lambda certified, pred: certified and pred == 0,
                verdict=RELEASE)
    assert rule("review_unsettled", when=lambda certified: not certified,
                verdict=ASK)


REGRESSION_THRESHOLD = 180.0


@dsl
class RegressionPolicy(DecisionPolicyBase):
    """Gate on the whole conformal interval relative to the decision threshold."""

    assert rule("release_safe_interval",
                when=lambda hi: hi <= 180.0, verdict=RELEASE)
    assert rule("deny_high_interval", when=lambda lo: lo > 180.0, verdict=DENY)
    assert rule("review_widening",
                when=lambda lo, hi: lo <= 180.0 and hi > 180.0, verdict=ASK)


# ------------------------------------------------------------------ helpers --
def rows_of(X, names):
    return [dict(zip(names, map(float, x))) for x in X]


def split_clf(X, y, seed=0):
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.4,
                                              stratify=y, random_state=seed)
    Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                            stratify=yrest, random_state=seed)
    return (Xtr, ytr), (Xcal, ycal), (Xte, yte)


def split_reg(X, y, seed=0):
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.4,
                                              random_state=seed)
    Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                            random_state=seed)
    return (Xtr, ytr), (Xcal, ycal), (Xte, yte)


def _claim_rate(disps):
    from mjolnir_ml.vendors.wire import DENY_W, RELEASE
    return sum(1 for d in disps if d.verdict in (RELEASE, DENY_W)) / len(disps)


def _certified_share(disps):
    return sum(1 for d in disps
               if (d.prediction or {}).get("certified") is True) / len(disps)


def _ratio(a, b):
    return None if b is None else (a / b)


def bench_binary(dataset, learners, alpha=0.10):
    X, y, names = dataset
    (Xtr, ytr), (Xcal, ycal), (Xte, yte) = split_clf(X, y)
    safe = [int(v) == 0 for v in yte]          # benign is the safe outcome
    dec = [True] * len(safe)
    out = []
    for name, make in learners:
        t0 = time.perf_counter()
        est = make().fit(Xtr, ytr)
        dm = DecisionModel(est, BinaryPolicy, features=tuple(names))
        dm.fit_regime(rows_of(Xtr, names)).calibrate(
            rows_of(Xcal, names), ycal, alpha=alpha)
        disps = [dm.dispose(r) for r in rows_of(Xte, names)]

        # ungated: the same fitted learner as a total decision model
        proba = est.predict_proba(Xte)[:, list(est.classes_).index(1)]
        ungated = [RELEASE if p < 0.50 else DENY for p in proba]

        report = audit(disps, safe, dec)
        upair = omega_pair(ungated, safe, dec)
        out.append({
            "dataset": "breast_cancer", "kind": "binary", "learner": name,
            "alpha": alpha, "n_test": len(safe),
            "harness": {**report},
            "ungated": {**upair.as_dict(),
                        "prec_ci": wilson(
                            upair.n_cert, upair.n_claim)},
            "claim_rate": _claim_rate(disps),
            "certified_share": _certified_share(disps),
            "degenerate": degenerate_controls(safe, dec),
            "seconds": round(time.perf_counter() - t0, 2),
        })
    return out


def bench_multiclass(dataset, learners, alpha=0.10):
    X, y, names = dataset
    (Xtr, ytr), (Xcal, ycal), (Xte, yte) = split_clf(X, y)
    safe = [int(v) == 0 for v in yte]
    dec = [True] * len(safe)
    out = []
    for name, make in learners:
        t0 = time.perf_counter()
        est = make().fit(Xtr, ytr)
        dm = DecisionModel(est, MulticlassPolicy, features=tuple(names))
        dm.fit_regime(rows_of(Xtr, names)).calibrate(
            rows_of(Xcal, names), ycal, alpha=alpha)
        disps = [dm.dispose(r) for r in rows_of(Xte, names)]
        pred = est.predict(Xte)
        ungated = [RELEASE if p == 0 else DENY for p in pred]

        report = audit(disps, safe, dec)
        upair = omega_pair(ungated, safe, dec)
        out.append({
            "dataset": "wine", "kind": "multiclass", "learner": name,
            "alpha": alpha, "n_test": len(safe),
            "harness": {**report},
            "ungated": {**upair.as_dict(),
                        "prec_ci": wilson(
                            upair.n_cert, upair.n_claim)},
            "claim_rate": _claim_rate(disps),
            "certified_share": _certified_share(disps),
            "degenerate": degenerate_controls(safe, dec),
            "seconds": round(time.perf_counter() - t0, 2),
        })
    return out


def bench_regression(dataset, learners, alpha=0.10):
    X, y, names = dataset
    (Xtr, ytr), (Xcal, ycal), (Xte, yte) = split_reg(X, y)
    tau = REGRESSION_THRESHOLD
    safe = [float(v) <= tau for v in yte]
    dec = [True] * len(safe)
    out = []
    for name, make in learners:
        t0 = time.perf_counter()
        est = make().fit(Xtr, ytr)
        dm = DecisionModel(est, RegressionPolicy, features=tuple(names))
        dm.fit_regime(rows_of(Xtr, names)).calibrate(
            rows_of(Xcal, names), ycal, alpha=alpha)
        disps = [dm.dispose(r) for r in rows_of(Xte, names)]
        pred = est.predict(Xte)
        ungated = [RELEASE if p <= tau else DENY for p in pred]

        report = audit(disps, safe, dec)
        upair = omega_pair(ungated, safe, dec)
        out.append({
            "dataset": "diabetes", "kind": "regression", "learner": name,
            "alpha": alpha, "n_test": len(safe),
            "harness": {**report},
            "ungated": {**upair.as_dict(),
                        "prec_ci": wilson(
                            upair.n_cert, upair.n_claim)},
            "claim_rate": _claim_rate(disps),
            "certified_share": None,
            "mean_interval_width": sum(
                (d.prediction or {}).get("hi", 0) - (d.prediction or {}).get("lo", 0)
                for d in disps) / len(disps),
            "degenerate": degenerate_controls(safe, dec),
            "seconds": round(time.perf_counter() - t0, 2),
        })
    return out


def _fmt(x, nd=3):
    return "  ⊥  " if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    binary = (load_breast_cancer(return_X_y=False).data,
              load_breast_cancer().target,
              list(load_breast_cancer().feature_names))
    multi = (load_wine().data, load_wine().target,
             list(load_wine().feature_names))
    reg = (load_diabetes().data, load_diabetes().target,
           list(load_diabetes().feature_names))

    records = []
    records += bench_binary(
        binary, [("logistic", lambda: make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=2000))),
                 ("random_forest", lambda: RandomForestClassifier(
                     n_estimators=200, random_state=0)),
                 ("hist_gradient", lambda: HistGradientBoostingClassifier(
                     random_state=0))], alpha=args.alpha)
    records += bench_multiclass(
        multi, [("logistic", lambda: make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=2000))),
                ("random_forest", lambda: RandomForestClassifier(
                    n_estimators=200, random_state=0)),
                ("hist_gradient", lambda: HistGradientBoostingClassifier(
                    random_state=0))], alpha=args.alpha)
    records += bench_regression(
        reg, [("ridge", lambda: Ridge(alpha=1.0)),
              ("random_forest", lambda: RandomForestRegressor(
                  n_estimators=200, random_state=0)),
              ("hist_gradient", lambda: HistGradientBoostingRegressor(
                  random_state=0))], alpha=args.alpha)

    print(f"\nOmega pair vs the pinned oracle — alpha={args.alpha}, "
          f"harness vs ungated total model\n")
    hdr = (f"{'dataset':<14}{'learner':<15}{'n':>5}"
           f"{'prec':>8}{'rec':>8}{'prec_u':>9}{'rec_u':>8}"
           f"{'claims':>8}{'cert':>7}{'ask':>6}{'over':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        h, u = r["harness"], r["ungated"]
        p, pc = h["pair"], h["verdicts"]
        cert = ("  -  " if r["certified_share"] is None
                else f"{r['certified_share']:.2f}")
        print(f"{r['dataset']:<14}{r['learner']:<15}{r['n_test']:>5}"
              f"{_fmt(p['prec']):>8}{_fmt(p['rec']):>8}"
              f"{_fmt(u['prec']):>9}{_fmt(u['rec']):>8}"
              f"{p['n_claim']:>8}{cert:>7}"
              f"{pc.get('ask', 0):>6}{p['n_overclaim']:>6}")

    print("\nDegenerate controls (constructibility table):")
    seen = set()
    for r in records:
        key = r["dataset"]
        if key in seen:
            continue
        seen.add(key)
        d = r["degenerate"]
        print(f"  {key:<14} ALWAYS prec={_fmt(d['ALWAYS']['prec'])} "
              f"rec={_fmt(d['ALWAYS']['rec'])} · "
              f"NEVER prec={_fmt(d['NEVER']['prec'])} "
              f"rec={_fmt(d['NEVER']['rec'])}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = outdir / f"bench_{stamp}.json"
    path.write_text(json.dumps({"alpha": args.alpha, "records": records},
                               indent=2, sort_keys=True))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
