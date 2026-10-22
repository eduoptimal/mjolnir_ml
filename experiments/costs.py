"""Cost-sensitive verdicts on real data: bands vs a fixed 0.5 threshold.

breast_cancer with a calibrated probability (CalibratedClassifierCV), a
conformal κ on top, and declared costs:

    C_FR = 20   releasing a malignant case
    C_FD = 5    denying a benign case
    C_A  = 2    asking a human

Band optimum: p < 0.10 -> release, p > 0.60 -> deny, else ask.
Realized cost uses the TRUE label (the only honest score), not the posterior.

    python experiments/costs.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mjolnir_ml import (ASK, ABSTAIN, DENY, RELEASE, Costs, DecisionModel,
                        cost_policy)
from mjolnir_ml.vendors import wire

COSTS = Costs(c_false_release=20.0, c_false_deny=5.0, c_ask=2.0)


def realized(verdict, y, costs):
    """Realized cost of one decision given the truth (y=1 means unsafe)."""
    if verdict == wire.RELEASE:
        return costs.c_false_release if y == 1 else 0.0
    if verdict == wire.DENY_W:
        return costs.c_false_deny if y == 0 else 0.0
    return costs.c_ask                      # ask / abstain


def summarize(name, verdicts, ys, costs, ps=None):
    n = len(verdicts)
    cost = sum(realized(v, int(y), costs) for v, y in zip(verdicts, ys)) / n
    exp = (sum(costs.expected_cost(v, p) for v, p in zip(verdicts, ps)) / n
           if ps is not None else None)
    fr = sum(1 for v, y in zip(verdicts, ys) if v == wire.RELEASE and int(y) == 1)
    fd = sum(1 for v, y in zip(verdicts, ys) if v == wire.DENY_W and int(y) == 0)
    ask = sum(1 for v in verdicts if v in (wire.ASK, wire.ABSTAIN_W))
    rel = sum(1 for v in verdicts if v == wire.RELEASE)
    return {"arm": name, "realized_cost_per_1000": 1000 * cost,
            "expected_cost_per_1000": None if exp is None else 1000 * exp,
            "false_release": fr, "false_deny": fd, "asked": ask,
            "released": rel, "n": n}


def main():
    d = load_breast_cancer()
    names = tuple(n.replace(" ", "_") for n in d.feature_names)
    X, y = d.data, d.target
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.5,
                                              stratify=y, random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                            stratify=yrest, random_state=0)
    rows = lambda A: [dict(zip(names, map(float, x))) for x in A]  # noqa: E731

    est = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(LogisticRegression(max_iter=2000),
                               method="sigmoid", cv=5)).fit(Xtr, ytr)
    p_unsafe = est.predict_proba(Xte)[:, 1]

    plain = cost_policy(COSTS, name="CostPolicy")
    certified = cost_policy(COSTS, require_certified=True,
                           name="CostPolicyCertified")
    arms = {}
    for name, policy in (("cost_bands", plain), ("cost_bands+certified", certified)):
        dm = (DecisionModel(est, policy, features=names)
              .fit_regime(rows(Xtr)).calibrate(rows(Xcal), ycal, alpha=0.10))
        arms[name] = summarize(name, [dm.dispose(r).verdict for r in rows(Xte)],
                               yte, COSTS, ps=p_unsafe)

    arms["fixed_0.5"] = summarize(
        "fixed_0.5", [RELEASE if p < 0.5 else DENY for p in p_unsafe], yte,
        COSTS, ps=p_unsafe)
    arms["always_ask"] = summarize("always_ask", [ASK] * len(yte), yte, COSTS,
                                   ps=p_unsafe)
    arms["always_deny"] = summarize("always_deny", [DENY] * len(yte), yte,
                                    COSTS, ps=p_unsafe)
    arms["always_release"] = summarize("always_release", [RELEASE] * len(yte),
                                       yte, COSTS, ps=p_unsafe)

    print(f"\nCost-sensitive verdicts — breast_cancer, {len(yte)} held-out rows")
    print(f"costs: C_FR={COSTS.c_false_release:g} C_FD={COSTS.c_false_deny:g} "
          f"C_A={COSTS.c_ask:g}  bands: release < {COSTS.thresholds()[0]:.2f} < ask "
          f"< {COSTS.thresholds()[1]:.2f} < deny\n")
    hdr = (f"{'arm':<22}{'exp/1000':>10}{'real/1000':>11}{'falseRel':>10}"
           f"{'falseDeny':>11}{'asked':>8}{'released':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in arms.values():
        exp = r["expected_cost_per_1000"]
        print(f"{r['arm']:<22}{(f'{exp:.1f}' if exp is not None else '-'):>10}"
              f"{r['realized_cost_per_1000']:>11.1f}"
              f"{r['false_release']:>10}{r['false_deny']:>11}"
              f"{r['asked']:>8}{r['released']:>10}")

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"costs_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"costs": COSTS.as_dict(), "arms": arms},
                               indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()