"""Policy completeness against an EXTERNAL hazard oracle (CERTIFIABILITY.md).

breast_cancer, with `worst_concave_points` held out of the model's inputs so a
hazard defined on it is outside sigma(X) — the Chow ceiling.

    H1  out_of_spec_area   worst_area > 1500            observable
    H2  hidden_concavity   worst_concave_points > 0.2   NOT observable (held out)

Two policies over the same model and calibration:
    FULL  denies the out-of-spec hazard explicitly
    GAP   omits that rule (the ablation: a declaration-level completeness gap)

Measure precision (no false certificates), recall (hazards caught), coverage,
and the ceiling. Removing a declared hazard MUST cost recall and precision;
a hidden hazard must sit at the ceiling regardless of policy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mjolnir_ml import (ASK, DENY, RELEASE, DecisionModel, DecisionPolicyBase,
                        HazardLedger, dsl, measure_completeness, rule)

HELD_OUT = "worst_concave_points"


@dsl
class FullSafetyPolicy(DecisionPolicyBase):
    """Declares the out-of-spec hazard as a hard refusal."""

    assert rule("deny_out_of_spec", when=lambda worst_area: worst_area > 1500,
                verdict=DENY)
    assert rule("release_certified_benign",
                when=lambda certified, p_positive: certified and p_positive < 0.50,
                verdict=RELEASE)
    assert rule("review_unsettled", when=lambda certified: not certified,
                verdict=ASK)


@dsl
class GapSafetyPolicy(DecisionPolicyBase):
    """Same policy with the hazard rule removed — the completeness ablation."""

    assert rule("release_certified_benign",
                when=lambda certified, p_positive: certified and p_positive < 0.50,
                verdict=RELEASE)
    assert rule("review_unsettled", when=lambda certified: not certified,
                verdict=ASK)


def main():
    d = load_breast_cancer()
    names = tuple(n.replace(" ", "_") for n in d.feature_names)
    X, y = d.data, d.target
    Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.5,
                                              stratify=y, random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                            stratify=yrest, random_state=0)
    all_rows = lambda A: [dict(zip(names, map(float, x))) for x in A]  # noqa: E731
    tr_rows, cal_rows, te_rows = all_rows(Xtr), all_rows(Xcal), all_rows(Xte)
    model_feats = tuple(n for n in names if n != HELD_OUT)
    mask = [i for i, n in enumerate(names) if n != HELD_OUT]
    est = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000)).fit(Xtr[:, mask],
                                                               ytr)

    ledger = (HazardLedger()
              .add("out_of_spec_area",
                   lambda r: r["worst_area"] > 1500.0, "spec sheet §4.1")
              .add("hidden_concavity",
                   lambda r: r[HELD_OUT] > 0.20,
                   "incident report #77 (feature not wired to the model)",
                   observable=False))

    results = {}
    for name, policy in (("FULL", FullSafetyPolicy),
                         ("GAP (rule removed)", GapSafetyPolicy)):
        dm = (DecisionModel(est, policy, features=model_feats)
              .fit_regime(tr_rows).calibrate(cal_rows, ycal, alpha=0.10))
        disps = [dm.dispose(r) for r in te_rows]
        results[name] = measure_completeness(disps, te_rows, ledger)
        results[name]["policy_digest"] = policy.policy().digest

    print(f"\nCompleteness vs the external hazard oracle — breast_cancer, "
          f"{len(te_rows)} held-out rows, alpha=0.10\n")
    print("hazard ledger:", ledger.version, ledger.digest[:19] + "…")
    hdr = (f"{'policy':<20}{'prec':>8}{'recall':>8}{'coverage':>10}"
           f"{'hazard':>8}{'false':>7}{'ceiling':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        prec = "   ⊥  " if r["precision"] is None else f"{r['precision']:.3f}"
        rec = "   ⊥  " if r["recall"] is None else f"{r['recall']:.3f}"
        ceil = "   ⊥  " if r["ceiling"] is None else f"{r['ceiling']:.3f}"
        print(f"{name:<20}{prec:>8}{rec:>8}{r['coverage']:>10.3f}"
              f"{r['n_hazard']:>8}{r['n_false_certificate']:>7}{ceil:>9}")
    print("\nper-hazard (rows / released):")
    for name, r in results.items():
        for code, v in sorted(r["per_hazard"].items()):
            flag = "observable" if v["observable"] else "HIDDEN (outside sigma(X))"
            print(f"  {name:<20} {code:<20} {v['rows']:>3} / {v['released']:>3}"
                  f"   {flag}")
    full = results["FULL"]
    print(f"\nFULL: precision CI {full['precision_ci']}, "
          f"recall CI {full['recall_ci']}")
    print(f"observable recall {full['recall_observable']:.3f} "
          f"({full['n_observable_hazard']} rows) · "
          f"hidden-only recall {full['recall_hidden_only']:.3f} "
          f"({full['n_hidden_only']} rows)")
    print(f"independence ceiling = {full['ceiling']:.3f} (share of hazard "
          f"rows with an observable hit); hidden-only catches are incidental "
          f"correlation, not policy coverage")

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"completeness_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"ledger": ledger.as_dict(),
                                "results": results}, indent=2,
                               default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()