"""Quickstart: a fitted sklearn model -> a fail-closed decision model.

    python examples/quickstart.py
"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mjolnir_ml import (ASK, DENY, RELEASE, DecisionModel, DecisionPolicyBase,
                        audit, dsl, rule)


@dsl
class MammographyPolicy(DecisionPolicyBase):
    """Release only a CERTIFIED benign call; deny a confident malignant one."""

    assert rule("deny_confident_malignant",
                when=lambda p_positive, conf: p_positive >= 0.80 and conf >= 0.80,
                verdict=DENY)
    assert rule("release_certified_benign",
                when=lambda certified, p_positive: certified and p_positive < 0.50,
                verdict=RELEASE)
    assert rule("review_unsettled", when=lambda certified: not certified,
                verdict=ASK)


def main():
    data = load_breast_cancer()
    names = tuple(data.feature_names)
    Xtr, Xrest, ytr, yrest = train_test_split(data.data, data.target,
                                              test_size=0.4, stratify=data.target,
                                              random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xrest, yrest, test_size=0.5,
                                            stratify=yrest, random_state=0)
    est = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000)).fit(Xtr, ytr)

    def rows(X):
        return [dict(zip(names, map(float, x))) for x in X]

    dm = (DecisionModel(est, MammographyPolicy, features=names)
          .fit_regime(rows(Xtr))                    # fail-closed regime floor
          .calibrate(rows(Xcal), ycal, alpha=0.10))  # conformal kappa

    print("model digest       ", dm.model_digest)
    print("policy digest      ", dm.policy.digest)
    print("calibration digest ", dm.kappa.digest)

    for i in range(6):
        row = rows(Xte)[i]
        d = dm.dispose(row)
        print(f"\nrow {i}: y={yte[i]} (0=benign, 1=malignant)")
        print(f"  verdict   {d.verdict:<8} basis={list(d.basis)} "
              f"guarantee={d.guarantee}")
        print(f"  say       {d.say}")
        print(f"  cert      {d.certificate_id[:26]}…  alpha={d.alpha}")

    disps = [dm.dispose(r) for r in rows(Xte)]
    report = audit(disps, safe=[int(v) == 0 for v in yte],
                   decidable=[True] * len(yte))
    print("\npair:", report["pair"])
    print("prec 95% CI:", report["prec_ci"], " rec 95% CI:", report["rec_ci"])
    print("verdicts:", report["verdicts"])
    print("\nreplay:", all(dm.replay(d, r) for d, r in zip(disps, rows(Xte))))


if __name__ == "__main__":
    main()
