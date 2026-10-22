"""Third-party audit: signed certificates replayed WITHOUT the learner.

An auditor receives (certificate, policy artifact, key). No estimator, no
training data, no live stack. The certificate publishes the fact table, so the
verdict is re-derived with Policy.dispose_facts and checked against the HMAC.

    python experiments/audit.py
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

from mjolnir_ml import (ASK, DENY, RELEASE, CertificateError, DecisionModel,
                        DecisionPolicyBase, certify, dsl, rule,
                        verify_certificate)

KEY = b"regulator-demo-key"


@dsl
class MammographyPolicy(DecisionPolicyBase):
    assert rule("deny_confident_malignant",
                when=lambda p_positive, conf: p_positive >= 0.80 and conf >= 0.80,
                verdict=DENY)
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
    rows = lambda A: [dict(zip(names, map(float, x))) for x in A]  # noqa: E731
    est = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    dm = (DecisionModel(est, MammographyPolicy, features=names)
          .fit_regime(rows(Xtr)).calibrate(rows(Xcal), ycal, alpha=0.10))

    te_rows = rows(Xte)[:120]
    certs = [certify(dm.dispose(r), key=KEY) for r in te_rows]
    policy = MammographyPolicy.policy()

    verified = sum(verify_certificate(c, key=KEY, policy=policy) is True
                   for c in certs)
    replayable = sum(1 for c in certs if c["replayable"])

    # tamper 1: change the stated facts of a SIGNED certificate
    bad_sig = json.loads(json.dumps(certs[0]))
    bad_sig["facts"]["mean_radius"] = 999.0
    sig_ok = True
    try:
        verify_certificate(bad_sig, key=KEY, policy=policy)
    except CertificateError as ex:
        sig_ok = False
        sig_err = str(ex)

    # tamper 2: flip the verdict of an UNSIGNED certificate (replay catches it)
    unsigned = certify(dm.dispose(te_rows[0]))
    unsigned["verdict"] = RELEASE if unsigned["verdict"] != RELEASE else DENY
    replay_ok = True
    try:
        verify_certificate(unsigned, policy=policy)
    except CertificateError as ex:
        replay_ok = False
        replay_err = str(ex)

    # tamper 3: wrong key
    key_ok = True
    try:
        verify_certificate(certs[0], key=b"wrong-key", policy=policy)
    except CertificateError:
        key_ok = False

    print(f"\nThird-party audit — {len(certs)} certificates, policy-only replay\n")
    print(f"verified with key + policy : {verified}/{len(certs)}")
    print(f"replayable (facts suffice) : {replayable}/{len(certs)}")
    print(f"tampered facts rejected    : {not sig_ok}  ({sig_err})")
    print(f"tampered verdict rejected  : {not replay_ok}  ({replay_err})")
    print(f"wrong key rejected         : {not key_ok}")
    print("\nsample certificate:")
    sample = dict(certs[0])
    sample["signature"] = sample["signature"][:26] + "…"
    sample["facts"] = dict(list(sample["facts"].items())[:3]) | {"…": "…"}
    print(json.dumps(sample, indent=2, default=str)[:1400])

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"audit_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({
        "n_certificates": len(certs), "verified": verified,
        "replayable": replayable, "tampered_facts_rejected": not sig_ok,
        "tampered_verdict_rejected": not replay_ok,
        "wrong_key_rejected": not key_ok,
        "policy_digest": policy.digest,
        "sample": {k: v for k, v in certs[0].items() if k != "signature"}},
        indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()