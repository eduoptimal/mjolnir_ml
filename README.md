<p align="center">
  <img src="assets/mjml.png" alt="mjolnir_ml" width="420">
</p>

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-401%20passing-brightgreen.svg)](#validation)
[![Audit coverage](https://img.shields.io/badge/audit%20coverage-100%25-brightgreen.svg)](#validation)
[![Coverage](https://img.shields.io/badge/package%20coverage-100%25-yellowgreen.svg)](#validation)
[![Audit](https://img.shields.io/badge/audit-Ed25519%20replay-black.svg)](#developer-and-auditor-surfaces)
[![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-orange.svg)](LICENSE)

**Turn a fitted predictor into a fail-closed decision model — and give a third
party everything needed to replay the decision without the model.**

A predictor answers *what will happen*. `mjolnir_ml` answers *may this run*.
It wraps a fitted estimator in a declared policy, calibrated trust facts, and a
signed, replayable audit bundle. Anything that cannot be certified abstains.

```text
estimator proposes facts  ->  policy disposes  ->  signed bundle  ->  independent audit
```

## Why it exists

High-risk automation fails in the gap between a prediction and a decision.
`mjolnir_ml` makes that gap explicit:

- the **model never disposes**; it only supplies facts;
- a **declared policy** is the only source of release, ask, deny, or abstain;
- missing, out-of-regime, or uncalibrated facts **fail closed**;
- every decision carries pinned digests and a certificate;
- an auditor can re-dispose from the published facts with **no model, no training
  data, and no live `DecisionModel`**.

## Install

```bash
pip install mjolnir-ml              # core library
pip install "mjolnir-ml[audit]"     # auditor path: no model, no sklearn
pip install "mjolnir-ml[all]"       # issuer + auditor + sklearn examples
```

The audit path is model-free. Importing it does not load `DecisionModel`,
`sklearn`, or `numpy`.

## Quickstart

### 1. Declare the policy

```python
from mjolnir_ml import (DecisionModel, DecisionPolicyBase, DENY, RELEASE,
                        dsl, rule)


@dsl
class PaymentPolicy(DecisionPolicyBase):
    facts = ("amount", "credit_limit")

    assert lambda amount: amount > 0, "amount_positive"
    assert lambda credit_limit: credit_limit > 0, "limit_positive"

    assert rule(
        "hard_deny",
        when=lambda amount, credit_limit: amount > credit_limit * 2,
        verdict=DENY,
    )
    assert rule(
        "release_certified",
        when=lambda certified, p_positive: certified and p_positive < 0.20,
        verdict=RELEASE,
    )
```

Policies are compiled from source into one canonical IR. Predicates are pure
over their declared parameters. Missing facts abstain; they never default.

### 2. Wrap the fitted estimator

```python
row = {"amount": 100, "credit_limit": 1000}

dm = DecisionModel(
    fitted_estimator,
    policy=PaymentPolicy,
    features=("amount", "credit_limit"),
)

dm.fit_regime(train_rows)
dm.calibrate(cal_rows, y_cal, alpha=0.10)

decision = dm.dispose(row)
print(decision.verdict)          # release | ask | deny | abstain
print(decision.basis)            # firing policy statement codes
print(decision.certificate_id)
```

For action risk rather than conformal coverage, use `calibrate_risk(...)` and
policies over `p_event`, `risk_release`, and `risk_deny`.

### 3. Issue a signed audit bundle

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from mjolnir_ml import WarrantLedger

signing_key = Ed25519PrivateKey.generate()
ledger = WarrantLedger.issue(decision, PaymentPolicy, row, signing_key)
ledger.write("decision.bundle.json")
```

The bundle contains the certificate, policy IR, facts, row, row hash, key id,
spec version, and Ed25519 signature. It is self-contained.

### 4. Audit independently

```python
from mjolnir_ml import WarrantLedger
from mjolnir_ml.warrant.signing import key_id, public_key_b64

public_key = signing_key.public_key()
trust = {
    "keys": {
        key_id(public_key): public_key_b64(public_key),
    }
}

report = WarrantLedger.from_bundle(
    "decision.bundle.json",
    trust=trust,
).audit()

assert report.valid
assert report.replayable
print(report.verdict, report.basis)
```

Or from the command line:

```bash
python -m mjolnir_ml.auditor verify decision.bundle.json \
  --trust trust.json --json
```

```json
{
  "schema": "warrantledger.audit/1",
  "valid": true,
  "replayable": true,
  "checks": {
    "structure": true,
    "signature": true,
    "binding": true,
    "replay": true
  },
  "verdict": "release",
  "basis": ["release_certified"],
  "missing": null,
  "errors": []
}
```

## Developer and auditor surfaces

| Role | Interface | Depends on model |
|---|---|---|
| Developer / issuer | `DecisionModel`, `@dsl`, `certify`, `WarrantLedger.issue` | yes |
| Auditor / disposer | `WarrantLedger.from_bundle(...).audit()`, `python -m mjolnir_ml.auditor` | **never** |

The auditor path has its own canonicalization, policy-IR evaluator, and replay
logic. It does not import the model, policy implementation, or scientific stack.

## Package layout

| Package | Responsibility |
|---|---|
| `mjolnir_ml.core` | policy, DSL, canonical model, frozen verdict/wire kernel, canonical JSON |
| `mjolnir_ml.calibration` | conformal calibration, risk, regime |
| `mjolnir_ml.release` | gates, feasibility, hazards, pair, reporting |
| `mjolnir_ml.operations` | live operation, discovery, causal analysis |
| `mjolnir_ml.warrant` | certificates, issuance, ledger, signing |
| `mjolnir_ml.auditor` | independent public-key replay |

## Core guarantees

| Guarantee | Behavior |
|---|---|
| Fail closed | Missing, invalid, out-of-regime, model-error, or uncovered facts abstain |
| Model never disposes | A confident estimator cannot manufacture a verdict |
| Only tighten | Ask, deny, and premises can only make the outcome stricter |
| Unknown is not zero | Uncalibrated groups yield no certified fact; required facts are absent |
| Replayable | Digests, row hash, policy binding, and fact-table replay are independently checked |
| Auditable | Ed25519 public-key signature, stable audit report, fail-closed exit codes |

## Audit bundle

```text
bundle_schema     warrantledger.bundle/1
spec_version      warrantledger.replay/1
certificate       disposition certificate
policy_ir         canonical policy table
facts             complete replay table
row               original stated input
row_hash          claim digest
key_id            Ed25519 key id
signature         Ed25519 signature over the bundle
```

Audit checks:

```text
structure  ->  signature  ->  binding  ->  replay
```

`replayable=false` is reported as attested-only. It is never presented as a
successful replay.

## Capabilities

- fail-closed decision models over fitted estimators;
- source-compiled `@dsl` policies with canonical IR;
- split conformal, normalized conformal, CQR, one-sided envelopes, and
  Mondrian groups;
- Conformal Risk Control for action-risk budgets;
- multivariate regime floors;
- readiness reports, hazard ledgers, live sessions, containment, promotion, and
  change gates;
- causal analysis tools;
- signed certificates and independent Ed25519 audit bundles.

## Requirements

- Python 3.10+
- `cryptography` for signed bundles
- `scikit-learn` only for the sklearn integration and examples
- auditor path: no model, no training data, no network

## Validation

```bash
python -m pytest tests -q
python -m pytest tests -q --cov=mjolnir_ml --cov-report=term-missing
```

Validation result: **401 tests passed, 2 skipped**. Auditor path: **100% line coverage**. Full package: **100% line coverage**.

## Scope

`mjolnir_ml` does not decide truth, legal authority, or domain policy. It
enforces a declared policy over facts, calibrates what it can certify, and fails
closed when it cannot. The caller owns the estimator, the policy, the
calibration data, and the authority to release.

## License

PolyForm Noncommercial 1.0.0. Commercial licensing: **reach@optimality.ai**.
