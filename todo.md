# TODO — one distribution, two roles

**Decision:** keep one package, `mjolnir-ml`, and split by **import boundary and
dependency extra**, not by wheel.

```text
pip install mjolnir-ml[audit]   # auditor only: no model, no sklearn
pip install mjolnir-ml[all]     # issuer + auditor + sklearn/examples
```

**Implementation status:** Phase 1–3 core is implemented. `WarrantLedger.issue`,
`WarrantLedger.from_bundle`, `WarrantLedger.audit`, the independent policy-IR
evaluator, Ed25519 bundle signatures, CLI, and lazy audit import path exist. The
remaining work is the published `spec/` documents/vectors, key rotation and
revocation, fuzz/size limits, and the one-page auditor guide.

There are exactly two roles:

```text
proposer / issuer   mjolnir_ml live path
disposer / auditor  mjolnir_ml audit path
```

A developer can replay locally. That dev replay has no authority and is not the
auditor module.

---

## 1. Why one distribution

- One version, one changelog, one spec.
- Same `WarrantLedger` facade for both roles.
- The auditor installs `[audit]`; model dependencies are absent.
- Functional separation is the import graph, not the wheel count.

The hard requirement:

```text
import mjolnir_ml[audit]  ->  must not import model, DecisionModel, sklearn, numpy
```

This only works if `mjolnir_ml/__init__.py` becomes lazy. Today it eagerly imports
the full library. That is the first refactor.

---

## 2. Target layout

```text
mjolnir_ml/
  __init__.py              # lazy exports; no eager model/sklearn imports
  ledger.py                # WarrantLedger facade: issue/load/audit
  issue.py                 # issuer path; local model imports only when called
  dev_replay.py            # developer replay; explicitly non-audit
  certificate.py           # certify + legacy verify; developer/compat
  audit/
    __init__.py            # auditor-only imports
    bundle.py              # load/validate bundle
    canonical.py           # own canonical JSON/hash
    policy_ir.py           # own restricted policy-IR evaluator
    dispose.py             # re-dispose from facts + policy IR
    verify.py              # signature, binding, replay
    report.py              # AuditReport
    cli.py                 # python -m mjolnir_ml.audit

spec/                      # not installed; wire contract and vectors
  warrantledger-bundle-v1.md
  warrantledger-replay-v1.md
  schemas/
  vectors/
```

Dependency rules:

```text
mjolnir_ml.audit  -X-> mjolnir_ml.model
mjolnir_ml.audit  -X-> mjolnir_ml.policy
mjolnir_ml.audit  -X-> sklearn / numpy
mjolnir_ml.ledger -X-> model at import time
```

`mjolnir_ml.ledger` may import model only inside the issuer method.

---

## 3. OO interface

```python
from mjolnir_ml import WarrantLedger

# Issuer / proposer
ledger = WarrantLedger.issue(
    disposition=disp,
    policy=policy,
    row=row,
    signer=ed25519_signer,
)
ledger.write("bundle.json")

# Auditor / independent disposer
ledger = WarrantLedger.from_bundle(
    "bundle.json",
    trust="trust.json",
)
report = ledger.audit()

assert report.valid
assert report.replayable
print(report.verdict, report.basis, report.missing)
```

Rules:

- `WarrantLedger.audit()` is the auditor path.
- It accepts no `DecisionModel`, `Policy`, estimator, or callback.
- `audit()` must work with only `[audit]` installed.
- `WarrantLedger.issue()` may import `DecisionModel`/policy lazily; that is the
  developer/issuer path.
- `Audit` is a method name issue: use `audit()` per Python convention.

A separate `dev_replay` exists only for development:

```python
from mjolnir_ml.dev_replay import replay_local
replay_local(disposition, row, policy)   # not an audit API
```

---

## 4. Bundle contract

The issuer bundles everything the disposer needs:

```json
{
  "bundle_schema": "warrantledger.bundle/1",
  "spec_version": "warrantledger.replay/1",
  "certificate": {},
  "policy_ir": {},
  "facts": {},
  "row": {},
  "row_hash": "sha256:...",
  "issuer": "mjolnir_ml",
  "key_id": "ed25519:...",
  "issued_at": "2026-11-16T00:00:00Z",
  "signature": "ed25519:..."
}
```

Rules:

- `policy_ir` must hash to the certificate’s `policy_digest`.
- `row` must hash to `claim_digest` and `row_hash`.
- `facts` must be complete for replay.
- unknown schema/spec fails closed;
- the audit artifact is the bundle, not the certificate alone.

### Audit report

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
  "basis": ["..."],
  "missing": null,
  "errors": [],
  "spec_version": "warrantledger.replay/1",
  "key_id": "ed25519:...",
  "certificate_id": "..."
}
```

`valid` is true only when every check passes. `replayable=false` is attested-only,
not a replay success.

Exit codes:

```text
0  valid
2  invalid / replay mismatch
3  unsupported schema/spec
4  signature or trust failure
5  malformed bundle / IO failure
```

---

## 5. Extras and dependency surface

`pyproject.toml`:

```toml
[project.optional-dependencies]
audit = ["cryptography>=42"]
sklearn = ["scikit-learn>=1.3"]
examples = ["scikit-learn>=1.3", "numpy>=1.24", "pandas>=2.0",
            "phart>=2.1", "networkx>=3.3"]
all = ["cryptography>=42", "scikit-learn>=1.3", "numpy>=1.24",
       "pandas>=2.0", "phart>=2.1", "networkx>=3.3"]
```

Definition of done:

- `pip install mjolnir-ml[audit]` can import `WarrantLedger` and run `audit()`.
- `sklearn`, `numpy`, and `mjolnir_ml.model` are **not** in `sys.modules` after
  importing the audit path.
- `pip install mjolnir-ml[all]` runs the full test suite.

---

## 6. Functional separation

| Concern | Issuer path | Auditor path |
|---|---|---|
| Model execution | yes, lazily imported | never |
| Policy evaluation | live | independent replay |
| Truth/calibration fitting | yes | never |
| Bundle creation | yes | never |
| Signature | sign Ed25519 | verify public key |
| Trust store | declares key id | owns trust |
| Audit report | no | yes |
| `DecisionModel` | yes | never |
| Network | optional | no |
| Database | optional | no |

The auditor may be a clone of the semantics. That is fine. It must not import
the issuer’s live implementation.

Agreement is enforced by `spec/vectors`, not by a shared runtime kernel.

---

## 7. Work plan

### Phase 0 — Freeze the wire contract

- [ ] Write `spec/warrantledger-bundle-v1.md`.
- [ ] Write `warrantledger-replay-v1.md`: policy-IR semantics, error codes,
      attested vs replayed.
- [ ] Publish schemas and vectors.
- [ ] Vector classes: valid, tampered signature, tampered row, wrong policy,
      unsupported spec, attested-only, missing fact, revoked key.
- [ ] Freeze compatibility for `mjolnir_ml.certificate.verify_certificate`.

### Phase 1 — Make the package lazy

- [x] Replace eager `mjolnir_ml/__init__.py` imports with lazy `__getattr__`.
- [x] Ensure `import mjolnir_ml` does not import sklearn/numpy.
- [x] Ensure `from mjolnir_ml import WarrantLedger` imports only the ledger
      facade.
- [x] Add import-graph tests.

### Phase 2 — Issuer path

- [x] Add `WarrantLedger.issue(...)`.
- [x] Add `mjolnir_ml/issue.py` with local model imports.
- [x] Sign with Ed25519; keep HMAC legacy.
- [x] Include policy IR, facts, row, row hash, key id, spec version.
- [x] Keep `certify(...)` compatible.
- [x] Separate `dev_replay.py` from audit claims.

### Phase 3 — Auditor path

- [x] Add `mjolnir_ml/auditor` with no model imports.
- [x] Implement own canonicalization and schema validation.
- [x] Implement restricted policy-IR evaluator; no `eval`.
- [x] Implement `WarrantLedger.from_bundle(...)` and `.audit()`.
- [x] Implement audit report and exit codes.
- [x] Add CLI:
      `python -m mjolnir_ml.audit verify bundle.json --trust trust.json --json`.
- [ ] Add key rotation and revocation checks.

### Phase 4 — Conformance and hardening

- [ ] Both issuer and auditor pass every `spec/vectors` case.
- [x] Clean-venv test with `[audit]` only.
- [x] Import-boundary CI:
      no `model`, `policy`, sklearn, numpy in audit import graph.
- [x] Tamper and version-skew tests.
- [ ] Fuzz and size-limit tests.
- [ ] Deterministic replay and no-network tests.
- [x] Mark `verify_certificate` as developer/compat, not audit.
- [ ] One-page auditor guide.

### Phase 5 — Optional service

- [ ] Wrap `WarrantLedger.from_bundle(...).audit()` in a stateless endpoint.
- [ ] Service imports only the audit path.
- [ ] Keep CLI/library primary.

---

## 8. Acceptance criteria

- [x] `pip install mjolnir-ml[audit]` runs `WarrantLedger.from_bundle(...).audit()`.
- [x] No `sklearn`, `numpy`, `model`, or `DecisionModel` in the audit import graph.
- [x] No `eval()` on bundle/policy input in the auditor.
- [x] Ed25519 public-key verification works; HMAC is legacy/internal.
- [x] Tampering with any bound field fails closed.
- [x] Unsupported spec returns exit `3`.
- [x] Invalid replay returns exit `2`.
- [x] Valid bundle returns exit `0` with a deterministic report.
- [x] `replayable=false` is attested-only, never “valid replay.”
- [ ] Issuer and auditor pass the same `spec/vectors`.
- [x] Developer replay is explicitly non-audit.
- [ ] A third party can audit from a bundle, a public key, and a one-page guide.

---

## 9. Non-goals

- No separate wheel.
- No shared runtime replay kernel.
- No model in the auditor.
- No training or calibration in the auditor.
- No network dependency in the CLI.
- No L3 authority claim.
- No silent compatibility with unknown schemas.

---

## 10. Main risks

| Risk | Control |
|---|---|
| `[audit]` still imports the full library | Lazy `__init__` + import-graph test |
| Semantic drift | Spec + conformance vectors |
| Untrusted policy executes code | Restricted evaluator; no `eval` |
| HMAC mistaken for third-party proof | Ed25519 default; HMAC legacy |
| Auditor accepts issuer verdict | Re-dispose from facts, never trust verdict |
| Key compromise | Rotation, revocation, unknown-key fail-closed |
| Version skew | Explicit versions; unknown fails |
| Bundle too large | Size/depth/fact limits |
| “Valid” claimed for attested-only | Separate `valid` and `replayable` |

---

## 11. Recommended first commit

1. Freeze `spec/` and vectors.
2. Make `mjolnir_ml/__init__.py` lazy.
3. Add `WarrantLedger.issue` and bundle emission.
4. Add `mjolnir_ml/auditor` with independent replay and `.audit()`.
5. Add the clean `[audit]` environment test.

One package, two roles, one wire contract.
