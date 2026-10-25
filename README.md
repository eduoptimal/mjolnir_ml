# mjolnir_ml — turn a fitted predictor into a fail-closed decision model

A predictor answers **what will happen**. A decision model answers **may this
run**. This package is the transform between the two: keep the learner, put a
declared policy, a calibrated trust index, and a replayable certificate around
it — and make every decision that cannot be certified **abstain**, by
construction.

```python
@dsl
class MammographyPolicy(DecisionPolicyBase):
    # release only a CERTIFIED benign call; deny a confident malignant one
    assert rule("deny_confident_malignant",
                when=lambda p_positive, conf: p_positive >= 0.80 and conf >= 0.80,
                verdict=DENY)
    assert rule("release_certified_benign",
                when=lambda certified, p_positive: certified and p_positive < 0.50,
                verdict=RELEASE)
    assert rule("review_unsettled", when=lambda certified: not certified,
                verdict=ASK)

dm = (DecisionModel(fitted_estimator, MammographyPolicy, features=features)
      .fit_regime(train_rows)                  # regime floor (I3)
      .calibrate(cal_rows, y_cal, alpha=0.10)) # conformal trust index κ

d = dm.dispose(row)          # release | ask | deny | abstain + basis + certificate
cert = certify(d, key=b"audit-key")
verify_certificate(cert, key=b"audit-key", policy=MammographyPolicy,
                   row=row)  # structure · signature · binding · replay
```

`pred`/`conf`/`p_positive` are model facts; `certified`/`set_size` are the conformal
facts; features are stated input facts. The learner **proposes facts**; only
the declared policy disposes.

---

## 1. The type and the law

One object, from the Ω-TD purest form (`mjolnir_cartographer/beta_release`),
instantiated for supervised learners:

```
Contract     Γ = (policy, alpha, regime)        declared, digested
Index        κ = conformal calibration           per-region, finite-sample
Kernel       M = the fitted estimator           swappable, never trusted
Representation R = features + training bounds   input contract
Certificate  C = (claim, policy, model, κ [, risk, regime]) digests
               + verdict + basis + guarantee/assurance
```

```
Adm(x) = { release  : every premise holds ∧ engine ROUTE ∧ a release rule fires
                      ∧ the conformal fact the rule needs is present
           ask      : an escalation rule fired, or a premise declined
           deny     : a deny rule fired (never softened by an ask)
           abstain  : missing fact / out of regime / model error / uncovered }
```

**Law.** `release ⇒ a firing release statement + a clean engine ROUTE`. Every
property follows (pinned by `tests/test_model.py`). The disposition's
`guarantee` is **`declared`** (the deciding statements read only stated facts)
or **`model_backed`** (a deciding statement read estimator output); the
conformal proof mark is the separate fact `certified` (a prediction-set
singleton). The two never share a name. `assurance` goes further and records
what the deciding statements actually used: `declared`, `model_backed`,
`pointwise_coverage` (conformal set/interval facts), `action_risk_control`
(CRC risk facts), or `dual_track` (a release fired from both coverage and CRC
rules). It is derived from the policy and the fired basis, never authored.

| property | why it is a corollary |
|---|---|
| fail-closed | uncovered, missing, non-finite, out-of-regime, model error ⇒ abstain |
| model never disposes | verdicts come only from the policy; a confident learner cannot manufacture one |
| only-tighten | ask/deny/premises can only make the outcome stricter than a release |
| unknown ≠ zero | uncalibrated group ⇒ no certified set/interval; required-and-absent ⇒ abstain |
| replayable | pinned digests (claim/policy/model/calibration, plus risk/regime when present); `verify()` rebuilds the id, hashes an optional row, and replays the policy over the published facts |

**Two-phase compile.** The frozen engine has no `release` verdict (`ROUTE` means
"no obligation fired"). Premises compile to abstain-rules (phase 1), ask/deny to
engine rules (severity max: deny beats ask), and release rules are evaluated
**only over a clean ROUTE** — a firing release rule is the only way to release;
no firing release ⇒ `abstain(uncovered)`.

---

## 2. `@dsl`: policies are compiled from source, not executed

```python
@dsl
class Policy(DecisionPolicyBase):
    facts = ("amount", "credit_limit")            # optional, compile-tight

    assert lambda amount: amount > 0, "amount_positive"   # premise (violation -> abstain)
    assert rule("hard_deny", when=lambda amount, credit_limit:
                amount > credit_limit * 2, verdict=DENY)
    assert rule("auto_ok", when=lambda p_positive, conf:
                p_positive < 0.05 and conf >= 0.95, verdict=RELEASE)

    @rule(verdict=ASK)                            # method form, unit-testable
    def escalate_mid(self, conf):
        return conf < 0.80
```

Why source-harvest, not runtime introspection:

* class-body `assert` leaves nothing in `cls.__dict__`; runtime registration
  inside asserts is invisible or (under `python -O`) **deleted**;
* one AST pass harvests asserts, `rule(...)` assignments and `@rule` methods in
  source order; the IR is the same table every frontend produces, and the
  digest is over canonical JSON of the IR (never bytecode);
* predicates are **pure over their parameters**: no calls, attributes,
  subscripting or free names — the whitelist makes them replayable. A policy
  the compiler cannot read (REPL/exec) is refused, never silently degraded;
* `python -O` immunity is pinned by a subprocess test.

Errors are fail-closed at authoring time: missing premise codes, unknown facts,
forbidden verdicts, duplicate codes, non-literal rule metadata.

---

## 3. κ: certified claims, not confident ones

`DecisionModel.calibrate(rows, y, alpha, group=None)` fits conformal
quantiles per group (Mondrian):

* **classifier** — prediction set `{y : 1 − p_y(x) ≤ q}`; `certified` means the
  set is a **singleton** at the declared α; `set_size` travels with the
  disposition. Empty set ⇒ not certified (never a free claim);
* **regressor** — interval `[pred − q, pred + q]`; policies gate on `lo`/`hi`, so
  an action is released only when its whole interval clears the declared band.
  The width is selectable without changing the guarantee: `method="normalized"`
  takes a fitted scale estimator and returns `[pred ± q·σ(x)]`; `method="cqr"`
  takes fitted lower/upper quantile estimators and returns `[l(x) − q, u(x) + q]`.
  Both are calibrated by `fit_kappa` exactly like split conformal, pin their
  auxiliary estimators in the calibration digest, and fail closed (no interval)
  when the auxiliary model emits a non-positive/non-finite width;
  `one_sided=True` separates the tails: `lo` is calibrated at
  `alpha_lower` (so `P(Y >= lo) >= 1 - alpha_lower`) and `hi` at `alpha_upper`.
  That is the two-track form a policy needs when release reads only `hi` and
  deny reads only `lo`; it is a **decision envelope**, not a joint interval
  (joint coverage is only `1 - alpha_lower - alpha_upper` by the union bound),
  and `report()` labels it that way;
* **unknown group** — no calibrated set/interval: classifier reports
  `certified=False`, regressor omits `lo`/`hi` so a rule requiring them ⇒
  `abstain(missing:lo)`. Unknown is never zero;
* the conformal floor `n ≥ ceil(1/α)−1` is enforced **per group**; too little
  data is a `CalibrationError`, not a silent uncalibrated claim.

The disposition pins `alpha` and `calibration_digest`; recalibrating changes the
digest and invalidates stale certificates — a feature, not a bug.

### 3b. The risk track: `calibrate_risk` (action risk, not coverage)

Coverage answers *where Y lies for this row*. It does not answer *whether an
action is safe for the cohort I act on*. The second question is the decision
risk, and it is a different guarantee:

```
coverage   P(Y in C(x)) >= 1 - alpha          per row
risk       E[loss | action] <= alpha          over the selected cohort
```

`DecisionModel.calibrate_risk(rows, y, tau=..., estimator=crossing)` fits a
probabilistic **crossing model** `p_event(x) = P(Y > tau | x)` on held-out rows
and calibrates two action thresholds by Conformal Risk Control (CRC):

```
release_ok(p)  <=>  p <= lambda_release      loss = 1[Y > tau]
deny_ok(p)     <=>  p >  lambda_deny         loss = 1[Y <= tau]
                   lambda_release <= lambda_deny
```

The two thresholds are chosen **jointly** on the same CRC grid, so the release
and deny sets are disjoint by construction; the middle `(lambda_release,
lambda_deny]` is `ask`. The pair is the one that maximizes the acted cohort
while both corrected empirical risks satisfy their budgets. When the budgets
cannot support a gap, the ask band is empty (a binary split is the honest
outcome); when one action cannot be certified, its threshold is `None` and the
action is never taken (fail closed).

The resulting facts are `p_event`, `risk_release`, `risk_deny`; a policy that
requires them gets `assurance="action_risk_control"` (the coverage facts
`certified`/`lo`/`hi` get `pointwise_coverage`; a release that fires both
tracks gets `dual_track`). The CRC guarantee is on the **expected risk of the
selected cohort** under exchangeability, not a per-row certainty claim. Per-
action budgets are marginal; a joint guarantee over both actions needs
`alpha_release + alpha_deny <= alpha_total` (union bound), and the report
states them separately.

For a classifier the same call applies: pass a 0/1 event (e.g. `y != y_hat`,
or `y == dangerous_class`) and a fitted `predict_proba` estimator. That is
selective classification with a finite-sample risk budget.

### 3c. Feasibility: a rule that cannot fire is a contract defect

`feasibility(dm, rows, threshold=...)` audits every compiled statement against
the observed fact box:

```
observed    rows that actually fired the statement
reachable   observed, or reachable in the observed fact box
dead        neither — a declared rule that cannot fire
```

For a regressor threshold the decisive evidence is the attainable bound:
`deny_lo > tau` is dead when `max(lo) <= tau`; `release_hi <= tau` is dead when
`min(hi) > tau`. For the risk track it is `max(p_event) < lambda_deny` /
`min(p_event) > lambda_release`. `report()` prints a `FEASIBILITY` section and
names `DEAD RULES`; `readiness(..., thresholds=ReadinessThresholds(
require_reachable_rules=True))` can make deployment fail on one.

The honest conclusion is often that an action is unreachable for the model and
threshold — e.g. a 90% lower bound for `Y > tau` cannot clear `tau` when the
conditional spread is large. That is information for the policy owner (change
the model, the threshold, or the action), not something a calibration method
can manufacture.

### 3d. Multivariate regime floor

The default `fit_regime(rows)` is an axis-aligned box: a point is out of regime
if **any** feature leaves its training min/max. Correlated points routinely
cross one axis while staying in the joint support, so this over-abstains.

```python
dm.fit_regime(train_rows, method="mahalanobis",   # or "isolation_forest"
              alpha_ood=0.02, calibration_rows=cal_rows)
```

`mahalanobis` fits a robust center/covariance (`MinCovDet`) and scores each row
by its robust distance; `isolation_forest` scores by isolation depth. The
cutoff `tau` is the conformal quantile of the calibration scores, so the
false-abstention rate is bounded at `alpha_ood` under exchangeability (with the
usual `n >= ceil(1/alpha_ood)-1` floor). `method="custom"` accepts a
`score_fn(matrix) -> scores` for any detector. The calibrated cutoff and
detector configuration are pinned in `regime_digest` and travel with every
disposition and certificate. The report shows
`regime required method=mahalanobis alpha_ood=0.02 tau=...`.

The box remains the default (and is still recorded for reporting), so existing
models and categorical features are unchanged.

### One artifact for review: `report()`

`report(dm, rows, safe=..., decidable=..., ledger=..., key=...)` renders the whole picture as a
fixed-width text/table (and `report_data` returns the same as JSON):

```
CONTRACT   policy/model/calibration digests, α, regime, features, every compiled statement
DECISIONS  verdict census, claim/certified rates, abstention causes named
OMEGA PAIR precision/recall + Wilson CIs + ALWAYS/NEVER degenerate controls
HAZARDS    external-ledger completeness, per hazard, observable vs hidden-only recall
PREDICTIONS conformal set size / interval width
CERTIFICATES signed & verified, replayable-from-facts counts
SAMPLE     per-row verdict, basis, guarantee, model facts
```

Two executed notebooks walk through it end to end:
`examples/random_forest_regressor.ipynb` (diabetes, all features) and
`examples/random_forest_classifier.ipynb` (breast_cancer binary + wine multiclass,
RandomForestClassifier, all features).

---

## 4. The Ω pair: what a high-risk deployment actually reports

`omega_pair` / `audit` / `audit_model` implement Definition 1 from
`omega_certifiability`, for decisions:

```
claim      verdict ∈ {release, deny}        (ask/abstain are the reject option)
certified  the claim matches the oracle's required side
A          the oracle settles the row
prec = cert/claim  (0 claims -> ⊥, never 1)      rec = cert/|A|
```

plus `degenerate_controls` (ALWAYS/NEVER) and Wilson intervals. The ALWAYS
policy makes every scalar look mediocre; NEVER (the policy that says nothing)
reports **⊥, 0** while every scalar metric would call it safe — the
constructibility table, reproduced on your batch.

### Measured on real sklearn data (`experiments/bench.py`, α = 0.10)

Three decision shapes × three learners; **harness vs. the same fitted learner as
a total decision model** (which never abstains):

| dataset (kind) | learner | harness prec / rec | ungated prec / rec | claims | certified |
|---|---|---:|---:|---:|---:|
| breast_cancer (binary) | logistic | **0.990** / 0.912 | 0.982 / 0.982 | 105/114 | 0.92 |
| breast_cancer | random_forest | **0.980** / 0.860 | 0.956 / 0.956 | 100/114 | 0.92 |
| breast_cancer | hist_gradient | **0.981** / 0.886 | 0.947 / 0.947 | 103/114 | 0.90 |
| wine (multiclass) | logistic | **1.000** / 0.444 | 0.972 / 0.972 | 16/36 | 0.67 |
| wine | random_forest | **1.000** / 0.306 | 0.944 / 0.944 | 11/36 | 0.61 |
| wine | hist_gradient | **1.000** / 0.361 | 0.917 / 0.917 | 13/36 | 0.50 |
| diabetes (regression) | ridge | ⊥ / 0.000 | 0.798 / 0.798 | 0/89 | – |
| diabetes | random_forest | **1.000** / 0.045 | 0.742 / 0.742 | 4/89 | – |
| diabetes | hist_gradient | **1.000** / 0.022 | 0.708 / 0.708 | 2/89 | – |

Zero overclaims on every dataset. The reading is the papers' reading: the pair
buys **soundness at the price of coverage** — precision rises to ~1.0 (the only
wrong claim in the table is one benign row the harness released), recall falls
because `ask`/`abstain` are honest. Diabetes is the honest negative: the
learners' calibrated intervals are ~206–240 target units wide against a 320-unit
range, so the harness abstains almost everywhere. That is the floor working —
wide uncertainty cannot be certified into narrow decisions.

Reproduce: `python experiments/bench.py` (writes JSON under
`experiments/results/`).

---

## 5. Prospect: why this unlocks high-risk applications

The regulated world does not buy "a model with 0.97 AUC". It buys a **decision
procedure with per-decision evidence**. That is the artifact this package
produces — and the learner becomes a swappable component inside it.

1. **Credit, insurance, lending.** Adverse-action notices need reason codes; the
   disposition's `basis` *is* the reason set, and the certificate pins the
   policy version that produced it. Unseen categories (an employment type the
   model never saw) abstain instead of being silently scored. A model refresh
   changes `model_digest` only; policy semantics stay audited.
2. **Clinical decision support.** "Release only when the conformal prediction
   set is a singleton at α = 0.05, escalate otherwise" is implementable policy
   text, with a replayable proof per case. The device's failure mode is
   *declining to decide*, not a confident wrong call.
3. **Fraud / AML blocking.** `deny` requires stated facts plus confidence;
   `ask` routes to review; the Ω pair against an external hazard oracle makes
   each release's precision/recall measurable, with Wilson intervals, instead of
   a vendor accuracy number.
4. **Industrial, energy, actuarial.** Regression intervals gate actuation: an
   action runs only if the whole conformal interval clears the declared band —
   Chow's reject option with a finite-sample guarantee.
5. **Model risk management / EU AI Act style audits.** The pinned digests
   (claim/policy/model/calibration, plus risk/regime when present) and the
   certificate id per decision are the evidence trail: the auditor rebuilds the
   id, checks the input hash, and replays the policy over the published facts
   without the model's code. A policy/calibration drift is a digest mismatch,
   not a discovery.

The economic claim: this is the layer that converts *"we cannot put this model
in a regulated process"* into *"we deploy the harness; the model is one
replaceable part"* — and gives a go/no-go metric (Ω pair + abstention rate)
that survives the degenerate policies every scalar metric fails.

---

## 6. The value ladder, measured

Four rungs build on the base harness. Each has a runnable experiment and tests.

### 6.1 Adaptive conformal for drift (ACI)

Static conformal is a promise about exchangeable data. Under drift it degrades —
the beta_release **G4** finding. `calibrate(..., adaptive=True)` fits an ACI
index: after each decision, `observe(row, y)` updates the per-group risk level
from realized miscoverage (`alpha_{t+1} = alpha_t + gamma*(alpha0 - err_t)`),
so the prediction set widens after errors and narrows when it over-covers.

`experiments/drift.py` (breast_cancer, concept drift: 45% label flips after row 70):

| arm | coverage pre | coverage post | mean set pre | mean set post |
|---|---:|---:|---:|---:|
| static | 0.84 | **0.40** | 0.84 | 0.70 |
| adaptive (ACI) | 0.86 | **0.64** | 0.86 | **1.28** |

`alpha_t` falls 0.100 → 0.037. The repair is real and priced: coverage recovers,
width grows. The unit test pins the mechanism (error → alpha down → wider set;
coverage → alpha up → narrower set) and the monotone recovery direction.

### 6.2 Policy completeness vs an external hazard oracle

`HazardLedger` + `measure_completeness()` implement the CERTIFIABILITY protocol:
**H** is external (geometry, spec sheets, incident records — never the policy);
the ledger is union-only (retiring a hazard needs a refutation record);
precision = 1 − false certificates, recall = hazards caught, coverage = price.

`experiments/completeness.py` (breast_cancer; `worst_concave_points` held out of
the model, so `hidden_concavity` is outside σ(X)):

| policy | precision | recall | coverage | false certs | ceiling |
|---|---:|---:|---:|---:|---:|
| FULL (declares the hazard) | 0.742 | 0.692 | 0.217 | 8 | 0.654 |
| GAP (rule removed) | **0.500** | **0.115** | 0.322 | 23 | 0.654 |

Observable recall: **1.000** (full) vs 0.118 (gap) on the 17 out-of-spec rows;
hidden-only recall 0.111 — incidental correlation, not coverage. Removing a
declared hazard costs both axes, exactly as the paper's table shows; the hidden
hazard sits at the independence ceiling.

### 6.3 Cost-sensitive verdicts

`Costs(C_FR, C_FD, C_A)` prices the reject option; `cost_policy` compiles the
optimal Chow bands into the same IR every other policy uses, and
`require_certified=True` adds the conformal singleton gate.

`experiments/costs.py` (breast_cancer, calibrated probabilities; C_FR=20,
C_FD=5, C_A=2 → release < 0.10 < ask < 0.60 < deny):

| arm | expected cost /1000 | realized cost /1000 | falseRel | falseDeny | asked |
|---|---:|---:|---:|---:|---:|
| cost bands | **803.4** | 468.5 | 0 | 1 | 31 |
| cost bands + certified | **803.4** | 468.5 | 0 | 1 | 31 |
| fixed 0.5 | 915.0 | **104.9** | 0 | 3 | 0 |
| always ask | 2000.0 | 2000.0 | 0 | 0 | 143 |
| always release | 11995.5 | 12587.4 | 90 | 0 | 143 |

The bands are posterior-optimal (expected cost beats every fixed rule), but on
this 143-row draw the realized cost favors the fixed threshold: the ask band
charged 31 reviews while only 3 false denies occurred. That disagreement is
information — either the calibration is sharp enough that the ask band is not
worth its price at these costs, or the costs are mispriced. Report both axes;
a single realized number on a small sample is not the decision.

### 6.4 Does a better model make better decisions? (`experiments/model_ladder.py`)

Two axes, seed-paired: **training fraction** 15→100% × **RandomForest capacity**
(poor = 10 trees/depth 3/leaf 20 · medium = 100/depth 10/leaf 5 · rich =
300/unlimited/leaf 1). The conformal calibration rows are fixed across the sweep,
so kappa never sees more data; 3 seeds per cell; means with 95% CIs. Each cell
reports decision precision/recall, claim rate, hazard precision/recall and the
model metric — never a scalar.

| dataset | model metric | decision recall | decision precision | hazard precision |
|---|---:|---:|---:|---:|
| wine (multiclass) | acc 0.907 → **0.981** | 0.269 → **0.370** (+0.102 ± 0.079) | 0.970 → **1.000** | 0.316 → **0.204** (−0.112 ± 0.027) |
| diabetes (regression) | MAE 46.2 → 46.8 (n.s.) | 0.154 → **0.060** (−0.094 ± 0.092) | 0.972 → 1.000 | 1.000 → 1.000 |

Two findings worth keeping:

1. **Wine: a better learner improves decisions and exposes a policy gap.** Accuracy
   and decision recall rise significantly with no precision loss — but hazard
   precision *falls*, because more certified releases land inside the external
   hazard band the policy declares too narrowly (ledger: alcohol > 13.5; rule:
   > 14.5). A better model does not fix an incomplete policy; it makes the gap
   measurable.
2. **Diabetes: capacity is not capability.** The richer forest does not lower MAE
   (difference not separable from seed noise), and decision recall falls with the
   claim rate. The harness reports the truth instead of assuming "bigger is better".

Spearman(model quality, decision recall): wine **+0.51**, diabetes **−0.25** over
the swept cells. Plots: `experiments/results/figures/model_ladder_wine.png` and
`model_ladder_diabetes.png`. The poor model at small data has **0 claims**, where
hazard recall reads 1.0 — the coverage/claims column is what exposes that vacuity.

### 6.5 Signed certificates for third-party audit

`certify(disp, key=...)` produces an HMAC-SHA256 certificate carrying the fact
table, the pinned digests (claim/policy/model/calibration, plus risk/regime when
present), α and the certificate id. `verify_certificate(cert, key=...,
policy=..., row=...)` checks four things, each fail-closed:
**structure**; **signature** (HMAC over the canonical certificate);
**binding** — the certificate id is recomputed from the pinned digests +
verdict/basis/guarantee, and a supplied `row` is hashed against
`claim_digest`; and — for decisions over a complete fact table —
**replay**: the verdict/basis/missing are re-derived from the published facts
with `Policy.dispose_facts` (the same function live `dispose()` ran), and the
`guarantee`/`assurance` tags are re-derived from the policy. No learner, no
training data, no repository live stack — and nothing estimator-class-specific:
the policy replays whatever facts the table carries (regressor `lo`/`hi`,
classifier `conf`/`certified`/`set_size`, risk `p_event`/…).

```python
verify_certificate(cert, key=KEY, policy=Policy, row=row)
# structure · signature · binding · replay; raises CertificateError on any miss
```

The binding check rebuilds `certificate_id` from the pinned digests with the
same recipe live `_pack` uses, and hashes the supplied `row` against
`claim_digest`. Replay re-derives verdict/basis/missing AND the
`guarantee`/`assurance` tags from the policy, so an unsigned certificate cannot
relabel them even after the public id recipe is recomputed.

`experiments/audit.py` (120 certificates):

```
verified with key + policy : 120/120
replayable (facts suffice) : 111/120
tampered facts rejected    : True  (signature does not match the certificate)
tampered verdict rejected  : True  (certificate_id does not match the pinned digests (binding check failed))
wrong key rejected         : True
```

Harness-stage abstentions that depend on the model or training bounds
(`model_error`, `out_of_regime*`, `regime_not_fitted`, missing/invalid facts)
are marked `replayable=False`: signature and digests still bind them, and the
certificate says so instead of pretending a third party can recompute them.

---

## 7. The flywheel: readiness, live, containment, promotion

Two verbs, deliberately separate:

- **`readiness(dm, rows, safe=..., decidable=..., ledger=..., key=...)`** —
  one-shot deployment gate. Returns a `ReadinessReport` with per-gate verdicts
  (`calibration_present`, `overclaim`, `precision`, `recall`, `coverage`,
  `abstain_rate`, optional hazard precision/recall, certificate replay rate), the
  Ω pair, hazard completeness, a deterministic `readiness_id`, and a signed
  evidence bundle. It mutates nothing: no calibration, bounds, policy or journal.
- **`LiveSession(dm, journal=..., ledger=..., detectors=..., approval_verifier=...)`**
  — a journaled runtime. `Journal` is append-only, hash-chained and tamper-evident.
  `decide` journals the row, fact table, verdict and digests; `observe` joins the
  delayed outcome and updates ACI only; `status(window_batches=…)` returns verdicts,
  coverage, MAE/accuracy, hazard precision and the **human review queue** of
  abstentions; `tick()` is deterministic; `contain`/`uncontain`, `retract`, and
  versioned `ChangeSet` promotion complete the loop.

```python
rep = readiness(dm, rows, safe=safe, decidable=decidable, ledger=ledger, key=KEY)

live = LiveSession(dm, journal="journal.jsonl", ledger=ledger,
                   detectors=detectors, approval_verifier=authority.verify)
for row, y in batch:
    d = live.decide(row, batch=b)
    live.observe(d.certificate_id, y)          # ACI: kappa only
    live.auto_act(live.tick(), batch=b)        # hazard release -> contain
st = live.status(window_batches=1)            # verdicts, coverage, queue

change = ChangeSet(lever="contract", reason="add missing hazard rule",
                   policy=PolicyV2.policy())
live.promote(change, oracle=oracle, ledger=ledger)   # tightening: no approval
live.uncontain(authority.sign(live.containment.change_digest))  # loosening: signed
```

The law, enforced rather than documented: **tightening is free, loosening is signed.**
`contain()` may only turn a release into an ask and is unconditional. A `ChangeSet`
that adds a release on any logged row is refused unless an `Approval` verifies against
a verifier the session holds — the signing key lives outside the session.
`tick()` recomputes detections from the journal (same journal ⇒ same alarms) using
declared sequential CUSUMs, each with a target, slack, threshold, minimum n and a
declared minimum detectable effect. Warnings never act; the only critical detector is
a released external hazard. The hazard detector is epoch-aware: after a repair lifts
containment, history is retracted, not re-paged. `retract(model_digest=…,
policy_digest=…)` lists the certificates issued under a faulty version for human
triage; nothing is deleted.

A `ChangeSet` is evaluated by replaying the journal's stored facts through
`Policy.dispose_facts` (no model), or by re-scoring rows with a new estimator and a
paired calibration split. Promotion requires all of: hazard releases not increased,
precision and recall non-regression, at least one strict improvement, and — if the
candidate adds a release on any logged row (`may_loosen`) — an external approval
that verifies against the session's verifier. Promotion creates a new immutable
version (`with_policy` / `with_estimator`); the old version remains for rollback.

`experiments/flywheel.py` deploys a diabetes regressor over six batches with staged
drift, injects an age-band hazard, and runs the whole loop. Measured run:

| batch | release | ask | deny | abstain | coverage | MAE | alpha_t | hazard rel. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 10 | 0 | 2 | 0.833 | 56.3 | 0.060 | 0 |
| 2 | 2 | 8 | 0 | 2 | 0.917 | 48.2 | 0.080 | 0 |
| 3 | 4 | 7 | 0 | 1 | 0.917 | 63.5 | 0.100 | 0 |
| 4 | 1 | 11 | 0 | 0 | 0.833 | 78.1 | 0.058 | **1** |
| 5 | 0 | 12 | 0 | 0 | 0.833 | 142.0 | 0.047 | 0 |
| 6 | 0 | 9 | 1 | 1 | 0.818 | 103.9 | 0.007 | 0 |

Batch 4: hazard release → **containment**; the repair ChangeSet is **promoted**
(tightening, no approval); containment is **lifted with an external approval**;
48 v1 certificates are **retracted** for triage. Batch 6's second hazard is **denied**
by v2, not contained. ACI paid for coverage in width (α 0.100 → 0.007). The run also
found two design bugs that are now fixed and documented in the generated report: the
Ω oracle must encode the external hazard (otherwise a correct repair fails the Pareto
gate), and the hazard detector must be epoch-aware (otherwise history re-contains the
session after every repair).

Outputs: `experiments/results/flywheel_journal.jsonl`, results JSON/CSV, two figures
under `experiments/results/figures/` (`fig_flywheel_batches.png`,
`fig_flywheel_actions.png`), and an auto-written `flywheel_report_*.md` with the
batch table, detections, actions, human queue, design lessons and limits.

A self-contained walkthrough is in `examples/flywheel_live.ipynb`: readiness →
four live batches → containment → replay/Pareto repair → signed uncontain →
retraction, with the review queue printed at each batch.

## 8. Non-claims (read these before deploying)

* Coverage is conformal, **relative to exchangeability and the declared α**.
  ACI restores coverage under drift only approximately in a finite window, and
  pays for it in width (0.84 → 1.28 labels on the drift experiment). Regroup,
  recalibrate or adapt; never quote a static coverage number under drift.
* The policy is declared, not verified: an incomplete policy misses hazards.
  Completeness is measurable *against an external hazard oracle*, never against
  itself. The numeric ceiling assumes hidden hazards are independent of the
  inputs; when they correlate, hidden-only catches are incidental and are
  reported separately. Hazards invisible in both X and H remain invisible.
* Cost-sensitive bands are only as good as the probability they consume: use a
  calibrated classifier, and treat expected cost (posterior-optimal) and
  realized cost (small-sample) as different evidence.
* The regime floor is a training-support box, not a density model: it catches
  unseen values/categories, not adversarial in-support inputs.
* HMAC is symmetric. An auditor holding the key can also sign; for adversarial
  multi-party settings, replace `_signature` with an asymmetric scheme. The
  certificate format does not change.
* `@dsl` predicates are pure over declared facts; they cannot express
  cross-row logic, temporal conditions or cost trade-offs inside one predicate.
* Discovery is a *claim-audit* loop, not invention: a candidate is only as good
  as the declared criterion and the audit population. The twin bounds
  overreach; it does not prove completeness. A `REFUTED -> CERTIFIED` move is
  evidence in conflict with the record, which is why it needs a signature
  rather than a better argument.

---

### Naming contract

The public vocabulary is differentiated and load-bearing — one verb, one job:

| name | meaning |
|---|---|
| `fit_regime(rows, method=..., calibration_rows=...)` | record the support: per-feature box (default) or a calibrated multivariate floor |
| `calibrate(rows, y, alpha, ...)` | fit κ (conformal; `one_sided`, `normalized`/`cqr`, `adaptive=True` for ACI) |
| `calibrate_risk(rows, y, tau=..., estimator=...)` | fit the CRC action thresholds (`p_event`/`risk_release`/`risk_deny`) |
| `observe(row, y)` | consume a realized outcome into κ |
| `dispose(row)` | decide one row — the only disposer |
| `replay(disp, row)` | re-derive a disposition from (policy, facts) |
| `fit_kappa(...)` | fit κ directly, without a `DecisionModel` |
| `fit_risk_index(...)` | fit the CRC thresholds directly |
| `feasibility(dm, rows, threshold=...)` | audit whether every declared rule can fire |
| `certify` / `verify_certificate` | issue / check a signed proof (structure, signature, binding, replay) |
| `report` / `report_data` | render / structure the whole run |
| `cost_policy(costs)` | compile cost thresholds into a `Policy` |
| `measure_completeness(disps, rows, ledger)` | score a batch against an external hazard oracle |
| `propose(proposer)` | journal UNLABELED candidates from a producer (never a label) |
| `evaluate(candidate, oracle)` | the independent oracle measures the four-number census move |
| `promote(candidate, oracle, approval=...)` | evidence-gated admission into the knowledge ledger |
| `retire(id, refutation)` | union-only retirement: record the refutation, revert the claims |
| `certify_admission` / `verify_admission_certificate` | pin / check the move (structure, HMAC, binding, census replay) |
| `do(var, value)` / `CausalClaim` | a `do()`-estimand with a declared acceptance region and assumptions |
| `Strategy(kind, ...)` | an identification strategy: `backdoor`, `iv`, `did`, `rdd`, `ci`, `its` |
| `CausalSession` | the discovery loop over causal claims; strategies are capabilities |
| `event_study` / `series_from_journal` | interrupted time series over the live journal (pre-trend + placebo gates) |

Model facts: `pred`, `conf`, `p_positive` (binary classifiers only),
`certified`, `set_size`, `lo`, `hi`, `p_event`, `risk_release`, `risk_deny`.
Verdicts: `release | ask | deny | abstain`. Guarantee tags: **`declared`** /
**`model_backed`** — deliberately different from the conformal fact
`certified`. Assurance tags (derived): `declared`, `model_backed`,
`pointwise_coverage`, `action_risk_control`, `dual_track`.

## 9. Standalone, vendored, tested

`mjolnir_ml` imports nothing else from this repo at runtime. It depends on two
clones under `mjolnir_ml/vendors/`:

* `vendors/core/verdict.py` — **byte-identical** clone of the frozen rule engine
  (`src/mjolnir_router/core/verdict.py`), pinned by a test;
* `vendors/wire.py` — clone of `mjolnir_disposition/engine.py` with exactly one
  delta: the import line.

```bash
cd mjolnir_research/mjolnir_ml
uv pip install -e .            # or: pip install -e .
python -m pytest tests -q      # 210 tests
python examples/quickstart.py
# notebooks (already executed, outputs committed; pip install 'mjolnir-ml[examples]'):
#   examples/random_forest_regressor.ipynb
#   examples/random_forest_classifier.ipynb
#   examples/flywheel_live.ipynb
#   examples/discovery_example.ipynb
#   examples/causal_analysis.ipynb
python experiments/bench.py        # 3 datasets x 3 learners
python experiments/drift.py        # ACI under concept drift
python experiments/completeness.py # external hazard ledger
python experiments/costs.py        # cost-sensitive bands
python experiments/audit.py        # signed certificate audit
python experiments/model_ladder.py # learner → decision learning curves
python experiments/flywheel.py    # readiness, live batches, containment, repair
```

Layout:

```
mjolnir_ml/
  dsl.py           @dsl, rule(), AST harvest, whitelist, IR
  policy.py        Statement/Policy, canonical digest, engine compile,
                   dispose_facts (the estimator-free audit path)
  model.py         DecisionModel, Disposition, bind/regime/dispose/verify/observe
  calibration.py   Kappa + AdaptiveKappa (split conformal, Mondrian, ACI,
                   normalized/CQR, one-sided tails)
  risk.py          CRC action thresholds, RiskIndex, p_event/risk_release/risk_deny
  regime.py        multivariate regime floors (mahalanobis/isolation forest/custom)
  feasibility.py   reachability audit, dead rules, readiness gate
  _digest.py       shared canonical digest helpers
  pair.py          Omega pair, Wilson, audit, degenerate controls
  report.py        report() / report_data(): one text/table for the whole run
  readiness.py     readiness() / ReadinessReport: the one-shot gate
  live.py          Journal, LiveSession, detectors, containment, retraction,
                   ChangeSet/Evaluation/Promotion, ApprovalAuthority
  costs.py         Costs, Chow thresholds, cost_policy -> Policy
  hazards.py       HazardLedger (union-only) + measure_completeness()
  certificate.py   certify / verify_certificate (structure + HMAC + binding
                   + policy replay; row-integrity check)
  discovery.py     DiscoverySession: Census (four numbers), unlabeled Candidate,
                   independent-oracle gates (tests + twin), KnowledgeLedger
                   (union-only), containment, certify_admission
  causal.py        do()/CausalClaim/Strategy: backdoor, IV, DiD, RDD,
                   conditional independence, journal event study; placebos as
                   twins; markov_class; CausalSession
  vendors/         the two clones
tests/             210 tests: laws, DSL failures, calibration, ACI, costs,
                   completeness, certificates, reports, sklearn integration,
                   notebook validity, model-ladder protocol, readiness,
                   live session/containment/change control, discovery laws,
                   causal identification, clones
examples/          quickstart, five executed notebooks (flywheel, discovery,
                   causal analysis) + causal_analysis.md
experiments/       bench, drift, completeness, costs, audit, model_ladder
                   + results JSON/CSV and curve PNGs
```

## 10. The second loop: `DiscoverySession`

The flywheel decides rows; discovery admits **capabilities**. `DiscoverySession`
runs the same laws over a claim corpus — a prover registry export, a decline
census, any `claim -> verdict` mapping:

```python
session = DiscoverySession(corpus, reasons=declines_by_reason,
                           approval_verifier=authority.verify)   # no key held
candidate = session.propose(mine_conjunction_rules)[0]   # unlabeled
ev = session.evaluate(candidate, audit_rule)             # independent oracle
adm = session.promote(candidate, audit_rule)             # evidence-gated
cert = certify_admission(adm, key=KEY)                   # the move, pinned
```

| law | encoding |
|---|---|
| proposer never disposes | a `Candidate` is unlabeled (`evidence["expect"]` is refused); an oracle whose identity equals `proposed_by` is refused, journaled, and reports a **critical** `self_certification` alarm |
| four numbers, never a scalar | `Census` is `(certified, declared, declined, refuted)`; `as_dict()` has exactly those keys, and the frontier is the decline census by reason |
| twins are mandatory | `tests` (what the rung must newly certify) and `twin` (the near-miss battery) are checked by the oracle; either absent or failing ⇒ no admission |
| tightening free, loosening signed | certifying a previously `REFUTED` claim is `may_loosen` and needs an `Approval` verified by a verifier the session holds; a declared frontier class without reason data fails closed |
| union-only ledger | `KnowledgeLedger` never reuses ids; `retire(id, refutation)` records the refutation and reverts the admitted claims to their pre-admission verdicts |
| critical alarms contain | `tick()` recomputes `new_refuted`, `self_certification` and `decline_growth` from the journal; the criticals contain the session (admissions refused) until a signed `uncontain`, and the epoch keeps history from re-alarming |
| the move is certifiable | `certify_admission` pins the candidate, both censuses and every gate under an HMAC binding id, **replays** the census-arithmetic gates, and lists the oracle-dependent gates as `attested` |

The claim axis is deliberately a different vocabulary from the disposition axis
(`certified/declared/declined/refuted` vs `release/ask/deny/abstain`), so a
disposition can never be mistaken for a proof.

End to end on `breast_cancer` — classification rules mined from the training
split, audited on a held-out split, with the frontier move, the signed
loosening, containment, the certificate and the retirement:
`examples/discovery_example.ipynb` (executed, outputs committed).

## 11. The causal layer: `do()`, identification, placebos

`causal.py` puts causal estimands on the discovery loop. A **strategy is a
capability**; a claim is a `do()` estimand with a declared acceptance region; a
**placebo is the twin**; and the frontier is the decline census by identification
failure (`unidentified:*`, `positivity_violation`, `weak_instrument`,
`assumption_violated:*`, `underpowered`, `ambiguous_ci`).

```python
from mjolnir_ml import CausalClaim, CausalSession, Strategy, do

claim = CausalClaim(kind="ate", outcome="y", treatment="x",
                    arms=(do("x", 1), do("x", 0)),
                    criterion={"min": 0.8, "max": 1.2},
                    assumptions=("no unmeasured confounding for the "
                                 "declared adjustment set",))
session = CausalSession([claim], table=rows)
strategy = Strategy("backdoor", adjust=("z",))
adm = session.promote(session.candidate(strategy=strategy, targets=[claim]),
                      session.strategy(strategy))
```

| claim kind | estimand |
|---|---|
| `association` | `E[Y|X=x1] − E[Y|X=x0]` — no identification claim, refutable on its own |
| `ate` | `ATE[Y | do(X=x1), do(X=x0)]` |
| `late` | `LATE[Y | do(X); Z]` — complier effect under monotonicity |
| `att` | `ATT[Y | group, time ≥ post]` — DiD |
| `rdd` | `LATE[Y | running, cutoff]` — sharp design |
| `independence` | `X ⟂ Y | S` — bounded by a declared MDE |
| `intervention_effect` | effect of an event in a journal series (tier 1) |

**What gates the verdict, and what is merely declared.** Testable shadows gate:
positivity per cell, instrument relevance (first-stage t), DiD pre-trends,
RDD covariate continuity, CI power, the ITS pre-trend and placebo date. Untestable
parts ship as `assumptions` on the claim and are listed by `assumptions()`:
no unmeasured confounding, exclusion, monotonicity, parallel trends, continuity,
no simultaneous shock. A causal claim with no declared assumptions does not
exist — the type refuses it.

**`CERTIFIED` is narrow on purpose**: *under the declared assumptions, the
identified interval lies inside the claim's acceptance region*. It is not "the
effect is true". `REFUTED` requires an identified interval entirely outside the
region; a straddling interval is `DECLINED` (the ambiguous band). Certifying a
claim the record holds as `REFUTED` (a prior study's verdict) is `may_loosen` and
needs an external signature.

**Structure is an equivalence class.** `CausalSession.markov_class(strategy)`
reports the skeleton and colliders the independences identify, and states that
edge orientation is not identified from observational data. There is no DAG and
no edge-score.

Run end to end on the **same `breast_cancer` classifier as the discovery
example** — the causal chain is rendered in the notebook as plain-text
diagrams with `phart` (an examples-only dependency; the harness never imports
it):
`examples/causal_analysis.ipynb` (executed) with
`examples/causal_analysis.md` as its extended explanation — etiology declined,
over-adjustment contained, the paired mechanistic ATE with an exact placebo,
bounded independence and a Berkson collider, and the audit stream as an
interrupted time series. `tests/test_causal.py` also pins IV/Wald, DiD and RDD
on declared synthetic SCMs.
