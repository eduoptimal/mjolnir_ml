# Causal analysis — explained from first principles

Companion to [`causal_analysis.ipynb`](causal_analysis.ipynb). The notebook runs
end to end on the **same dataset and classifier** as `discovery_example.ipynb`:
`breast_cancer`, a depth-3 decision tree on `worst concave points` and
`worst radius`, 312 training rows and 257 audit rows.

You do not need a causal-inference background to read it. This document defines
every term the notebook uses, with the concrete `breast_cancer` example, and
then walks through each result. The short version: the notebook asks two kinds
of question — *does changing something change the outcome?* and *did an
intervention change the process?* — and it is equally careful about the answers
it **cannot** give.

---

## 1. The vocabulary, in plain language

### Claims and estimands

| term | plain meaning | in this notebook |
|---|---|---|
| **claim** | a precise statement with a range in which the answer must fall to count as confirmed | `ATE[malignant \| do(concave_high=1), do(concave_high=0)] ∈ [0.6, 0.8]` |
| **estimand** | the quantity a claim is about — the thing being estimated | "the average change in malignancy probability if concavity were set high" |
| **estimate** | the number the data produced | `0.3496` |
| **interval (CI)** | the range of values consistent with the data and the assumptions; printed as `[lo, hi]` | `[0.2609, 0.4382]` |
| **acceptance region** | the interval the *claim itself* says the effect must lie in | `[0.6, 0.8]` |
| **association** | a pattern visible in the data, with no claim about cause | malignant rate is ≈ 0.70 higher when concavity is high |

An `Estimate(...)` line in the notebook reads as:

```
Estimate(-0.03891 ∈ [-0.07275, -0.005072], n=257, paired-difference(paired(pair)))
          ↑ estimate    ↑ 95% interval               ↑ rows  ↑ how it was computed
```

### The `do()` operator, and why it is the whole game

| term | plain meaning | in this notebook |
|---|---|---|
| **`do(x = v)`** | *set* x to v, as an action — as opposed to merely *observing* x = v | `do(radius = 17.2)` sets the feature for every audit row and re-scores it |
| **intervention** | an action that changes a variable | tightening a policy threshold, or setting a model input |
| **counterfactual / potential outcome** | what *would* have happened under the other action; only one is observed per unit | for one audit row: the model's call with radius observed, and with radius set |
| **ATE** (average treatment effect) | the average change in the outcome if everyone were moved from one action to the other | average change in malignant calls when radius is set to 17.2 |
| **etiology** | the study of **causes of disease** — does X cause Y, in the world? | "does concavity cause malignancy?" — asked, then declined |
| **mechanistic effect** | an effect inside a known mechanism (here, the classifier), not in the patient | setting a model input is well posed; setting a tumor feature is not |

The crucial distinction in the notebook: `do(concave)` **on a patient** is not a
sensible action — concavity is a *symptom* of the tumor, not a lever. `do(concave)`
**on the classifier** is sensible, because the classifier is a deterministic
mechanism whose inputs we control. The notebook analyses the second and refuses
the first.

### Identification: can the data answer the question at all?

| term | plain meaning | in this notebook |
|---|---|---|
| **identification** | whether the data plus declared assumptions are enough to pin the causal quantity down at all | the etiological ATE is *not identified* here |
| **unidentified / declined** | no method can answer from this data; this is **not** the same as "the answer is no" | the etiology claim is `DECLINED`, not `REFUTED` |
| **confounder** | a common cause of both the action and the outcome; it creates a misleading association | a variable that makes high-concavity cases also high-risk |
| **backdoor adjustment** | compare action vs no-action *within* groups that share the confounders, then average — this blocks the misleading path | `backdoor({radius})` in the notebook (which turns out to be the wrong thing to adjust for) |
| **descendant / mediator** | a variable *caused by* the action (or by the disease), sitting on the path | `worst radius` is another manifestation of the same tumor |
| **over-adjustment** | adjusting for a descendant/mediator, which distorts the effect instead of cleaning it | adjusting the etiology claim for `radius` drops the estimate from ≈ 0.70 to 0.35 and **refutes** the claim |
| **positivity / overlap** | every group must contain both action and no-action units; otherwise there is nothing to compare | checked per cell; too few rows ⇒ `positivity_violation` |
| **assumption** | a statement that cannot be tested from this data but the result depends on it; declared and shipped with the claim | "concavity is not a descendant of malignancy" |
| **sensitivity / placebo** | a check that *should* show nothing; if it shows something, the design is broken | an intervention on `mean symmetry`, which the model ignores |

### Negative controls and the twin

| term | plain meaning | in this notebook |
|---|---|---|
| **placebo / negative control** | an intervention or outcome where the true effect is known to be zero | `do(mean symmetry)` — not an input of the model |
| **twin** | the adversarial companion claim that must keep its expected verdict; the gate refuses the capability if it does not | the placebo claim, or the conditioned independence that must stay `REFUTED` |
| **full compliance** | the action was actually applied to every unit | every row is scored under both feature settings, by construction |

The placebo here is stronger than "statistically zero": the contrast is **exactly
0 for every pair**, so its interval is `[0, 0]`. An action on a variable with no
arrow into the outcome cannot move it.

### Structure: independence, colliders, DAGs

| term | plain meaning | in this notebook |
|---|---|---|
| **independence** | two variables carry no information about each other | `texture` and `smoothness` (r ≈ −0.02) |
| **conditional independence** | independence *within* each level of a third variable | `texture ⟂ symmetry \| radius` |
| **conditioning** | looking only inside a subgroup (fixing a variable) | "among diagnosed and undiagnosed cases separately" |
| **collider** | a variable caused by two others; conditioning on it makes those two look related even when they are not (Berkson's paradox) | the **diagnosis** is a collider between `texture` and `smoothness` |
| **DAG** | a directed acyclic graph: arrows, no cycles — a picture of assumed causes | the small classifier diagram in the notebook |
| **skeleton** | which pairs are directly linked, ignoring arrow direction | recovered from certified independences |
| **Markov equivalence class** | the set of DAGs that imply exactly the same independences; observational data identify only this class | why the notebook refuses to draw a DAG |
| **edge orientation** | the direction of an arrow | explicitly listed as unidentified |

### Time series (tier 1)

| term | plain meaning | in this notebook |
|---|---|---|
| **interrupted time series (ITS)** | measure a process before and after a known event, and ask whether the level changed | policy tightened at batch 20 |
| **segmented regression** | the standard ITS estimator: fit a line before the event, a jump at the event, and a change of slope after | `y = b0 + b1·t + b2·D + b3·t·D` |
| **pre-trend** | the slope *before* the event; if the process was already moving, the post-change cannot be attributed to the event | gated: the interval must contain zero |
| **placebo date** | pretend the event happened earlier, where nothing happened; the effect should be zero | two placebo-date claims |
| **batch** | a group of decisions processed together (the unit of the time series) | 32 batches of ~17 audit rows |
| **hazard / hazard-release rate** | an outcome that must not be released, and the share of releases that carry it; defined **outside** the model | released a case with `concave ≥ p60` |
| **journal** | the append-only, hash-chained record the harness writes | `journal.verify() == True` |

### Reading the verdicts

| verdict | means | does **not** mean |
|---|---|---|
| **CERTIFIED** | under the declared assumptions, the interval lies inside the acceptance region | "true" |
| **DECLINED** | the data/strategy cannot decide: unidentified, too little overlap, too little power, or the interval straddles the region | "false" |
| **REFUTED** | an *identified* interval lies entirely outside the region | "the design is wrong" (though that is the usual cause — see containment) |
| **DECLARED** | a premise, not a measurement | "verified" |

The **census** is always four counts — `certified / declared / declined /
refuted` — never a single score. The **frontier** is the declined claims grouped
by the reason they were declined (e.g. `unidentified:no-strategy`), because that
reason is what a new capability must address.

A **capability** is one identification strategy. A **gate** is a check that must
pass before a capability is admitted: the positive battery, the twin, "no new
refutations", "the census moved", and "the declared frontier class shrank".

---

## 2. Setup: what is being analysed

The classifier is the causal system under study. It has two inputs — concavity
and radius — and `symmetry` is deliberately present in the data but never read
by the model, which makes it a perfect placebo lever.

The notebook draws four compact plain-text diagrams with
[`phart`](https://github.com/scottvr/phart) — boxed Unicode output, no browser:

**(1) the intervention chain — what `do()` acts on**

```text
┌────────────────────┐    ┌───────────────────┐
│  do(concave=.155)  │    │  do(radius=17.2)  │
└────────────────────┘    └───────────────────┘
           └───┐                ┌───┘
               │                │
               v                v
         ┌───────────┐    ┌──────────┐
         │  concave  │    │  radius  │
         └───────────┘    └──────────┘
               │       ┌────────┘
               └───────┤
                       v
               ┌──────────────┐
               │  prediction  │
               └──────────────┘
```

**(2) the decision chain — prediction → release → hazard rate**

```text
        ┌──────────────┐
        │  prediction  │
        └──────────────┘
                │
                │
                v
    ┌───────────────────────┐
    │  release? (policy τ)  │
    └───────────────────────┘
                │
                │
                v
┌───────────────────────────────┐
│  hazard rate (external rule)  │
└───────────────────────────────┘
```

**(3) the structure the etiology claim would need — and cannot have**

```text
      ┌──────────────────┐
      │  latent disease  │
      └──────────────────┘
                ├─────────┐
        ┌───────┘         │
        v                 v
┌──────────────┐    ┌───────────┐
│  smoothness  │    │  texture  │
└──────────────┘    └───────────┘
        │       ┌─────────┘
        └───────┤
                v
         ┌─────────────┐
         │  diagnosis  │
         └─────────────┘
```

**(4) the placebo lever — a box with no edges**

```text
┌─────────────────┐
│  mean symmetry  │
└─────────────────┘
```

Reading them:

* in (1), `do(concave)` and `do(radius)` set the model's inputs, and the two
  paths meet at the prediction;
* in (2), the policy threshold gates the release and the hazard rule is
  **external** to the model — it defines the outcome, not the mechanism;
* in (3), both texture and smoothness are effects of the latent disease, and the
  diagnosis is their **collider**: conditioning on it creates the Berkson
  dependence the notebook tests. There is deliberately **no arrow from an
  intervention into the diagnosis** — that arrow is exactly the etiological
  claim, and the notebook declines it as unidentified rather than drawing it;
* in (4), `mean symmetry` has no edge into the prediction, which is why an
  intervention on it must have exactly zero effect.

```
rows 569 | train 312 | audit 257
audit malignant-call rate: 0.354
```

"Malignant call" means the model predicts class 0 (malignant); 35.4% of audit
rows get that call.

## 3. The record: association certifies, etiology does not

The registry is audited once with `Strategy("none")` — no identification
strategy at all.

```
certified E[malignant|concave_high=1] − E[malignant|concave_high=0] ∈ [0.6, 0.8]
          reason: certified
declined  ATE[malignant|do(concave_high=1), do(concave_high=0)] ∈ [0.6, 0.8]
          reason: unidentified:no-strategy

census  : {'certified': 1, 'declared': 0, 'declined': 1, 'refuted': 0}
frontier: {'unidentified:no-strategy': 1}
```

**Association** is the observed difference in malignancy rates between
high- and low-concavity audit rows: ≈ 0.70. The interval fits inside `[0.6, 0.8]`,
so it certifies. Nothing causal has been said.

**Etiology** asks the causal version: if concavity were *set*, would malignancy
change? This cannot be answered from a cross-sectional table of symptoms. The
claim is **declined** — the frontier records why (`unidentified:no-strategy`).

This distinction is the point of the whole exercise. "We cannot tell" is a
different statement from "it is false", and the harness keeps them apart.

## 4. Over-adjustment manufactures a refutation, and containment catches it

A plausible-looking capability adjusts for `worst radius`.

```
over-adjusted: refuted refuted Estimate(0.3496 ∈ [0.2609, 0.4382], n=257,
                                    regression-adjustment(backdoor({radius})))

admission: False - gate failed: labelled_tests, twin_holds, no_new_refuted,
                    certified_gained, targeted_decline_moved
  new_refuted        alarm=True  critical 1 evaluation(s) with after.refuted > before.refuted
  auto_act -> ['1 evaluation(s) with after.refuted > before.refuted']

unsigned lift refused: uncontain requires a valid external approval for <containment digest>
lifted with a signature; tick: [('new_refuted', False), ('self_certification', False), ('decline_growth', False)]
```

Why does adjusting make things worse? `radius` is not a confounder here — it is
another **manifestation** of the same disease, i.e. a descendant of the thing we
are trying to explain. Comparing high- vs low-concavity cases *at the same
radius* compares cases that are no longer comparable, and the estimate collapses
to 0.35. That is **over-adjustment**.

The loop's response is not to argue: a capability that produces a new
**REFUTED** is treated as a defect signal. Admission fails, `tick()` raises a
critical alarm, `auto_act()` **contains** the session (admissions are refused),
and lifting containment requires an external signature. The failure that an
analyst would otherwise publish cannot leave the loop without a human.

## 5. Mechanistic ATE: a paired potential-outcomes design

For a model intervention the notebook builds a **potential-outcomes table**:
every audit row appears twice, once observed and once with the feature set. The
**paired** strategy takes the mean within-pair difference — the correct analysis
when the same unit is measured under both actions.

```
do(radius=17.2): certified Estimate(-0.03891 ∈ [-0.07275, -0.005072], n=257,
                                   paired-difference(paired(pair)))

gates: labelled_tests=PASS, twin_holds=PASS, no_new_refuted=PASS,
       certified_gained=PASS, targeted_decline_moved=PASS
admission: True - admitted

placebo do(symmetry):   Estimate(0 ∈ [0, 0], n=257, paired-difference(paired(pair)))
test do(concave=0.155): Estimate(-0.02335 ∈ [-0.05378, 0.007084], n=257, paired-difference(paired(pair)))
```

Setting `radius` to 17.2 moves **3.9%** of audit cases away from a malignant
call, and the whole interval is below zero, so the directional claim
(`effect ≤ 0`) certifies. This is a statement about the **model**, not about
patients: it says the classifier's decision boundary responds to that input by
that much.

Two supporting claims:

* the **test** (`do(concave = 0.155)`) is a second intervention under the same
  capability, predicted to certify as well;
* the **placebo** (`do(symmetry)`) is an action on a variable with no arrow into
  the outcome. Every pair differs by exactly 0, so the interval is `[0, 0]`.

## 6. Structure: bounded independence and a Berkson collider

```
tex/smooth             independent  r=-0.023 p=0.57785 detectable_r=0.1172
tex/sym                independent  r=+0.071 p=0.08883 detectable_r=0.1172
tex/fractal            independent  r=-0.076 p=0.06845 detectable_r=0.1172
tex/smooth|malignant   dependent    r=-0.203 p=0.0     detectable_r=0.1173
tex/sym|radius         independent  r=+0.006 p=0.8784  detectable_r=0.1173

separated   : ['texture ⟂ smooth ∈ [-1, 1]', 'texture ⟂ symmetry ∈ [-1, 1]',
               'texture ⟂ fractal ∈ [-1, 1]', 'texture ⟂ symmetry | radius ∈ [-1, 1]']
dependent   : ['texture ⟂ smooth | malignant ∈ [-1, 1]']
colliders   : [{'at': 'malignant', 'left': 'texture', 'right': 'smooth'}]
unidentified: ['orientation of non-collider edges',
               'edges into colliders from outside the triple',
               'anything not in the measured variables']
```

**Bounded independence.** `detectable_r = 0.117` is the smallest correlation
this sample can reliably detect at the declared **MDE** (minimum detectable
effect) of 0.12. A certified independence is therefore "no dependence of size
≥ 0.12", not "independent, full stop". A test that could only detect larger
effects is `DECLINED` as underpowered rather than certified.

**The collider.** `texture` and `smoothness` are unrelated in the population,
but *within* the diagnosed and undiagnosed groups they are related (r ≈ −0.20).
Both are effects of the same latent disease, and the diagnosis — a collider —
makes them look connected once you condition on it (**Berkson's paradox**). The
twin is the conditioned claim, which must be `REFUTED`; that is what admits the
capability.

**What is not claimed.** The certified independences pin the **skeleton** (which
pairs are linked) and the collider, but they do not pin **edge orientation**.
Several DAGs imply the same independences — the **Markov equivalence class** —
so the report lists what remains unidentified instead of drawing a graph. There
is no DAG and no confidence score.

## 7. Tier 1: the audit stream as an interrupted time series

The offline audit is replayed as 32 **batches**. At batch 20 the policy tightens
(`τ: 0.50 → 0.99`) — the event. The outcome is the **hazard-release rate**: the
share of releases that carry an external **hazard** (`concave ≥ p60`), a rule
defined outside the model.

```
containment  certified    est=-0.114 CI[-0.205, -0.022] pre_slope_ci=[-0.0037, 0.0105] placebo=-0.0279
placebo_12   certified    est=-0.028 CI[-0.148, +0.092] pre_slope_ci=[-0.0084, 0.0126] placebo=0.2118
placebo_8    certified    est=+0.008 CI[-0.183, +0.199] pre_slope_ci=[-0.0249, 0.0193] placebo=None

admission: True - admitted
```

The estimator is a **segmented regression**:

```
y = b0 + b1·t + b2·D + b3·t·D,     t = batch − event,  D = 1[t ≥ event]
```

`b1` is the **pre-trend** (gated: its interval must contain zero), and the
post-event effect is `b2 + b3·mean(t_post)` with an analytic interval. The
containment estimate is −0.114 with the interval entirely below zero, and the
pre-trend is flat.

The **placebo dates** are claims, not hidden checks: pretend the event happened
at batch 12 or 8, where nothing happened, and the effect must be null. Both
certify. Note the widths: ±0.19 for the earlier date versus ±0.12 for the later
one — fewer pre-event batches buy a wider null region. The power cost of a short
window is visible in the interval instead of being hidden.

## 8. The four numbers, and the declared tier

```
etiology audit : {'certified': 1, 'declared': 0, 'declined': 1, 'refuted': 0} | contained then lifted
mechanistic    : {'certified': 1, 'declared': 0, 'declined': 0, 'refuted': 0}
structure      : {'certified': 3, 'declared': 0, 'declined': 2, 'refuted': 0}
event study    : {'certified': 1, 'declared': 0, 'declined': 2, 'refuted': 0}

assumptions certified (mechanistic):
  - do() assigns the feature for every unit (full compliance)
  - the classifier is a deterministic mechanism
assumptions certified (structure):
  - faithfulness
  - linear-Gaussian
```

Each audit is its own session (one table or one series), so the censuses are per
audit. The **assumptions in force** are listed beside the results because
`CERTIFIED` is only ever relative to them. `linear-Gaussian` and `faithfulness`
are the technical premises behind the partial-correlation independence test;
`full compliance` and `deterministic mechanism` are the premises behind the
model intervention.

## 9. Non-claims: what the notebook refuses

`CERTIFIED` means *under the declared assumptions, the identified interval lies
inside the acceptance region*. It does not mean "true". Refused on purpose:

* **A DAG.** Observational independences identify an equivalence class; the
  collider is named, the edge orientations are not.
* **Etiology from this table.** A `do()` on a manifestation is not an
  intervention. The claim stays in the frontier instead of being answered with
  a made-up strategy.
* **Over-adjustment.** Adjusting for a descendant manufactures a refutation;
  containment stops it and a human must sign the lift.
* **Positivity and support.** No smooth extrapolation into regions with no
  overlap; cells below `min_cell` are `DECLINED`.
* **Power.** Independence is bounded by a declared MDE; underpowered tests are
  declined.
* **Hidden placebo checks.** The placebo is a claim, so it appears in the
  journal, the census and the certificate like everything else.
* **Scores.** No BIC, no edge confidence, no "causal strength" — four words and
  an interval.

What remains is the honest version of the original question. The harness
certifies *"this rule is precise on held-out rows"*; the causal layer certifies
*"under assumptions A, changing X would change Y by δ inside the support where
both occur, and the placebo says we are not fooling ourselves about why."*

---

## Reproduce

```bash
cd mjolnir_ml
pip install "mjolnir-ml[examples]"    # sklearn + phart + networkx, for the notebook
python -m jupyter nbconvert --to notebook --execute --inplace examples/causal_analysis.ipynb
```

The notebook is self-contained (sklearn + stdlib). The development suite also
pins the estimators this dataset cannot exercise — IV/Wald (unmeasured
confounding), two-by-two DiD (parallel trends), and sharp RDD (continuity) — on
declared synthetic SCMs with known answers.
