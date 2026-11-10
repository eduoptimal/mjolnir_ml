# MJ-Eval v0.1 — measuring certifiable trustfulness

A benchmark for **trustworthiness-as-verifiability**, scored by the mjolnir
authority. Unlike pass@k or compile checks, MJ-Eval asks whether a model's
*claims are calibrated to what an exact verifier can certify* — and treats honest
abstention as a first-class, positively-scored outcome.

> **Fail-closed ML — the inverse of a hallucinating AI.** Abstention is a
> positively-scored outcome here, not a failure to be scored.

## What it measures — 2×2 calibration

Every task carries a **required contract** and an **oracle class** (decidable /
undecidable, machine-validated). The model must either produce a function that
encodes the contract, or output `ABSTAIN: <reason>`. The authority disposes it.

|                    | oracle **decidable** (Suite A) | oracle **undecidable** (Suite C) |
|--------------------|-------------------------------|----------------------------------|
| model **claims**   | `CertYield@1` ↑               | `OverclaimRate` ↓                |
| model **abstains** | `FalseAbstention` ↓           | `AbstentionRecall` ↑             |

A trustworthy model proves when it can and abstains when it cannot. Reporting only
yield is gameable (vacuous claims); reporting only abstention ignores capacity.
The vector is the measurement.

## Metrics

Primary (calibration):
- **OverclaimRate** — tasks where the model asserts a claim the authority refutes
  (hard) or asserts an unprovable claim instead of declining (soft).
- **AbstentionRecall** — on undecidable tasks, fraction correctly `ABSTAIN`ed.
- **FalseAbstention** — on decidable tasks, fraction wrongly `ABSTAIN`ed.
- **AbstentionPrecision / ProofPrecision** — declines for good reasons; claim discipline.

Secondary (capacity): **CertYield@1**, **CertYield@K**, **SpuriousClaimRate**, **SNR**.

Invariants: **FAULT count = 0**, **system false approvals = 0** (guaranteed by the
authority regardless of the model).

### What this benchmark does NOT measure — read before citing

**Suite A is authority-shaped, not open-world.** Every A rung exists because the
current authority has a route for it, and A's vocabulary mirrors that: 92 tasks
mention a product, 13 a floor division, and **zero** mention `min`, `max`, `abs`
or `%`. So a high CertYield@1 means *the model can state claims this verifier
handles* — not that the model reasons well. Anyone using MJ-Eval to rank models
against each other is on firm ground; anyone reading it as a general code-
reasoning measure is not, and the 2×2 matrix is precisely what keeps it honest
by showing the decline side beside the capability side.

The 0-dropped machine check is what stops the suite from silently flattering
itself: a task is kept only if the authority honours its label, so if a future
change made an A task undecidable or a C task decidable, that task is **dropped**,
not mislabelled. `python -m rlvf gate-suite` re-checks this under the authority
in force.

## Construction — labels are machine-checked

Tasks are emitted by `generator.py` from parameterized families, then **validated
through the authority** by `build.py`: a Suite A reference must CERTIFY; a Suite C
naive attempt must NOT certify. Only candidates whose oracle label holds are kept.
Across the full set: **184 tasks, 0 dropped, 0 oracle mismatches**.

Composition/contract-propagation (Suite B) is intentionally absent in v0.1 — it
requires exposing `module_source` in `certify_unit` (see `rlvf.md`).

**The authority pin matters when citing results.** `manifest.json` carries the
authority fingerprint; the 2026-10-01 soundness fix moved it from
`99aa74145d05` to `808cade9caab`. All 42 recorded runs were re-disposed under the
new pin and were **byte-identical** (`rlvf/data/evals_rebaseline/`), so every
result in `hypothesis_validation.md` stands.

## Suites and rungs

| suite | n | rungs |
|---|---|---|
| A (decidable) | 120 | straight-line, loop summaries (Faulhaber), premises, bounds/linear-box, linear systems, combos |
| C (undecidable) | 64 | mutation, subscript, method, membership, identity, type, boolop, slice, call, exception, unbounded loop, operator, attribute, premise-gap, near-miss, **non-polynomial** |

The **non-polynomial** rung (19 tasks) was added 2026-10-01. `nf` is a
POLYNOMIAL normal form, so `min`/`max`/`abs`/`%` are outside it and no contract
over them is decidable — measured at 8 of 14 candidate contracts refused. The
decline side was thinnest exactly where the authority's limits are sharpest (one
task against a 57% refusal rate), which is the shape that flatters a model: a
suite shaped by what the grader can handle.

## Schema (`tasks/suite{A,C}.jsonl`)

```json
{"id": "...", "suite": "A|C", "oracle": "decidable|undecidable",
 "rung": "...", "fn": "...", "signature": "...",
 "task": "natural-language task",
 "contract": "the required assert predicate",
 "source": "generator-v1"}
```

`oracle.jsonl` holds, per task, the reference code and the observed authority
status. `manifest.json` pins counts, split sizes, per-task content hashes, and the
**authority fingerprint** — the verdicts define the labels, so a run must pin the
authority version.

## Scoring

```
# build/rebuild the benchmark (offline; validates labels through the authority)
python -m rlvf bench-build

# evaluate any proposer on the frozen held-out split
python -m rlvf rollout --model <hf_model_or_local> --backend mlx \
  [--adapter <mlx_adapter>] --suite all --split eval --k 8 --run-id eval-<name>
python -m rlvf bench --run-id eval-<name>
```

Frozen splits: `splits/train.jsonl` (91) and `splits/eval.jsonl` (59), by content
hash. Training data must come from `train` only.

## Reference numbers (Qwen2.5-Coder-1.5B-Instruct, eval split, K=4)

Authority fingerprint `233a2a7b…`; 0 FAULTs. Runs `pre-1.5b-eval` / `post-1.5b-eval`.

| model | A CertYield@1 | A Overclaim | A FalseAbst | C Overclaim | C AbstentionRecall |
|---|---|---|---|---|---|
| base | 38.6% | 30.7% | 7.4% | 58.3% | 8.3% |
| + SFT (self-imitation) | 79.5% | 14.2% | 0.0% | 86.7% | 3.3% |
| + DPO (self-pairs) | 80.1% | 14.8% | 0.0% | 80.0% | 3.3% |
| + SFT (oracle-derived) | **95.5%** | **4.5%** | 0.0% | **30.0%** | **70.0%** |

**Reads as a trade, not a win:** one SFT round raises capacity (A yield 39% → 80%,
false-abstention → 0) but *lowers* calibration (C overclaim 58% → 87%,
abstention-recall 8% → 3%) — imitating the certified-positive distribution teaches
"always claim". Ω-certifiability needs the negative signal (abstain-vs-overclaim),
not just SFT.

## Files

```
benchmark/
  README.md            this spec
  generator.py         deterministic task families
  build.py             authority validation + write + manifest
  tasks/suite{A,C}.jsonl
  oracle.jsonl         reference code + observed status per task
  splits/{train,eval}.jsonl
  manifest.json        counts, hashes, authority fingerprint
```

## Provenance and license

Tasks are synthetic and generated from templates (no third-party code); released
for benchmarking. The authority is mjolnir (`mjolnir/lib`), used read-only. Because
labels depend on the authority's fragment, reproduce numbers with the pinned
`authority_fingerprint`.
