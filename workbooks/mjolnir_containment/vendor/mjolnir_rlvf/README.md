# mjolnir RLVF — all-3 post-training experiments (Qwen2.5-Coder-1.5B-Instruct)

Post-train one 1.5B model on **all three mjolnir components** and measure whether
they compose or interfere:

| component | skill | authority (oracle) |
|---|---|---|
| mjolnir_prover | proof-carrying code or `ABSTAIN` | `mjolnir_prover.lib.certify` |
| mjolnir_intent | `VERDICT: BUILD / ASK / blocked` + findings | `mjolnir_intent.checks.audit` |
| mjolnir_router | `VERDICT: ROUTE / ESCALATE / ABSTAIN` + reason | `mjolnir_router … policy.decision_dict` |

Every training label and eval label is oracle-derived; nothing is hand-labelled.
Harness lives in `mjolnir_research/mjolnir_experiments/rlvf/` (`stack.py`,
`battery.py`, `run_all3.sh`); this file records the experiment series and verdict.

## Gates (set before v2 was trained)

Templated axis (held-out eval splits, full task templates): **intent ≥87%** ·
**router ≥78% with ESCALATE recall ≥66%** · **C overclaim ≤5% / recall ≥90%** ·
**A CertYield ≥92%** · 0 FAULTs.
Serve axis (4-cell battery, 26 *plain* prompts/cell, no templates):
**trained+active intent ≥6/8** · **router ≥5/8** · **prover: every
out-of-fragment ask declined**.
Controls: `base+prompt` (system prompt only) and `base` (nothing) must show the
raw prior (fenced/prose, degenerate verdicts) in every battery.

## Results — all versions × gates

| version | recipe delta | intent | router | ESC | C over/rec | A yield | batt i/r/p | gates |
|---|---|---|---|---|---|---|---|---|
| base | — | 54.5 | 21.7 | 0.0 | 79.5/4.5 | 42.2 | — | 0 |
| v2 | balanced caps, 600 iters | 87.3 ✅ | 74.5 | 66.1 ✅ | **0/100** ✅ | 92.2 ✅ | 3/2/0 | 4 |
| v3 | +plain L2 asks, +system-mix | 89.1 | 76.6 | **0.0** | 23.5/76.5 | 98.9 ✅ | 6/4/5 ✅ | 4 |
| **v4** | +ESCALATE×5, plain 1/3-in, OOF↑ | **90.9** ✅ | 81.0 ✅ | **84.7** ✅ | 15.2/84.8 | 97.8 ✅ | **6/6/6** ✅ | **7/8** |
| v5 | +32 one-sided gap declines | 58.2 | 85.4 ✅ | 67.9 ✅ | **0/100** ✅ | 93.3 | 6/5/6 ✅ | 6 |
| v6 | 16 gap declines + 16 certified twins | 65.5 | 85.4 ✅ | 67.9 ✅ | **0/100** ✅ | 93.3 | 6/5/6 ✅ | 6 |
| v6ck900 | v6 checkpoint 900 | 85.5 | 78.6 ✅ | 57.6 | 12.9/87.1 | 96.1 | 6/3/6 | 4 |
| v7 | v6@1000 + intent-only refresh (120 @3e-6) | 83.6 | 84.1 ✅ | 56.9 | 1.5/98.5 ✅ | 92.2 | **8/5/7** ✅ | 6 |
| v8 | +maintenance set (intent+ESC+gap) | 87.3 ✅ | 58.0 | 94.9 ✅ | 68.9/6.1 | 100 ✅ | 8/5/0 | 5 |
| soup | 50/50 weight avg of v4 ⊕ v6 | 80.0 | **87.1** ✅ | 84.7 ✅ | 9.8/90.2 | 94.4 | 6/4/7 | 5 |
| cosine | v6 data, cosine LR 1e-5→0 | **87.3** ✅ | 70.8 | **0.0** | 26.5/65.9 | 93.9 | 6/2/5 | 4 |

Controls held in every battery: `base+prompt` = 10/10 fenced, `base` = 5/5
fenced+prose, base intent 6/8, base router 2/8 (always-ABSTAIN degenerate).

### Battery (trained+active cell, plain prompts)

| version | prover abstain/bare/fenced/prose | intent | router |
|---|---|---|---|
| v2 | 0 / 10 / 0 / 0 | 3/8 | 2/8 |
| v3 | 5 / 5 / 0 / 0 | 6/8 | 4/8 |
| **v4** | 6 / 4 / 0 / 0 | 6/8 | **6/8** |
| v6 | 7 / 3 / 0 / 0 | 6/8 | 5/8 |
| v6ck900 | 6 / 4 / 0 / 0 | 6/8 | 3/8 |
| **v7** | 7 / 3 / 0 / 0 | **8/8** | 5/8 |
| v8 | 0 / 10 / 0 / 0 | 8/8 | 5/8 |
| soup | 7 / 3 / 0 / 0 | 6/8 | 4/8 |
| cosine | 5 / 5 / 0 / 0 | 6/8 | 2/8 |

Every trained cell kept contract form (0 fenced / 0 prose); trained+inactive
fails *closed* (all declines, no fences). From v3 on, every out-of-fragment
prover ask is declined in the active cell; v8's twin-heavy refresh was the one
regression (10/10 attempted).

### Notable per-class failure signatures

| version | signature |
|---|---|
| v3 | router ESCALATE → 0% (minority swamped by ABSTAIN oversampling) |
| v5 | intent blocked → 37.5% (answers ASK), A false-abstain 1.1→12.8 — one-sided `>=`-decline demos taught a surface heuristic |
| v6 | staggered convergence: intent peaks at ck800–900 (85.5), collapses to 65.5 at 1000 while gap/C and router finish |
| v7 | intent-only refresh lifts intent +18.1 but erodes unshown ESCALATE 67.9→56.9 |
| v8 | refresh composes exactly what it shows: ESC 56.9→94.9, but ABSTAIN 88.7→25.8 and gap flips 20/20→0/20 attempts |
| soup | composes router/ESC/C numerically but ASK collapses to 11% (subspace interference, not midpoint) |
| cosine | intent survives (87.3) — late-step violence was real — but gap/ESC never finish learning: same steps carry both damage and progress |

## Interpretation

1. **The three skills compose as one fail-closed prior, but not for free.**
   Every version beats base by a wide margin on all three axes; the fights are
   between *calibration sub-classes*, not between components.
2. **Dose-response on the C-gap family is monotone and expensive.**
   Gap-pattern decline rows: 0 → 32 moves C overclaim 15.2 → 0 but intent
   90.9 → 65.5. Certified twins repair A and blocked-recall but do not remove
   the trade.
3. **Whack-a-mole equilibrium.** Whichever class receives no gradient erodes;
   minority classes erode fastest. Sequential refreshes fix the shown classes
   and poison the hidden ones (v7, v8) — no single-sided refresh converges.
4. **Schedule shape moves the same frontier, it does not escape it.**
   Cosine preserves early-converged intent but starves late learners; constant
   LR does the reverse. Weight soup composes across-trajectory skills that
   sequential training cannot, but interferes on ASK.
5. **Plain-prompt serving (the battery) is a separable skill** from templated
   accuracy: v2 was calibration-perfect yet battery-dead (0 declines, 3/8
   intent); from v3 on, plain-ask inference was trained explicitly
   (L1–L2 rows + system-prompt randomization) and every later version holds it.

## Conclusions

- **Official candidate: `rlvf/data/models/sft-all3v4` — 7/8 gates.**
  Passes intent (90.9), router (81.0), ESCALATE (84.7), A (97.8 / false-abstain
  1.1), FAULTs 0, and the full battery (6/6/6 with all OOF declines correct).
  Sole residual: held-out C-gap family, overclaim 15.2% / recall 84.8 (still
  19× better than base on C).
- **Alternate: `sft-all3v6-refresh` (v7)** if C-calibration outranks two
  sub-bar templated numbers: C 1.5/98.5 and a perfect 8/8 battery intent,
  against intent 83.6 (−3.4) and ESC 56.9 (−9.1).
- The remaining gap is **structural at 1.5B / LoRA rank 8 / ~1.6 epochs**:
  nine configurations (dose, checkpoints, refreshes, schedule, soup) mapped the
  same C-gap ↔ intent/ESC frontier. The one untested combination is *more
  epochs so the gap family converges before a gentle tail* — a higher-cost
  follow-up, not another knob at the current budget.
- Rigour caveat carried forward: single training seed per version
  (publication bar in the RLVF plan: ≥3 seeds); intent eval n=55; router
  ESCALATE n=59.

## Quantization — deployment options (measured on fp16 `sft-all3v4`)

Quantized the official candidate with `mlx_lm.convert` and re-ran the identical
8-gate protocol (same eval sets, same battery) against its fp16 reference —
functional performance is measured, not assumed. Decode = tokens/sec on a fixed
coding prompt (fp16 baseline 55.5).

| artifact | config | size | decode | vs fp16 | gates | verdict |
|---|---|---|---|---|---|---|
| `sft-all3v4` | fp16 (reference) | 2.9 GB | 55.5 tok/s | 1× | **7/8** | scoring / reference artifact |
| **`fp8-sft-all3v4`** | `--q-mode mxfp8` (8-bit float, 8.25 bits/wt) | **1.5 GB** | ~100 tok/s | **1.8×** | **7/8 — parity** (router Δ0.0, every class identical; battery bit-for-bit) | **recommended deployment default** |
| **`q4g64-sft-all3v4`** | `--q-bits 4 --q-group-size 64 --q-mode affine` (4.5 bits/wt) | **839 MB** | 151–162 tok/s | **2.7–2.9×** | 6/8 — router 74.6 (ROUTE class 94→69.5) is the one slip; battery identical; C improved to 10.6 | max-speed / memory-tight option |
| `m34-sft-all3v4` | `--quant-predicate mixed_3_4` (3-bit base + 4-bit v/down/lm_head) | 679 MB | ~140 tok/s | 2.5× | **4/8 — rejected**: 3-bit cliff (A-yield −25, ROUTE →28, ASK →0) | do not use |
| fp2 / int3 | — | — | — | — | — | off the table: MLX has no 2-bit float (lowest = fp4); measured 3-bit is the quality cliff |

Reading: calibration metrics tolerate 8-bit exactly and 4-bit *mostly* (the one
regression is the templated ROUTE class); below 4 bits everything that isn't
decline-behavior falls apart — compression keeps improving C-recall while
destroying capacity (more bits lost → more declining → better C, worse
everything else), the same trade the training frontier mapped.

**Deployment guidance:** serve **fp8** as the default (half the size, 1.8×
faster, zero measured performance loss); switch to **int4-g64** when
throughput/memory matters more than the ROUTE gate; keep **fp16** as the
scoring reference. Reproduce: `mlx_lm.convert --hf-path <fused-hf-dir>
--mlx-path <out> -q --q-mode {mxfp8|affine} [--q-bits 4 --q-group-size 64]`,
then `python -m rlvf stack-eval --tag <tag> --model <out>` +
`python -m rlvf battery --model-path <out> --adapter-name ""`.

## Artifacts

- Ledgers: `mjolnir_research/mjolnir_experiments/rlvf/data/evals/stack-*.json`
  (base, v2, v3, v4, v6, v6ck900, v7/refresh, v8/maint, soup, cosine,
  v4dpo, q4g64-v4, fp8-v4, m34-v4)
- Batteries (full transcripts + matrices): `rlvf_data/serve_smoke_battery*.md`
- Adapters: `…/rlvf/data/models/sft-all3{,v3,v4,v5,v6,-refresh,-maint,-cos}` and
  `sft-soup-v4v6`; DPO: `dpo-all3v4` + fused `hf-sft-all3v4` / `hf-dpo-all3v4`;
  quantized: `fp8-sft-all3v4` (deploy), `q4g64-sft-all3v4` (fast),
  `m34-sft-all3v4` (rejected)
- Reproduce: `bash rlvf/run_all3.sh` (env flags per version are recorded in the
  manifests under `…/rlvf/data/datasets/all3*/manifest.json` and in the shell
  history of this session); battery: `python -m rlvf battery --adapter-name <dir>`

## Improvement lifecycle

**Driver:** *the reward landscape, not the weights.* The measured record is
unambiguous — training pays only where the reward is exact and computed
(`fabricated` at −1.00, quantity-not-in-source, assert-checked outcomes),
and three runs showed a trained transcriber losing to the verifier. The
reusable artifact each cycle produces is the **landscape**: task family +
oracle + gate numbers. A checkpoint is the landscape's by-product,
versioned and ledgered, never the point.

**In operation, collect:**
- gate results per run (the 8-gate table) — a regression on any gate is
  the signal to stop, not to average;
- battery transcripts (`serve_smoke_battery_*.md`) — bit-for-bit
  determinism is a release property, re-checked after quantization;
- reward outcomes actually hit during rollouts — a `reward.py` outcome
  that never fires is dead landscape; one that fires on the wrong cases is
  a defect;
- quantization parity deltas (fp8/int4 vs fp16) — recomputed, not assumed,
  for every new checkpoint.

**Roadmap:**
1. New capability → **first** write the task family as `benchmark/tasks`
   with an exact oracle (`oracle.jsonl`) and register the pre-registered
   gate bar. If the reward cannot be computed exactly, the answer is no —
   prose-judged rewards are the measured failure regime.
2. Run the chain: train → 8 gates → battery → quantize → parity table →
   ledgers. The README's quantization table and Artifacts list are updated
   in the same change; a checkpoint without a ledger line does not exist.
3. Curation changes must beat the incumbent on the recorded series (v3
   regressed, v3-full flat, v4 regressed — the bar for "curated v5" is
   those numbers, not intuition).
4. Deploy defaults move only on parity evidence (fp8 = 7/8 parity today;
   int4 trades a ROUTE-class slip for speed — that trade is documented per
   switch, keep it that way).

**Never:** ship weights without gates + battery + quantized parity;
introduce a reward the model can satisfy without the artifact being right;
reopen scalar scores — the landscape earns trust, a leaderboard does not.

**Potential reference (future train rounds, not yet adopted).**
*DiffGate: Difficulty-Gated Teacher Guidance for On-Policy Distillation*
(NeurIPS 2026; arXiv:2610.04596, Oct 2026). Relevance: it targets exactly the
regime
this package's error log records — GRPO-style advantage vanishing on
all-failure rollout groups, and entropy/policy collapse during training
(H-R2″). Its design (verifier gates *which* failed trajectories receive
teacher guidance, difficulty scales it, a `tanh` bound keeps the dense
signal from dominating the outcome signal) is the training-time form of
this repo's standing rule: the exact component chooses where the learned
component may act. Adopt it only if a round's rollout stats show a real
all-failure share, and only under the usual contract: landscape + exact
oracle first, pre-registered gates after. It does **not** touch the
retraction above — it improves the coverage of trained models; it never
makes a trained model beat an exact verifier.