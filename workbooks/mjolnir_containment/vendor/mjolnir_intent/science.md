# The science behind mjolnir

**One sentence:** mjolnir tests whether a *response* still honours the commitments
carried in its *prompt* — verification of **fidelity**, where existing tooling only
verifies **correctness**.

A rename that drops every call site parses, type-checks, passes lint and passes the
tests. Nothing in the correctness toolchain is broken by it, because nothing in the
correctness toolchain has ever seen the prompt. That gap is the subject of this
document.

---

## 1. The chain, and which links are decidable

```
prompt  ->  inferred intent  ->  implementation  ->  response/diff  ->  gate
```

Three segments, three different epistemic statuses. The status is not an opinion; it
follows from what each segment *is*:

| segment | status | why |
|---|---|---|
| **prompt → intent** | **largely exact** | A prompt carries concrete, checkable claims: numbers with units, negated tokens, named artifacts. These are extractable sets, and set-conservation is exactly decidable — the same shape as the invariants the parent project proves at 1.000 across the `intent → diff` boundary. |
| **intent → diff / response** | **exactly decidable for narrow classes** | "Was `retry` added?" is a string question with a yes/no answer. "Is this response a faithful implementation of that request?" is semantic equivalence — undecidable in general, and no engineering makes it otherwise. |
| **prompt → diff (whole)** | **not decidable** | Semantic equivalence of free text. The correct response is not to attempt it but to **instrument** it: measure how large the undecidable slice is, and attribute every caught failure to the link that broke. |

The design consequence: **every check in mjolnir is entitled to exactly as much
authority as the decidability of its segment.** Checks that cannot decide say
`unverifiable` — recorded, never failed. A check that lies about what it can decide is
worse than no check, because it manufactures the feeling of coverage.

---

## 2. The five checks as an epistemic hierarchy

The checks are ordered by how much they are allowed to *do* — gate, report, or
intervene. That ordering was not designed in; it was **measured**, and the numbers
below are the reason each check sits where it sits.

### Check 1 — Unit adjacency (gate)

`30 seconds` → `30 minutes` conserves the number and destroys the meaning. A numeric
diff **passes** this transformation. So does a compiler (`30 == 30`), so does a test
suite, so does `re.sub`. This is the cheapest check in the file and the only one that
sees a failure class invisible to the entire downstream toolchain.

**Why it can gate:** it compares two concrete extracted sets (prompt units vs response
units) over a fixed dimension table. No semantics required, no judgment call. On the
real transcript it fires on **22 findings across 3 turns (3%)**, and every one was a
real `unit_dropped` — e.g. prompt says `3s`, response carries the bare `3`.

### Check 2 — Negation target (gate, against a diff only)

`without retries` → was `retry` in the *added* lines? Against a **diff** this is a
string question with a yes/no answer: exactly decidable. Against **prose** it is not —
a response may legitimately *name* the thing it declines to do ("I did not add
retries"), and failing that would be a false positive on correct behaviour.

The rule follows the decidability boundary, not convenience:

```
diff available   ->  add of a negated token        = FAILURE (gate)
prose only       ->  mention under local negation  = respected (declined)
                   ->  bare mention                 = unverifiable (recorded, never failed)
```

### Check 3 — Conservation (advisory — and the measurement that forced that)

The prompt commits to names and numbers; they should survive downstream. In principle
the cleanest check of all. In practice it is the one that taught this project its
central lesson about gates:

| scoping | conservation findings on 113 real turns | turns affected |
|---|---|---|
| every mention anywhere | 887 `number_lost` + 415 `entity_lost` | **50% of turns** |
| **directive sentences only** (shipped) | 415 `number_lost` + 135 `entity_lost` | **30% of turns** |
| + copula forms (`the budget is 4096`) | 780 `number_lost` + 191 `entity_lost` | 42% of turns |

A gate firing on half of all real inputs is not a gate; it is noise, and a gate with a
50% false-positive rate gets switched off inside a month. The scoping principle that
cut it in half:

> **Conservation applies to a COMMITMENT, not to a mention.** A commitment lives in a
> directive sentence — one that asks for something to change. A number inside an aside,
> or inside a results table the user pasted *for analysis*, is data, not a promise.

**The copula row is a measured rejection, not an oversight.** Adding `is/are/equals`
recovers exactly one labelled case (`"the budget is 4096"` is a commitment) at the cost
of **+89% `number_lost` findings on real data**. Two scoping attempts is the signal to
stop and report; tuning until one transcript is quiet manufactures false confidence.
The case is marked `KNOWN` in `selftest` rather than deleted, and the fix's cost is
recorded in this document rather than left for someone to rediscover.

That residual 30% is almost entirely the same error class: **data pasted for analysis
(`0.000`, `0.394`, `5.6`) being treated as a commitment.** Telling a commitment from
data is a semantic judgment — this is precisely what regex is bad at (§4) and what the
encoder probe (§4.1) measured. Until that classifier clears its bar, check 3 reports and
does not block. The honest status: **advisory by measurement, not by timidity.**

### Check 4 — Attribution (report, derived)

Every finding is labelled by *which link broke*: `downstream_of_prompt` (the prompt was
determinate, the response violated it), `intent_inference` (the extraction scoped
wrong), or `unattributed`. The unattributed bucket is not a bug to fix — it is the
**measured size of the undecidable slice**, and therefore the number that says how much
of the risk *any* gate can reach.

### Check 5 — Ambiguity (intervention, not a gate)

`ASK` fires *before* the divergence rather than reporting it afterwards. It is
deliberately narrow — over-triggering makes a system a nuisance and people route
around it — so it fires only on provably under-determined claims: a unitless number
(`set the limit to 4096`), a vague adjective with nothing to measure against (`make it
faster`), a referenced artifact that does not exist, a constraint with no concrete
target. On the real transcript it returns `ASK` on **39/113 turns (35%)**.

> **Caveat (2026-10-09):** the 39/113 was measured under the original
> whole-prompt vague rule, which let any unrelated number suppress the finding.
> Vague detection is now clause-local and `negation_targets` recovers the object
> of `never use X` (two cases donated by the contract-lowering experiment). Both
> changes move individual turns between verdicts; the transcript number has not
> been re-run and is quoted as historically measured, not as current.

---

## 3. The evidence

### 3.1 Labelled cases — ground truth by construction

14 prompt/response pairs where the right answer is known *by construction*. Both
directions are scored, because a check that fires on all of them is as useless as one
that fires on none:

```
13/14 pass   ·   1 documented KNOWN miss (copula commitment, §2)   ·   0 unexpected failures
```

The suite runs inside `mjolnir selftest` and again in `tests/`, where it gates the
test run — a regression in any check fails the build, not a code review.

### 3.2 The real transcript — the false-positive audit

113 turns of a genuine human–AI conversation, every user prompt paired with the
response it produced. There is no ground truth here — **that is the point**. This
measures the *noise* on real language, which is the number that decides whether a
check can ever be a gate rather than a report:

```
turns                 113
verdicts              BUILD 74 / ASK 39
gate findings on      3/113 turns   =  3%     <- gate-grade requires this to be low
advisory findings on  34/113 turns  = 30%

findings:  unit_dropped 22 (gate) · number_lost 415, entity_lost 135 (advisory)
```

The 3% / 30% split **is** the gate/advisory boundary. It was measured, not chosen.

### 3.3 Controls

- **Determinism** — identical inputs produce byte-identical reports (tested).
- **Both-direction scoring** — every labelled case declares what must fire *and* what
  must not; a check that fires everywhere fails the suite.
- **Noise floor on real data** — the transcript audit exists to catch the failure mode
  where a check looks perfect on examples and screams on real language. It caught two
  such checks during development (before scoping: 50% fire rate).

---

## 4. Why this cannot be a regex — the intent result

The deeper question the parent project measured: when the task is *"does this text
carry intent X?"*, how does a keyword/regex baseline compare to a frozen encoder?

**The corpus:** 15,960 commits examined across 16 real repositories; **948 single-action
commits** qualify (one structural detector fires, all others silent), yielding 935 rows
over 887 distinct messages. **Labels come from what the diff does**, never from what
the message says — ground truth that is not derived from the thing being predicted.

**The result (five seeds, macro-F1, zero optimised parameters):**

```
frozen encoder (MiniLM, 22.7M)   0.3576 ± 0.0321
keyword/regex baseline           0.090 – 0.117
EMPIRICAL chance floor           0.1228   (simulated over 3000 draws, not asserted)

GAIN over regex                  +0.2504 ± 0.0384   (all five seeds pass)
```

**The load-bearing detail: the regex scores below the chance floor.** Not weak — *below
guessing*. The reason is visible in the corpus itself. Real messages are indirect:

```
add_docstring      "Fix #6628 - JSONDecodeError are not deserializable"
rename_variable    "no path by default"
```

Nothing lexical survives. So when this project's early check prototypes reached for
regex to classify commitments — the same discrimination, "set the timeout to 30" vs
"we measured 0.394" — the measured prior said that was the wrong tool *for that class
of question*, and the 550-finding false-positive storm that followed was the proof
rather than an accident of tuning.

The encoder bake-off adds the second lesson: **15× more parameters buys absolute
accuracy, not a different capability** (bge-large 335M scores 0.4117 vs MiniLM's
0.3576; SimCSE — trained specifically for sentence similarity — is *worst* of the
three). Capacity is not the axis; the semantic discrimination is.

### 4.1 The commitment classifier — where check 3's gate would come from

The natural rescue for check 3: classify every candidate claim as
**COMMITMENT / REFERENCE / DATA**, trained on the 948 commits, with the **diff as the
non-circular label** (did the diff act on the claim? then it was a commitment).
`DATA` is exactly the class regex can never get, and it is what the residual advisory
findings are made of — the 415 `number_lost` are overwhelmingly pasted measurements,
the 135 `entity_lost` overwhelmingly referenced-but-unchanged artifacts.

Measured result, honest status: **direction-positive, statistically inconclusive.**
Across five seeds and two encoders (22.7M and 335M), the encoder's macro-F1 beats the
keyword control on **all ten runs** (mean gain +0.046 / +0.066), and on the target
class the gap is large — keyword COMMITMENT-F1 averages ~0.05, the encoder ~0.21 — but
the paired bootstrap clears p<0.05 in only 3 of 10 runs, and the significant runs are
precisely the largest test splits. The 15× larger encoder adds nothing on the target
class (COMMITMENT-F1 0.206 vs 0.227 for the small one).

**Diagnosis:** the binding constraint is corpus size (505 claims, 111 positives, test
splits of 60–164), not encoder capacity. The probe either rescues check 3 into a gate
after more labels, or closes it honestly. It is neither yet, and check 3 ships advisory
until it is one or the other.

---

## 5. Position: what this is and what it is not

### It is not a linter

The product boundary (parent project §2.13) is one filter:

> **If a check does not need the diff — the before/after pair — it is not ours.**

Naming, import hygiene, complexity, dead code are *file* properties: one snapshot, no
transformation, and `ruff`/`pylint` are free, mature and better at them. Every check in
mjolnir needs both sides — the prompt's claims and the response's handling of them.
It sits inside `propose → execute → verify`, and it checks **fidelity**: was the edit
faithful to what was asked? Correctness tooling answers *"does this work?"*;
fidelity is a different question with a different evidence base.

### It is a verifier with a measured authority limit

The parent project proves the downstream links tightly: the `intent → diff` gate at
**1.000** on invariants, the `diff → message` audit at **25/25**. mjolnir covers the
two links to the *left* — the ones that were never gated — and its authority
correspondingly stops at decidability:

```
exact and gateable   unit adjacency · negation (against a diff)
exact but too noisy  conservation          -> report, measured 30%
undecidable          semantic equivalence  -> instrumented via attribution
judgment            ambiguity             -> intervene (ASK), never block
```

### Why fidelity is the interesting half

Code-editing models are **25% silently wrong** — parse-valid, unfaithful,
compiler-blind — replicated across two model families (parent §2.12), and the rate
*falls* with file size, so scale does not dissolve it. Fail-closed verification
pipelines for AI agents exist and ship (parent `FAILED.md` §9) — but they ask only
*"does this work?"*. The wedge this project measures is that **the silent 25% is a
fidelity failure wearing a correctness costume**, and fidelity starts at the prompt.

---

## 6. What this document does not claim

Stated plainly so it is not implied by omission:

- **One corpus, one transcript.** The 3%/30% figures come from a single real
  conversation. They are a measurement, not a guarantee.
- **Conservation misses copula phrasing** — measured, fix rejected on noise grounds
  (§2). The labelled miss is visible in `selftest`, not hidden.
- **Negation is undecidable without a diff.** Prose mentions are recorded
  `unverifiable`; no claim is made about them in either direction.
- **Check 3's classifier is inconclusive** (§4.1), so check 3 is advisory. The path to
  a gate exists and is measured to be *label-limited*, but it is not walked yet.
- **ASK at 35% is a judgment call.** It is narrow by design; whether that is the right
  width is a deployment question, decided by whether people route around it.
- **No comparison against existing prompt-fidelity tooling.** The claim is the
  *category* — fidelity vs correctness — is unoccupied where measured, not that no
  competitor exists anywhere.

---

## Appendix: reproduction

```bash
pip install .
mjolnir selftest                    # 16 labelled cases, both directions scored
mjolnir session path/to/chat.md     # gate 3% / advisory 30% on the real transcript
python tests/test_mjolnir.py        # 15 tests: labelled cases + CLI exit contracts
```

All numbers in this document come from those commands and from the research repo's
`docs/STATE.md` (§2.11 intent, §2.12 silent breakage, §2.13 product boundary),
`docs/FAILED.md` (rejected fixes, four ways a number lies), and the `claimcls`
experiment logs. Nothing here is asserted without a measurement behind it; anything
that failed is recorded as a failure, which is the only way the passing numbers mean
anything.
