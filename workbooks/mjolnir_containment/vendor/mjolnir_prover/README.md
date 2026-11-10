# mjolnir_prover

The proof engine: mine claims from code, answer each one, keep the record.

```
mjolnir verify <dir>        fragments.jsonl -> certificates.json -> report
mjolnir prover harvest      mine claims into the storage corpus
mjolnir prover check        re-check an existing corpus
```

**The vocabulary (what a reader is allowed to conclude):**

| verdict | meaning |
|---|---|
| `CERTIFIED` | proved by the kernel — the only word that means "true" |
| `declared` | a recorded contract (premise tier) — checked, resting on something verified elsewhere; counted separately, never as a proof |
| `REFUTED` | proved **false**, with a witness (exit 1, open it) |
| `UNVERIFIABLE` | the engine declined — `certificates.json` says why (`no_assert_spec`, `runtime_observation` → see `enforce_candidates.json`, `numeric_domain:ieee_float`, …). Declined is not failed |

Storage (`storage/data`, mined corpora) is deliberately **not** wheel
package-data: created on first use by the runtime path resolver. The
contract DSL and report templates ship; the corpora do not.

## frontend_nl — Contractual Lowering (the NL frontend)

`lib/frontend_nl.py` is the prose member of the frontend family
(`frontend_sql`, `frontend_pytest`, `frontend_lean2`): it lowers a user's
*prompt* — the specification, not source code — into closed `Claim[]` records
and hands them to `dispose_claims`. PROPOSE / GROUND / DISPOSE:

- **propose** — seed regex segmentation by default; the documented `PROPOSER`
  seam takes a model segmenter whose output is **offsets only** `(start, end)`;
- **ground** — exact code rejects any span that does not slice the prompt
  verbatim (a model cannot inject an obligation); extraction (units, entities,
  negations, vague predicates) is model-free over shared lexica from
  `mjolnir_intent.checks`; undecidable remainders become typed ASK sinks and
  unknown action verbs HALT;
- **dispose** — only this authority does, via `claim_ir.dispose_claims`.

**Activation:** called by `mjolnir intent check --strict` and the
contract-lowering experiment/benchmark
(`mjolnir_research/mjolnir_experiments/contract_lowering`). **Not** imported
by `mjolnir decide serve`, `mjolnir guard serve`, or the default `mjolnir`
pipeline — served verdicts are unchanged by construction. Test:
`python -m mjolnir_prover.lib.test_frontend_nl`.

## Improvement lifecycle

**Driver:** *the UNVERIFIABLE taxonomy.* The declined set is the roadmap —
every reason in `certificates.json` names a missing structure (no spec, no
venue, wrong numeric domain), and each distinct reason is a candidate
tactic, kernel extension, or promote-to-venue path. Real runs are the
census: the banking-SQL run (3,864 claims → 502 proved, 1,538 declared,
1,824 declined, 0 refuted) is the baseline the next version is measured
against, not a target to inflate.

**In operation, collect:**
- every `REFUTED` — investigate before anything else; a refutation with a
  bad witness is a harness error, not a product event;
- declined reasons grouped by tactic-id — a reason that appears at scale
  outranks a reason that is merely interesting;
- `enforce_candidates` accepted/rejected — the database-venue path is the
  promotion route for `runtime_observation` claims;
- any certificate whose claim and artifact were produced by the same
  function (the CWE lesson: oracle and subject must be independent).

**Roadmap:**
1. Pick one declined reason by volume → write the tactic/claim-miner
   support → the claim type mines, certifies, and has a labelled test
   **before** it counts anywhere in a report.
2. Premise-tier hygiene: anything not kernel-checked enters as `declared`
   with its resting place named. Promoting a premise tier is a versioned,
   evidenced act — never a rename.
3. Storage/ledger changes (verdict history, audit fields) ship with **twins
   that fail when the change is reverted** — the audit/ directory exists
   for exactly this; the demotion-audit blocker (no verdict history) was a
   missing field, not a missing idea.
4. Re-run the census corpus after each tactic lands; report proved /
   declared / declined / refuted as four numbers, never a single score
   (scalar trust is a closed question).

**Never:** count `declared` as proof; quote a certificate count without its
declined/ refuted companions; let a self-check certify its own input.
