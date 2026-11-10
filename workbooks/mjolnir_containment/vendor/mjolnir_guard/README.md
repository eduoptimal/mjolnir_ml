# mjolnir_guard

**The critical decider.** One question — *may this action run without a
human?* — answered binary (`noul` 0/1) with a named stage. Same OpenRouter
envelope as `mjolnir_decide`; the `model` field tells them apart.

```
authority (mjolnir_router policy — exact, rules-as-data, free)
  ESCALATE / ABSTAIN  ->  0 + the rule or missing fact
  ROUTE               ->  veto lane
veto-reader (trained qwen, fp8, temp 0, battery-exact templates)
  ROUTE               ->  1  stage "released"
  anything else       ->  0  stage "vetoed-by-reader"
```

The reader can only tighten. It never sees a non-ROUTE, and a reader error
fails closed. `--reader off` degrades to the authority alone (stage
`released-authority-only`) — every answer names which keys signed it.

## Reading the answer

| `noul` / `decision.stage` | means |
|---|---|
| `1` / `released` | authority ROUTE **and** reader ROUTE — two keys |
| `1` / `released-authority-only` | authority ROUTE, reader not deployed |
| `0` / `blocked-by-authority:<rule>` | a policy rule fired |
| `0` / `abstained-by-authority:<fact>` | a needed fact was never stated (incl. `unknown_verb:*`) |
| `0` / `vetoed-by-reader:<verdict>` | reader saw a stated hazard on an allowed action |
| `0` / `reader-error` | reader failed — fail-closed |

## Use

```bash
pip install -e ".[guard]"                       # mlx-lm
python -m mjolnir_guard serve --http 8080       # POST / | /v1/decide
python -m mjolnir_guard serve --reader off      # authority-only
python -m mjolnir_guard selftest                # offline, stub reader
```

Reader weights: `MJOLNIR_GUARD_READER_PATH` (the quantized qwen-v4
artifact). The dialect table (`dialect.py:DIALECT`) is data — adding a verb
is a policy change, not a code change. States with no parseable
`Proposed tool call:` line are not decidable here and abstain. The FIRST hop
of the vocabulary quarantine (prose verb → tool verb) lives beside the
vocabulary it resolves into — `actions_tau_bench/prose_dialect.py` (see that
module's README); guard consumes only this second hop, and no served path
imports the prose hop.

## Limits (its own)

Authority vocabulary is closed: anything it cannot classify goes to a
human, by design. The reader is coupled to one checkpoint/template regime
(~0.5 s when it fires, GPU). Measured on the Mercury-100 battery (round-5
record): **0 real danger-misses**, per-stage audit trace, monotone
two-key release; its noise-lax cells were the battery's own documented
array-noise cases.

## Boundary with mjolnir_decide

`decide` proposes numbers (triage, ranking, offline). **A decide noul never
overrides a guard 0, and a guard 1 never depends on decide.** If you need
the number, call decide; if you need the answer, call guard.

## Improvement lifecycle

**Driver:** *coverage without loosening.* Every refusal the pipeline emits
is backlog in one of three queues — a dialect gap, a fact the authority
cannot derive, or a reader call the reader gets wrong — and improvement is
moving items out of those queues as **DATA first** (a `DIALECT` row or a
policy `Rule`), a reader checkpoint second, and never as a code-level `if`
in the authority. The two-key invariant is not part of the backlog.

**In operation, collect:**
- the `decision` block of every answer — it is the dataset. Watch the
  stage histogram: `abstained-by-authority` reasons (`missing:*`,
  `unknown_verb:*`) size the dialect/fact queues; `vetoed-by-reader` lines
  size the reader queue; any `reader-error` is an incident, not a metric;
- human outcomes on escalations — released-but-shouldn't vs
  blocked-but-shouldn't, the only two labels that matter;
- reader latency and trigger rate (only ROUTE reaches it — if that share
  grows, the authority's coverage question, not the reader's speed).

**Roadmap:**
1. **Stage → queue, every time.** `unknown_verb:X` → add `X` to
   `DIALECT` as a reviewed policy change (map to an authority verb that
   classifies it honestly — never to a verb that dodges a rule). `missing:fact`
   → decide whether the frontend can derive it or the abstention is the
   correct product answer. False reader vetoes → battery cases for the next
   reader round.
2. **Reader rounds are battery-gated, like every other trained thing in
   this repo:** collect veto/release states + human outcomes → build the
   battery with the reference-free hazard method and repeat-determinism
   cells → train template-faithful → re-run the Mercury battery + the
   no-loosen tests. Bar (register before the run): zero real
   danger-misses, monotonicity on every release, 0 determinism violations.
3. **Authority rule changes** ship as data + differential test against the
   recorded action logs (the router's 746-action corpus); escalation rate
   is reported as a bound, never optimized as a target.
4. Version coupling: the reader is married to one checkpoint + byte-exact
   templates; a new reader model means the battery reruns, not a config
   bump.

**Never:** add an `if` to the authority (policy is DATA); let the reader
see or upgrade a non-ROUTE (the `no-loosen` selftest enforces this — it
stays green or the release does not ship); fail open on `reader-error`;
answer a state with no parseable tool call.
