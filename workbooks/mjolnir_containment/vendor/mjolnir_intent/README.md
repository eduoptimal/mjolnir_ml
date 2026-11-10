# mjolnir

**Does the response still honour what the prompt committed to?**

A prompt is a specification. Someone (a user, a ticket, an agent instruction)
commits to things in it — *30 seconds*, *without retries*, *`parse_config` in
`engine.py`*. Mjolnir reads those commitments and checks what came back
against them, then tells you one of three things:

| verdict | means |
|---|---|
| **BUILD** | every committed claim survived — go ahead |
| **ASK** | the prompt itself is under-determined — clarify first |
| **blocked** (exit 1) | a committed claim was violated — see findings |

It is a **contract check between prompt and output**, not a code linter. It
never reads your code's style, complexity, or correctness — only whether the
thing you *asked for* is the thing you *got*.

**The evidence behind every claim in this README — which checks may block, which
may only report, and why — is in [science.md](science.md).**

Zero dependencies. Python ≥ 3.9. One file of logic.

---

## Install

```bash
pip install .          # from this folder; installs the `mjolnir` command
```

Or skip installing entirely:

```bash
python -m mjolnir_intent --help   # from the repository root
```

## Use

```bash
# one prompt/response pair — text or file paths, both work
mjolnir intent check -p "set the timeout to 30 seconds" -r "Set timeout to 30 minutes"
mjolnir intent check -p prompt.txt -r response.txt

# with a diff — this is what enables exact "forbidden thing" checking
mjolnir intent check -p "add caching, without retries" -r done -d change.patch

# audit a whole chat transcript (## User / ## Assistant markdown)
mjolnir intent session transcript.md

# prove the checks still behave
mjolnir intent selftest
```

Flags for `check`:

| flag | effect |
|---|---|
| `-p / --prompt` | the request (inline text or file path) — required |
| `-r / --response` | what was built (inline text or file path) |
| `-d / --diff` | unified diff; unlocks exact negation checking |
| `--context DIR` | a directory of existing files, enables "missing artifact" detection |
| `--json` | machine-readable report |
| `--strict` | advisory findings block too **+ Contractual Lowering** (see below) |

### Exit codes

| code | meaning |
|---|---|
| 0 | BUILD — no blocking findings |
| 1 | blocked — a committed claim was violated |
| 2 | ASK — prompt is under-determined; clarify before building |
| 3 | usage error |

Exit codes are stable and scriptable. Advisory findings never change the exit
code unless you pass `--strict`.

### Library

```python
from mjolnir import audit

r = audit("set the timeout to 30 seconds", "Set timeout to 30 minutes")
r["verdict"]    # 'BUILD'
r["gate"]       # [{'check': 'unit_adjacency', 'prompt_unit': 's', 'got_unit': 'min', ...}]
r["advisory"]   # findings worth reporting, not worth blocking
r["unverifiable"]  # things only a diff can decide — recorded, never failed
```

---

## The five checks, and what each is allowed to decide

| # | check | verdict | scope |
|---|---|---|---|
| 1 | **Unit adjacency** — `30 seconds` → `30 minutes` conserves the number and destroys the meaning. A numeric diff passes this. So does a compiler. | **gate** | response or diff |
| 2 | **Negation target** — `without retries` → was `retry` added? | **gate**, only when a diff is given | diff (prose is recorded `unverifiable`, never failed) |
| 3 | **Conservation** — numbers/names the prompt *committed to* must survive | **advisory** | directive sentences only |
| 4 | **Attribution** — every finding labelled by which link broke | report | derived from 1–3 |
| 5 | **Ambiguity** — under-determined prompt → `ASK` instead of guessing | **verdict** | intervention, not a gate |

The gate/advisory split is not a preference; it is the measured line. On a real
113-turn transcript (see below) the gate checks fire on **3% of turns** while
advisory conservation fires on **30%** — most of it data the user pasted for
analysis (`0.000`, `0.394`, `5.6`), which is not a commitment to reproduce it.

---

## Contractual Lowering: what is default, what needs activation

Two different things live in this module:

**Always on** (part of `audit`, every surface — CLI, session, the `mjolnir`
pipeline, RLVF verdicts): three clause-local repairs — negation objects
(`never use retries` → `retries`, previously lost to the stopword filter),
clause-local vague detection (a number in *another* clause no longer suppresses
`faster`), and `lexicon_uncovered` (a prompt outside the seed lexicon ASKs
instead of silently passing as BUILD).

**Opt-in (`--strict`):** *Contractual Lowering* — the prompt is lowered into
closed claims over T_Q and disposed by a second, independent authority
(`mjolnir_prover.lib.frontend_nl` → `claim_ir.dispose_claims`). A refuted claim
reports `claim_refuted` → exit 1; an unmaterialised commitment reports
`commitment_unmaterialised` (advisory, promoted by `--strict`).

| surface | lowering active? |
|---|---|
| `mjolnir intent check` (default) | no — regex gates only |
| `mjolnir intent check --strict` | **yes** |
| `mjolnir intent session` | no |
| `mjolnir` bare/full pipeline | no (`audit` only) |
| `mjolnir decide serve` / `mjolnir guard serve` | **never** — no import; served verdicts unchanged |
| library `frontend_nl.lower_prompt(...)` | explicit call |

Activation is deliberate: lowering can move exit codes, and every number here
is quoted (3% gate, 35% ASK on the real transcript). Default-on requires
re-running the labelled suite and that transcript through both modes and
recording the delta first. Unknown-language prompts → `ASK lexicon_uncovered`;
unknown action verbs → halt `unknown_verb`; the vocabulary grows only by
reviewed data rows (`prose_dialect.review_queue`), never by a guess.

---

## Measured behaviour

**16 labelled cases** (prompt/response pairs where the right answer is known by
construction): **13 pass, 1 documented miss** (`selftest` — both directions
scored; a check that fires on everything is as useless as one that fires on
nothing).

**113-turn real transcript** (`mjolnir session`):

```
turns                 113
verdicts              BUILD 74 / ASK 39
gate findings on      3/113 turns   =  3%     <- gate-grade: fire rarely or not at all
advisory findings on  34/113 turns  = 30%     <- report-only by design

findings by check:  unit_dropped 22 (gate) · number_lost 415, entity_lost 135 (advisory)
```

Every gate finding in that transcript was a real `unit_dropped` (e.g. prompt
says `3s`, response carries the bare `3`). None were spurious.

---

## Known limitations — read before trusting it

- **Conservation misses copula phrasing.** `"the budget is 4096"` is treated as
  a mention, not a commitment, because only directive sentences (*set, change,
  ensure…*) count. Adding `is/are/equals` was built and measured: it recovers
  exactly this labelled case at **+89% `number_lost` findings on real data
  (415 → 780)**. The fix was rejected; the labelled case is marked `KNOWN` in
  `selftest` rather than quietly deleted.
- **Negation needs a diff.** Against prose alone, a mention of the forbidden
  thing is recorded `unverifiable` — never a failure. A check that lies about
  what it can decide is worse than no check.
- **Advisory fires on pasted data.** Check 3 cannot yet tell a commitment from a
  table you pasted for analysis. That is why it is advisory. (The classifier
  that would make it a gate — commitment/reference/data, trained on mined
  commits — exists in the research repo and is measured inconclusive at current
  corpus size: not a gate yet, not closed either.)
- **ASK is a judgment call at ~35% of real turns.** Narrow by design; if it
  becomes a nuisance people will route around it, which is worse than silence.
- **One corpus, one session.** The transcript numbers come from a single real
  conversation. Treat them as a measurement, not a guarantee.

---

## Layout

```
mjolnir_intent/
├── README.md              install/usage, five-check table, measured numbers, honest limitations
├── science.md             why each check has the authority it has — the evidence in full
├── pyproject.toml         `mjolnir` console entry point, no deps, Python ≥3.9
├── mjolnir/
│   ├── __init__.py        public API: audit()
│   ├── __main__.py        python -m mjolnir_intent
│   ├── checks.py          all five checks — the only file with logic
│   ├── cli.py             check / session / selftest verbs
│   └── selftest.py        the 16 labelled cases
└── tests/test_mjolnir.py 15 tests, runs with plain python (no pytest needed)
```

Run the tests: `python tests/test_mjolnir.py` (no pytest needed) or
`pytest` if you have it.

This folder is standalone — copy it anywhere and `pip install .` works.

## Improvement lifecycle

**Driver:** *labelled cases.* A check earns its place by discriminating a
case it previously got wrong (or could not decide). The improvement unit
is not a model and not a rule patch — it is a labelled case in
`selftest.run()` plus, when the science matters, a claim in `science.md`.
`unverifiable` is a first-class answer, so "cannot check this from prose"
is backlog (usually: a diff, a declared unit table, a named referent),
not a failure to paper over.

**In operation, collect:**
- every real `BLOCKED` (exit 1) a user disputes and every `ASK` (exit 3)
  they answer anyway — each becomes a labelled case, pass side and fail
  side;
- every `needs a diff` / unverifiable finding — grouped by check, that
  histogram is the exactness roadmap;
- advisory findings that users acted on (they may belong in the gate) and
  gate findings they ignored (they may be wrong — investigate before
  defending).

**Roadmap:**
1. Write the labelled case first (expected verdict, gate vs advisory) and
   watch it fail — the case is the spec, the check is the implementation.
2. Implement the smallest exact check that passes it; classify gate vs
   advisory deliberately (gate findings block, advisory never changes the
   exit code) — misclassification is a product defect, not a nuance.
3. If prose cannot settle it, the honest move is extending the *declared*
   surface (diff input, session format), not a heuristic guess — a guess
   here produces the error-log shape: a green number measuring the
   harness.
4. Keep the standalone property: no new imports beyond stdlib + this
   folder. Tests: `python tests/test_mjolnir.py` (pytest optional).

**Never:** let an advisory silently block or a gate silently pass (the
exit-code contract is the API); add a check with no labelled case;
"resolve" an unverifiable by assuming.
