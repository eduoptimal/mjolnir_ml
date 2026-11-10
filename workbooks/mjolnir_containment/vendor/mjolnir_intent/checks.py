"""mjolnir_intent.checks — the five intent-contract checks.

THE CHAIN
---------
    prompt -> inferred intent -> implementation -> response/diff -> gate

A prompt is a specification.  This module checks whether what came back still
honours it:

  1 UNIT ADJACENCY   "30 seconds" becoming "30 minutes" conserves the number and
                     destroys the meaning.  A numeric diff PASSES this.  Nothing
                     else in the toolchain catches it.
  2 NEGATION TARGET  "without retries" -> `retry` in the added lines.  Exactly
                     checkable whenever a diff is available.
  3 CONSERVATION     numbers and named entities the prompt COMMITTED to must
                     survive downstream.  Scoped to directive sentences: pasting
                     a results table is not a commitment to reproduce it.
  4 ATTRIBUTION      every failure labelled by WHICH LINK broke.
  5 AMBIGUITY        detect prompts that are under-determined and return ASK
                     instead of guessing.  An intervention, not a gate.

THE HONEST SPLIT
----------------
Negation is exactly checkable against a DIFF.  It is NOT exactly checkable
against PROSE — a response may name the thing it declines to do.  Prose matches
are recorded `unverifiable`, never as failures.  A check that lies about what it
can decide is worse than no check.

MEASURED BEHAVIOUR (113-turn real transcript, 16 labelled cases)
----------------------------------------------------------------
  gate checks (1, 2)        fire on 3% of real turns; 6/6 labelled gate cases pass
  report checks (3)         fires on 30% of real turns — advisory, not a gate
  ambiguity (5)             ASK on 35% of real turns
  labelled total            13/14 — the known miss is documented in README
"""
from __future__ import annotations

import re
from collections import Counter

GATE_CHECKS = {"unit_adjacency", "unit_dropped", "negation_added"}

# ============================================================ claim extraction ==
UNITS = {
    "second": "time", "seconds": "time", "sec": "time", "secs": "time", "s": "time",
    "minute": "time", "minutes": "time", "min": "time", "mins": "time",
    "hour": "time", "hours": "time", "hr": "time", "hrs": "time",
    "day": "time", "days": "time", "week": "time", "weeks": "time",
    "month": "time", "months": "time", "year": "time", "years": "time",
    "ms": "time", "millisecond": "time", "milliseconds": "time",
    "microsecond": "time", "us": "time",
    "byte": "size", "bytes": "size", "kb": "size", "mb": "size", "gb": "size",
    "tb": "size", "kib": "size", "mib": "size", "gib": "size",
    "%": "ratio", "percent": "ratio", "pct": "ratio",
}
# FULL WORDS, LONGEST FIRST.  The short-form-only list shipped earlier could not
# match "seconds" (alternation tried `s`, then \b failed on the following `e`),
# so a correctly-conserved "30 seconds" was reported as unit_dropped and a real
# 30s->30min swap was mislabelled.  Both labelled failures traced to this one
# list; fixing it cost zero additional findings on the real transcript.
PAIRS = {
    "time": ["milliseconds", "millisecond", "microsecond", "seconds", "second",
             "secs", "sec", "minutes", "minute", "mins", "min", "hours", "hour",
             "hrs", "hr", "days", "day", "weeks", "week", "months", "month",
             "years", "year", "ms", "us", "s", "h"],
    "size": ["bytes", "byte", "kib", "kb", "mib", "mb", "gib", "gb", "tb", "b"],
}
CANON = {"seconds": "s", "second": "s", "sec": "s", "secs": "s",
         "minutes": "min", "minute": "min", "mins": "min",
         "hours": "h", "hour": "h", "hr": "h", "hrs": "h",
         "days": "day", "week": "week", "weeks": "week", "month": "month",
         "months": "month", "year": "year", "years": "year",
         "kb": "kb", "mb": "mb", "gb": "gb", "tb": "tb", "bytes": "b",
         "byte": "b", "millisecond": "ms", "milliseconds": "ms",
         "microsecond": "us"}

NUM_UNIT = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*"
    r"(seconds|second|secs|sec|s|minutes|minute|mins|min|hours|hour|hrs|hr|"
    r"days|day|weeks|week|months|month|years|year|ms|millisecond|milliseconds|"
    r"microsecond|us|bytes|byte|kb|mb|gb|tb|kib|mib|gib|%)\b", re.I)

BARE_NUM = re.compile(r"(?<![\w.$])(\d+(?:\.\d+)?)(?![\w.])")

ENTITY = re.compile(
    r"`([^`\n]{1,60})`"
    r"|\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][\w.]*)\b"
    r"|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"
    r"|\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")

NEG_PATTERNS = [
    re.compile(r"\bwithout\s+(?:any\s+)?([a-z][a-z0-9_]{2,})", re.I),
    # optional head verb: "never use retries" must yield `retries`, not `use`
    # (the old capture took `use`, then the stopword filter dropped it, so the
    # concrete artifact was lost and check 2 could never gate this prompt).
    re.compile(r"\b(?:don't|do not|never)\s+(?:[a-z]+\s+)?([a-z][a-z0-9_]{2,})",
               re.I),
    re.compile(r"\bno\s+([a-z][a-z0-9_]{2,})", re.I),
    re.compile(r"\bavoid\s+([a-z][a-z0-9_]{2,})", re.I),
    re.compile(r"\bstop\s+([a-z][a-z0-9_]{2,})", re.I),
]
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your", "not", "any",
    "have", "will", "can", "are", "was", "but", "all", "its", "our", "use",
    "using", "need", "want", "make", "get", "set", "add", "run", "let", "one",
    "two", "way", "now", "just", "also", "then", "than", "them", "they",
}
VAGUE = re.compile(r"\b(faster|better|cleaner|nicer|improved|optimi[sz]e|"
                   r"robust|efficient|simpler|tighter)\b", re.I)

# ================================================== COMMITMENT SCOPING =========
# Conservation applies to a COMMITMENT, not to a mention.  A commitment lives in
# a DIRECTIVE sentence — one that asks for something to change.  The unscoped
# version fired on 50% of real turns (887 number_lost, 415 entity_lost); scoping
# to directives cut that to 30% with no loss on the labelled cases.  Adding
# copula forms ("the budget is 4096") to this list was MEASURED and REJECTED:
# it recovered one labelled case at +89% number_lost on real data (415 -> 780).
DIRECTIVE = re.compile(
    r"\b(set|make|use|keep|change|update|rename|add|remove|drop|bump|scale|limit|"
    r"target|move|switch|replace|increase|decrease|reduce|raise|lower|should|must|"
    r"need to|needs|want|wants|please|ensure|migrate|refactor|rewrite|hardcode)\b",
    re.I)

# a prose mention preceded by a local negation is the response DECLINING the
# thing, not asserting it — respected, not merely unverifiable
_DECLINE_LEAD = re.compile(
    r"\b(?:not|never|without|no|don't|didn't|did not|avoid|refrained?)\b"
    r"[^.;]{0,30}$", re.I)


def _sentences(text):
    out, cur = [], []
    for ln in text.splitlines():
        cur.append(ln)
        if re.search(r"[.!?:]\s*$", ln):
            out.append("\n".join(cur))
            cur = []
    if cur:
        out.append("\n".join(cur))
    return out


def directive_text(text):
    """Only the sentences that ask for a change.  Everything else is context."""
    return "\n".join(s for s in _sentences(text) if DIRECTIVE.search(s))


def numbers_with_units(text):
    out = []
    for m in NUM_UNIT.finditer(text):
        v = float(m.group(1))
        u = m.group(2).lower()
        out.append((v, CANON.get(u, u), UNITS.get(u, "?"), m.group(0)))
    return out


def bare_numbers(text):
    taken = {m.group(1) for m in NUM_UNIT.finditer(text)}
    return [m.group(1) for m in BARE_NUM.finditer(text) if m.group(1) not in taken]


def committed_numbers(text):
    """Unitless numbers that appear in a DIRECTIVE sentence.

    "the budget is 4096" is a commitment only under copula scoping, which was
    measured and rejected (see DIRECTIVE).  "set the budget to 4096" is one.
    """
    d = directive_text(text)
    return [n for n in bare_numbers(d) if len(n) > 1]


def committed_entities(text):
    """Backticked names, file paths and snake_case identifiers in a directive."""
    d = directive_text(text)
    out = set()
    for m in re.finditer(r"`([^`\n]{1,60})`", d):
        if len(m.group(1)) > 2:
            out.add(m.group(1).strip())
    for m in re.finditer(r"\b([\w./-]+\.(?:py|md|json|yaml|yml|txt|cfg|toml|ts|js|sql))\b", d):
        out.add(m.group(1))
    for m in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", d):
        if m.group(1).lower() not in STOPWORDS:
            out.add(m.group(1))
    return out


def entities(text):
    out = set()
    for m in ENTITY.finditer(text):
        for g in m.groups():
            if g and len(g) > 2 and g.lower() not in STOPWORDS:
                out.add(g.strip())
    return out


def negation_targets(text):
    out = Counter()
    for pat in NEG_PATTERNS:
        for m in pat.finditer(text):
            w = m.group(1).lower()
            if w in STOPWORDS or len(w) < 3:
                continue
            out[w] += 1
    return out


def _numstr(v):
    return str(int(v)) if float(v) == int(v) else str(v)


# =============================================================== CHECK 1: units ==
def check_unit_adjacency(prompt, response, diff=None):
    """A number must not change its unit class, and must not lose its unit.

    A numeric diff PASSES a unit swap.  So does a compiler, and so does the test
    suite, because 30 is still 30.  The cheapest check here and the only one that
    catches a class nothing else in the toolchain sees.
    """
    fails = []
    src = response if not diff else diff
    for val, unit, dim, surface in numbers_with_units(prompt):
        if dim == "?" or dim == "ratio":
            continue
        n = _numstr(val)
        pat = re.compile(rf"(?<![\w.]){re.escape(n)}\s*({'|'.join(PAIRS[dim])})\b",
                         re.I)
        m = pat.search(src)
        if m:
            got = CANON.get(m.group(1).lower(), m.group(1).lower())
            if got != unit:
                fails.append(dict(check="unit_adjacency", surface=surface,
                                  prompt_unit=unit, got_unit=got, dimension=dim))
        elif re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", src):
            fails.append(dict(check="unit_dropped", surface=surface,
                              prompt_unit=unit, got_unit=None, dimension=dim))
    return fails


# ============================================================ CHECK 2: negation ==
def check_negation(prompt, response, diff=None):
    """A concretely-negated token must not be introduced into a DIFF.

    Prose is NOT decidable and is reported `unverifiable` rather than failed —
    unless the mention itself sits under a local negation ("I did not add
    retries"), which is the response declining the thing and is respected
    silently.  Only a diff can fail this check.
    """
    fails, unver = [], []
    targets = negation_targets(prompt)
    if not targets:
        return fails, unver
    added = ""
    if diff:
        added = "\n".join(l for l in diff.splitlines()
                          if l.startswith("+") and not l.startswith("+++"))
    for tok in targets:
        if added:
            if re.search(rf"\b{re.escape(tok)}\w*", added, re.I):
                fails.append(dict(check="negation_added", token=tok))
            continue
        m = re.search(rf"\b{re.escape(tok)}\w*", response, re.I)
        if not m:
            continue
        lead = response[max(0, m.start() - 50):m.start()]
        if _DECLINE_LEAD.search(lead):
            continue
        unver.append(dict(check="negation_prose_unverifiable", token=tok))
    return fails, unver


# ================================================ CHECK 3: claim conservation ==
def check_conservation(prompt, response, diff=None):
    """COMMITTED entities and numbers from the prompt must survive downstream.

    Advisory by design: on real transcripts this fires on 30% of turns, mostly
    on data the user pasted for analysis.  Report it; do not gate on it.
    """
    src = f"{response}\n{diff or ''}"
    fails = []
    for e in sorted(committed_entities(prompt)):
        if not re.search(re.escape(e), src, re.I):
            fails.append(dict(check="entity_lost", entity=e))
    for n in sorted(set(committed_numbers(prompt))):
        if len(n) <= 1:
            continue
        if not re.search(rf"(?<![\w.$]){re.escape(n)}(?![\w.])", src):
            fails.append(dict(check="number_lost", number=n))
    return fails


# ================================================== CHECK 4: link attribution ==
def attribute(fails):
    """Label each failure by WHICH LINK broke.

    `unattributed` is the size of the undecidable slice (prompt -> diff).  It is
    not a bug to fix — it is the number that says how much of the risk any gate
    can reach.
    """
    out = Counter()
    for f in fails:
        c = f["check"]
        if c in ("unit_adjacency", "unit_dropped", "negation_added"):
            out["downstream_of_prompt"] += 1
        elif c in ("number_lost", "entity_lost"):
            out["intent_inference"] += 1
        else:
            out["unattributed"] += 1
    return out


# ================================================ CHECK 5: ambiguity detector ==
def detect_ambiguity(prompt, context=None):
    """Would a competent engineer have to ASK before building this?

    Returns `(verdict, reasons)` where verdict is `BUILD` or `ASK`.
    An INTERVENTION, not a gate: it prevents the divergence rather than
    reporting it afterwards.  Deliberately narrow — over-triggering makes a
    system a nuisance and people route around it.
    """
    reasons = []
    bn = committed_numbers(prompt)
    if bn:
        reasons.append(dict(kind="unitless_number", value=bn[:4],
                            ask=f"What unit applies to {bn[0]}?"))
    # vague detection is CLAUSE-LOCAL: the whole-prompt rule
    # (`VAGUE and not numbers_with_units(prompt)`) let ANY unrelated number
    # suppress the finding -- "set the timeout to 5 minutes ... make it faster"
    # reported no vague_adjective at all, because `5 minutes` silenced it. The
    # unit that defeats a vague claim must sit in the SAME clause ("make it 30%
    # faster" stays BUILD; "5 minutes ... make it faster" now ASKs).
    for _part in re.split(r"[.;]\s*|\s*,\s*|\s+and\s+", prompt, flags=re.I):
        _m = VAGUE.search(_part)
        if _m and not numbers_with_units(_part):
            reasons.append(dict(kind="vague_adjective", term=_m.group(0),
                                ask=f"Which specific property should change "
                                    f"('{_m.group(0)}') and measured how?"))
            break
    for e in sorted(entities(prompt)):
        if "." in e and e.endswith((".py", ".md", ".json", ".yaml", ".ts", ".js")):
            if context is not None and not re.search(re.escape(e), context):
                reasons.append(dict(kind="missing_artifact", entity=e,
                                    ask=f"{e} does not exist -- create it, or "
                                        f"did you mean a different path?"))
    nt = negation_targets(prompt)
    if re.search(r"\b(?:don't|do not|never|without|avoid|no)\b", prompt, re.I) and not nt:
        reasons.append(dict(kind="unresolvable_negation",
                            ask="The constraint names no concrete artifact, so it "
                                "cannot be verified. What exactly is forbidden?"))
    # LEXICON COVERAGE: silence is not ignorance. A prompt whose obligations
    # are written in words the seed lexica do not know would otherwise read as
    # "no obligations" and exit 0 -- a fail-open pass exactly where the
    # pipeline claims to fail closed. Non-ASCII letters + zero recognized
    # directive sentences = the sieve cannot see this language; reclassify the
    # silence as ASK. Same lifecycle as stage 3's unknown_verb halt: counted,
    # then cleared by reviewed lexicon rows, never by a guess.
    if (any(ord(ch) > 127 and ch.isalpha() for ch in prompt)
            and not directive_text(prompt)):
        reasons.append(dict(kind="lexicon_uncovered",
                            ask="No recognized obligation markers in this text: "
                                "the prompt's language is outside the seed "
                                "lexicon. Clarify, or add reviewed lexicon rows."))
    return ("ASK" if reasons else "BUILD"), reasons


# ================================================================ the report ====
def audit(prompt, response="", diff=None, context=None):
    """Run all five capabilities and return a single report.

    `failures` split into gate-grade (checks 1-2) and advisory (check 3) is the
    caller's job — see `GATE_CHECKS`.
    """
    units = check_unit_adjacency(prompt, response, diff)
    neg, unver = check_negation(prompt, response, diff)
    cons = check_conservation(prompt, response, diff)
    fails = units + neg + cons
    verdict, amb = detect_ambiguity(prompt, context)
    return dict(
        verdict=verdict,
        ambiguity=amb,
        failures=fails,
        gate=[f for f in fails if f["check"] in GATE_CHECKS],
        advisory=[f for f in fails if f["check"] not in GATE_CHECKS],
        unverifiable=unver,
        attribution=dict(attribute(fails)),
        n_claims=dict(
            numbers_with_units=len(numbers_with_units(prompt)),
            bare_numbers=len(bare_numbers(prompt)),
            entities=len(entities(prompt)),
            negations=len(negation_targets(prompt)),
        ),
    )
