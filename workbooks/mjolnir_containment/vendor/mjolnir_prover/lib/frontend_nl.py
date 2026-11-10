"""mjolnir_prover.lib.frontend_nl -- Contractual Lowering: prose -> Claim IR.

The frontend rule (frontend.py): parse -> `claim_ir.Claim` -> `dispose_claims`.
Every other frontend parses a MACHINE language (Python, Java, SQL, Lean, TPTP).
This one parses the user's specification: NATURAL LANGUAGE, projected into the
decidable fragment before any oracle may look at it.

    Contractual Lowering := structurally project undecidable prose into a
    decidable algebraic fragment. A check is entitled to exactly as much
    authority as the decidability of its segment.

PROPOSE / GROUND / DISPOSE -- the model gets attention, never authority
-----------------------------------------------------------------------
PROPOSE   `scope(prompt, proposer=PROPOSER)`. The default proposer is the seed:
          exact regex segmentation (imperative markers + commitment payload),
          English, free, deterministic. The PROPOSER seam (documented, like
          `ENTAILMENT_HOOK` in tactics/nl_certify.py) takes a model-backed
          segmenter -- an encoder such as SmolLM/Qwen -- whose output is SPANS
          ONLY: (start, end) offsets into the prompt. Offsets, not text: a span
          that does not slice back the prompt verbatim is rejected at grounding,
          so a hallucinated obligation cannot enter. A model that segments
          obligation clauses in any language extends the sieve multilingually
          without ever being trusted about meaning.

GROUND    exact code, on every path: the span must be in bounds and slice
          verbatim; extraction is model-free regex over the grounded clause
          (units, entities, negation objects, vague predicates). The lexica
          (UNITS via mjolnir_intent, VAGUE, negation markers) are DATA: more
          languages are more rows, never more code.

DISPOSE   grounded commitments become CLOSED Claim records over T_Q and are
          handed to `claim_ir.dispose_claims` (`decide_eq`, residue, interval).
          Undecidable remainders (a vague predicate, an inexact unit, an
          unmaterialised commitment) become TYPED SINKS (ASK) -- never a guess,
          never silence. An unknown action verb (with a `dialect` supplied) is a
          structural HALT: quarantine, not translation.

Dependencies: mjolnir_intent (single source for the NL lexica -- intent stays
zero-dependency; this direction introduces no cycle). Domain dialect tables
(prose verb -> tool verb for a specific platform) are INJECTED by the caller --
this module owns no platform vocabulary, consistent with "everything
domain-specific lives in its own module" (actions_tau_bench/vocab.py).
"""
from __future__ import annotations

import re
from fractions import Fraction

from mjolnir_intent import checks as intent

from mjolnir_prover.lib import claim_ir, term as T

# Documented seam. None = the seed proposer (exact regex segmentation).
# A model-backed proposer is callable(prompt) -> list[(start, end)], returning
# offsets into the prompt for each clause the model believes is an obligation.
# It may only POINT; grounding below decides whether the pointer is real.
PROPOSER = None

# =========================================================== unit algebra (DATA) ==
# Canonicalise every unit to an EXACT base quantity: Fraction arithmetic, never
# float ("5 minutes" -> 300 s with no rounding). Units with no exact conversion
# to the base (a month, a year) are deliberately ABSENT -- decline rather than
# invent 30 days. More languages: more rows here, still DATA.
BASE = {"time": "s", "size": "b", "ratio": "%"}
FACTOR = {
    "s": Fraction(1), "ms": Fraction(1, 1000), "us": Fraction(1, 10 ** 6),
    "min": Fraction(60), "h": Fraction(3600), "day": Fraction(86400),
    "week": Fraction(604800),
    "b": Fraction(1), "kb": Fraction(1024), "mb": Fraction(1024 ** 2),
    "gb": Fraction(1024 ** 3), "tb": Fraction(1024 ** 4),
    "%": Fraction(1),
}

CONSTRAINT_VERBS = {"ensure", "make", "keep", "require", "limit", "target",
                    "cap", "allow", "want", "need", "use", "must", "should"}

# The seed's imperative markers extend the shipped DIRECTIVE set with the
# read-side verbs of this domain. (A model proposer replaces this whole test.)
EXTRA_IMPERATIVES = re.compile(
    r"\b(read|fetch|query|select|list|show|display|inspect|deploy|run|kill|"
    r"revoke|nuke|purge|wipe)\b", re.I)

_SENT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE = re.compile(r"\s*,\s*|\s+\band\b\s+|\s+\bthen\b\s+", re.I)
_LEAD_CJ = re.compile(r"^(?:and|also|then|plus|please|kindly)\s+", re.I)

_WORDS = re.compile(r"[a-z_]+")
_SKIP_WORDS = {
    "hey", "hi", "hello", "ok", "okay", "please", "kindly", "also", "then",
    "and", "you", "we", "i", "must", "should", "shall", "never", "always",
    "just", "now", "could", "would", "will", "can", "the", "a", "an",
}

# Clause-local negation WITH its object: "never use retries" yields `retries`
# (the whole-prompt extractor captured `use` and dropped it as a stopword).
_NEG_OBJ = re.compile(
    r"\b(?:never|don't|do not|without|avoid|no|stop)\s+"
    r"(?:(?:any\s+)?[a-z]+\s+)?([a-z][a-z0-9_]{2,})", re.I)
_NEG_LEAD = re.compile(
    r"^[\s\W]*(?:also\s+|and\s+|please\s+)*"
    r"(?:never|don't|do not|without|avoid|no|stop)\b", re.I)

_MODAL = re.compile(r"\b(must|should|shall|never|don't|do not|without|avoid|"
                    r"required?|forbidden)\b", re.I)

_ENTITY = re.compile(
    r"`([^`\n]{1,60})`"
    r"|\b([\w./-]+\.(?:py|md|json|yaml|yml|toml|cfg|txt|ts|js|sql))\b"
    r"|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")


def entities_of(text: str) -> list:
    out, seen = [], set()
    for m in _ENTITY.finditer(text or ""):
        for g in m.groups():
            if g and len(g) > 2 and g.lower() not in _SKIP_WORDS and g not in seen:
                seen.add(g)
                out.append(g.strip())
    return out


def split_sentences(text: str) -> list:
    """Prose sentences. `intent._sentences` is line-oriented (transcripts)."""
    out = []
    for s in _SENT.split((text or "").strip()):
        s = s.strip().rstrip(".!?;:").strip()
        if s:
            out.append(s)
    return out


def split_clauses(sentence: str) -> list:
    out = []
    for c in _CLAUSE.split(sentence):
        c = _LEAD_CJ.sub("", c.strip(" ,;.!?")).strip()
        if c:
            out.append(c)
    return out


def payload_types(clause: str) -> list:
    """Commitment payload of a clause. Empty payload => not an obligation."""
    kinds = []
    if intent.NUM_UNIT.search(clause):
        kinds.append("unit")
    if entities_of(clause):
        kinds.append("entity")
    if _MODAL.search(clause):
        kinds.append("modal")
    if _NEG_OBJ.search(clause):
        kinds.append("negation")
    if intent.VAGUE.search(clause):
        kinds.append("vague")
    return kinds


def leading_verb(clause: str) -> str:
    for w in _WORDS.findall((clause or "").lower()):
        if w not in _SKIP_WORDS:
            return w
    return ""


# ============================================================== stage 1: scope ==
def scope(prompt: str, proposer=None) -> dict:
    """The epistemic sieve. kept clauses are the obligations; the rest is
    rhetoric, discarded.

    proposer=None  seed (exact, English): sentences -> clauses -> imperative
                   marker AND commitment payload.
    proposer set   model-backed segmenter: returns (start, end) offsets; GROUND
                   rejects anything that does not slice the prompt verbatim, and
                   the payload/imperative tests are skipped (they are English
                   seed rules -- the model's claim is WHICH spans are
                   obligations, and its only failure mode is a dropped
                   obligation, never a fabricated one).
    """
    kept, discarded = [], []
    if proposer is None:
        for sent in split_sentences(prompt):
            for clause in split_clauses(sent):
                payload = payload_types(clause)
                imperative = bool(intent.DIRECTIVE.search(clause)
                                  or EXTRA_IMPERATIVES.search(clause))
                if imperative and payload:
                    kept.append({"clause": clause, "verb": leading_verb(clause),
                                 "payload": payload})
                else:
                    why = ("no_imperative" if not imperative
                           else "no_commitment_payload")
                    discarded.append({"clause": clause, "reason": why})
        return {"kept": kept, "discarded": discarded, "proposer": "seed"}

    for span in proposer(prompt) or []:
        # GROUND: offsets must be in bounds and slice a non-empty span of the
        # prompt. The slice is verbatim by construction -- a proposer that
        # returns text instead of offsets, or offsets that land nowhere, is
        # rejected here and can never fabricate an obligation.
        try:
            start, end = int(span[0]), int(span[1])
            ok = 0 <= start < end <= len(prompt)
            text = prompt[start:end].strip() if ok else ""
            ok = ok and bool(text)
        except Exception:                                     # noqa: BLE001
            ok, text = False, ""
        if ok:
            kept.append({"clause": text, "verb": leading_verb(text),
                         "payload": payload_types(text), "span": (start, end)})
        else:
            discarded.append({"clause": str(span), "reason": "ungrounded_span"})
    return {"kept": kept, "discarded": discarded, "proposer": "model"}


# ================================================= stage 2: commitment extraction ==
def _first_entity(clause: str):
    ents = entities_of(clause)
    return ents[0] if ents else None


def _vague_subject(clause: str, term: str):
    m = re.search(r"\b([a-z_][a-z0-9_]*)\s+(?:is|are|gets|becomes|remains|"
                  r"runs)\b[^.]*?\b" + re.escape(term) + r"\b", clause, re.I)
    return m.group(1) if m else None


def extract(scoped: dict) -> list:
    """Grounded clauses -> typed commitments
    (value | negation | action | vague | unlowerable). Model-free."""
    out = []
    for item in scoped["kept"]:
        clause, verb = item["clause"], item["verb"]

        units = intent.numbers_with_units(clause)
        if units:
            val, unit, dim, surface = units[0]
            ent = _first_entity(clause)
            if unit not in FACTOR or dim not in BASE:
                out.append(dict(kind="unlowerable", clause=clause, verb=verb,
                                artifact=ent, surface=surface, unit=unit,
                                dimension=dim, reason="inexact_unit"))
            elif ent is None:
                out.append(dict(kind="unlowerable", clause=clause, verb=verb,
                                artifact=None, surface=surface, unit=unit,
                                dimension=dim, reason="no_named_artifact"))
            else:
                out.append(dict(kind="value", clause=clause, verb=verb,
                                artifact=ent, surface=surface, unit=unit,
                                dimension=dim,
                                value=Fraction(str(val)) * FACTOR[unit]))
        elif intent.VAGUE.search(clause):
            term = intent.VAGUE.search(clause).group(0)
            out.append(dict(kind="vague", clause=clause, verb=verb,
                            term=term, subject=_vague_subject(clause, term)))

        for m in _NEG_OBJ.finditer(clause):
            obj = m.group(1).lower()
            if obj in _SKIP_WORDS or obj in CONSTRAINT_VERBS:
                out.append(dict(kind="unresolvable_negation", clause=clause,
                                verb=verb, object=obj,
                                reason="negation_names_no_artifact"))
            else:
                out.append(dict(kind="negation", clause=clause, verb=verb,
                                object=obj, required=Fraction(0)))

        neg_led = bool(_NEG_LEAD.match(clause))
        if (not neg_led and verb and verb not in CONSTRAINT_VERBS
                and not (intent.VAGUE.search(clause) and not units)):
            out.append(dict(kind="action", clause=clause, verb=verb))
    return out


# ================================================== stage 4: lower to claim IR ==
def _as_quantity(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return Fraction(str(v))
    if isinstance(v, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", v.strip()):
        return Fraction(v.strip())
    return None


def _proposal_value(proposal: dict, bind: dict, artifact: str):
    key = bind.get(artifact, artifact)
    cfg = proposal.get("config") or {}
    if key in cfg:
        return _as_quantity(cfg[key]), key, "config"
    for line in (proposal.get("diff") or "").splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        m = re.search(rf"\b{re.escape(key)}\b\s*[:=]\s*(-?\d+(?:\.\d+)?)", line)
        if m:
            return Fraction(m.group(1)), key, "diff"
    return None, key, None


def _canonical_action(nl_verb: str, args: dict) -> str:
    scope_str = ""
    for v in (args or {}).values():
        if isinstance(v, str) and v:
            scope_str = v
            break
    return f"{nl_verb.upper()}({scope_str})"


def _frac_str(v):
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _object_of(clause: str):
    ents = entities_of(clause)
    return ents[0] if ents else ""


def lower(commitments: list, proposal: dict, bind: dict | None = None,
          dialect: dict | None = None) -> dict:
    """Grounded commitments + the agent's proposal -> CLOSED Claim records.

    The proposal's canonical value is substituted HERE, at the frontend, so the
    IR sees two exact terms and nothing else. `dialect` (prose verb -> tool verb)
    enables the action leg; with dialect=None, action commitments produce no
    claim (the caller -- e.g. `mjolnir intent` -- gates artifacts, not actions).
    An action verb that the dialect does not contain is a structural HALT.
    """
    bind = bind or {}
    claims, sinks, lowered, halt = [], [], [], None

    for c in commitments:
        if c["kind"] == "value":
            proposed, key, src = _proposal_value(proposal, bind, c["artifact"])
            if proposed is None:
                sinks.append(dict(kind="missing_proposal_value",
                                  artifact=c["artifact"], key=key,
                                  reason="the commitment was never materialised "
                                         "by the agent"))
                continue
            claims.append(claim_ir.eq(
                T.const(proposed), T.const(c["value"]), line=len(claims) + 1,
                is_premise=False,
                source=f"{c['artifact']} == {_frac_str(c['value'])} "
                       f"{BASE.get(c['dimension'], '')}".strip()))
            lowered.append(dict(kind="value", artifact=c["artifact"],
                                required=_frac_str(c["value"]),
                                proposed=_frac_str(proposed),
                                unit=BASE.get(c["dimension"]),
                                source=src, key=key))

        elif c["kind"] == "negation":
            proposed, key, src = _proposal_value(proposal, bind, c["object"])
            if proposed is None:
                claims.append(_claim(0, 0, f"{c['object']} == 0 (absent)",
                                     len(claims) + 1))
                lowered.append(dict(kind="negation", artifact=c["object"],
                                    required="0", proposed="0 (absent)",
                                    source="absent"))
            else:
                claims.append(_claim(proposed, 0, f"{c['object']} == 0",
                                     len(claims) + 1))
                lowered.append(dict(kind="negation", artifact=c["object"],
                                    required="0", proposed=_frac_str(proposed),
                                    source=src, key=key))

        elif c["kind"] == "action":
            if dialect is None:
                continue                      # caller gates artifacts only
            tool = (dialect.get(c["verb"]) if hasattr(dialect, "get")
                    else dialect(c["verb"]))   # dict of rows, or a lookup fn
            if tool is None:
                if halt is None:
                    halt = {"reason": f"unknown_verb:{c['verb']}",
                            "clause": c["clause"]}
                continue
            if any(v["kind"] == "value" and v["clause"] == c["clause"]
                   for v in commitments):
                continue          # the verb edits a value: not an execution leg
            match = next((call for call in (proposal.get("tool_calls") or [])
                          if call.get("name") == tool), None)
            if match is None:
                sinks.append(dict(kind="missing_proposal_action", verb=c["verb"],
                                  tool=tool,
                                  reason="the prompt committed to an action the "
                                         "agent never proposed"))
                continue
            required = _canonical_action(c["verb"],
                                         {"obj": _object_of(c["clause"])})
            proposed = _canonical_action(c["verb"], match.get("args") or {})
            claims.append(_claim(proposed, required,
                                 f"action == {required}", len(claims) + 1))
            lowered.append(dict(kind="action", verb=c["verb"], tool=tool,
                                required=required, proposed=proposed))

        elif c["kind"] == "vague":
            sinks.append(dict(kind="vague_adjective", term=c["term"],
                              subject=c.get("subject"),
                              reason="vague adjective with no measurable property",
                              ask=f"Which exact property should change "
                                  f"('{c['clause']}') and by how much?"))
        elif c["kind"] == "unlowerable":
            sinks.append(dict(kind="unlowerable", reason=c["reason"],
                              surface=c.get("surface"), unit=c.get("unit"),
                              artifact=c.get("artifact"),
                              ask="State a value in a unit with an exact "
                                  "conversion to the base unit, naming the "
                                  "artifact to change."))
        elif c["kind"] == "unresolvable_negation":
            sinks.append(dict(kind="unresolvable_negation",
                              reason=c["reason"], object=c.get("object"),
                              ask="Name the concrete artifact that is forbidden."))

    unit, verdicts = "UNVERIFIABLE", []
    if claims:
        unit, results, _trace = claim_ir.dispose_claims(claims)
        verdicts = [dict(source=cl.source_text, verdict=v, why=w)
                    for cl, v, w in results]
    return {"claims": [cl.to_dict() for cl in claims], "verdicts": verdicts,
            "unit": unit, "sinks": sinks, "lowered": lowered, "halt": halt}


def _claim(a, b, source, line):
    """One closed claim: constants only. The model never appears here."""
    return claim_ir.eq(T.const(a), T.const(b), line=line, is_premise=False,
                       source=source)


def lower_prompt(prompt: str, proposal: dict, bind: dict | None = None,
                 proposer=None, dialect: dict | None = None) -> dict:
    """The composition: scope -> extract -> lower. The full lowering trace."""
    sc = scope(prompt, proposer=proposer)
    commits = extract(sc)
    ir = lower(commits, proposal, bind, dialect=dialect)
    return {"scope": sc, "commitments": commits, "ir": ir,
            "halt": ir.get("halt")}