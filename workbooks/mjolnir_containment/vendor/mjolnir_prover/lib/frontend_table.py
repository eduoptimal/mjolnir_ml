"""mjolnir_prover.lib.frontend_table — stage 1+2 for MACHINE-READABLE sources.

A table row is the candidate; the column name is the referent; the cell is the
evidence. That is the whole of the referent resolution, and it is exact because
the source declares its own identifiers.

MEASURED (`tactics/table_pipeline.py`): 60/60 with 0 wrong and 0 unresolved
on a document with half its values MUTATED — so stage 1, stage 2 and stage 3 all
tracked the SOURCE rather than the ground truth. Stage 1 here is the SAME
extractor the prose path used; it did not need replacing. The source was the
variable.

SCOPE, and the reason prose is NOT in this module: on `sessions/*.md` the same
extractor reaches a 0.079 referent-resolution rate and the pipeline produces 0
certificates and 297 refutations. Prose is a DECLINED class, and the evidence
for declining it lives in `docs/analsysis/natural_language.md` S9 — not here,
so that the decline stays falsifiable.
"""
from __future__ import annotations

import re
from .disclosure import Claim

ROW = re.compile(r"^\s*\|(.+)\|\s*$")
KV = re.compile(r"^\s*([A-Za-z_][A-Za-z_0-9 .+-]*?)\s*[:=]\s*(-?[\d.,]+)\s*$")
NUMERIC = re.compile(r"-?[\d.,]+")
# CSV/TSV: a header naming the columns, then one row per observation. This is the
# format a first-time user is most likely to have, and it was DECLINED: the only
# readers were a markdown pipe-row and a `key: value` line. Found by walking the
# first-user path with the files a real user would actually have.
CSV_SPLIT = re.compile(r"[,\t;]")


def _split(line: str) -> list[str]:
    """Split one delimited row. Minimal quoting support (RFC4180 double quotes),
    because a quoted numeric is still a fact and a quoted key is still a
    referent."""
    out, cur, q = [], "", False
    for ch in line:
        if ch == '"':
            q = not q
        elif ch in ",\t;" and not q:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


def parse_csv(text: str, key_col: int | None = None,
              value_col: int | None = None) -> tuple[dict, dict]:
    """A delimited file -> ({quantity: value}, repeated).

    Column selection: with two columns they are key and value. With more, the
    LAST column is the value column and the first is the referent, UNLESS the
    caller names them -- and the caller being able to name them is the point,
    because a wide CSV has no single correct reading and guessing one silently
    would be a coverage number standing in for a decision.
    """
    rows = [l for l in text.splitlines() if l.strip()]
    if not rows:
        return {}, {}
    header = _split(rows[0])
    if len(header) < 2:
        return {}, {}
    kc = 0 if key_col is None else key_col
    vc = len(header) - 1 if value_col is None else value_col
    if not (0 <= kc < len(header)) or not (0 <= vc < len(header)):
        return {}, {}
    facts: dict = {}
    every: dict = {}
    for line in rows[1:]:
        cells = _split(line)
        if len(cells) <= max(kc, vc):
            continue
        k, v = cells[kc].strip(), cells[vc].strip()
        if not k or not NUMERIC.fullmatch(v):
            continue
        q = _ident(k)
        facts[q] = v
        every.setdefault(q, []).append(v)
    return facts, {q: vs for q, vs in every.items() if len(vs) > 1}


def parse_table(text: str) -> dict:
    """A markdown table or `key: value` block -> {quantity: rendered value}."""
    facts: dict = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            cells = [c.strip() for c in m.group(1).split("|")]
            if len(cells) == 2 and re.fullmatch(r"-?[\d.,]+", cells[1]):
                facts[_ident(cells[0])] = cells[1]
            continue
        m = KV.match(line)
        if m:
            facts[_ident(m.group(1))] = m.group(2)
    return facts


def _ident(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def extract(facts: dict, template: str = "{q} is {v}") -> list:
    """One candidate claim per declared quantity. The claim is a TEMPLATE over
    the source's own vocabulary — no generation, no paraphrase, nothing to
    hallucinate, because the claim cannot say anything the source does not."""
    out = []
    for q, v in facts.items():
        c = Claim(text=template.format(q=q.replace("_", " "), v=v),
                  quantity=q, value=str(v).strip().replace(",", ""))
        out.append(c)
    return out


def read(path: str) -> dict:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return parse_table(fh.read())


def parse_all(text: str) -> tuple[dict, dict]:
    """Return `(facts, repeated)` where `repeated[q]` is EVERY value stated for
    `q`, in document order.

    `parse_table` returns a dict, so a referent stated twice keeps only the LAST
    value. That is not merely lossy: a claim asserting the dropped value is
    REFUTED against a document that states it. Measured, not hypothesised --
    so the conflict is now REPORTED rather than resolved silently, and a
    relation-valued source is available to callers that want the whole set.
    """
    facts: dict = {}
    every: dict = {}
    for line in text.splitlines():
        m = ROW.match(line)
        pair = None
        if m:
            cells = [c.strip() for c in m.group(1).split("|")]
            if len(cells) == 2 and re.fullmatch(r"-?[\d.,]+", cells[1]):
                pair = (_ident(cells[0]), cells[1])
        if pair is None:
            m = KV.match(line)
            if m:
                pair = (_ident(m.group(1)), m.group(2))
        if pair is None:
            continue
        q, v = pair
        facts[q] = v                      # LAST wins, exactly as parse_table
        every.setdefault(q, []).append(v)
    repeated = {q: vals for q, vals in every.items() if len(vals) > 1}
    return facts, repeated


def read_all(path: str) -> tuple[dict, dict]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    # A CSV/TSV is tried ONLY when it does not read as a markdown table or a
    # key: value block. Order matters: a pipe-table must never be re-read as
    # delimited, or the reader would be guessing between two readings of the
    # same file and silently picking one.
    facts, repeated = parse_all(text)
    if facts:
        return facts, repeated
    return parse_csv(text)
