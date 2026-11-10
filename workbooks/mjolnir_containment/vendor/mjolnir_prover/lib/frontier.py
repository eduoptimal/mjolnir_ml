"""mjolnir_prover.lib.frontier — size the ABSTENTION POPULATION, not a hand-picked set.

`theory_extensions_v3.md` orders the rungs from a corpus of 13 tasks it already
reports on (4 `subscript`, 3 `mutation`, 3 `method`, 3 `boolop`). That is a
sample, not the population. This module measures the population: every
`out_of_fragment` abstention the gate emits over a corpus is classified by the
construct that blocks it, so a rung is funded on the count of instances it
would move rather than on the count of tasks someone happened to write down.

The classifier is deliberately conservative and transparent — it is a *triage*
over raw claim text, not a decision procedure. It over-counts candidates: a
claim tagged `subscript` may still be undecidable for other reasons. It never
under-counts silently, because every untagged claim is reported in `unclassified`
and the classes are meant to be read as upper bounds.

Read the output as "instances of this construct exist in the abstention
population", and treat a rung's ROI as that number divided by its build cost.
"""
from __future__ import annotations

import re
from collections import Counter

from mjolnir_prover.lib import countability as CT

# Ordered: the first match wins, so the classes are exclusive and stable.
# Order matters and is not arbitrary: floats before attribute (otherwise `3.14`
# reads as a field access), subscript before call (`d.get(k)[0]` is blocked by
# the index, not the call), attribute before the generic call (so `self.x` does
# not read as a method), container literals before setops.
_CLASSES = (
    # --- the closure rungs under consideration ---
    ("subscript", "subscript",
     re.compile(r"[A-Za-z_][A-Za-z_0-9_]*\s*\[")),
    ("method", "method",
     re.compile(r"\.\w+\s*\(")),
    ("boolop", "boolop",
     re.compile(r"\b(?:and|or|not)\b")),
    ("mutation", "mutation",
     re.compile(r"\b(?:append|extend|insert|remove|pop|sort|update|setdefault)\s*\(")),
    # --- other blocking constructs, so the backlog is complete ---
    ("call", "calls",
     re.compile(r"[A-Za-z_][A-Za-z_0-9_]*\s*\(")),
    ("float", "floats",
     re.compile(r"\d+\.\d+|\d+e-?\d+")),
    ("attribute", "attributes",
     re.compile(r"\.\w+")),
    ("container_literal", "containers",
     re.compile(r"(?:==|!=)\s*[\[\{\(]|^\s*[\[\{\(]")),
    ("setop", "setops",
     re.compile(r"[|&^]|\bset\(|\bfrozenset\(")),
    ("lambda", "lambdas",
     re.compile(r"\blambda\b")),
    ("comprehension", "comprehensions",
     re.compile(r"\bfor\b|\bwhile\b")),
    ("modulo", "modulo",
     re.compile(r"%")),
    ("assignment", "mutation",
     re.compile(r"(?<![=!<>])[-+*/%&|^]?=(?!=)")),
    ("string", "strings",
     re.compile(r"['\"]")),
    # residual: a plain comparison between two names or expressions. Often a
    # TEST ORACLE (`dist == expected`, `theo == act`) — an assertion against a
    # computed answer rather than a mathematical statement. No amount of
    # algebra decides these, and they are the single biggest reason the
    # abstention class will not shrink by adding rungs.
    ("equality", "none (oracle-shaped)",
     re.compile(r"==|!=|<=|>=|<|>")),
)


def classify_claim(text: str) -> tuple[str, str]:
    """(class, implied rung) for one raw claim, or ('unclassified', 'unknown')."""
    if not text or text == "?":
        return "unclassified", "unknown"
    for name, rung, rx in _CLASSES:
        if rx.search(text):
            return name, rung
    return "unclassified", "unknown"


def abstentions(certs) -> list[dict]:
    """Every placeholder abstention row: the population under triage."""
    out = []
    for c in certs:
        unit = c.get("unit", {})
        for o in c.get("obligations", []):
            if CT.is_placeholder(o):
                out.append({
                    "unit": f"{unit.get('path','')}:{unit.get('name','')}",
                    "line": o.get("line"),
                    "status": c.get("status"),
                    "claim": o.get("premise", ""),
                })
    out.sort(key=lambda r: (r["unit"], r["line"] if r["line"] is not None else -1,
                            r["claim"]))
    return out


# --------------------------------------------------------------------------- #
# The second axis: is this claim decidable BY ANY RUNG?
#
# The construct axis above answers "which operator is missing". This one answers
# the prior question: "could any operator decide this at all?". Most of the
# population cannot be, and the reason is not a gap in the term algebra — the
# claim is about a value that only exists at runtime.
# --------------------------------------------------------------------------- #

# Names that are decidable operations, not operands. A claim consisting only of
# these and literals has nothing unknown in it.
_OPS = {"len", "sorted", "sum", "repr", "list", "set", "str", "abs", "min", "max",
        "int", "float", "bool", "tuple", "dict", "round", "all", "any", "type",
        "range", "frozenset", "reversed", "hex", "sqrt", "pow", "divmod"}
_WORD = re.compile(r"[A-Za-z_]\w*")
_UNIT_CACHE: dict = {}


def reset_cache() -> None:
    """Drop the parsed-unit cache.

    The cache is keyed by `path:name`, so a file that changes on disk keeps its
    stale symbols until this is called. Production runs read a fixed tree and
    never need it; anything that rewrites a source between calls must.
    """
    _UNIT_CACHE.clear()


def _unit_symbols(unit: str):
    """(params, literal-bound names, is_test, declared premises) for
    `path:name`, cached per file. Premises are the asserts in the unit's own
    top-level body, EXCLUDING the claim's own text."""
    if unit in _UNIT_CACHE:
        return _UNIT_CACHE[unit]
    import ast
    path, _, name = unit.rpartition(":")
    out = None
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except Exception:
        tree = None
    if tree is not None:
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if fn is not None:
            params = {a.arg for a in fn.args.args}
            lits = set()
            for n in ast.walk(fn):
                if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                        and isinstance(n.targets[0], ast.Name) \
                        and isinstance(n.value, (ast.List, ast.Dict)):
                    lits.add(n.targets[0].id)
            prems = []
            for st in fn.body:                    # top level only, ordered
                if isinstance(st, ast.Assert) and isinstance(st.test, ast.Compare) \
                        and len(st.test.ops) == 1:
                    try:
                        prems.append(ast.unparse(st.test))
                    except Exception:
                        pass
            out = (params, lits, name.startswith("test_"), prems)
    _UNIT_CACHE[unit] = out
    return out


_EQ = re.compile(r"^([^=<>!]+)==(.*)$")
_NUM = re.compile(r"-?\d+(\.\d+)?$")


def _split_eq(text: str):
    m = _EQ.match(text)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def premise_contradiction(claim: str, premises) -> str | None:
    """A premise asserting `f(...) == k1` while the claim asserts `f(...) == k2`.

    Sound because the premise is an ASSERT in the unit: the runtime enforces it.
    If it holds, the claim cannot, so the two asserts are mutually inconsistent
    and the unit is refuted rather than undecided. No shape abstraction is used
    — the LHS expressions must be textually identical, because an abstraction
    that ignores arguments turns `abs_val(34) == 34` and `abs_val(0) == 0` into
    a "contradiction" that is not one.
    """
    cs = _split_eq(claim)
    if not cs:
        return None
    cl, cr = cs
    if not _NUM.fullmatch(cr):
        return None
    for p in premises:
        ps = _split_eq(p)
        if not ps:
            continue
        pl, pr = ps
        if pl == cl and _NUM.fullmatch(pr) and pr != cr:
            return f"premise asserts {pl} == {pr}, claim asserts {cl} == {cr}"
    return None


def decidability(row: dict) -> tuple[str, str]:
    """(class, explanation) for one abstention, independent of its construct.

      premise        — a declared premise contradicts the claim: the unit's own
                       asserts are inconsistent, which is decidable without any
                       value for the operand
      decidable      — every operand has a value the checker can know
      runtime_state  — an operand is a unit PARAMETER: a value that only exists
                       at run time. No rung decides this; it is a scope limit,
                       not a capability gap.
      oracle         — the claim compares against a computed answer
                       (`dist == expected`); an assertion about the execution
                       rather than about the program text.

    Order matters: the premise rule runs FIRST, because a contradiction decides
    the claim regardless of whether its operand has a known value. That is the
    whole point of crediting premises — it is the one route by which a claim
    about an unknown value is still decidable.
    """
    claim, unit = row.get("claim", ""), row.get("unit", "")
    syms = _unit_symbols(unit)
    params, lits, is_test, prems = syms if syms else (set(), set(), False, [])
    prems = [p for p in prems if p != claim]
    contra = premise_contradiction(claim, prems)
    if contra:
        # NOT framework-addressable. This is a contradiction between two RAW
        # ASSERTS, but `assert len(G) == 4` is not a premise under the
        # framework: it is an opaque-atom claim mjolnir itself abstains on. Of
        # the instances this class catches, ZERO have a premise mjolnir records,
        # so no table-governed rule can discharge them today. Reporting them as
        # addressable would quote a number nothing can deliver — the same error
        # the rung ROI made. The directive exists and ships disabled
        # (classification.json -> premise_contradiction); this class is the
        # measurement that justified leaving it disabled.
        return ("premise_broader", contra +
                " — but the premise is not framework-declared (mjolnir "
                "abstains on it), so no directive can use it")
    names = {n for n in _WORD.findall(claim)} - _OPS
    # The ONLY other way a claim is addressable is if every operand is bound to
    # a literal the unit itself assigns. A parameter is not a binding — it is
    # the absence of one, which is why it has to be excluded from `lits` rather
    # than treated as known.
    unbound = {n for n in names if n not in lits}
    if unbound:
        if unbound & params:
            return ("runtime_state",
                    "operand is a unit parameter: no rung can decide it")
        return ("runtime_state",
                "operand is never bound to a known value in the unit")
    if is_test and re.search(r"==|!=", claim):
        return ("oracle",
                "all operands known, but the claim asserts a computed answer")
    return ("decidable", "every operand is bound to a known value")


def population(certs) -> dict:
    """The measured rung backlog: instances per blocking construct.

    `distinct_claims` counts unique claim TEXTS (the corpus repeats identical
    assertions across many units — 2148 rows can be far fewer distinct
    problems). Both are reported because the product cares about instances
    ("this control class, N instances, not certified") and the science cares
    about distinct shapes.
    """
    rows = abstentions(certs)
    by_class = Counter()
    claims_by_class = {}
    by_dec = Counter()
    dec_instances: dict[str, set] = {}
    for r in rows:
        cls, rung = classify_claim(r["claim"])
        by_class[cls] += 1
        claims_by_class.setdefault(cls, set()).add(r["claim"])
        dec, _why = decidability(r)
        by_dec[dec] += 1
        # a rung may only claim instances it could actually move
        if dec in ("decidable", "premise", "premise_broader"):
            dec_instances[cls] = dec_instances.get(cls, 0) + 1
    classes = sorted(by_class, key=lambda k: (-by_class[k], k))
    return {
        "population": len(rows),
        "distinct_claims": len({r["claim"] for r in rows}),
        "units": len({r["unit"] for r in rows}),
        "by_class": [
            {"class": c,
             "rung": classify_claim(next(iter(claims_by_class[c])))[1]
                     if c != "unclassified" else "unknown",
             "instances": by_class[c],
             "distinct_claims": len(claims_by_class[c]),
             # instances a rung of this type could ACTUALLY move
             "addressable": dec_instances.get(c, 0)}
            for c in classes
        ],
        "by_decidability": [
            {"class": d, "instances": by_dec[d],
             "distinct_claims": len({r["claim"] for r in rows
                                     if decidability(r)[0] == d})}
            for d in sorted(by_dec, key=lambda k: -by_dec[k])
        ],
        "ceiling": {
            "decidable_instances": by_dec.get("decidable", 0),
            "note": "no rung moves an instance outside `decidable`; a coverage "
                    "target must be stated over this subset or it is "
                    "unreachable at any capability",
        },
    }


def _roadmap_costs() -> dict:
    from . import theory as TH
    return dict(TH.load("roadmap")["costs"])


def rung_roi(certs, costs=None) -> list[dict]:
    """Rank the candidate rungs by ADDRESSABLE instances per unit of build cost.

    This is the ROI that does not lie. Ranking on raw instance counts is wrong:
    96.9% of the `method`/`call` population is blocked by an operand that only
    exists at runtime, so no rung moves it however the rung is built. A rung may
    only claim the instances a rung of its type could actually decide, so the
    `instances` column here is the *addressable* count and `blocked` is what the
    rung would leave on the table.

    `costs` is the v3 plan's qualitative cost, mapped to small=2, medium=3,
    large=5. A crude number on purpose: the point is to make the ordering
    *derived from a measurement* rather than asserted from a 13-task sample.
    """
    pop = population(certs)
    # `costs` is keyed by RUNG; the population is keyed by CLASS. Several classes
    # can imply the same rung, so aggregate the class counts per rung — mixing
    # the two vocabularies is what made this silently return nothing.
    by_rung_addr: dict[str, int] = {}
    by_rung_total: dict[str, int] = {}
    for c in pop["by_class"]:
        rung = c["rung"]
        if rung == "unknown" or rung.startswith("none"):
            continue
        by_rung_addr[rung] = by_rung_addr.get(rung, 0) + c["addressable"]
        by_rung_total[rung] = by_rung_total.get(rung, 0) + c["instances"]
    if costs is None:
        # POLICY is DATA (theories/roadmap.json), not a dict written here.
        # Hard-coding it in the analysis module is exactly the move the tables
        # exist to prevent: the judgement becomes invisible and unhashed.
        costs = _roadmap_costs()
    rows = [{"rung": r,
             "instances": by_rung_addr.get(r, 0),
             "blocked": by_rung_total.get(r, 0) - by_rung_addr.get(r, 0),
             "cost": costs.get(r, 3)}
            for r in costs if by_rung_total.get(r, 0)]
    for r in rows:
        r["instances_per_cost"] = (r["instances"] / r["cost"]) if r["cost"] else None
    rows.sort(key=lambda r: (-(r["instances_per_cost"] or 0), r["rung"]))
    return rows
