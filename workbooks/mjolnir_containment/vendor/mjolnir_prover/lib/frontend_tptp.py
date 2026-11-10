"""mjolnir_prover.lib.frontend_tptp — ILTP/TPTP propositional problems as claims.

Shape follows the sibling frontends (`frontend_sql.py`, `frontend_lean2.py`):
parse a declarative text into premise/claim pairs with provenance and a trust
story. The difference is structural and deliberate:

    a mined candidate carries `canonical_unit` — a PYTHON function of asserts,
    which then goes through the arithmetic certifier.

That does not work here. A propositional theorem is not arithmetic: lowered to
`assert p & q`, the arithmetic fragment abstains on all 274 problems. So this
frontend emits **problems** (axioms + conjecture) for the propositional
decision procedure instead of Python wrappers, and the certificate is built
directly. Same provenance discipline, different target theory.

Trust story: ILTP is a BENCHMARK, not a runtime-enforced contract. The axioms
are not "the database guaranteed this on every write"; they are premises of a
problem, and the benchmark supplies the Theorem/Non-Theorem label. That label is
the point — it is the first corpus in this stack whose verdicts can be scored by
something we did not write.
"""
from __future__ import annotations

import os
import re
import sys
import zlib

from . import term as T

_CID = lambda *p: "tptp::" + format(
    zlib.crc32("|".join(str(x) for x in p).encode()) & 0xffffffff, "08x")


# ------------------------------------------------------------- file scan ---
def formulas(text: str):
    """Yield (name, role, formula_text) for every `fof(...)` in `text`.

    Balanced-paren scanning, not a regex: TPTP formulas nest arbitrarily and a
    regex gives up on exactly the problems worth reading.
    """
    i = 0
    while True:
        i = text.find("fof(", i)
        if i < 0:
            return
        j = i + 4

        def endarg(k: int, depth: int = 0) -> int:
            while k < len(text):
                c = text[k]
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth == 0:
                        return k
                    depth -= 1
                elif c == "," and depth == 0:
                    return k
                k += 1
            return k

        k = endarg(j)
        name = text[j:k].strip()
        j = k + 1
        k = endarg(j)
        role = text[j:k].strip()
        j = k + 1
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        if j >= len(text) or text[j] != "(":
            return
        d = 0
        s = j
        while j < len(text):
            if text[j] == "(":
                d += 1
            elif text[j] == ")":
                d -= 1
                if d == 0:
                    break
            j += 1
        yield name, role, text[s + 1:j]
        i = j + 1


def strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("%"))


# ------------------------------------------------------------- the lexer ---
_TOK = re.compile(r"\s*(<=>|=>|[&|~()!?,]|\$true|\$false|[A-Za-z_][A-Za-z_0-9]*)")
_TERM = {"&": "pand", "|": "por", "~": "pnot", "=>": "pimp",
         "<=>": "piff", "!": "pforall", "?": "pexists"}


class _ParseError(ValueError):
    pass


# Structural sharing. `ciff(a, b)` mentions a and b TWICE each, so building
# plain tuples duplicates both operands once per biconditional: a 3.5 KB ILTP
# file with 28 `<=>` expanded to 1.07 BILLION nodes (measured — it is 2^30, the
# doubling, exactly). Parsing is linear; the explosion was entirely in
# construction.
#
# Interning on (op, id-of-children) makes those shared references share one
# object, turning the expansion back into a DAG of linear size. Keying on id()
# rather than on the child tuples keeps every lookup O(1) — hashing a deep tuple
# would reintroduce the cost we are removing.
_INTERN: dict = {}


def _reset_intern():
    _INTERN.clear()


def _mk(op, *kids):
    """Interned constructor — one object per distinct subformula."""
    key = (op,) + tuple(id(k) for k in kids)
    got = _INTERN.get(key)
    if got is None:
        got = (op,) + kids
        _INTERN[key] = got
    return got


def _mkand(a, b):   return _mk("pand", a, b)


def _mkor(a, b):    return _mk("por", a, b)


def _mknot(a):      return _mk("pnot", a)


def _lex(s: str):
    out, i = [], 0
    while i < len(s):
        m = _TOK.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise _ParseError(f"lex:{s[i]!r}")
        out.append(m.group(1))
        i = m.end()
    return out


def parse_formula(s: str):
    """TPTP propositional formula text -> a term from the boolop rung.

    Precedence, loosest first:  <=>   then  =>   then  |   then  &   then  ~
    Quantifiers `!`/`?` raise: they are outside the propositional theory, and
    guessing would be unsound.
    """
    # ILTP nests deeply (hundreds of levels) and this descent costs ~12 frames
    # per level, so the DEFAULT 1000-frame limit overflows well before a legal
    # problem. Raise it for the parse and restore it after; a formula that STILL
    # overflows returns None via the RecursionError guard, i.e. abstains. A
    # corpus walk must never die on one pathological file.
    _prev_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(_prev_limit, 20000))
    try:
        return _parse_formula_inner(s)
    except RecursionError:
        raise _ParseError("too_deep")
    finally:
        sys.setrecursionlimit(_prev_limit)


def _parse_formula_inner(s: str):
    toks = _lex(s)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        t = toks[pos]
        pos += 1
        return t

    def iff():
        left = imp()
        while peek() == "<=>":
            take()
            left = T.ciff(left, imp())
        return left

    def imp():
        left = orr()
        if peek() == "=>":
            take()
            _r = orr()
            return _mkor(_mknot(left), _r)
        return left

    def orr():
        left = andd()
        while peek() == "|":
            take()
            left = _mkor(left, andd())
        return left

    def andd():
        left = un()
        while peek() == "&":
            take()
            left = _mkand(left, un())
        return left

    def un():
        if peek() in ("~", "!", "?"):
            op = take()
            if op in ("!", "?"):
                raise _ParseError("quantified: outside the propositional theory")
            return _mknot(un())
        return atom()

    def atom():
        tok = take()
        if tok == "(":
            inner = iff()
            if peek() != ")":
                raise _ParseError("unbalanced")
            take()
            return inner
        if tok == "$true":          # TPTP's boolean constants
            return ("c", 1)
        if tok == "$false":
            return ("c", 0)
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tok):
            return T.catom(tok)
        raise _ParseError(f"atom:{tok}")

    out = iff()
    if pos != len(toks):
        raise _ParseError(f"trailing:{toks[pos:]}")
    return out


# ------------------------------------------------------------- the mining ---
def _includes(text: str):
    """Yield the `include('path')` targets in `text`."""
    out, i = [], 0
    while True:
        i = text.find("include(", i)
        if i < 0:
            return out
        j = text.find("'", i)
        k = text.find("'", j + 1) if j >= 0 else -1
        if j < 0 or k < 0:
            return out
        out.append(text[j + 1:k].strip())
        i = k + 1


def load_includes(text: str, base_dir: str, *, max_files: int = 64):
    """-> (texts, resolved, unresolved).

    THE REASON THE ILTP SCORE WAS ZERO, AND IT IS WORTH NAMING PRECISELY.

    An ILTP problem file is a `conjecture` and a list of `include('Axioms/X.ax')`
    lines. This parser read the `fof(...)` entries and stopped. So every one of the
    2,670 first-order problems arrived with ZERO axioms -- and with no axioms,
    `axioms |= conjecture` is vacuously TRUE for every conjecture in existence.

    That is a false-certificate factory, not a coverage gap. Had the decision
    procedure been reachable from the empty-axiom case, this module would have
    certified 2,670 theorems it had read none of the premises for. The guard is
    in `propositional.decide` as well, but the fix belongs here: a parser that
    silently drops the premises of a proof obligation is worse than one that
    refuses the file.
    """
    texts, resolved, unresolved, seen = [], 0, [], set()
    stack = [(base_dir, p) for p in _includes(text)]
    while stack and len(texts) < max_files:
        d, rel = stack.pop(0)
        path = os.path.normpath(os.path.join(d, rel))
        if path in seen:
            continue
        seen.add(path)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError:
            unresolved.append(rel)
            continue
        resolved += 1
        texts.append(body)
        stack.extend((os.path.dirname(path), p) for p in _includes(body))
    for rel in stack:
        unresolved.append(rel)
    return texts, resolved, unresolved


def theory_ceiling(text: str, base_dir: str | None = None) -> str | None:
    """-> None if this problem is IN this theory, else the reason it is not.

    MEASURED, NOT ASSUMED. Over the 2,670 ILTP first-order problems:

        quantifier anywhere in the include chain      0.0%
        EQUALITY anywhere in the include chain       87.1%
        neither -- i.e. decidable here               12.9%

    So the reachable ceiling on ILTP first-order with a propositional decision
    procedure is 345 problems, and no amount of parser work changes that. The
    binding constraint is the THEORY, not the reader -- which is the opposite of
    what the 0-decisions measurement suggested, and it took resolving the include
    chains to see.

    Equality is the whole story. `a = b` is not an opaque atom; a theory that
    treats it as one is not proving what the problem asks, so a certificate from
    such a run would be a certificate about a different problem. Abstaining is the
    correct answer and it is the reason this function exists.

    `[` appears alongside equality in TPTP for bracketed term lists and is
    treated the same way: not this theory's syntax.
    """
    body = strip_comments(text)
    if base_dir:
        texts, _r, _u = load_includes(body, base_dir)
        body = body + "\n" + "\n".join(texts)
    import re as _re
    if _re.search(r"\b(?:forall|exists)\b", body):
        return "outside_theory:quantified"
    if _re.search(r"(?<![<>=!])=(?!=)", body):
        return "outside_theory:equality"
    if "[" in body:
        return "outside_theory:bracketed_terms"
    return None


def parse_problem(text: str, base_dir: str | None = None):
    """-> (axioms, conjecture, meta). Never returns None.

    WHY IT USED TO RETURN NONE
    --------------------------
    It returned `None` for 1,028 of 1,200 ILTP first-order problems -- 86% -- and
    did so SILENTLY. The caller could not distinguish "this file is not one
    conjecture" from "I could not read this language", and the first scorer to
    touch it unpacked the `None` and reported a `TypeError`, which reads like a
    crash in the parser rather than a coverage hole. A parser that gives up on
    86% of its input without saying so is worse than one that raises, because the
    number it produces looks like a measurement.

    So: no more None. Every outcome is a `meta` with an explicit `shape`, and the
    caller is expected to branch on it.
    """
    body = strip_comments(text)
    extra, resolved, unresolved = [], 0, []
    if base_dir:
        inc, resolved, unresolved = load_includes(body, base_dir)
        extra = inc
    ax, cn = [], []
    for chunk in [body] + extra:
        for name, role, f in formulas(chunk):
            if role == "axiom":
                ax.append(f)
            elif role == "conjecture":
                cn.append(f)
    meta = {"raw_axioms": len(ax), "includes_resolved": resolved,
            "includes_unresolved": unresolved[:3], "error": None,
            "shape": None}
    if len(cn) != 1:
        meta["shape"] = (f"conjectures:{len(cn)}")
        return [], None, meta
    if not ax:
        # NEVER an empty premise set. Entailment from nothing is vacuous, and a
        # vacuous certificate on 2,670 problems would look like a perfect score.
        meta["shape"] = "no_axioms"
        return [], None, meta
    try:
        _reset_intern()
        axioms = [parse_formula(a) for a in ax]
        conjecture = parse_formula(cn[0])
    except RecursionError:
        meta["shape"] = "recursion_limit"
        return [], None, meta
    except _ParseError as exc:
        meta["shape"] = f"parse_error:{exc}"
        return [], None, meta
    except Exception as exc:  # noqa: BLE001
        meta["shape"] = f"parse_error:{type(exc).__name__}"
        return [], None, meta
    meta["shape"] = "ok"
    return axioms, conjecture, meta


def iltp_labels(text: str) -> dict:
    """ILTP's own verdicts, kept SEPARATE by logic.

    A header can carry two statuses, and conflating them silently inverts the
    only external ground truth in the stack:

        % Status             : Theorem          <- CLASSICAL
        % Status (intuit.)   : Non-Theorem      <- INTUITIONISTIC

    LCL181 (Principia Mathematica 2.15) is exactly this case. Scoring classical
    reasoning against the intuitionistic label produced 42 apparent "false
    certificates" that were all this one mistake. So both are returned, named,
    and only the classical one is comparable to a two-valued decision procedure.

    `classical` is the ground truth. `intuitionistic` is recorded because it is
    a genuinely different theory, not because this module decides it.
    """
    out = {"classical": None, "intuitionistic": None}
    m = re.search(r"%\s*Status\s*:\s*([A-Za-z\-]+)", text)
    if m:
        out["classical"] = m.group(1).replace("-", "").lower()
    m = re.search(r"%\s*Status\s*\(\s*intuit\.\s*\)\s*:\s*([A-Za-z\-]+)",
                  text)
    if m:
        out["intuitionistic"] = m.group(1).replace("-", "").lower()
    return out


def iltp_label(text: str) -> str | None:
    """The CLASSICAL verdict — the one comparable to a two-valued procedure."""
    return iltp_labels(text)["classical"]


def mine_tptp_file(path: str, repo: str = "iltp", commit: str = "") -> list:
    """Mine one .p file into a decidable problem (or nothing, if out of model)."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    parsed = parse_problem(text)
    if parsed is None:
        return []
    axioms, conjecture, meta = parsed
    name = os.path.basename(path)
    return [dict(
        id=_CID(repo, commit, path), family="tptp_propositional",
        tier="claim", predicate="entailment",
        axioms=axioms, conjecture=conjecture,
        label=iltp_label(text),
        labels=iltp_labels(text),
        canonical_unit=None,          # NOT a python unit: see module docstring
        origin=dict(repo=repo, commit=commit, file=path, construct="tptp_fof",
                    line=0),
        provenance=dict(source="benchmark", confirmed=True,
                        trust="external_labels"),
    )]


def mine_tptp_repo(root, repo_name=None, commit=None) -> list:
    """Every problem under `root`, sorted for a deterministic manifest."""
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            if f.endswith(".p"):
                out += mine_tptp_file(os.path.join(dirpath, f),
                                      repo_name or "iltp", commit)
    return out