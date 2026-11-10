"""mjolnir_prover.lib.frontend_sql — SQL DDL frontend: schema constraints as premises.

SQL CHECK/NOT NULL/UNIQUE constraints are runtime-enforced input contracts —
the same trust class as code guards, in declarative text. This frontend
lowers them into the canonical claim format (same IR as python/java).

Shapes handled (honest MVF):
  CHECK (col > N) / (col >= N) / (<, <=)         -> premise  col OP N
  CHECK (col BETWEEN a AND b)                    -> premise  a <= col <= b
  CHECK (col <> 0) / (col != 0)                  -> premise  col != 0
  NOT NULL on col                                -> monitor  (non-numeric)
  PRIMARY KEY / UNIQUE                           -> premise  col unique (monitor)
  CHECK (end > start)                            -> premise  param-param pair

Trust story: the database enforced these on every write for the schema's
lifetime — battle-tested natural labels, in clean declarative syntax.
"""
from __future__ import annotations
import os, re, zlib

_CID = lambda *parts: "sq::" + format(zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xffffffff, "08x")

# table alias for cross-table refs (o.amount -> amount): kept as the column name
_COL = r"([A-Za-z_][A-Za-z0-9_]*)"
_NUM = r"(-?\d+(?:\.\d+)?)"


_CREATE_TABLE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:GLOBAL\s+|LOCAL\s+)?"
    r"(?:TEMPORARY\s+|TEMP\s+|TRANSIENT\s+|VOLATILE\s+)?TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:IDENTIFIER\s*\(\s*:?([\w]+)\s*\)|[`\"\[]?([\w.]+)[`\"\]]?)", re.I)


def _extract_tables(ddl: str):
    """Yield (table_name, body) using BALANCED-paren scanning — dialect-proof
    (works for T-SQL GO batches, Oracle, Snowflake `CREATE OR REPLACE TEMPORARY
    TABLE IDENTIFIER(:var)`, multi-line constraints)."""
    seen_tables = set()
    for m in _CREATE_TABLE.finditer(ddl):
        name = (m.group(1) or m.group(2) or "").split(".")[-1].strip('`"')
        if not name or name.lower() in seen_tables:
            continue
        seen_tables.add(name.lower())
        i = m.end()
        while i < len(ddl) and ddl[i] != "(":
            i += 1
        depth, j = 0, i
        while j < len(ddl):
            if ddl[j] == "(":
                depth += 1
            elif ddl[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        # the body's ABSOLUTE start: line attribution needs to know where in the
        # file this table lives, not just what its columns are.
        yield name, ddl[i + 1:j], i + 1


def _table_index(ddl: str):
    """{table_lower: (body, 1-based line of its first body line)} — built once
    per file so every claim can be located without rescanning."""
    idx = {}
    for name, body, start in _extract_tables(ddl):
        idx.setdefault(name.lower(),
                       (body, ddl.count("\n", 0, start) + 1))
    return idx


def _col_line(index, table, column) -> int:
    """1-based line where `column` of `table` is declared, else 0.

    Two passes: the declaration line (`NAME <type>` at a line start), then any
    mention of the column in that table's body. Line 0 only when the column is
    not in the body at all — which would itself be a parse gap worth seeing
    rather than a number a reviewer has to interpret.
    """
    entry = index.get((table or "").lower())
    if not entry:
        return 0
    body, base = entry
    m = re.match(r'^[ \t]*"?%s"?[ \t]+\S' % re.escape(column), body,
                 re.I | re.M)
    if not m:
        m = re.search(r"\b%s\b" % re.escape(column), body, re.I)
    if not m:
        return 0
    return base + body.count("\n", 0, m.start())


def _expr_line(ddl: str, expr: str) -> int:
    """1-based line of a CAST expression, matched on a distinctive slice."""
    key = " ".join((expr or "").split())[:40]
    if not key:
        return 0
    for n, raw in enumerate(ddl.splitlines(), 1):
        if key in " ".join(raw.split()):
            return n
    return 0


def _split_cols(body: str):
    """Split a table body on commas at paren-depth 0 (so NUMERIC(10,0) survives)."""
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


_CONSTRAINT_LINE = re.compile(
    r"^\s*(CONSTRAINT|PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK|KEY|INDEX)\b", re.I)
_COL_DECL = re.compile(
    r"^\s*[`\"\[]?([A-Za-z_]\w*)[`\"\]]?\s+([A-Za-z][A-Za-z0-9_]*)"
    r"\s*(?:\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\))?", re.I)


_KNOWN_TYPES = {
    "STRING", "VARCHAR", "VARCHAR2", "CHAR", "CHARACTER", "TEXT", "CLOB",
    "NUMERIC", "DECIMAL", "NUMBER", "MONEY", "INT", "INTEGER", "BIGINT",
    "SMALLINT", "TINYINT", "INT2", "INT4", "INT8", "FLOAT", "FLOAT4", "FLOAT8",
    "DOUBLE", "REAL", "BOOLEAN", "BOOL", "DATE", "DATETIME", "TIMESTAMP",
    "TIMESTAMP_NTZ", "TIMESTAMP_TZ", "TIMESTAMP_LTZ", "TIME", "VARIANT",
    "OBJECT", "ARRAY", "BINARY", "VARBINARY", "BYTEA", "UUID", "JSON", "JSONB",
    "GEOGRAPHY", "GEOMETRY", "BYTEINT", "SERIAL", "BIGSERIAL",
}


def parse_columns_all(ddl: str):
    """Declared columns: (table, col, TYPE, precision, scale, not_null) from every
    table body — `scale` is the second NUMERIC(p,s) argument when present, else
    None; `not_null` reflects an explicit NOT NULL on the declaration. Skips
    constraint lines (PK/FK/CHECK/UNIQUE) and any `NAME X` whose X is not a known
    SQL type (the body scan can over-capture in long procedures)."""
    out = []
    for table, body, _start in _extract_tables(ddl):
        for part in _split_cols(body):
            if _CONSTRAINT_LINE.match(part):
                continue
            m = _COL_DECL.match(part)
            if not m:
                continue
            typ = m.group(2).upper()
            if typ not in _KNOWN_TYPES:
                continue
            not_null = bool(re.search(r"\bNOT\s+NULL\b", part, re.I))
            out.append((table, m.group(1), typ, m.group(3), m.group(4),
                        not_null))
    return out


def _find_checks(body: str):
    """Yield CHECK(...) expressions from a table body (balanced, case-insens)."""
    for cm in re.finditer(r"CHECK\s*\(", body, re.I):
        i = cm.end()
        depth, j = 1, i
        while j < len(body) and depth > 0:
            if body[j] == "(":
                depth += 1
            elif body[j] == ")":
                depth -= 1
            j += 1
        yield body[i:j - 1]


def _numeric_comparisons(expr: str):
    """Comparison pairs inside a constraint expression: numeric (param vs
    const) and param-vs-param (e.g. end_date > start_date — premisable)."""
    out = []
    for m2 in re.finditer(
            rf"{_COL}\s*(>=|<=|<>|!=|=|>|<)\s*{_NUM}|{_NUM}\s*(>=|<=|<>|!=|=|>|<)\s*{_COL}",
            expr):
        if m2.group(1):
            out.append((m2.group(1), m2.group(2), m2.group(3)))
        else:
            out.append((m2.group(5), m2.group(4), m2.group(6)))
    for m3 in re.finditer(rf"{_COL}\s+BETWEEN\s+{_NUM}\s+AND\s+{_NUM}", expr, re.I):
        out.append((m3.group(1), ">=", m3.group(2)))
        out.append((m3.group(1), "<=", m3.group(3)))
    # param-vs-param comparisons (both names, no literal)
    for m4 in re.finditer(rf"{_COL}\s*(>=|<=|<|>)\s*{_COL}\b", expr):
        a, o, b = m4.group(1), m4.group(2), m4.group(3)
        if a != b:
            out.append((a, o, b))
    return out


def parse_checks_all(ddl: str):
    """All (table, col, op, val) from CHECK constraints anywhere in the DDL —
    table-level, column-level, ALTER TABLE ADD CONSTRAINT, GO-batched T-SQL."""
    out = []
    seen_keys = set()
    for table, body, _start in _extract_tables(ddl):
        for expr in _find_checks(body):
            for col, op, val in _numeric_comparisons(expr):
                out.append((table, col, op, val))
    # ALTER TABLE ... ADD CONSTRAINT ... CHECK (...) — and Oracle ADD (...)
    for am in re.finditer(r"ADD\s+(?:CONSTRAINT\s+[\w\[\]\.]+\s*)?CHECK\s*\(", ddl, re.I | re.S):
        i = am.end()
        depth, j = 1, i
        while j < len(ddl) and depth > 0:
            if ddl[j] == "(":
                depth += 1
            elif ddl[j] == ")":
                depth -= 1
            j += 1
        expr = ddl[i:j - 1]
        # table context: the preceding ALTER TABLE <name>
        tm = list(re.finditer(r"ALTER\s+TABLE\s+[`\"\[]?([\w.\[\]]+)", ddl[:am.start()], re.I))
        table = tm[-1].group(1).split(".")[-1].strip('`"[]') if tm else "unknown"
        for col, op, val in _numeric_comparisons(expr):
            key = (table, col, op, str(val))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append((table, col, op, val))
    return out


def parse_col_checks(ddl: str):
    return parse_checks_all(ddl)      # column-level CHECKs are inside the body scan


def parse_checks(ddl: str):
    return parse_checks_all(ddl)


def canonical_premise(col, op, val):
    """SQL operator -> canonical premise text (the engine's IR)."""
    op = {"<>": "!=", "=": "=="}.get(op, op)
    return f"assert {col} {op} {val}"


def wrapper_unit(fn_name, col, op, val):
    return f"def {fn_name}({col}):\n    {canonical_premise(col, op, val)}\n"


_NUM_T = ("NUMERIC", "DECIMAL", "NUMBER")
_CAST_RE = re.compile(
    r"CAST\s*\(\s*([^()]{1,60}?)\s+AS\s+(NUMERIC|DECIMAL|NUMBER)\s*"
    r"\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)\s*\)", re.I | re.S)


def decimal_claims(ddl: str, rel: str, repo: str, commit: str) -> list:
    """The DECIMAL rung, lowered — prover unchanged (see banking_audit.md).

    integer-at-scale: a column of scale `s` is exact at scaled integers, so the
    *lossless cast* `v == k / 10^s` under the scale premise `v * 10^s == k` is a
    theorem the fragment already proves. Rounding-bearing `CAST`s are emitted as
    **monitors** (declared, never certified) — the guardrail against a silent
    false approval on money paths."""
    out = []
    tidx = _table_index(ddl)
    for t, c, typ, p, s, _nn in parse_columns_all(ddl):
        if typ not in _NUM_T:
            continue
        sc = int(s) if s is not None else (int(p) if p else 0)
        mul = 10 ** sc
        fn = f"dec_{t}_{c}".lower()
        sid = "sq::" + format(
            zlib.crc32(f"{repo}|{rel}|{t}.{c}|dec|{sc}".encode()) & 0xffffffff,
            "08x")
        out.append(dict(
            id=sid, family="theorem", tier="theorem",
            predicate=f"lossless_cast {c} scale={sc}",
            annotation="sql-decimal",
            canonical_unit=(f"def {fn}(v, k):\n    assert v * {mul} == k\n"
                            f"    assert v == k / {mul}\n"),
            expect="CERTIFIED",
            origin=dict(repo=repo, commit=commit, file=rel, table=t,
                        column=c, line=_col_line(tidx, t, c),
                        construct="sql_decimal"),
            provenance=dict(source="mined", confirmed=False)))
    for m in _CAST_RE.finditer(ddl):
        expr, typ, p, s = (m.group(1).strip(), m.group(2).upper(),
                           m.group(3), m.group(4) or "0")
        sid = "sq::" + format(
            zlib.crc32(f"{repo}|{rel}|cast|{expr}|{typ}|{p}|{s}".encode())
            & 0xffffffff, "08x")
        out.append(dict(
            id=sid, family="monitor", tier="monitor",
            predicate=f"cast({expr[:30]}) :: {typ}({p},{s})",
            annotation="sql-cast",
            canonical_unit="def cast_monitor():\n    return 0\n",
            expect="UNVERIFIABLE",
            origin=dict(repo=repo, commit=commit, file=rel, table="",
                        column=expr[:30], line=_expr_line(ddl, expr),
                        construct="sql_cast"),
            provenance=dict(source="mined", confirmed=False)))
    return out


_NULL_SITES = [
    ("sql_coalesce", r"\b(?:COALESCE|NVL|IFNULL|NULLIF)\s*\(", "null-default"),
    ("sql_nullpred", r"\bIS\s+(?:NOT\s+)?NULL\b", "null-predicate"),
    ("sql_nullcase", r"CASE\s+WHEN[^;]{0,80}?\bIS\s+(?:NOT\s+)?NULL\b",
     "case-on-null"),
]


def mine_sql_text(ddl: str, rel: str, repo: str, commit: str,
                  index: dict = None) -> list:
    """Mine one SQL file's DDL into premise candidates.

    `index` is the corpus-level declaration index (see `sql_index`): NOT NULL
    contracts and column order may be declared in a different file than the
    write path that must satisfy them, and verify/harvest/transfer MUST emit
    identical ids for the same tree -- so the index is built by the caller
    over the whole corpus, deterministically, and only defaults to this
    file's own declarations when absent.
    """
    out = []
    seen = set()
    tidx = _table_index(ddl)          # for line attribution of every claim
    for t, c, o, v in parse_checks_all(ddl):
        key = (t, c, o, v)
        if key in seen:
            continue
        seen.add(key)
        v_is_num = isinstance(v, (int, float)) or (
            isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v))
        if v_is_num:
            claim = canonical_premise(c, o, v)
            args = c
        else:
            # param-vs-param: premise like "start_date < end_date"
            opn = {"<": "le", ">": "ge"}.get(o, o)
            claim = f"assert {c} {o} {v}" if opn not in ("<", ">") else \
                (f"assert {c} < {v}" if o == "<" else f"assert {c} > {v}")
            claim = (f"assert {c} <= {v}" if o == "<=" else
                     f"assert {c} >= {v}" if o == ">=" else
                     f"assert {c} < {v}" if o == "<" else
                     f"assert {c} > {v}" if o == ">" else claim)
            args = f"{c}, {v}"
        fn = f"check_{t}_{c}".lower()
        wrapper = f"def {fn}({args}):\n    {claim}\n"
        sid = "sq::" + format(zlib.crc32(f"{repo}|{rel}|{t}.{c}|{o}|{v}".encode()) & 0xffffffff, "08x")
        out.append(dict(
            id=sid, family="premise", tier="premise",
            predicate=claim, annotation="sql-check",
            canonical_unit=wrapper, expect=None,
            origin=dict(repo=repo, commit=commit, file=rel, table=t,
                        column=c, line=_col_line(tidx, t, c),
                        construct="sql_check"),
            provenance=dict(source="mined", confirmed=False)))

    # Declared columns -> schema contracts (monitor class: DB-enforced, not
    # provable). This is what reaches procedure bodies: Snowflake temp tables
    # carry their types/scales as `CREATE ... TEMPORARY TABLE (col NUMERIC(p,s))`.
    NUMERIC_T = ("NUMERIC", "DECIMAL", "NUMBER")
    seen_cols = set()
    for t, c, typ, p, s, not_null in parse_columns_all(ddl):
        scale = s if s is not None else (p if typ in NUMERIC_T else None)
        key = (t, c, typ, p, s)
        if key in seen_cols:
            continue
        seen_cols.add(key)
        if typ in NUMERIC_T and scale is not None:
            pred = f"{c} :: {typ}({p},{scale})"
        elif typ in NUMERIC_T and p:
            pred = f"{c} :: {typ}({p})"
        else:
            pred = f"{c} :: {typ}"
        if not_null:
            pred += " NOT NULL"
        fn = f"col_{t}_{c}".lower()
        sid = "sq::" + format(
            zlib.crc32(f"{repo}|{rel}|{t}.{c}|{typ}|{p}|{s}".encode()) & 0xffffffff,
            "08x")
        # NOT NULL is a DECLARED contract — the database enforces it on every
        # write — so with the nullness rung it states its real proposition
        # (`col is not None`) and disposes as CERTIFIED-DECLARED (premise
        # tier, counted separately from proofs). A bare TYPE declaration
        # states no proposition at all; it stays a monitor (inventory,
        # declined honestly), because inventing a claim from a type would be
        # the file asserting what the file contains.
        if not_null:
            unit = f"def {fn}({c}):\n    assert {c} is not None\n"
            fam, tier_, expect = "premise", "premise", "CERTIFIED"
        else:
            unit = f"def {fn}({c}):\n    return {c}\n"
            fam, tier_, expect = "monitor", "monitor", "UNVERIFIABLE"
        out.append(dict(
            id=sid, family=fam, tier=tier_,
            predicate=pred, annotation="sql-column",
            canonical_unit=unit,
            expect=expect,
            origin=dict(repo=repo, commit=commit, file=rel, table=t,
                        column=c, line=_col_line(tidx, t, c),
                        construct="sql_column"),
            provenance=dict(source="mined", confirmed=False)))

    out.extend(decimal_claims(ddl, rel, repo, commit))
    out.extend(range_claims(ddl, rel, repo, commit))
    out.extend(write_claims(ddl, rel, repo, commit, index=index))
    return out


def mine_sql_repo(root, repo_name=None, commit=None):
    import subprocess
    if commit is None:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                                    cwd=root, capture_output=True,
                                    text=True).stdout.strip()
        except Exception:
            commit = "unknown"
    repo_name = repo_name or os.path.basename(root)
    pairs = []
    for dp, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "target", ".venv")]
        for f in sorted(fs):
            if not f.endswith((".sql", ".ddl")):
                continue
            full = os.path.join(dp, f)
            rel = os.path.relpath(full, root)
            try:
                ddl = open(full, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            pairs.append((rel, ddl))
    index = sql_index(pairs)          # corpus-wide contracts before per-file mining
    return [c for rel, ddl in pairs
            for c in mine_sql_text(ddl, rel, repo_name, commit, index=index)]




def range_claims(ddl: str, rel: str, repo: str, commit: str) -> list:
    """Split the PRECISION object out of the conflated declaration.

    `AMOUNT :: NUMBER(12,2)` states two separable things: a representation
    theorem (exact at scale 2 -- `decimal_claims`, PROVED) and a range bound
    (|v| < 10^(p-s) -- a declared contract this text cannot prove about
    stored rows, and the input other proofs consume as a bound). The
    conflated string was the only home for both, so the provable half was
    mined while the bound half stayed unreachable: one unverifiable row, no
    premise. This emits the bound as its own premise-tier claim -- declared,
    never counted as a proof, driving no injection until it is confirmed
    through the ledger.
    """
    out = []
    num_t = ("NUMERIC", "DECIMAL", "NUMBER")
    tidx = _table_index(ddl)
    for t, c, typ, p, s, _nn in parse_columns_all(ddl):
        if typ not in num_t or not p or not str(p).isdigit():
            continue
        pi = int(p)
        if s is not None and str(s).isdigit():
            e = pi - int(s)
        else:
            e = pi
        if not (0 <= e <= 60):        # sanity: a bound, not a bomb literal
            continue
        bound = "1" + "0" * e
        fn = re.sub(r"\W", "_", f"rng_{t}_{c}").lower()
        sid = "sq::" + format(
            zlib.crc32(f"{repo}|{rel}|{t}.{c}|range|{p}|{s}".encode())
            & 0xffffffff, "08x")
        out.append(dict(
            id=sid, family="premise", tier="premise",
            predicate=f"range {c} |v| < 10^{e}",
            annotation="sql-range",
            canonical_unit=(f"def {fn}({c}):\n"
                            f"    assert {c} < {bound}\n"
                            f"    assert {c} > -{bound}\n"),
            expect="CERTIFIED",
            origin=dict(repo=repo, commit=commit, file=rel, table=t,
                        column=c, line=_col_line(tidx, t, c),
                        construct="sql_range"),
            provenance=dict(source="mined", confirmed=False)))

    # The STRING analog: VARCHAR(n)/CHAR(n) states a length bound the text
    # cannot prove about stored rows -- and, until this rung, the length was
    # not even a proposition (the unbounded string columns still are not).
    # Measured constituency on BANKING_SQL: 887 length-bounded columns, the
    # histogram's largest decidability-shaped abstain class. Same venue as
    # the numeric range: a declared contract, premise tier, never a proof.
    str_t = ("VARCHAR", "VARCHAR2", "NVARCHAR", "NVARCHAR2", "CHAR",
             "BPCHAR", "STRING", "TEXT")
    for t, c, typ, p, s, _nn in parse_columns_all(ddl):
        if typ.upper() not in str_t or not p or not str(p).isdigit():
            continue
        n = int(p)
        if not (1 <= n <= 8000):      # a bound, not a bomb literal
            continue
        fn = re.sub(r"\W", "_", f"len_{t}_{c}").lower()
        sid = "sq::" + format(
            zlib.crc32(f"{repo}|{rel}|{t}.{c}|lenrange|{p}".encode())
            & 0xffffffff, "08x")
        out.append(dict(
            id=sid, family="premise", tier="premise",
            predicate=f"range {c} len <= {n}",
            annotation="sql-range",
            canonical_unit=f"def {fn}({c}):\n    assert len({c}) <= {n}\n",
            expect="CERTIFIED",
            origin=dict(repo=repo, commit=commit, file=rel, table=t,
                        column=c, line=_col_line(tidx, t, c),
                        construct="sql_range"),
            provenance=dict(source="mined", confirmed=False)))
    return out


# ----------------------------------------------------------- write paths ----
# Route A for NOT NULL: the declaration is the CONTRACT, the write path is
# the CLAIM. `supplies T.C at insert` quantifies over program statements --
# inside the fragment -- where the declaration itself quantified over table
# rows, which live in the database, not the file. Disposition is exact:
#   literal / NULL keyword -> decided here (certified / refuted with the
#                             statement as the witness);
#   anything else          -> instance claim with NO unit -> honest
#                             `runtime_observation` abstain (the venue is
#                             runtime; the enforcement bridge hands over the
#                             query). Never a guess, never a vacuous pass.

_INS_RE = re.compile(r"\bINSERT\s+INTO\s+", re.I)
_UPD_RE = re.compile(r"\bUPDATE\s+", re.I)
_TABLE_TOK = re.compile(
    r"IDENTIFIER\s*\(\s*:?[\w]+\s*\)|`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[\w.$#]+",
    re.I)


def _norm_table(name: str) -> str:
    """IDENTIFIER(:var) / quoting / schema prefix -> comparable tail."""
    m = re.fullmatch(r"IDENTIFIER\s*\(\s*:?(\w+)\s*\)", (name or "").strip(),
                     re.I)
    if m:
        return m.group(1).lower()
    return (name or "").strip().strip("`\"[]").split(".")[-1].lower()


def _split_top(s: str) -> list:
    """Split on depth-0 commas: quote-aware, paren-balanced."""
    parts, buf, depth, i, q = [], [], 0, 0, None
    while i < len(s):
        ch = s[i]
        if q:
            buf.append(ch)
            if ch == q:
                if q == "'" and i + 1 < len(s) and s[i + 1] == "'":
                    buf.append(s[i + 1])
                    i += 2
                    continue
                q = None
            i += 1
            continue
        if ch in "'\"":
            q = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _balanced_group(s: str, start: int):
    """Parse `(...)GROUP` at s[start]; -> (inner_text, end_index)."""
    if start >= len(s) or s[start] != "(":
        return None, start
    depth, j = 1, start + 1
    while j < len(s) and depth:
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
        j += 1
    if depth:
        return None, start
    return s[start + 1:j - 1], j


def _value_kind(v: str):
    """SQL value -> ('null'|'literal'|'other', python literal text|None)."""
    vs = (v or "").strip()
    if re.fullmatch(r"NULL", vs, re.I):
        return "null", None
    if re.fullmatch(r"-?\d+", vs):
        return "literal", vs
    if re.fullmatch(r"'([^']|'')*'", vs, re.S):
        return "literal", repr(vs[1:-1].replace("''", "'"))
    return "other", None


def _sql_statements(ddl: str):
    """Yield (kind, stmt_text, stmt_start) for INSERT/UPDATE statements."""
    for rx, kind in ((_INS_RE, "insert"), (_UPD_RE, "update")):
        for m in rx.finditer(ddl):
            end = ddl.find(";", m.start())
            if end == -1 or end - m.start() > 8000:
                end = min(len(ddl), m.start() + 8000)
            yield kind, ddl[m.start():end], m.start()


def _line_of(ddl: str, off: int) -> int:
    return ddl.count("\n", 0, off) + 1


def sql_index(pairs) -> dict:
    """Corpus-level declaration index: NOT NULL contracts + column order.

    The contract and the write path routinely live in different files (a DDL
    tree vs procedure bodies), and three surfaces mine this tree -- verify,
    harvest, transfer -- which must produce IDENTICAL claim ids for the same
    input or shape collapse forks. Deterministic: pairs sorted by rel, first
    declaration wins.
    """
    index = {"nn": {}, "order": {}}
    for rel, ddl in sorted(pairs, key=lambda p: p[0]):
        tidx = _table_index(ddl)
        for t, c, _typ, _p, _s, not_null in parse_columns_all(ddl):
            nt = _norm_table(t)
            slot = index["order"].setdefault(nt, {"file": rel, "cols": []})
            if slot["file"] == rel:
                slot["cols"].append((t, c))
            if not_null:
                index["nn"].setdefault((nt, c.lower()), dict(
                    file=rel, line=_col_line(tidx, t, c),
                    table=t, column=c))
    return index


def write_claims(ddl: str, rel: str, repo: str, commit: str,
                 index: dict = None) -> list:
    """Mine write paths against the corpus NOT NULL contracts (Route A).

    origin points at the DECLARATION (file:line of the contract -- a column
    claim lands on the line that declares the column); write_file/write_line
    point at the statement. Disposition is exact: a NULL literal refutes
    (the statement is the witness), a literal certifies, anything else
    abstains honestly as a runtime observation.
    """
    if index is None:
        index = sql_index([(rel, ddl)])
    nn, order = index["nn"], index["order"]
    out = []

    def _emit(key, stmt_start, kind, row_i, vkind, py_lit):
        decl = nn[key]
        c = decl["column"]
        wline = _line_of(ddl, stmt_start)
        wfile = os.path.basename(rel)
        expect = {"null": "REFUTED",
                  "literal": "CERTIFIED"}.get(vkind, "UNVERIFIABLE")
        sid = "sq::" + format(zlib.crc32(
            f"{repo}|{decl['file']}|{decl['table']}.{c}|write|{kind}"
            f"|{rel}|{stmt_start}|{row_i}".encode()) & 0xffffffff, "08x")
        unit = None
        if vkind == "null":
            unit = "def w():\n    assert None is not None\n"
        elif vkind == "literal":
            unit = f"def w():\n    assert {py_lit} is not None\n"
        out.append(dict(
            id=sid, family="instance", tier="instance",
            predicate=(f"supplies {decl['table']}.{c} at {kind} "
                       f"{wfile}:{wline}"),
            annotation="sql-write",
            canonical_unit=unit,
            expect=expect,
            origin=dict(repo=repo, commit=commit,
                        file=decl["file"], line=decl["line"],
                        table=decl["table"], column=c,
                        write_file=rel, write_line=wline,
                        construct="sql_write"),
            provenance=dict(source="mined", confirmed=False)))

    def _cols_of(ntab, cols):
        if cols is not None:
            return cols
        slot = order.get(ntab)
        return [c for _t, c in slot["cols"]] if slot else []

    for kind, stmt, stmt_start in _sql_statements(ddl):
        if kind == "insert":
            m = _INS_RE.match(stmt)
            rest = stmt[m.end():]
            tm = _TABLE_TOK.match(rest)
            if not tm:
                continue
            ntab = _norm_table(tm.group(0))
            rest = rest[tm.end():].lstrip()
            cols = None
            if rest.startswith("("):
                inner, j = _balanced_group(rest, 0)
                if inner is None:
                    continue
                cols = [x.strip().strip("`\"[]") for x in _split_top(inner)]
                rest = rest[j:].lstrip()
            vm = re.match(r"VALUES\s*", rest, re.I)
            if vm:
                pos = rest[vm.end():].lstrip()
                rows, cursor = [], 0
                while True:
                    while cursor < len(pos) and pos[cursor] in " \t\r\n,":
                        cursor += 1
                    inner, j = _balanced_group(pos, cursor)
                    if inner is None:
                        break
                    rows.append(_split_top(inner))
                    cursor = j
                if not rows:
                    continue
                cols = _cols_of(ntab, cols)
                for ri, vals in enumerate(rows):
                    if len(vals) != len(cols):
                        continue          # arity mismatch: no honest zip
                    for col, val in zip(cols, vals):
                        key = (ntab, col.lower())
                        if key not in nn:
                            continue
                        vkind, py_lit = _value_kind(val)
                        _emit(key, stmt_start, kind, ri, vkind, py_lit)
            elif re.match(r"(SELECT|WITH|EXEC)\b", rest, re.I):
                for col in _cols_of(ntab, cols):
                    key = (ntab, col.lower())
                    if key in nn:
                        # value comes from a query: not decidable from text
                        _emit(key, stmt_start, kind, 0, "other", None)
            continue

        # UPDATE t SET c = expr, ...
        m = _UPD_RE.match(stmt)
        rest = stmt[m.end():]
        tm = _TABLE_TOK.match(rest)
        if not tm:
            continue
        ntab = _norm_table(tm.group(0))
        sm = re.search(r"\bSET\b", rest[tm.end():], re.I)
        if not sm:
            continue
        set_txt = rest[tm.end() + sm.end():]
        for piece in _split_top(set_txt):
            am = re.match(r"(`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_]\w*)"
                          r"\s*=\s*(.+)", piece, re.S)
            if not am:
                continue
            col = am.group(1).strip("`\"[]")
            key = (ntab, col.lower())
            if key not in nn:
                continue
            vkind, py_lit = _value_kind(am.group(2))
            _emit(key, stmt_start, kind, 0, vkind, py_lit)
    return out
