"""mjolnir_guard.dialect — structural action extraction + the dialect map.

Two DATA tables, no model:

1. the structural parser (`Proposed tool call: name(args)` -> name/args) —
   copied self-contained from the battery harness that measured it; parsing
   is structural, never a rewrite of the prose.
2. DIALECT: the SQL-flavoured tool verbs seen in decision states mapped to the
   authority's vocabulary.  Recorded from the authority run
   (`ours_decide_results_authority.jsonl`, n=100): ROUTE 48 / ABSTAIN 48 /
   ESCALATE 4 -- those numbers describe the PREVIOUS map, in which
   delete_rows/drop_table were laundered through retail verbs and every delete
   abstained `missing:money` (a reason true of the map, not of a SQL statement).
   As of 2026-10-14 the destructive SQL verbs map to the authority's OWN
   sql_* vocabulary (`sql_delete_rows`, `sql_drop_table`: scoped/backup/fanout
   facts read off the statement and the state), so deletes abstain or escalate
   with the right reason.  A verb NOT in this table is an action the authority
   cannot classify -> ABSTAIN to a human.  That is the closed-vocabulary
   contract, and adding a row here is a policy change, not a code change.
"""
from __future__ import annotations

import re

DIALECT = {
    "select_rows": "get_order_details",
    "count_rows": "get_order_details",
    "update_rows": "modify_user_address",
    "delete_rows": "sql_delete_rows",
    "drop_table": "sql_drop_table",
    "grant_role": "modify_user_address",
}

_CALL_RE = re.compile(r"Proposed tool call:\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*$",
                      re.M)


def _split_args(blob: str) -> list:
    parts, depth, cur, quote = [], 0, "", None
    for ch in blob:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            cur += ch
        elif ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _coerce(value: str):
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def extract_action(state: str) -> dict | None:
    """-> {"name": ..., "args": {...}} from a prose state, or None."""
    m = _CALL_RE.search(state or "")
    if not m:
        return None
    name, blob = m.group(1), m.group(2).strip()
    args = {}
    if blob:
        for part in _split_args(blob):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            args[k.strip()] = _coerce(v)
    return {"name": name, "args": args}


def to_tau(action: dict | None) -> str | None:
    """dialect verb -> the authority's verb, or None (cannot classify)."""
    if not isinstance(action, dict):
        return None
    return DIALECT.get(str(action.get("name") or ""))
