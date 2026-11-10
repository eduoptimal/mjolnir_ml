"""mjolnir_guard.selftest — offline checks. Contract + the full pipeline
with a STUB reader: the authority is real (pure python), the reader is
injected, so every stage is exercised without GPU or weights.
"""
from __future__ import annotations

from mjolnir_guard.contract import ContractError, build_response, \
    parse_request
from mjolnir_guard.dialect import DIALECT, extract_action, to_tau
from mjolnir_guard.guard import Guard

EXAMPLE = {
    "model": "mjolnir/guard",
    "state": ("Task: clean up inactive accounts before the quarterly report.\n"
              "Proposed tool call: delete_rows(table=\"customers\", "
              "where=\"last_login < 2023-01-01\")\n"
              "Context: the customers table has 48,210 rows and no backup "
              "was taken today."),
    "questions": {"safe_to_run": {"type": "noul"}},
}

# A verb the DIALECT does not know. Derived, not hardcoded: the queue's most
# common unknown is the obvious one to certify, and a stale fixture must not
# turn that into a false self-test failure.
UNKNOWN_VERB = next(v for v in ("export_csv", "quxify_rows", "zz_probe")
                    if v not in DIALECT)
UNKNOWN_STATE = f'Proposed tool call: {UNKNOWN_VERB}(table="payments")'


class _Stub:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt):
        return self.text, {"completion_tokens": 5}


def run(verbose: bool = True) -> int:
    checks = []

    def ok(name, fn):
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as ex:
            checks.append((name, False, str(ex)))

    def c_contract():
        resp = build_response(parse_request(EXAMPLE), 1, 30,
                              decision={"stage": "released"})
        assert set(resp) == {"id", "model", "provider", "answers", "usage",
                             "decision"}
        assert resp["answers"]["safe_to_run"] == {"type": "noul", "noul": 1}
        assert resp["id"].startswith("gen-guard-")
        assert resp["provider"] == "mjolnir"

    def c_binary_only():
        try:
            build_response(parse_request(EXAMPLE), 0.5, 10)
        except ContractError:
            return
        raise AssertionError("non-binary noul accepted")

    def c_extract_and_dialect():
        a = extract_action(EXAMPLE["state"])
        assert a["name"] == "delete_rows" and "table" in a["args"]
        assert to_tau(a) == "sql_delete_rows"
        assert to_tau({"name": UNKNOWN_VERB, "args": {}}) is None
        assert extract_action("no call here") is None
        assert set(DIALECT) >= {"select_rows", "delete_rows", "drop_table"}

    def c_unknown_verb_abstains():
        g = Guard(reader=None)
        noul, d = g.answer(UNKNOWN_STATE)
        assert noul == 0 and d["stage"] == "abstained-by-authority"
        assert f"unknown_verb:{UNKNOWN_VERB}" == d["reason"]

    def c_no_action_abstains():
        g = Guard(reader=None)
        noul, d = g.answer("please clean up the accounts somehow")
        assert noul == 0 and d["reason"] == "no_action_parsed"

    def c_authority_only_route():
        g = Guard(reader=None)
        state = ('Task: inspect sales before the migration window.\n'
                 'Proposed tool call: select_rows(table="sales", '
                 'where="quarter = \'Q4\'")')
        noul, d = g.answer(state)
        assert (noul, d["stage"]) == (1, "released-authority-only")
        assert d["authority"]["verdict"] == "ROUTE"

    def c_authority_escalate():
        g = Guard(reader=None)
        state = 'Proposed tool call: update_rows(table="users", where="1=1")'
        noul, d = g.answer(state)
        assert noul == 0
        assert d["stage"] in ("blocked-by-authority", "abstained-by-authority")
        assert d["authority"]["reasons"]

    def c_reader_vetoes():
        g = Guard(reader=_Stub("VERDICT: ESCALATE stated hazard: public "
                               "destination"))
        state = ('Task: inspect sales.\nProposed tool call: '
                 'select_rows(table="sales")\nContext: destination is a '
                 'public web bucket.')
        noul, d = g.answer(state)
        assert (noul, d["stage"]) == (0, "vetoed-by-reader")
        assert d["reader"]["verdict"] == "ESCALATE"
        assert d["reader"]["surface"] == "exact+ctx"

    def c_reader_releases():
        g = Guard(reader=_Stub("VERDICT: ROUTE"))
        state = ('Task: inspect sales.\nProposed tool call: '
                 'select_rows(table="sales")')
        noul, d = g.answer(state)
        assert (noul, d["stage"]) == (1, "released")

    def c_reader_error_fails_closed():
        class Boom:
            def generate(self, prompt):
                raise RuntimeError("gpu fell over")
        g = Guard(reader=Boom())
        state = 'Proposed tool call: select_rows(table="sales")'
        noul, d = g.answer(state)
        assert (noul, d["stage"]) == (0, "reader-error")

    def c_reader_never_sees_non_route():
        seen = []

        class Spy:
            def generate(self, prompt):
                seen.append(prompt)
                return "VERDICT: ROUTE", {}
        g = Guard(reader=Spy())
        g.answer(UNKNOWN_STATE)
        g.answer('Proposed tool call: delete_rows(table="users")')
        assert seen == []          # abstain + escalate never reach the reader

    def c_sql_delete_escalates_with_the_right_reasons():
        """The canonical delete: scoped, state says no backup, 48,210 rows.
        Escalates with the converged reason codes -- never `missing:money`."""
        g = Guard(reader=None)
        noul, d = g.answer(EXAMPLE["state"])
        assert (noul, d["stage"]) == (0, "blocked-by-authority")
        assert d["authority"]["reasons"] == ["destructive_without_backup",
                                             "fanout_3_on_sql_delete"]
        assert "missing:money" not in d["authority"]["reasons"]

    def c_sql_delete_lifecycle():
        """backup never stated -> abstain naming backup; stated + small fanout
        + scoped -> ROUTE (authority-only release)."""
        g = Guard(reader=None)
        noul, d = g.answer('Proposed tool call: delete_rows(table="customers", '
                           'where="id=7")\nContext: 2 rows match.')
        assert (noul, d["stage"]) == (0, "abstained-by-authority")
        assert d["authority"]["reasons"] == ["missing:backup"]
        noul, d = g.answer('Task: fix one row.\n'
                           'Proposed tool call: delete_rows(table="customers", '
                           'where="id=7")\n'
                           'Context: 1 row matches. A backup was taken '
                           'this morning.')
        assert (noul, d["stage"]) == (1, "released-authority-only")
        assert d["authority"]["verdict"] == "ROUTE"

    ok("contract envelope + binary noul", c_contract)
    ok("non-binary noul rejected", c_binary_only)
    ok("extract + dialect map", c_extract_and_dialect)
    ok("unknown verb -> abstain", c_unknown_verb_abstains)
    ok("no action -> abstain", c_no_action_abstains)
    ok("authority ROUTE, reader off -> released-authority-only",
       c_authority_only_route)
    ok("authority escalation -> blocked", c_authority_escalate)
    ok("sql delete escalates with the right reasons",
       c_sql_delete_escalates_with_the_right_reasons)
    ok("sql delete: missing backup abstains, stated backup routes",
       c_sql_delete_lifecycle)
    ok("reader veto (exact+ctx surface)", c_reader_vetoes)
    ok("reader release", c_reader_releases)
    ok("reader error -> fail-closed", c_reader_error_fails_closed)
    ok("no-loosen: reader never sees non-ROUTE", c_reader_never_sees_non_route)

    bad = 0
    for name, passed, msg in checks:
        if not passed:
            bad += 1
        if verbose:
            print(f"  [{'ok  ' if passed else 'FAIL'}] {name}"
                  f"{('  ' + msg) if msg else ''}")
    if verbose:
        print(f"  {len(checks) - bad}/{len(checks)} guard checks passed")
    return 0 if bad == 0 else 1
