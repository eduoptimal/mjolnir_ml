"""Tests for mjolnir — the labelled cases plus CLI exit-code contracts.

Run:  python -m pytest            (or just)  python tests/test_mjolnir_intent.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import mjolnir_intent                      # noqa: E402
from mjolnir_intent import checks, selftest  # noqa: E402


# ------------------------------------------------- labelled cases (core) -----
def test_labelled_cases():
    assert selftest.run(verbose=False) == 0


# ------------------------------------------------- gate vs advisory split ----
def test_gate_split():
    r = checks.audit("set the timeout to 30 seconds", "Set timeout to 30 minutes")
    assert [f["check"] for f in r["gate"]] == ["unit_adjacency"]
    assert r["advisory"] == []

    r = checks.audit("update `parse_config` in `engine.py`", "updated the parser")
    assert r["gate"] == []
    assert [f["check"] for f in r["advisory"]] == ["entity_lost"] * 2


def test_diff_enables_negation():
    p = "add caching, without retries"
    r = checks.audit(p, "", diff="+ retries = 3")
    assert [f["check"] for f in r["gate"]] == ["negation_added"]
    # without a diff it is unverifiable, never a failure
    r = checks.audit(p, "we considered retries")
    assert r["gate"] == []
    assert len(r["unverifiable"]) == 1


def test_declined_prose_is_respected():
    r = checks.audit("add caching, without retries",
                     "I did not add retries; only the cache.")
    assert r["gate"] == []
    assert r["unverifiable"] == []


def test_verdicts():
    assert checks.audit("make the pipeline faster")["verdict"] == "ASK"
    assert checks.audit("set the limit to 4096")["verdict"] == "ASK"
    assert checks.audit("rename `foo` to `bar`")["verdict"] == "BUILD"


def test_deterministic():
    a = checks.audit("set timeout to 30 seconds", "timeout is 30 minutes")
    b = checks.audit("set timeout to 30 seconds", "timeout is 30 minutes")
    assert a == b


def test_version_export():
    assert mjolnir_intent.__version__ == "0.1.0"


# ------------------------------------------------- CLI exit-code contract ----
def _cli(*args):
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-m", "mjolnir_intent", *args],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    return p.returncode, p.stdout + p.stderr


def test_cli_clean_build_is_0():
    code, out = _cli("check", "-p", "rename `foo` to `bar`", "-r", "renamed foo to bar")
    assert code == 0, out


def test_cli_gate_is_1():
    code, out = _cli("check", "-p", "set timeout to 30 seconds",
                     "-r", "timeout is 30 minutes")
    assert code == 1, out
    assert "unit changed" in out


def test_cli_ask_is_2():
    code, out = _cli("check", "-p", "make it faster", "-r", "done")
    assert code == 2, out
    assert "ASK" in out


def test_cli_diff_negation_is_1():
    code, out = _cli("check", "-p", "add caching, without retries",
                     "-r", "done", "-d", "+ retries = 3\n+ cache = {}")
    assert code == 1, out
    assert "negation violated" in out


def test_cli_json():
    code, out = _cli("check", "-p", "set timeout to 30 seconds",
                     "-r", "timeout is 30 minutes", "--json")
    assert code == 1
    rep = json.loads(out)
    assert rep["verdict"] == "BUILD"
    assert rep["gate"][0]["check"] == "unit_adjacency"


def test_cli_strict_blocks_advisory():
    code, _ = _cli("check", "-p", "update `parse_config` now",
                   "-r", "done", "--strict")
    assert code == 1
    # advisory alone never blocks
    code, _ = _cli("check", "-p", "update `parse_config` now", "-r", "done")
    assert code == 0


# -------------------------------- Contractual Lowering via `intent --strict` ----
def test_cli_strict_refutes_unit_swap():
    # 5 minutes vs a proposal of 3000 s: the WHOLE-PROMPT regex cannot see this
    # (it looks for the literal "5" downstream), the lowered claim can.
    code, out = _cli("check", "-p", "set the query_timeout to 5 minutes",
                     "-r", '{"timeout_seconds": 3000}', "--strict")
    assert code == 1, out
    assert "lowered claim refuted" in out and "query_timeout == 300 s" in out
    # the same pair without --strict is untouched: no second authority engaged
    code, out = _cli("check", "-p", "set the query_timeout to 5 minutes",
                     "-r", '{"timeout_seconds": 3000}')
    assert code == 0, out
    assert "lowered" not in out


def test_cli_strict_certifies_conforming_proposal():
    code, out = _cli("check", "-p", "set the query_timeout to 5 minutes",
                     "-r", "query_timeout now 5 minutes (timeout_seconds: 300)",
                     "-d", "+timeout_seconds: 300", "--strict")
    assert code == 0, out
    assert "authority CERTIFIED" in out


def test_cli_selftest():
    code, out = _cli("selftest")
    assert code == 0, out
    # counts grow as cases are added; the contract is "everything passed"
    assert " pass" in out and "UNEXPECTED FAIL" not in out, out


def test_cli_prompt_from_file(tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("set timeout to 30 seconds")
    code, out = _cli("check", "-p", str(f), "-r", "timeout is 30 seconds")
    assert code == 0, out


def _make_tmp():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp())


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(*[_make_tmp() for _ in inspect.signature(fn).parameters])
            print(f"  PASS {fn.__name__}")
        except Exception as e:                                  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
