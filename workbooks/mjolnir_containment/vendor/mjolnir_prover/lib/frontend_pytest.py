"""mjolnir_prover.lib.frontend_pytest — the pytest test-framework ADAPTER.

WHY THIS IS A FRONTEND, NOT IN `differential.py`
-------------------------------------------------
The differential scorer is language-independent: it wants ground truth about a
claim and nothing more. Supplying that ground truth is LANGUAGE-SPECIFIC, so it
belongs behind the same boundary as every other language-specific concern —
`frontend.py`, `frontend_lean2.py`, `frontend_sql.py`, `frontend_tptp.py` — and
not inline in the scorer. pytest invocation sitting in `differential.py` made the
core depend on a third-party test framework, which is the coupling this package
layout exists to prevent.

THE MECHANISM IS GENERAL; THE ADAPTER IS NOT
--------------------------------------------
The mechanism is: **ask the language's own test framework to supply the runtime
value the claim is about.** A guessed value is unsound in both directions — a
guess on which the claim holds proves nothing about the real value, and a guess
on which it fails refutes nothing either. But the framework has the real value:

    Python / pytest    run the test, read the outcome      (this adapter)
    Rust  / cargo test run the test, read the outcome      (not written)
    Go    / go test    same                                (not written)
    Java  / JUnit      same                                (not written)

So the capability is "test-framework adapters", of which this is the first. The
mechanism generalises; only the invocation is per-framework.

DEPENDENCY POSTURE
------------------
mjolnir does NOT import pytest in-process. The `import pytest` below is inside a
string — a script written to a temp file and run in a **subprocess**. So:

  * pytest is not a dependency of the package (not in pyproject/requirements);
  * if pytest is absent, this adapter reports UNAVAILABLE and the route reports
    NO VERDICT for every unit, rather than crashing the scorer.

That is deliberate. A differential oracle that cannot run must say "I don't
know", not "it passed".
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

NAME = "pytest"

# Outcomes the adapter can report. Only PASSED and FAILED are VERDICTS; every
# other value is explicitly "no verdict" and must be surfaced as such.
VERDICTS = ("PASSED", "FAILED")
NO_VERDICT = ("SKIPPED", "ERROR", "TIMEOUT", "NOTCOLLECTED",
              "UNAVAILABLE", "COLLECTION_ERROR")


def available() -> bool:
    """Can this adapter run at all? Checked in a subprocess, so mjolnir never
    imports pytest itself."""
    try:
        r = subprocess.run([sys.executable, "-c", "import pytest"],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:                                       # noqa: BLE001
        return False


_RUNNER = r'''
import json, sys, os
# The test's own directory must be importable, and `-I` ignores PYTHONPATH, so
# the runner inserts it itself. Note the FLAGS: `-I` isolates from the ambient
# environment, but `-S` is NOT used — it would drop site-packages and with it
# pytest, which is exactly the dependency we need.
# APPEND, do not insert at 0. The corpus vendors whole package SOURCES, so a
# front-of-path entry makes `import numpy` pick the vendored tree instead of the
# installed one -- which fails with a circular import and makes the whole file
# NOTCOLLECTED. Appending keeps the test's own module importable while letting
# installed packages win, which is what a working checkout looks like.
for _p in sys.argv[1:]:
    if _p.endswith(".py") and os.path.isfile(_p):
        sys.path.append(os.path.dirname(os.path.abspath(_p)))
        break
out = {}


class _Rec:
    def pytest_runtest_logreport(self, report):
        if report.when in ("call", "setup"):
            if report.outcome == "failed":
                out[report.nodeid] = "FAILED"
            elif report.outcome == "skipped":
                out[report.nodeid] = "SKIPPED"
            elif report.outcome == "passed" and report.nodeid not in out:
                out[report.nodeid] = "PASSED"

    def pytest_collectreport(self, report):
        if report.outcome == "failed":
            out["<collection>"] = "ERROR"


import pytest
try:
    pytest.main(sys.argv[1:], plugins=[_Rec()])
except BaseException as e:
    out["<fatal>"] = "ERROR:" + type(e).__name__
print("@@JSON@@" + json.dumps(out))
'''


def run(path: str, names, timeout_s: float = 120.0) -> dict:
    """Run `names` from one test file; return {name: outcome}.

    Per-FILE rather than per-test: one pytest invocation per test would cost a
    process start each, and a file's tests share its imports and fixtures.
    """
    if not names:
        return {}
    if not available():
        return {n: "UNAVAILABLE" for n in names}
    with tempfile.TemporaryDirectory(prefix="mj_pytest_") as d:
        runner = os.path.join(d, "_runner.py")
        with open(runner, "w", encoding="utf-8") as fh:
            fh.write(_RUNNER)
        env = {"PATH": os.environ.get("PATH", ""),
               "HOME": d,
               "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": os.path.dirname(os.path.abspath(path))}
        # ABSOLUTE path: cwd is set to the test's own directory, so a relative
        # path would not resolve (every run reported "file or directory not
        # found" and came back as a blanket ERROR).
        # `-I` but NOT `-S`: isolation from the environment, while keeping
        # site-packages so pytest is importable. `-S` made every run ERROR.
        abspath = os.path.abspath(path)
        argv = [sys.executable, "-I", runner, "-q", "--no-header",
                "--tb=no", "-p", "no:cacheprovider", abspath]
        # NODE IDs, not `-k`. `-k` is a name filter with its own matching rules
        # and it silently DESELECTS on a miss ("8 deselected"), which looks like
        # a verdict and is not one. `path::test_name` selects exactly that test.
        argv += [f"{abspath}::{n}" for n in names]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True, text=True, timeout=timeout_s,
                cwd=os.path.dirname(abspath) or ".", env=env)
        except subprocess.TimeoutExpired:
            return {n: "TIMEOUT" for n in names}
        except Exception:                                    # noqa: BLE001
            return {n: "ERROR" for n in names}
        blob = None
        for line in proc.stdout.splitlines():
            if line.startswith("@@JSON@@"):
                blob = line[len("@@JSON@@"):]
        if blob is None:
            return {n: "ERROR" for n in names}
        import json
        try:
            raw = json.loads(blob)
        except (json.JSONDecodeError, TypeError, ValueError):
            # NARROW on purpose. A bare `except Exception` here silently turned
            # a missing import into a blanket ERROR for every unit — the run
            # looked like "the tests all errored" when nothing had run at all.
            # A NameError here should be loud.
            return {n: "ERROR:unparsable_runner_output" for n in names}
        by_name = {}
        for nodeid, outcome in raw.items():
            by_name[nodeid.rsplit("::", 1)[-1]] = outcome
        # a name pytest never reported was not collected -> no verdict
        return {n: by_name.get(n, "NOTCOLLECTED") for n in names}