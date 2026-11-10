"""mjolnir_prover.lib.differential — external scoring of the arithmetic fragment.

WHY THIS EXISTS. Every other corpus in the stack is self-authored or vendored,
so the ONLY correctness evidence mjolnir had was its own kernel re-deriving its
own verdicts — same code, twice. ILTP supplied external labels but for a logic
nobody's code is written in. This module supplies the missing property for the
corpus that actually matters: **an independent oracle**. Python's own evaluator
decides whether mjolnir's certificates and refutations are right, without
consulting mjolnir.

THE ONE RULE. Execution SCORES; it never DISCHARGES. A verdict in a certificate
must come from the prover. Promoting an executed result into a certificate is
the A-combobound defect wearing a lab coat — the certifier would be trusting the
program it is certifying. So this module produces a SCOREBOARD and nothing that
feeds back into the gate.

WHAT IS SCORED, per unit:

    mjolnir CERTIFIED, execution falsifies it   ->  FALSE CERTIFICATE  (critical)
    mjolnir REFUTED,  execution satisfies it    ->  WRONG REFUTATION
    mjolnir UNVERIFIABLE, execution satisfies it ->  headroom (conservative)
    mjolnir UNVERIFIABLE, execution falsifies it ->  correctly declined

SAMPLING IS NOT PROOF. "No falsifying sample found" is NOT "true". Only a
falsifying counterexample is conclusive, and only in one direction: a
false certificate found here is definitive, a clean run certifies nothing. The
`--exec` path runs vendored third-party source; it is OPT-IN for that reason.
"""
from __future__ import annotations

import ast
import json
import os

from . import certify

SAMPLES_DEFAULT = 16


# ------------------------------------------------------------- extraction ---
def claim_parts(code: str):
    """(setup_statements, claim_test_expr) for the LAST assert of a unit.

    That is mjolnir's own reading: the last assert is the claim, everything
    before it is premise (see certificate.py's `_is_last`). Scoring anything
    else would compare mjolnir against a different claim than it certified.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return None
    asserts = [i for i, st in enumerate(fn.body)
               if isinstance(st, ast.Assert)]
    if not asserts:
        return None
    last = asserts[-1]
    return fn.body[:last], fn.body[last].test


def _fresh(seed: int = 0):
    """A namespace with the math builtins a numeric assert may use.

    Deliberately NOT the full builtins: a differential oracle must not be able
    to import, open files, or reach the network, or a corpus unit could make it
    print anything it likes.
    """
    import math
    safe = {k: getattr(math, k) for k in
            ("sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan",
             "floor", "ceil", "fabs", "isqrt", "pow", "gcd", "hypot",
             "factorial", "inf", "nan", "pi", "e", "tau")}
    safe.update({"len": len, "min": min, "max": max, "abs": abs,
                 "sum": sum, "sorted": sorted, "reversed": reversed,
                 "round": round, "int": int, "float": float, "bool": bool,
                 "str": str, "list": list, "tuple": tuple, "set": set,
                 "dict": dict, "range": range, "enumerate": enumerate,
                 "zip": zip, "all": all, "any": any,
                 "True": True, "False": False, "None": None})
    return {"__builtins__": safe, "__name__": "__mjolnir_diff__"}


def evaluate_claim(code: str, samples: int = SAMPLES_DEFAULT, seed: int = 0):
    """SAFE-PATH scoring: evaluate the claim expression under random integer
    bindings. Reaches only claims that are pure arithmetic over the parameters
    — a small minority of the corpus, and it is reported rather than hidden.

    Returns dict(n, true, false, rate) or None when not scorable.
    """
    import random
    parts = claim_parts(code)
    if parts is None:
        return None
    setup, test = parts
    for node in ast.walk(test):
        if isinstance(node, (ast.Call,)):
            fn = node.func
            nm = fn.id if isinstance(fn, ast.Name) else None
            if nm not in ("len", "min", "max", "abs", "sum", "round",
                          "int", "float", "pow", "sorted"):
                return None
        elif isinstance(node, (ast.Attribute, ast.Lambda, ast.Subscript,
                               ast.ListComp, ast.DictComp, ast.SetComp,
                               ast.GeneratorExp, ast.Await, ast.Yield)):
            return None
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            return None
    try:
        fn = next(n for n in ast.parse(code).body
                  if isinstance(n, ast.FunctionDef))
    except StopIteration:
        return None
    params = [a.arg for a in fn.args.args]
    # setup statements must be trivial too, or the claim needs real fixtures
    for st in setup:
        for node in ast.walk(st):
            if isinstance(node, (ast.For, ast.While, ast.With, ast.Try,
                                 ast.If)):
                return None
    try:
        env = _fresh(seed)
        setup_code = compile(ast.Module(body=setup, type_ignores=[]),
                             "<setup>", "exec")
        test_code = compile(ast.Expression(test), "<claim>", "eval")
    except Exception:
        return None
    rng = random.Random(seed)
    edge = [0, 1, -1, 2, -2, 7, -3, 5, 100, -100]
    rows = []
    for i in range(samples):
        env_i = dict(env)
        assign = {}
        for j, p in enumerate(params):
            assign[p] = edge[(i + j * 3) % len(edge)] if i < len(edge) \
                else rng.randint(-50, 50)
        assign["__builtins__"] = env["__builtins__"]
        try:
            exec(setup_code, env_i, assign)
        except Exception:
            continue                       # setup failed: not scorable
        try:
            rows.append(bool(eval(test_code, env_i, assign)))
        except Exception:
            continue
    if not rows:
        return None
    return {"n": len(rows), "true": sum(rows), "false": len(rows) - sum(rows),
            "rate": sum(rows) / len(rows)}


def execute_claim(code: str, module_code: str | None = None,
                   timeout_s: float = 2.0):
    """EXEC-PATH scoring: actually run the unit's own body.

    Two reach problems, both measured rather than assumed:

    1. Only units that take no parameters are self-contained — anything with
       fixtures needs arguments we cannot invent.
    2. Even those reference MODULE-LEVEL globals (imports, constants, helper
       functions defined at file scope). Running the body in a fresh namespace
       raised NameError on nearly all of them, so the first version scored
       7 units in 8,000. Binding the module's own top level first is what makes
       the exec path worth having.

    Returns True/False (the claim held under execution) or None (could not run).
    """
    parts = claim_parts(code)
    if parts is None:
        return None
    setup, test = parts
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return None
    fn = next((n for n in parsed.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return None
    if fn.args.args or fn.args.vararg or fn.args.kwarg:
        return None                         # needs fixtures we cannot invent
    env = _fresh(0)
    g = dict(env)
    try:
        g["_TEST"] = compile(ast.Expression(test), "<claim>", "eval")
        g["_SETUP"] = compile(ast.Module(body=setup, type_ignores=[]),
                              "<setup>", "exec")
        if module_code:
            # Bind the module's own top level: its imports, constants and
            # helper defs are what the unit body actually reads.
            top = [n for n in ast.parse(module_code).body
                   if not isinstance(n, ast.FunctionDef)
                   or n.name == fn.name]
            g["_MODULE"] = compile(ast.Module(body=top, type_ignores=[]),
                                   "<module>", "exec")
    except Exception:
        return None
    try:
        import signal

        def _to(sig, frm):
            raise TimeoutError("unit_timeout")
        signal.signal(signal.SIGALRM, _to)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    except Exception:
        pass                                # not the main thread: no alarm
    try:
        if "_MODULE" in g:
            exec(g["_MODULE"], g)
        exec(g["_SETUP"], g)
        return bool(eval(g["_TEST"], g))
    except AssertionError:
        # an assert raised: the premise failed, so the claim is not reached
        return None
    except TimeoutError:
        return None
    except Exception:
        return None
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass


# --------------------------------------------------------------- scoring ----
def score_unit(name: str, code: str, line: int, path: str = "",
               module_code: str | None = None, *, exec_path: bool = False,
               samples: int = SAMPLES_DEFAULT):
    """Score one unit. Returns a row dict, or None if not scorable."""
    cert, _, _ = certify.certify_unit(name, code, line, path)
    verdict = cert["status"]
    obs = None
    if exec_path:
        obs = execute_claim(code, module_code=module_code)
        if obs is None:
            r = evaluate_claim(code, samples=samples)
            obs = None if r is None else (r["rate"] >= 0.999)
    else:
        r = evaluate_claim(code, samples=samples)
        obs = None if r is None else (r["rate"] >= 0.999)
    try:
        claim_src = ast.unparse(claim_parts(code)[1]) if \
            claim_parts(code) else ""
    except Exception:
        claim_src = ""                     # a row must stay JSON-serialisable
    row = {"path": path, "name": name, "line": line, "verdict": verdict,
           "reason": cert.get("reason", ""), "exec_holds": obs,
           "claim": claim_src}
    if obs is None:
        row["outcome"] = "unscored"
    elif verdict == "CERTIFIED":
        row["outcome"] = "false_certificate" if not obs else "certified_ok"
    elif verdict == "REFUTED":
        row["outcome"] = "wrong_refutation" if obs else "refuted_ok"
    else:
        row["outcome"] = "headroom" if obs else "declined_correctly"
    return row


def score_corpus(units, *, exec_path: bool = False,
                 samples: int = SAMPLES_DEFAULT, progress=None):
    """Score an iterable of (name, code, line, path[, module_code]) rows."""
    import collections
    rows = []
    counts = collections.Counter()
    for i, unit in enumerate(units):
        name, code, line, path = unit[:4]
        module_code = unit[4] if len(unit) > 4 else None
        r = score_unit(name, code, line, path, module_code,
                       exec_path=exec_path, samples=samples)
        if r is None:
            counts["not_scorable"] += 1
            continue
        rows.append(r)
        counts[r["outcome"]] += 1
        if progress and i % 500 == 0:
            progress(i, dict(counts))
    return rows, counts

# ===========================================================================
# The SOUND reach extension: REAL fixtures, via a test-framework adapter.
#
# The safe-eval and exec paths above both fail on the same population, and for
# the same reason: they must SUPPLY the value the claim is about. Guessing it is
# unsound in BOTH directions — a guessed value on which the claim holds proves
# nothing about the real value, and a guessed value on which it fails refutes
# nothing either.
#
# So: don't guess. Ask the language's own TEST FRAMEWORK to supply it. A
# framework reports PASSED or FAILED for a real run, which is ground truth we
# did not write.
#
# LANGUAGE INDEPENDENCE. This scorer knows nothing about pytest, or any
# framework. It asks an adapter registry; the invocation is per-framework and
# lives behind the same boundary as every other language-specific concern
# (`frontend_pytest.py`, alongside frontend_sql / frontend_lean2 / frontend_tptp).
# A future `cargo test` or JUnit adapter is a new file and no change here.
#
#     Python / pytest   frontend_pytest     (written)
#     Rust  / cargo     adapter             (not written)
#     Go    / go test   adapter             (not written)
#
# An adapter that cannot run reports UNAVAILABLE and every unit becomes NO
# VERDICT. A differential oracle that cannot run must say "I don't know" — never
# "it passed".
# ===========================================================================

#: name -> import path of a test-framework adapter
ADAPTERS = {"pytest": "mjolnir_prover.lib.frontend_pytest"}


def adapter(name: str = "pytest"):
    """Load a test-framework adapter, or None if it is not installed."""
    import importlib
    try:
        return importlib.import_module(ADAPTERS[name])
    except Exception:                                       # noqa: BLE001
        return None


def available_adapters() -> list:
    out = []
    for n in ADAPTERS:
        a = adapter(n)
        if a is not None and a.available():
            out.append(n)
    return out


def run_with_adapter(path: str, names, adapter_name: str = "pytest",
                     timeout_s: float = 120.0) -> dict:
    """{name: outcome} via `adapter_name`, or all-NO-VERDICT if unavailable."""
    a = adapter(adapter_name)
    if a is None or not a.available():
        return {n: "UNAVAILABLE" for n in names}
    return a.run(path, names, timeout_s=timeout_s)


def score_with_adapter(files, *, adapter_name: str = "pytest",
                       timeout_s: float = 120.0, progress=None):
    """Score every test unit by running it with ITS OWN fixtures.

    `files` is [(name, code, line, path)] grouped by path. One framework
    invocation per FILE: per-test invocations would pay a process start each,
    and a file's tests share imports and fixtures.

    Ground truth, by outcome:
        PASSED  -> the claim held for the real fixture values
        FAILED  -> the claim was false for the real fixture values
        anything else -> NO verdict, and it is reported as such

    The asymmetry is the point of this route: the safe-eval and exec paths
    could only reach self-contained units, and a GUESSED fixture would be
    unsound in both directions. Here the framework supplies the value, so a
    PASSED is real evidence about the real claim.
    """
    import collections
    import time
    by_file = collections.defaultdict(list)
    for unit in files:
        # accept (name, code, line, path) and the 5-tuple the CLI also builds
        by_file[unit[3]].append(unit[:4])
    rows = []
    # PRECISION is a separate axis from reach. Reach asks "how much did we
    # score?"; precision asks "of what mjolnir CERTIFIED, how much did an
    # oracle we did not write confirm?" A high reach with almost no
    # certificates is a high scoreboard over nothing, so the two are reported
    # together and never as one number.
    counts = collections.Counter()
    t0 = time.time()
    for i, (path, group) in enumerate(sorted(by_file.items())):
        names = [n for (n, _c, _l, _p) in group]
        outcomes = run_with_adapter(path, names, adapter_name=adapter_name,
                                    timeout_s=timeout_s)
        with _quiet():
            certs = {n: certify.certify_unit(n, c, l, p)[0]
                     for (n, c, l, p) in group}
        for (n, c, l, p) in group:
            o = outcomes.get(n, "NOTCOLLECTED")
            verdict = certs[n]["status"]
            if o == "PASSED":
                # the claim held for real values. A CERTIFIED here agrees; an
                # abstention is CONSERVATIVE, not wrong.
                outcome = ("certified_ok" if verdict == "CERTIFIED"
                           else "headroom")
            elif o == "FAILED":
                # the claim was false for real values. A CERTIFIED here would
                # be a FALSE CERTIFICATE.
                outcome = ("false_certificate" if verdict == "CERTIFIED"
                           else "declined_correctly")
            else:
                outcome = "unscored"
                counts[f"no_verdict:{o}"] += 1
            rows.append(dict(path=p, name=n, line=l, verdict=verdict,
                             framework=o, outcome=outcome,
                             claim=_claim_text(c)))
            counts[outcome] += 1
            if verdict == "CERTIFIED":
                counts["certificates_existing"] += 1
        if progress:
            progress(i, len(by_file), dict(counts), time.time() - t0)
    # precision = of the certificates that EXISTED, how many the oracle could
    # check at all, and how many it confirmed. Report the denominator too: an
    # oracle that verified 1 of 4 has a perfect record and almost no evidence.
    existing = counts.get("certificates_existing", 0)
    verified = counts.get("certified_ok", 0)
    checked = verified + counts.get("false_certificate", 0)
    counts["precision"] = dict(
        certificates_existing=existing,
        certificates_checkable=checked,
        certificates_confirmed=verified,
        false_certificates=counts.get("false_certificate", 0),
        certificates_with_no_verdict=existing - checked,
        precision=(verified / checked) if checked else None,
        coverage_of_certificates=(checked / existing) if existing else None)
    return rows, counts


class _quiet:
    def __enter__(self):
        import contextlib
        import io
        self._c = contextlib.redirect_stdout(io.StringIO())
        return self._c.__enter__()

    def __exit__(self, *a):
        return self._c.__exit__(*a)


def _claim_text(code: str) -> str:
    try:
        parts = claim_parts(code)
        return ast.unparse(parts[1]) if parts else ""
    except Exception:                                        # noqa: BLE001
        return ""
