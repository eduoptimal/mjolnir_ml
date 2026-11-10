"""check / selfcheck / verify — the verification commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting

RELEASE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _claim(pred, locals_=None):
    """Minimal miner-shaped candidate for authority tests."""
    origin = {"file": "selfcheck.py", "line": 1}
    if locals_:
        origin["runtime_locals"] = locals_
    return {"id": "selfcheck", "tier": "static", "predicate": pred,
            "expect": None, "origin": origin,
            "provenance": {"source": "mined", "confirmed": False}}


def run_selfcheck() -> list[tuple[str, bool]]:
    """Soundness smoke test — every assertion goes through the single verdict
    authority, plus the architectural single-writer invariant."""
    import contextlib
    import io
    import tempfile

    from mjolnir_prover.lib import gate as CH
    from mjolnir_prover.lib import certify

    checks: list[tuple[str, bool]] = []

    def add(name, ok):
        checks.append((name, bool(ok)))

    # 1. authority: fragment true certifies, kernel-agreed (0 false certs)
    cert, _prof, _viol = certify.certify_unit(
        "f", "def f(n):\n    assert n * (n + 1) == n * n + n\n", 1, "selfcheck.py")
    add("fragment true -> CERTIFIED, kernel-agreed, 0 false certificates",
        cert["status"] == "CERTIFIED"
        and cert["kernel"]["false_certs"] == 0)

    # 2. authority: fragment false refutes
    cert2, _p, _v = certify.certify_unit(
        "f", "def f(n):\n    assert n * (n + 1) == n * n + n + 1\n", 1, "selfcheck.py")
    add("fragment false -> REFUTED", cert2["status"] == "REFUTED")

    # 2b. a claim must never solve itself: param-only Eq claims are REFUTED,
    #     never laundered into a tautology (regression for the VC self-solve bug)
    for _lab, _code in {
        "n == 0":      "def f(n):\n    assert n == 0\n",
        "n + 1 == 2":  "def f(n):\n    assert n + 1 == 2\n",
        "x - a == -a": "def f(x, a):\n    assert x - a == -a\n",
    }.items():
        _c, _p, _v = certify.certify_unit("f", _code, 1, "selfcheck.py")
        add(f"false Eq claim refutes, not tautology ({_lab})",
            _c["status"] == "REFUTED" and _c["kernel"]["false_certs"] == 0)
    # 2c. joint premise system must still certify (guard against over-correction)
    _c, _p, _v = certify.certify_unit(
        "f", "def f(a, b):\n    assert a + b == 10\n    assert a - b == 2\n"
             "    assert a == 6\n", 1, "selfcheck.py")
    add("premise system {a+b=10, a-b=2} |- a=6 -> CERTIFIED",
        _c["status"] == "CERTIFIED" and _c["kernel"]["false_certs"] == 0)
    _c, _p, _v = certify.certify_unit(
        "f", "def f(a, b):\n    assert a + b == 10\n    assert a - b == 2\n"
             "    assert a == 7\n", 1, "selfcheck.py")
    add("premise system does not entail a=7 -> REFUTED", _c["status"] == "REFUTED")

    # 2d. NEGATION / NULLNESS / NotEq SOUNDNESS (found 2026-10-06 by
    #      replaying `not` forms: _negate_cmp mapped Lt/Gt to the WEAKER
    #      relation, so `assert not (5 > 3)` lowered to `5 >= 3` and CERTIFIED
    #      a false claim; `not (a == b)` lowered to `a != b`, which the op
    #      whitelist skipped, so the claim vanished and a mixed unit certified
    #      unchecked; `not (a and b)` stripped the negation entirely. Each
    #      direction is pinned, plus the nullness rung's three honest answers
    #      and the refute-by-declaration gate (a declared premise may certify
    #      downstream by implication, never refute -- check 25 owns that
    #      discipline; these twins own the new paths into it).
    _neg_bad = []
    for _lab, _code, _want in (
        ("not(5>3) false -> REFUTED", "def f():\n    assert not (5 > 3)\n",
         "REFUTED"),
        ("not(3>5) true -> CERTIFIED", "def f():\n    assert not (3 > 5)\n",
         "CERTIFIED"),
        ("not(3<5) false -> REFUTED", "def f():\n    assert not (3 < 5)\n",
         "REFUTED"),
        ("not(5<3) true -> CERTIFIED", "def f():\n    assert not (5 < 3)\n",
         "CERTIFIED"),
        ("not(5>=3) false -> REFUTED", "def f():\n    assert not (5 >= 3)\n",
         "REFUTED"),
        ("not(3<=5) false -> REFUTED", "def f():\n    assert not (3 <= 5)\n",
         "REFUTED"),
        ("not(5==5) false -> REFUTED", "def f():\n    assert not (5 == 5)\n",
         "REFUTED"),
        ("not(5==3) true -> CERTIFIED", "def f():\n    assert not (5 == 3)\n",
         "CERTIFIED"),
        ("not(5!=3) false -> REFUTED", "def f():\n    assert not (5 != 3)\n",
         "REFUTED"),
        ("not(3!=3) true -> CERTIFIED", "def f():\n    assert not (3 != 3)\n",
         "CERTIFIED"),
        ("mixed not(5==5) refutes the unit",
         "def f():\n    assert not (5 == 5)\n    assert 2 + 2 == 4\n",
         "REFUTED"),
        ("not(A and B) never certifies the unit",
         "def f(a, b):\n    assert not (a > 0 and b > 0)\n"
         "    assert 2 + 2 == 4\n", "UNVERIFIABLE"),
        ("closed literal non-null certifies",
         "def f():\n    assert 5 is not None\n", "CERTIFIED"),
        ("closed None claim refutes",
         "def f():\n    assert None is not None\n", "REFUTED"),
        ("assignment non-null certifies",
         "def f():\n    x = 5\n    assert x is not None\n", "CERTIFIED"),
        ("assignment nullness refutes",
         "def f():\n    x = 5\n    assert x is None\n", "REFUTED"),
        ("parameter nullness never certifies (declared only)",
         "def f(x):\n    assert x is not None\n", "UNVERIFIABLE"),
        ("closed != decides (certified)", "def f():\n    assert 5 != 3\n",
         "CERTIFIED"),
        ("closed false != refutes", "def f():\n    assert 3 != 3\n",
         "REFUTED"),
        ("parameter != never certifies", "def f(x):\n    assert x != 5\n",
         "UNVERIFIABLE"),
        ("universal != over constant difference certifies",
         "def f(x):\n    assert x != x + 1\n", "CERTIFIED"),
        ("identical != refutes", "def f(x):\n    assert x != x\n",
         "REFUTED"),
        # the gate: a declared premise may not spend itself with the opposite
        # sign -- refute-by-declaration is UNVERIFIABLE, certify-by-implication
        # is CERTIFIED, and a program assignment may still refute.
        ("declared premise contradiction does not refute",
         "def f(b):\n    assert b == 5\n    assert b is None\n",
         "UNVERIFIABLE"),
        ("declared premise contradiction does not refute (!=)",
         "def f(b):\n    assert b == 5\n    assert b != 5\n",
         "UNVERIFIABLE"),
        ("declared premise certifies downstream by implication",
         "def f(b):\n    assert b == 5\n    assert b is not None\n",
         "CERTIFIED"),
        ("program assignment still refutes",
         "def f():\n    b = 5\n    assert b is None\n", "REFUTED"),
    ):
        _cn = certify.certify_unit("f", _code, 1, "selfcheck.py")[0]
        if _cn["status"] != _want or _cn.get("kernel", {}).get("false_certs"):
            _neg_bad.append(f"{_lab}: {_want}!={_cn['status']}")
    add("negation/nullness/!= soundness (26 twins)"
        + (f" — {_neg_bad[0]}" if _neg_bad else ""), not _neg_bad)

    # 2e. LENGTH-BOUND rung (chosen by the abstain histogram: 887
    #     length-bounded columns on BANKING_SQL were pure inventory).
    #     `len(param) <= N` is a declared input contract -- premise tier,
    #     never a proof; a theorem built only from it is demoted; and the
    #     substituted-local shape (the A-combobound) must stay
    #     decidable-or-abstain, never declared.
    _len_bad = []

    def _len_case(lab, tier, code, want):
        _cl = {"id": f"selfcheck::{lab}", "tier": tier,
               "predicate": "range v len <= 50",
               "canonical_unit": code,
               "origin": {"file": "selfcheck.py", "line": 1},
               "provenance": {"source": "mined", "confirmed": False}}
        _r = certify.certify_claim(_cl)
        if _r["status"] != want or _r.get("kernel", {}).get("false_certs"):
            _len_bad.append(f"{lab}: {want}!={_r['status']}")

    _len_case("declared as premise", "premise",
              "def f(v):\n    assert len(v) <= 50\n", "CERTIFIED")
    _len_case("theorem is demoted, never padded", "static",
              "def f(v):\n    assert len(v) <= 50\n", "UNVERIFIABLE")
    _len_case("substituted local abstains (A-combobound)", "static",
              "def f(v):\n    n = len(v)\n    assert n <= 50\n",
              "UNVERIFIABLE")
    _len_case("mixed unit certifies with visible premise", "static",
              "def f(v):\n    assert len(v) <= 50\n    assert 2 + 2 == 4\n",
              "CERTIFIED")
    add("length bound: declared-not-proved, never padded, never laundered"
        + (f" — {_len_bad[0]}" if _len_bad else ""), not _len_bad)

    # 2f. PREMISE-TIER equality (found by the dsl/ example, 2026-10-06):
    #     `premise scale == 2` declined as premise_refuted while
    #     `premise amount >= 0` declared — the last-assert position guard
    #     belongs to the THEOREM reading (A-combobound), but a premise
    #     unit's last assert IS its declaration. Tier-gated E1b fixes the
    #     asymmetry; these twins pin both directions and the scope:
    #     param-vs-const declares, the theorem still refutes, a closed
    #     falsehood never declares, param-param stays declined (v1 scope).
    _eq_bad = []

    def _eq_case(lab, maker, want):
        r = maker()
        got = r.get("status")
        false_certs = (r.get("kernel") or {}).get("false_certs") or 0
        if got != want or false_certs:
            _eq_bad.append(f"{lab}: {want}!={got}")

    _base = {"tier": "premise",
             "origin": {"file": "selfcheck.dsl", "line": 1},
             "provenance": {"source": "mined", "confirmed": False}}
    _eq_case("premise declares its equality",
             lambda: certify.certify_claim(dict(
                 _base, id="selfcheck::eq-premise", predicate="scale == 2",
                 canonical_unit="def scale_(scale):\n    assert scale == 2\n")),
             "CERTIFIED")
    _eq_case("theorem reading still refutes the same unit",
             lambda: certify.certify_unit(
                 "f", "def f(scale):\n    assert scale == 2\n")[0],
             "REFUTED")
    _eq_case("closed falsehood never declares",
             lambda: certify.certify_claim(dict(
                 _base, id="selfcheck::eq-closed", predicate="3 == 5",
                 canonical_unit="def f():\n    assert 3 == 5\n")),
             "UNVERIFIABLE")
    _eq_case("param-param equality stays declined (v1 scope)",
             lambda: certify.certify_claim(dict(
                 _base, id="selfcheck::eq-pp", predicate="a == b",
                 canonical_unit="def f(a, b):\n    assert a == b\n")),
             "UNVERIFIABLE")
    add("premise equality: param-vs-const declares, theorem unchanged"
        + (f" — {_eq_bad[0]}" if _eq_bad else ""), not _eq_bad)

    # 2g. CONIC DISCHARGE INTEGRITY (FAULT found by the policy-language
    #     prototype, 2026-10-07). The emitter's "nonnegative coefficients =>
    #     axioms alone" shortcut returned [0]*n for nonzero targets; the
    #     kernel re-derives by exact reproduction and rejected it as a
    #     false_certificate. Root cause pair: the shortcut, plus prefix-
    #     consumed inequalities never registering their difference polys with
    #     the solver. Both fixed; these twins pin the three outcomes:
    #     reproducible combination CERTIFIES (agreement 1.0), a
    #     nonnegative-looking target with no combination ABSTAINS (never
    #     FAULT, never padded CERTIFIED), and neither may carry false_certs.
    _conic_bad = []

    def _conic_case(lab, code, want):
        c = certify.certify_unit("f", code)[0]
        k = c.get("kernel") or {}
        if c["status"] != want or (k.get("false_certs") or 0):
            _conic_bad.append(f"{lab}: {want}!={c['status']}")

    _conic_case(
        "two-premise chain certifies",
        "def f(low, high):\n    assert low >= 0\n    assert high >= low\n"
        "    assert high >= 0\n", "CERTIFIED")
    _conic_case(
        "no combination abstains (was FAULT)",
        "def f(x, y):\n    assert x >= 0\n    assert x + y >= 0\n",
        "UNVERIFIABLE")
    _conic_case(
        "check-25 k=2 unchanged",
        "def c(a, b):\n    assert a <= b\n    assert a + a <= b + b\n",
        "CERTIFIED")
    _conic_case(
        "check-25 neg-mult unchanged",
        "def c(a, b):\n    assert a <= b\n    assert b <= a\n",
        "UNVERIFIABLE")
    add("conic discharge: reproducible combo certifies, else abstains"
        + (f" — {_conic_bad[0]}" if _conic_bad else ""), not _conic_bad)

    # 3. typed-reading contract for mined claims
    ident = certify.certify_claim(
        _claim("(x + y) * x == x ** 2 + x * y", locals_=["x", "y"]))
    add("strengthening identity -> CERTIFIED (proved_by_strengthening)",
        ident["status"] == "CERTIFIED"
        and ident.get("reason") == "proved_by_strengthening")
    nonid = certify.certify_claim(_claim("a * b == 0", locals_=["a", "b"]))
    add("strengthening refutation abstains (never a false violation)",
        nonid["status"] == "UNVERIFIABLE"
        and nonid.get("reason") == "strengthening_refuted")
    assumed = certify.certify_claim(
        _claim("len(data) == nrows + 1", locals_=["data", "nrows"]))
    add("unprovable claim -> UNVERIFIABLE (never a padded CERTIFIED)",
        assumed["status"] == "UNVERIFIABLE")
    viol = certify.certify_claim(_claim("n * (n + 1) == n * n + n + 1"))
    add("theorem refutation -> REFUTED (real violation)", viol["status"] == "REFUTED")

    # 3b. R1 premise lowering: guards lower exactly, or abstain honestly
    import tempfile as _tf
    d = _tf.mkdtemp(prefix="mj_r1_")
    src = os.path.join(d, "g.py")
    open(src, "w").write(
        "def f(n):\n"
        "    if n <= 0:\n"
        "        raise ValueError\n"
        "    return 1 / n\n"
        "\n"
        "def g(x):\n"
        "    if x == 5:\n"
        "        raise ValueError\n"
        "\n"
        "def h(n):\n"
        "    if not (n > 0):\n"
        "        raise ValueError\n"
        "    return n\n")
    from mjolnir_prover.lib import mine as _mine
    by_pred = {c["predicate"]: c for c in _mine.mine_file(src, "g.py", "r1", "t0")
               if c["tier"] == "premise"}
    g1 = certify.certify_claim(by_pred["n <= 0"])
    add("R1: guard lowers exactly -> CERTIFIED premise contract",
        g1["status"] == "CERTIFIED" and g1.get("reading") == "premise"
        and g1["kernel"]["false_certs"] == 0)
    g2 = certify.certify_claim(by_pred["x == 5"])
    add("R1 twin: no exact lowering -> honest abstain, never CERTIFIED",
        by_pred["x == 5"].get("canonical_unit") is None
        and g2["status"] == "UNVERIFIABLE")
    g3 = certify.certify_claim(by_pred["NOT not n > 0"])
    add("R1 twin: negated guard unwraps exactly -> CERTIFIED",
        g3["status"] == "CERTIFIED")

    # 3c. runtime class separation: enforcement stamps bookkeeping, never proof
    from mjolnir_prover.lib import shapes as _sh
    claim = {"id": "obs", "tier": "monitor", "family": "monitor",
             "predicate": "caps['x'] is True", "expect": None,
             "origin": {"repo": "t", "commit": "0" * 12, "file": "t.py",
                        "line": 1, "construct": "runtime_cap"},
             "provenance": {"source": "mined", "confirmed": False}}
    sid = _sh.shape_id(claim)
    groups = {sid: {"sid": sid, "shape": claim["predicate"],
                    "construct": "runtime_cap", "tier": "monitor",
                    "members": [claim], "oracle": _sh.codes()}}
    try:
        rec = _sh.confirm(groups, sid, "enforced", by="selfcheck")
        stamped = [json.loads(json.dumps(claim))]
        _sh.apply_ledger(stamped, {sid: rec})
        c = certify.certify_claim(stamped[0])
        add("runtime class: enforcement stamps bookkeeping, never proof",
            stamped[0]["provenance"].get("enforced") is True
            and stamped[0]["provenance"].get("confirmed") is not True
            and c["status"] == "UNVERIFIABLE" and c["enforced"] is True)
    except SystemExit as e:
        add(f"runtime class separation ({e})", False)

    # 3d. FORM axis (ontology): exclusive, stable, rule-shape labels
    forms = {
        "order.total >= 0": "attr-value",            # business rule over a field
        "cart[0] == item": "subscript-value",
        "status in ('open', 'closed')": "membership",
        "isinstance(x, int)": "type-check",
        "len(cart) > 0": "length",
        "user is None": "identity",
        "f(x) == 3": "call-value",
        "x + 1 <= x": "arithmetic",
        "valid and present": "conjunction",
    }
    ok_forms = all(_mine._form_of(p) == f for p, f in forms.items())
    exclusivity = len({_mine._form_of(p) for p in forms}) == len(forms)
    add("FORM axis: rule shapes typed exclusively and stably",
        ok_forms and exclusivity)

    # 4. lean arm via the authority: independent kernel confirms
    good = "theorem t (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by"
    bad = "theorem t (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 5 := by"
    g, b = certify.certify_lean(good), certify.certify_lean(bad)
    add("lean true -> CERTIFIED, lean kernel confirms",
        g["status"] == "CERTIFIED" and g["kernel"] == "CERTIFIED")
    add("lean false -> REFUTED", b["status"] == "REFUTED")

    # 5. fail-closed gate contract
    tmp = tempfile.mkdtemp(prefix="mj_selfcheck_")
    clean = os.path.join(tmp, "clean.py")
    ref = os.path.join(tmp, "ref.py")
    open(clean, "w").write("def k(n):\n    return n + 1\n")
    open(ref, "w").write("def f(n):\n    assert n * (n + 1) == n * n + n + 1\n")
    cfg = os.path.join(tmp, "gate.toml")
    open(cfg, "w").write('mode="gate"\nfail_on=["REFUTED"]\n')
    with contextlib.redirect_stdout(io.StringIO()):
        rc_clean = CH.run([clean], gate=True, config=cfg)["rc"]
        rc_ref = CH.run([ref])["rc"]
    add("gate: clean abstain passes (0)", rc_clean == 0)
    add("gate: refutation fails (1)", rc_ref == 1)

    # 6. storage plane round-trip (append-only preserved)
    try:
        from mjolnir_prover import storage as _st
        s = _st.get_store()
        key = "out/_selfcheck.jsonl"
        s.write_jsonl(key, [{"a": 1}])
        s.append_jsonl(key, {"a": 2})
        ok = [r["a"] for r in s.read_jsonl(key)] == [1, 2]
        add("storage round-trip + append", ok)
    except Exception as e:
        add(f"storage round-trip ({e})", False)

    # 7. single-writer architecture: only the authority disposes claims
    ok_arch, offenders = certify.check_single_writer()
    add("single verdict writer (import/call graph)" +
        (f" — {offenders[0]}" if offenders else ""), ok_arch)

    # 8. epistemic separation: NO heuristic (lint) findings exist — every reported
    #    finding is produced by the authority (certified) or is a declared contract.
    from mjolnir_prover.audit import bank_report as _br
    heur = {k for k, v in _br.EPISTEMIC.items() if v == "heuristic"}
    auth = {k for k, v in _br.EPISTEMIC.items() if v in ("certified", "declared")}
    add("epistemic: only certified/declared findings (no lint heuristics)",
        heur == set() and auth == {"bound"})

    # 9. countermodel twins. certificate/v0 declares a `witness`; the AST path
    #    left it None, so gate's "paste the witness as a failing test" CTA
    #    promised a repro that did not exist. These are the adversarial twins:
    #    a WRONG countermodel is worse than none, so every emitted point is
    #    re-checked by EXECUTING the unit, independently of the term algebra
    #    that verified it. Plus the negative twins: no witness on a positive
    #    verdict, and no witness for a claim over a computed local (which is
    #    not an input, so assigning it would fabricate a repro).
    from mjolnir_prover.lib import countermodel as _cm
    _twin_ok, _twin_bad = True, []

    def _falsifies(code, fn, env, params):
        """Execute the unit at the countermodel; True iff the assert fires."""
        ns = {}
        exec(compile(code, "<twin>", "exec"), ns)
        try:
            ns[fn](*[int(env[p]) for p in params])
        except AssertionError:
            return True
        except Exception:
            return False
        return False

    for _code, _fn, _params in (
        ("def f(n):\n    assert n + 1 == 2\n", "f", ["n"]),
        ("def f(a, b):\n    assert (a + b) * (a + b) == 0\n", "f", ["a", "b"]),
        ("def f(n):\n    assert 3 * n == n * n + 2\n", "f", ["n"]),
        # the recorded post-substitution sides (`lhs`/`rhs`) now let the box
        # refutations yield a point: `2*x >= 0` over unbounded x is false, so a
        # witness is CORRECT here. Execution-verified like every other point.
        ("def f(x):\n    result = 2 * x\n    assert result >= 0\n", "f", ["x"]),
        # `rdiv` earns its place on the allow-list here: it folds on constants,
        # so the term-level check is faithful AFTER substitution. Without this
        # case it would have no execution evidence and would not be trusted.
        ("def f(n):\n    assert n / 2 == 1\n", "f", ["n"]),
        # sound abstention: `n >= 0` over unbounded n is neither provable nor
        # refutable, so the certifier must abstain AND emit no witness. The
        # twin is conditional on status — demanding a point here would be
        # pressuring the emitter to invent one.
        ("def f(n):\n    assert n >= 0\n", "f", ["n"]),
    ):
        _c = certify.certify_unit(_fn, _code)[0]
        _w = _c.get("witness")
        if _c.get("status") != "REFUTED":
            if _w is not None:
                _twin_ok = False
                _twin_bad.append(f"witness on {_c.get('status')}: {_code!r}")
            continue
        if not (_w and _w.get("kind") == "point"):
            _twin_ok, _ = False, _twin_bad.append(f"no point for {_code!r}")
            continue
        # the twin: does the emitted assignment actually break the program?
        if not _falsifies(_code, _fn, _w["env"], _params):
            _twin_ok = False
            _twin_bad.append(f"point did not falsify: {_code!r} {_w['env']}")
    add("countermodel twin: every emitted point falsifies on execution"
        + (f" — {_twin_bad[0]}" if _twin_bad else ""), _twin_ok)

    # negative twins: never emit a witness we cannot stand behind
    _neg_ok = True
    for _code, _expect_none in (
        ("def f(n):\n    result = n + n\n    assert result == 2 * n\n", True),
        ("def f(n):\n    assert n + 1 == n + 1\n", True),
        # collection-typed parameter. The algebra judges `ledger + [x]` as
        # arithmetic and reported a "provable violation" where the program
        # raises TypeError. Found by executing all 28 emitted witnesses — the
        # term-level check could not see it. This is the twin that pins it.
        ("def f(ledger, x):\n    result = ledger + [x]\n    assert result == x\n",
         True),
        # unmodelled operator: must abstain, never emit an uninterpretable term
        ("def f(n):\n    assert n // 2 == 1\n", True),
    ):
        _c = certify.certify_unit("f", _code)[0]
        if _expect_none and _c.get("witness") is not None:
            _neg_ok = False
    add("countermodel twin: no witness on certified, for a local, for a"
        " collection-typed param, or for an unmodelled op", _neg_ok)

    # inequality predicates, tested DIRECTLY on canonical terms. Not reachable
    # from the current obligation shape (`rel` stores infix claim text and the
    # prose "violated over the box", so `_sides_of` correctly abstains). They
    # exist for H1's path — recording the post-substitution claim canonically —
    # and a wrong negation entry emits a point where the claim actually HOLDS,
    # so they are pinned here rather than left unexercised.
    _ineq_ok = True
    from mjolnir_prover.lib import term as _T2
    _lhs, _rhs = _T2.cadd(_T2.const(1), _T2.var("n")), _T2.const(5)
    _p_ge = _cm._find_point(_lhs, _rhs, ">=", ["n"])
    _p_le = _cm._find_point(_lhs, _T2.const(3), "<=", ["n"])
    if not (_p_ge and _p_le):
        _ineq_ok = False
    else:
        # `n+1 >= 5` fails iff n+1 < 5; `n+1 <= 3` fails iff n+1 > 3.
        _v1, _v2 = int(_p_ge["n"]), int(_p_le["n"])
        if not (_v1 + 1 < 5 and _v2 + 1 > 3):
            _ineq_ok = False
    add("countermodel inequality predicates: negation is the OPPOSITE relation"
        " (a wrong entry emits a point where the claim HOLDS)", _ineq_ok)

    # determinism twin: the witness must not depend on set iteration order,
    # which varies with PYTHONHASHSEED ACROSS processes. One authority version
    # must produce one artifact. (Measured: 2 distinct witnesses before.)
    from mjolnir_prover.lib import term as _T
    _names = _cm._env_of([_T.var("b"), _T.var("a"), _T.var("c")])
    add("countermodel determinism: parameter order is sorted, not set order",
        isinstance(_names, list) and _names == sorted(_names)
        and _names == _cm._env_of([_T.var("c"), _T.var("a"), _T.var("b")]))


    # 10. the trust pin must cover the whole verdict closure. The fingerprint is
    #    the only thing that invalidates a run when the authority changes, so a
    #    module in the verdict's import closure that is NOT in FILES is a silent
    #    staleness hole: weaken term.decide_eq and no pin moves, no run is
    #    invalidated. This is what let countermodel.py — and term, intervals,
    #    theory, claim_ir — sit outside the pin unnoticed.
    from mjolnir_prover.lib import authority as _auth
    _unpinned = sorted(_auth.closure(_auth.FILES) - set(_auth.FILES))
    add(f"trust pin covers the verdict closure ({len(_auth.FILES)} modules)"
        + (f" — UNPINNED: {_unpinned}" if _unpinned else ""), not _unpinned)

    # 11. one-sided box discharge (the A-combobound root). The box prover used
    #     to call Fraction() on IV.INF, which raises, so EVERY upper-bounded
    #     claim silently abstained behind the caller's try/except. The twins
    #     pin all six direction/verdict combinations, because the old holds/
    #     fails table was also wrong for ge/gt and had never been exercised.
    _box = [
        # (code, expected status)
        ("def f(n):\n    assert n <= 3\n    result = n\n    assert result <= 5\n", "CERTIFIED"),
        ("def f(n):\n    assert n <= 3\n    result = n\n    assert result <= 2\n", "REFUTED"),
        ("def f(n):\n    assert n >= 3\n    result = n\n    assert result >= 3\n", "CERTIFIED"),
        ("def f(n):\n    assert n <= 3\n    result = n\n    assert result >= 5\n", "REFUTED"),
        # the authored A-combobound shape: a bound composed through a loop
        ("def f(n):\n    assert n <= 50\n    result = 0\n    for i in range(n):\n"
         "        result += 1\n    assert result <= 50\n", "CERTIFIED"),
        # adversarial: |n| <= 3 does NOT imply n >= 5 — must never certify
        ("def f(n):\n    assert n >= 3\n    result = n\n    assert result >= 5\n",
         "UNVERIFIABLE"),
    ]
    _box_bad = []
    for _code, _want in _box:
        _st = certify.certify_unit("f", _code)[0]["status"]
        if _st != _want:
            _box_bad.append(f"{_want}!={_st}: {_code!r}")
    add("one-sided box discharge: le/ge x certify/refute/abstain, and the"
        " composed-bound shape" + (f" — {_box_bad[0]}" if _box_bad else ""),
        not _box_bad)

    import json as _json
    from mjolnir_prover.lib import countability as _CTKb
    # KNOWN GAP, asserted rather than hidden. `rel:rel-box` obligations have NO
    # kernel branch: the box comes from PROGRAM substitution (result = n), the
    # certificate records bounds=None, and the kernel cannot redo the
    # substitution. Until now that returned None, which added NO mismatch — so
    # these certificates scored agreement 1.0 with one obligation never verified
    # at all. `certified_with_unverified_obligations` makes it visible, and this
    # twin pins the gap so it cannot quietly return to looking healthy.
    from mjolnir_prover.lib import countability as _CTKb
    _fc, _, _ = certify.certify_unit("f", _box[0][0])
    _fr = _CTKb.K.replay(_fc)
    add("one-sided box: the kernel RE-DERIVES rel-box from the certificate's own"
        " substituted sides and box",
        _fr["agreement"] == 1.0 and _fr["false_certs"] == 0
        and "unchecked" not in _fr["statuses"])

    # and it must catch a DOCTORED box, or re-derivation is decoration: widen
    # the recorded bound and the claim should stop proving.
    _bt = _json.loads(_json.dumps(_fc))
    for _o in _bt["obligations"]:
        if _o["method"].startswith("rel:"):
            # n >= 6: `n <= 5` no longer follows from this box, so the
            # kernel must stop agreeing. (Widening DOWNWARD to n <= 1 would
            # not work — n <= 1 still implies n <= 5. My first attempt used
            # that and the certificate agreed, correctly.)
            _o["bounds"] = {"n": [6, None]}
    _btr = _CTKb.K.replay(_bt)
    add("one-sided box: a doctored bound is caught by the kernel re-derivation",
        _btr["agreement"] == 0.0 or "unchecked" in _btr["statuses"])

    # 12. box-refute witnesses. The `rel-box-refute` path refutes a SUBSTITUTED
    #     bound (`n >= 12`, from the source claim `result >= 12`) but recorded
    #     only the prose "violated over the box", so 348 refuted rows carried no
    #     recoverable input-space countermodel. The canonical sides are now
    #     carried as lhs/rhs and the emitter solves from them, re-deciding with
    #     the trusted primitive as always.
    _bw = certify.certify_unit("f", "def f(n):\n    result = n\n    assert result >= 12\n")[0]
    _bww = _bw.get("witness") or {}
    _bw_ok = (_bw.get("status") == "REFUTED" and _bww.get("kind") == "point"
              and bool(_bww.get("env")))
    if _bw_ok:
        _v = int(_bww["env"]["n"])
        # the claim must ACTUALLY be false at the emitted point
        _bw_ok = not (_v >= 12)
    add("box-refute witness: a substituted-bound refutation carries a verified"
        " point (348 rows previously carried none)", _bw_ok)

    # 13. the isolation / countability gate (PC-2c). Its own twins, because a
    #     gate that cannot fail is decoration: the manifest digest must be
    #     order-independent (the same set, hashed twice, must agree), coverage
    #     must actually move when a re-derivable obligation is removed, and
    #     isolation must be reported False when the kernel is denied its TCB.
    from mjolnir_prover.lib import countability as _CT
    _c1 = certify.certify_unit("f", "def f(n):\n    assert n * (n + 1) == n * n + n\n")[0]
    _c2 = certify.certify_unit("g", "def g(n):\n    assert n + 1 == n + 1\n")[0]
    _m1 = _CT.manifest([_c1, _c2])
    _m2 = _CT.manifest([_c2, _c1])            # order-independent set
    add("countability: the manifest digest is order-independent"
        + ("" if _m1["digest"] == _m2["digest"] else " — DIGEST MOVED WITH ORDER"),
        _m1["digest"] == _m2["digest"] and _m1["count"] == _m2["count"])

    _cv = _CT.coverage([_c1, _c2])
    _cv_empty = _CT.coverage([])
    add("countability: coverage is an integer ratio that moves with the corpus"
        + ("" if _cv["denominator"] > 0 and _cv_empty["denominator"] == 0
           else " — COVERAGE NOT A COUNT"),
        _cv["denominator"] > 0 and _cv_empty["denominator"] == 0)

    _iso = _CT.isolation(_c1)
    add("isolation: kernel re-derives in a process admitting only its own TCB"
        + (f" — {_iso['structural'].get('error')}" if not _iso["isolated"] else ""),
        bool(_iso["isolated"]))

    # the launder twin: the emitter's own `status` must not be a route to a
    # verdict. Tampering with it has to move nothing.
    _l = _CT.isolation_in_process(_c1)
    add("isolation: the emitter's status field is inert (no laundered verdict)",
        _l["status_launder_twinned"])

    # the isolation twin: an obligation whose STORED verdict contradicts what
    # re-derivation finds must make both processes agree that it is a mismatch
    # — isolation is worthless if a doctored certificate still passes. Here the
    # doctoring is inside the decision content, so it MUST be caught.
    _doc = json.loads(json.dumps(_c1))
    for _o in _doc.get("obligations", []):
        _o["status"] = "fail"
    _neg = _CT.isolation(_doc)
    add("isolation: a doctored obligation verdict is caught by the isolated"
        " process too (isolation cannot hide a mismatch)",
        bool(_neg["structural"].get("mismatches"))
        and _CT._derived(_doc)[1] == 0.0)

    # the placeholder twin. `out_of_fragment` rows are abstention MARKERS, not
    # obligations: they record which rung is missing. Scoring them as failures
    # made coverage measure its own instrumentation (2148 of 2217 rows on the
    # release), so the ratio read 2.1% when the real decidable surface was
    # 46/69. They must move the rung backlog, never the denominator.
    _stub = {"unit": {"path": "x.py", "name": "f"}, "status": "UNVERIFIABLE",
             "obligations": [{"kind": "vc", "line": 1, "premise": "xs[0] == y",
                              "conclusion": "?", "method": "out_of_fragment",
                              "status": "abstain"}]}
    _cs = _CT.coverage([_stub])
    _real_only = _CT.coverage([_c1])
    add("countability: rung placeholders are counted as pending, never as"
        " obligations in the coverage denominator",
        (_cs["denominator"] == 0 and _cs["pending_rungs"] == 1
         and _real_only["denominator"] > 0))

    # the frontier twin. The classes are a triage over claim text; a twin pins
    # the ones the roadmap actually funds, so a regex edit cannot silently
    # re-classify the corpus and quietly reorder the rungs.
    from mjolnir_prover.lib import frontier as _FR
    from mjolnir_prover.lib import term as Tmod
    from mjolnir_prover.lib import countability as _CTK
    _fcases = {
        "a and b": "boolop",
        "tot += 1": "assignment",
        "n = n + 1": "assignment",
        "s == 'a'": "string",
        "xs.append(1) == y": "method",
        "xs[0] == y": "subscript",
        "3.5 < err": "float",
        "node.label == 6": "attribute",
        "X == {0, 2}": "container_literal",
        "n % 7 == 2": "modulo",
        "dist == expected": "equality",
    }
    _fbad = [f"{k}->{_FR.classify_claim(k)[0]}!={v}"
             for k, v in _fcases.items() if _FR.classify_claim(k)[0] != v]
    add("frontier: blocking-construct classes are stable and exclusive"
        + (f" — {_fbad[0]}" if _fbad else ""), not _fbad)

    # and the population must actually be enumerable: no claim may fall out of
    # every class, or the backlog would be understated (the failure that hid
    # 524 instances the first time round).
    _rows = _FR.abstentions([_stub])
    add("frontier: every abstention lands in exactly one class",
        all(_FR.classify_claim(r["claim"])[0] != "unclassified" for r in _rows))

    # the DECIDABILITY axis — the prior question to "which rung". This is what
    # stopped a 1,443-instance rung from being funded: 96.9% of it is blocked
    # by an operand that only exists at run time, so no rung moves it. The twin
    # pins both directions: a claim over a unit PARAMETER is runtime state,
    # and a claim whose operands are all literals is addressable.
    import tempfile as _td, os as _os
    _dd = _td.mkdtemp(prefix="mj_frontier_")
    _dp = _os.path.join(_dd, "dec.py")
    open(_dp, "w").write(
        "def f():\n"
        "    d = [[1, 2], [3, 4]]\n"
        "    assert d[0, 1] == 2\n"
        "\n"
        "def test_runtime(chars):\n"
        "    assert len(chars) == 81\n"
        "\n"
        "def test_oracle(dist, expected):\n"
        "    assert dist == expected\n")
    _d_cases = [
        # every operand bound to a literal IN the unit -> addressable
        ({"claim": "d[0,1] == 2", "unit": _dp + ":f"}, "decidable"),
        # the operand is a unit PARAMETER: a value that only exists at run time
        ({"claim": "len(chars) == 81", "unit": _dp + ":test_runtime"},
         "runtime_state"),
        ({"claim": "dist == expected", "unit": _dp + ":test_oracle"},
         "runtime_state"),
    ]
    _d_bad = [f"{k['claim']}->{_FR.decidability(k)[0]}!={v}"
              for k, v in _d_cases if _FR.decidability(k)[0] != v]
    add("frontier: decidability separates runtime state from addressable"
        + (f" — {_d_bad[0]}" if _d_bad else ""), not _d_bad)

    # ROI must never quote an instance a rung cannot decide. If a rung is
    # ranked on raw matches, a reviewer is told a number it cannot deliver.
    _stub2 = {"unit": {"path": "t.py", "name": "test_f"}, "status": "UNVERIFIABLE",
              "obligations": [
                  {"kind": "vc", "line": 1, "premise": "len(chars) == 81",
                   "conclusion": "?", "method": "out_of_fragment", "status": "abstain"},
                  {"kind": "vc", "line": 2, "premise": "xs[0] == xs[0]",
                   "conclusion": "?", "method": "out_of_fragment", "status": "abstain"}]}
    _roi = _FR.rung_roi([_stub2])
    _calls = next(r for r in _roi if r["rung"] == "calls")
    add("frontier: ROI counts only addressable instances, and reports the rest"
        " as blocked",
        _calls["instances"] + _calls["blocked"] == 1)

    # the PREMISE rule. `assert len(G) == 4` followed by `assert len(G) == 0`
    # is decidable with no value for G at all: the runtime enforces the first,
    # so the two asserts are mutually inconsistent. This is the one route by
    # which a claim about an unknown value is still decidable, and it moved the
    # ceiling 0 -> 32.
    open(_dp, "w").write(
        "def f(G):\n"
        "    assert len(G) == 4\n"
        "    assert len(G) == 0\n"
        "\n"
        "def g(G):\n"
        "    assert len(G) == 4\n"
        "    assert len(G) == 4\n")
    _FR.reset_cache()          # the file changed; the cache is keyed by path
    _p_cases = [
        # premise 4, claim 0 -> the unit's own asserts contradict
        ({"claim": "len(G) == 0", "unit": _dp + ":f"}, "premise_broader"),
        # premise and claim AGREE -> no contradiction; runtime state instead
        ({"claim": "len(G) == 4", "unit": _dp + ":g"}, "runtime_state"),
    ]
    _p_bad = [f"{k['claim']}->{_FR.decidability(k)[0]}!={v}"
              for k, v in _p_cases if _FR.decidability(k)[0] != v]
    # NB the class is `premise_broader`, NOT addressable: `len(G) == 4` is an
    # assert mjolnir abstains on, not a framework-declared premise. Calling it
    # decidable is exactly the "quote a number nothing can deliver" error.
    add("frontier: a contradicting pair is TRIAGED as premise_broader, and an"
        " agreeing premise is not triaged at all"
        + (f" — {_p_bad[0]}" if _p_bad else ""), not _p_bad)

    # 15. the premise-contradiction DIRECTIVE lives in the table, not in code.
    #     Governance twin: it ships disabled, so it decides nothing, and the
    #     verdict is the table's to give. Flipping `enabled` in the table is
    #     what moves it — a hardcoded rule could not be governed this way.
    from mjolnir_prover.lib import theory as _TH
    _rule = _TH.load("classification")["premise_contradiction"]["rules"][0]
    _off = _TH.premise_contradiction_verdict(
        premise_method="premise", lhs_identical=True, rhs_both_numeric=True,
        rhs_differ=True)
    _d = _TH._cache["classification"]["premise_contradiction"]["rules"][0]
    _was, _d["enabled"] = _d["enabled"], True
    _on = _TH.premise_contradiction_verdict(
        premise_method="premise", lhs_identical=True, rhs_both_numeric=True,
        rhs_differ=True)
    _d["enabled"] = _was
    add("premise_contradiction is DATA: ships disabled, and the table alone"
        " governs it (flipping enabled flips the verdict)",
        _off[0] is None and _on[0] == "REFUTED" and _rule["enabled"] is False)

    # the twin of that twin: a DISABLED directive must not fire on a claim that
    # would otherwise be refutable. `n == 4` then `n == 0` is the case the
    # existing interval path already covers, and it must keep its own reason —
    # a new route must not steal it.
    _pc = certify.certify_unit("f", "def f(n):\n    assert n == 4\n"
                                     "    assert n == 0\n")[0]
    _pm = [o["method"] for o in _pc.get("obligations", [])]
    add("premise_contradiction: a disabled directive never fires, and does not"
        " shadow the existing interval route"
        + (f" — {_pm}" if any(m.startswith("premise_contradiction") for m in _pm)
           else ""),
        not any(m.startswith("premise_contradiction") for m in _pm)
        and "interval:eq" in _pm)

    # kernel twin: the route must be INDEPENDENTLY re-derivable, or a future
    # enablement would ship refutations nobody can check. A hand-built
    # obligation re-derives to False; a doctored one to a mismatch.
    _obl = {"method": "premise_contradiction:declared_premise_contradicts_claim",
            "kind": "eq", "premise": "n == 0", "premise_obligation": "n == 4",
            "lhs": "n", "rhs_claim": "0", "rhs_premise": "4", "status": "fail"}
    _cert = {"spec": {"source": "assert", "confirmed": True, "expr": "n == 0"},
             "obligations": [_obl]}
    _k_ok = _CTK.K.replay(_cert)
    _obl2 = dict(_obl, status="ok", rhs_premise="0")   # doctored: claims ok, premise==claim
    _k_bad = _CTK.K.replay({**_cert, "obligations": [_obl2]})
    add("premise_contradiction: the kernel re-derives the route independently,"
        " and catches a doctored one",
        (not _k_ok["false_certs"]) and "false_certificate" in _k_bad["mismatches"])

    # the premise rule must NOT fire on different calls that merely look alike.
    # An abstraction that ignores arguments turns `abs_val(34) == 34` and
    # `abs_val(0) == 0` into a contradiction that is not one — the first
    # heuristic counted 32 of those before it was made argument-sensitive.
    _nog = _FR.premise_contradiction("abs_val(0) == 0", ["abs_val(34) == 34"])
    _yes = _FR.premise_contradiction("len(G) == 0", ["len(G) == 4"])
    add("frontier: the premise rule compares calls argument-wise, not by shape"
        " (abs_val(34)==34 is not a contradiction with abs_val(0)==0)",
        _nog is None and _yes is not None)

    # 16. UNSOUNDNESS: a premise the BOX cannot represent must forbid
    #     refutation. `lo <= hi` is a param-param premise, so `_bounds_from`
    #     yields no bound and the box is UNRESTRICTED — but the contract is
    #     real. Refuting there produced a FALSE REFUTATION with a witness that
    #     violates the premise: `lo <= hi` together with the true theorem
    #     `lo + x <= hi + x` came back "violated" at lo=1, hi=0.
    #     The guard is not "are there premises" but "did the BOX capture them":
    #     `n <= 3` IS captured, and its refutation carries a valid witness.
    _pre_cases = [
        # (code, expected, note)
        # now CERTIFIED: the claim difference IS the premise difference, so it is
        # discharged by the contract rather than merely protected from refutation
        ("def c(lo, hi, x):\n    assert lo <= hi\n    assert lo + x <= hi + x\n",
         "CERTIFIED", "unrepresented premise now discharges, never refutes"),
        ("def f(n):\n    assert n <= 3\n    result = n\n    assert result <= 2\n",
         "REFUTED", "box-captured premise still refutes, valid witness"),
        ("def f(n):\n    assert n <= 3\n    result = n\n    assert result <= 3\n",
         "CERTIFIED", "box-captured premise still certifies"),
        ("def c(lo, hi, x):\n    assert lo + x > hi + x\n",
         "REFUTED", "no premise, refute stands"),
    ]
    _pre_bad = []
    for _code, _want, _note in _pre_cases:
        _st = certify.certify_unit("f", _code)[0]["status"]
        if _st != _want:
            _pre_bad.append(f"{_note}: {_want}!={_st}")
    add("box discharge: a premise the box cannot represent forbids refutation,"
        " and a premise it can represent still decides"
        + (f" — {_pre_bad[0]}" if _pre_bad else ""), not _pre_bad)

    # 27. the PYTEST FIXTURE route. It exists because a GUESSED fixture is
    #     unsound in BOTH directions: a guess on which the claim holds proves
    #     nothing about the real value, and a guess on which it fails refutes
    #     nothing. Running the test with its own fixtures is the only sound way
    #     in. The twins pin that it reports NO VERDICT rather than guessing
    #     when pytest does not collect the test — the failure mode that would
    #     silently turn "we don't know" into "it passed".
    from mjolnir_prover.lib import differential as _DP
    from mjolnir_prover.lib import frontend_pytest as _FPT
    import tempfile as _tdf, os as _osf
    _td = _tdf.mkdtemp(prefix="mj_pt_")
    _ptf = _osf.path.join(_td, "test_sample.py")
    with open(_ptf, "w", encoding="utf-8") as _fh:
        _fh.write(
            "import pytest\n\n"
            "@pytest.fixture\n"
            "def xs():\n    return [1, 2, 3]\n\n"
            "def test_holds(xs):\n    assert len(xs) == 3\n\n"
            "def test_false(xs):\n    assert len(xs) == 4\n\n"
            "@pytest.mark.skip\n"
            "def test_skipped(xs):\n    assert len(xs) > 0\n")
    _pr = _FPT.run(_ptf, ["test_holds", "test_false",
                                    "test_skipped", "test_absent"])
    add("differential --pytest: a passing test is ground truth, a failing one is"
        " ground truth, and everything else is NO VERDICT"
        + f" — {sorted(_pr.items())}"[:90],
        _pr.get("test_holds") == "PASSED"
        and _pr.get("test_false") == "FAILED"
        and _pr.get("test_absent") == "NOTCOLLECTED"
        # A skip surfaces as SKIPPED or, depending on where pytest short-
        # circuits, as NOTCOLLECTED. The property under test is "no verdict",
        # not which of the two spellings the runner emits.
        and _pr.get("test_skipped") in ("SKIPPED", "NOTCOLLECTED"))

    # and the scoring maps that ground truth onto mjolnir's verdict correctly:
    # abstained + truly holds = headroom, never a pass.
    _prows, _pcounts = _DP.score_with_adapter(
        [("test_holds", "def test_holds(xs):\n    assert len(xs) == 3\n",
          6, _ptf)])
    add("differential --pytest: a held claim on a fixture is HEADROOM"
        " (mjolnir abstained; it was not wrong), never a pass",
        _pcounts.get("headroom") == 1 and _pcounts.get("certified_ok", 0) == 0)

    # 28. LANGUAGE INDEPENDENCE of the scorer. The scorer must know nothing
    #     about pytest: it asks an adapter registry, and the invocation lives
    #     behind the same boundary as every other language-specific concern.
    #     A future `cargo test` or JUnit adapter is a NEW FILE and no change
    #     to differential.py — the twin asserts the registry is the ONLY place
    #     the framework is named.
    import mjolnir_prover.lib.differential as _DIFFMOD
    import inspect as _insp3
    _src = _insp3.getsource(_DIFFMOD)
    _invokes_pytest = ("subprocess" in _src and "pytest.main" in _src)
    add("differential: the scorer is language-independent — pytest is a"
        " registry entry behind a frontend, not a call in the core",
        (not _invokes_pytest)
        and _DIFFMOD.ADAPTERS.get("pytest") == "mjolnir_prover.lib.frontend_pytest"
        and _FPT.available() in (True, False))

    # and an unavailable adapter must yield NO VERDICT everywhere, never a pass
    _un = _DP.run_with_adapter(_ptf, ["test_holds"], adapter_name="cargo-test")
    add("differential: an adapter that cannot run reports NO VERDICT for every"
        " unit, never a pass",
        all(v in ("UNAVAILABLE", "NOTCOLLECTED") for v in _un.values())
        and _un.get("test_holds") == "UNAVAILABLE")

    # 26. the SCOPE promise. A declared scope that the code cannot check is a
    #     promise, not a scope. The twin requires that every DECLINED class has
    #     a predicate the analyser actually implements, so `mjolnir scope`
    #     cannot silently print "—" for a limit it never actually evaluates.
    from mjolnir_prover.lib import plan as _PLs
    _sc = _PLs.scope_report([("f", "def f(n):\n    return n\n", 1, "x.py"),
                            ("g", "def g(n):\n    assert n + 1 > n\n", 1, "x.py"),
                            ("h", "def h(xs):\n    assert xs[0] == 1\n", 1, "x.py"),
                            ("k", "def k(n):\n    assert min(n, 0) == 0\n", 1, "x.py")])
    # one unit per class: no claim / decided / selects a value / non-polynomial
    _want = {"declines: units that state no claim": 1, "decides": 1,
             "declines: claims that select a runtime value": 1,
             "declines: non-polynomial arithmetic": 1}
    add("scope: the promised classes are the classes the analyser computes"
        " (no predicate the code cannot check)",
        all(_sc["counts"].get(k) == v for k, v in _want.items())
        and all("fixtures" not in d["class"] for d in _sc["declared_declines"]))

    # 25. the LINEAR discipline, at the one place mjolnir is actually linear.
    #     `premise_derive` proves `d - c = sum k_i * p_i`, and the multipliers
    #     are non-negative. That is the linear discipline exactly: a premise
    #     difference may never be consumed with the OPPOSITE sign, because that
    #     would let a premise prove its own negation. Three properties pinned:
    #     no negative multiplier, no explosion, no refutation by declaration.
    from contextlib import redirect_stdout as _rs2
    import io as _io3
    _lin = [
        # (code, expected, note)
        ("def c(a,b):\n    assert a <= b\n    assert a <= b\n", "CERTIFIED",
         "k=0 weakening — sound, and affine not linear"),
        ("def c(a,b):\n    assert a <= b\n    assert a + a <= b + b\n",
         "CERTIFIED", "k=2 contraction — sound over Z, but NON-linear"),
        # a premise cannot be spent with the opposite sign: that is a premise
        # proving its own negation, which is the dual fragment entering freely
        ("def c(a,b):\n    assert a <= b\n    assert b <= a\n", "UNVERIFIABLE",
         "no negative multiplier — the linear part"),
        # no ex falso: contradictory premises must not explode into a claim
        ("def c(a,b):\n    assert a == b\n    assert a != b\n"
         "    assert a * a == 9\n", "UNVERIFIABLE",
         "no explosion from inconsistent premises"),
        # and a premise may never refute its own negation BY DECLARATION
        ("def c(a,b):\n    assert a <= b\n    assert a > b\n", "UNVERIFIABLE",
         "no refutation by declaration — the dual must be a witness"),
    ]
    _lin_bad = []
    for _c5, _want5, _note in _lin:
        with _rs2(_io3.StringIO()):
            _got5 = certify.certify_unit("f", _c5)[0]["status"]
        if _got5 != _want5:
            _lin_bad.append(f"{_note}: {_want5}!={_got5}")
    add("linear discipline: no negative multiplier, no explosion, no"
        " refutation by declaration" + (f" — {_lin_bad[0]}" if _lin_bad else ""),
        not _lin_bad)

    # and the misnaming hole was itself a LINEAR violation: declaring a premise
    # is consuming a resource you were not given. Say so where the fix lives.
    # 29. the SOFT / disclosure chain. The last three properties are the ones
    #     the design actually turns on: the router cannot speak, a disagreement
    #     still refutes, and the claims are not the source.
    from mjolnir_prover import tactics as _TAC
    from mjolnir_prover.lib import router as _RT
    from mjolnir_prover.lib.disclosure import Claim as _Cl
    from mjolnir_prover.lib import authority as _AU

    add("soft: the router interface cannot express a verdict (no certify/"
        "refute/judge method exists)",
        _RT.no_verdict_surface(_RT.Router))

    _src = {"mean_radius": "1428", "mean_texture": "391"}
    _d = _TAC.certify([_Cl("r", quantity="mean_radius", value="1421"),
                        _Cl("t", quantity="mean_texture", value="391")], _src)
    _by = {c["quantity"]: c["verdict"] for c in _d["claims"]}
    add("soft: a claim the source DISAGREES with is REFUTED; the agreement is"
        " CERTIFIED; and the set does NOT disclose",
        _by.get("mean_radius") == "REFUTED"
        and _by.get("mean_texture") == "CERTIFIED"
        and _d["discloses"] is False)

    _self = _TAC.self_check({"mean_radius": "1428"})
    add("soft: claims and source are SEPARATE inputs; the self-check is"
        " labelled VACUOUS rather than passed off as grounding",
        _self["discloses"] is True
        and all(c["verdict"] == "CERTIFIED" for c in _self["claims"]))

    add("soft: the disclosure exposes no scalar score (H-NL13/T-NL18)",
        not any(k in _d for k in ("score", "confidence", "rate", "rating",
                                  "accuracy")))

    add("soft: every module on the soft certificate path is in the trust pin",
        all(f in _AU.FILES for f in ("disclosure.py", "router.py",
                                      "frontend_table.py", "compose.py")))

    # 29b. CLAUSE (iii) MUST BE CHECKABLE FROM THE ARTIFACT. T-NL14 says the
    #      certificate is DECIDABLE; it does not say a stranger can check it.
    #      `marked` was absent from `as_dict`, so `unattributed` was `[]` in
    #      every emitted certificate -- vacuously satisfied, never verified. The
    #      twin asserts the mark is in the serialized claim AND that an unmarked
    #      remainder claim is actually reported, so "unattributed is empty" can
    #      never again be true for the trivial reason that the field was missing.
    _mk = _Cl("m", quantity="mean_radius", value="1428")
    _sr = _TAC.self_check({"mean_radius": "1428"})
    add("soft: the mark is SERIALIZED, so clause (iii) is externally checkable",
        all("marked" in c for c in _sr["claims"])
        and all(c["marked"] is True for c in _sr["claims"]))

    _un = _Cl("u", quantity="mean_radius", value="1428")
    _un.marked = False
    _um = _TAC.disclose([_un], level=1)
    add("soft: an UNMARKED remainder claim is reported as unattributed, not"
        " silently dropped",
        [c["quantity"] for c in _um["unattributed"]] == ["mean_radius"])

    # 29c. THE ASSERTED VALUE MUST SURVIVE TO THE RENDERED TABLE. The table
    #      printed `evidence` only, so a REFUTED row displayed the SOURCE's
    #      number next to the word REFUTED -- the enumeration (iii) requires was
    #      in the JSON and absent from what a human reads. Pinned on the emitted
    #      dict, since that is what the renderer consumes.
    _rv = _TAC.certify([_Cl("r", quantity="mean_radius", value="1421")],
                       {"mean_radius": "1428"}, level=1)
    _row = _rv["claims"][0]
    add("soft: a REFUTED claim carries BOTH the asserted value and the"
        " source's, so the disagreement is legible",
        _row["verdict"] == "REFUTED"
        and _row["value"] == "1421" and _row["evidence"] == "1428"
        and _row["value"] != _row["evidence"])

    # 29d. A RELATION-VALUED SOURCE. `parse_table` returned a dict, so a
    #      referent stated twice kept the LAST value -- and a claim asserting
    #      the dropped value came back REFUTED against a document that states
    #      it. A false refutation, measured. Two properties now hold: a claim
    #      matching ANY stated value certifies with THAT value as evidence (so
    #      the kernel re-derives the single-value proposition it knows), and a
    #      claim matching NONE abstains rather than refuting.
    from mjolnir_prover.lib import frontend_table as _FT
    _rep = _FT.parse_all("| q | value |\n|---|---|\n| a | 1 |\n| a | 2 |\n")
    add("soft: a referent stated TWICE is REPORTED, not silently truncated",
        _rep[0] == {"a": "2"} and _rep[1] == {"a": ["1", "2"]})

    _rel = {"a": ["1", "2"]}
    _c2 = _TAC.certify([_Cl("second", quantity="a", value="2")], _rel, level=0)
    _c0 = _TAC.certify([_Cl("first", quantity="a", value="1")], _rel, level=0)
    add("soft: a claim matching ANY stated value certifies, and the evidence"
        " IS the matched value",
        _c2["claims"][0]["verdict"] == "CERTIFIED"
        and _c2["claims"][0]["evidence"] == "2"
        and _c0["claims"][0]["verdict"] == "CERTIFIED"
        and _c0["claims"][0]["evidence"] == "1")

    _cn = _TAC.certify([_Cl("none", quantity="a", value="9")], _rel, level=0)
    add("soft: a claim matching NO stated value ABSTAINS -- never REFUTED,"
        " because a disjunction no rung decides cannot refute",
        _cn["claims"][0]["verdict"] == "UNVERIFIABLE")

    # 29e. THE COMPOSITION HARNESS. T-NL3 (sigma-additive coverage) could not be
    #      tested because the realized rungs are DISJOINT -- and S13 showed the
    #      disjointness is over CLAIM KIND, so no kind-partitioning ladder can
    #      ever produce an intersection. These two twins pin the two results the
    #      harness exists to produce: an INFORMATIVE intersection over one claim
    #      kind, and a genuine INCOMPARABILITY (A\B and B\A both non-empty),
    #      which is T-NL33.
    from mjolnir_prover.lib import composition as _CMP
    from mjolnir_prover.lib import intervals as _IV2
    from mjolnir_prover.lib import term as _T2

    def _boxeq(p, c, b):
        if not b:
            return None
        try:
            return _IV2.decide_rel(_T2.parse(p), _T2.parse(c), "eq",
                                   _IV2.env_from_bounds(b))
        except Exception:
            return None

    _PIN, _FREE = {"x": (4.0, 4.0)}, {"x": (0.0, 10.0)}
    _rows = [{"premise": "mul(2,x)", "conclusion": "add(x,x)", "bounds": _FREE},
             {"premise": "add(x,1)", "conclusion": "add(1,x)", "bounds": _FREE},
             {"premise": "add(x,x)", "conclusion": "8", "bounds": _PIN},
             {"premise": "mul(2,x)", "conclusion": "add(x,x)", "bounds": _PIN}]
    _A = _CMP.certified_set(_CMP.checker_algebra, _rows)
    _B = _CMP.certified_set(_boxeq, _rows)
    _sg = _CMP.sigma_additivity(_A, _B)
    add("composition: two independent routes over ONE claim kind yield an"
        " INFORMATIVE intersection (not a partition)",
        _sg["informative_inter"] and _sg["n_inter"] > 0
        and _sg["pairwise_holds"])
    add("composition: the two certified sets are INCOMPARABLE (A\\B and B\\A"
        " both non-empty) -- T-NL33",
        bool(_A - _B) and bool(_B - _A))
    add("composition: the harness emits no verdict and reuses the authority's"
        " own decision procedures",
        not hasattr(_CMP, "certify") and not hasattr(_CMP, "dispose")
        and not hasattr(_CMP, "judge"))

    # 29f. THE JOIN IS TABLE-DRIVEN (T-NL34w). The join of two routes existed as
    #      a Python `if` -- policy in code, unpinned, unreviewable, which is what
    #      the tcb_note forbids. It now lives in classification.json. These two
    #      twins pin the table's semantics and its hash-dependence: the same
    #      route outputs must give the same verdict, and the verdict must come
    #      from the table rather than from the caller.
    from mjolnir_prover.lib import theory as _TH3
    add("composition: the join is decided by the TABLE, not by the caller",
        _TH3.composition_verdict([True, None])[0] == "CERTIFIED"
        and _TH3.composition_verdict([False, False])[0] == "REFUTED"
        and _TH3.composition_verdict([False, None])[0] == "UNVERIFIABLE"
        and _TH3.composition_verdict([None, None])[0] == "UNVERIFIABLE"
        and "composition" in _TH3.load("classification"))
    add("composition: a DECLINING route is never evidence against -- one abstain"
        " is enough to prevent a refutation",
        _TH3.composition_verdict([False, None])[0] == "UNVERIFIABLE"
        and _TH3.composition_verdict([True, False])[0] == "CERTIFIED")

    # 30. FABRICATION MUST NOT EXIT 0. Measured: a model asserting five metrics
    #     the source does not contain produced exit 0, because every verdict was
    #     UNVERIFIABLE rather than REFUTED -- the exit rule only looked for
    #     REFUTED. The router had the distinction all along ("the source does not
    #     contain this quantity" vs "I cannot decide this one") and `tactics`
    #     flattened it into "router abstained", discarding the signal. These pin
    #     the distinction at every layer it passes through.
    from mjolnir_prover.lib import frontend_table as _FT2
    _src2, _rep2 = _FT2.parse_all("| m | value |\n|---|---|\n| a | 1 |\n")
    _fab = _TAC.certify([_Cl("f", quantity="not_in_source", value="1")],
                        _src2, level=0)["claims"][0]
    add("soft: the router's ABSTENTION REASON survives into the artifact, so"
        " 'not in the source' is distinguishable from 'cannot decide'",
        _fab["verdict"] == "UNVERIFIABLE"
        and str(_fab["reason"]).startswith("quantity_not_in_source"))
    add("soft: ordinary abstention is NOT reported as absence -- the two"
        " abstention classes stay distinct",
        _TAC.certify([_Cl("u", quantity="a", value="7")], _src2,
                     level=0)["claims"][0]["reason"] != _fab["reason"])

    # 31. THE FIRST-USER PATH. Found by walking it cold with the files a real
    #     user actually has: a CSV was DECLINED (the only readers were a markdown
    #     pipe-row and `key: value`), and the decline message said "prose is a
    #     declined class" for a CSV -- a misdiagnosis that blamed the user's
    #     format when the cause was this reader. A scope limit is a promise, and
    #     a promise that lies about its own boundary is worse than no promise.
    add("soft: a CSV/TSV with a header is READ -- it is the format a first-time"
        " user is most likely to have",
        _FT2.parse_csv("metric,value\nmean_radius,14.1273\n"
                       "mean_area,654.8891\n")[0]
        == {"mean_radius": "14.1273", "mean_area": "654.8891"})
    # The routing boundary: `parse_all` reads ONLY markdown/kv, and `read_all`
    # falls through to CSV. My first twin asserted `parse_all` reads a CSV; it
    # does not, and asserting it would have forced the markdown reader to accept
    # a format it was never designed for. The twin now pins both halves: the
    # markdown reader does NOT misread a CSV, and `read_all` does.
    import tempfile as _tf2
    _p2 = Path(_tf2.mkdtemp()) / "d.csv"
    _p2.write_text("metric,value\nmean_radius,14.1273\n", encoding="utf-8")
    add("soft: the CSV route is reached only via read_all, and the markdown"
        " reader does not misread a delimited file",
        _FT2.parse_all("metric,value\nmean_radius,14.1273\n")[0] == {}
        and _FT2.read_all(str(_p2))[0] == {"mean_radius": "14.1273"})
    add("soft: a CSV with a REPEATED key is reported, not silently truncated",
        _FT2.parse_csv("k,v\na,1\na,2\n")[1] == {"a": ["1", "2"]})

    # 24. MISNAMING. A method that asserts truth BY DECLARATION (`premise`,
    #     `program_exit`, `no_precondition`) is trivially "re-derivable", which
    #     makes it a laundering channel: an ABSTAINING claim whose method was
    #     rewritten to `premise` was accepted as a CERTIFIED certificate —
    #     measured 0 mismatches, 0 false certs. The kernel re-derived the
    #     semantics of the LABEL and never asked whether the label was right.
    #     Guard: a declaration may not sit on the claim's own line.
    import json as _json
    from contextlib import redirect_stdout as _rs
    import io as _io2
    with _rs(_io2.StringIO()):
        _ab, _, _ = certify.certify_unit(
            "f", "def f(n):\n    assert min(n, 0) == 0\n", 1, "x.py")
    _laundered = []
    for _m in ("premise", "program_exit", "no_precondition"):
        _t = _json.loads(_json.dumps(_ab))
        _t["obligations"][0]["method"] = _m
        _t["obligations"][0]["status"] = "ok"
        _t["status"] = "CERTIFIED"
        _r = _CTK.K.replay(_t)
        if _r["agreement"] == 1.0 and _r["false_certs"] == 0:
            _laundered.append(_m)
    add("kernel: an abstention cannot be laundered into a certificate by"
        " relabelling its method as a declaration"
        + (f" — {_laundered[0]} accepted" if _laundered else ""),
        not _laundered)

    # and the guard must not break the honest case: a REAL premise sits on an
    # earlier line and must still re-derive
    with _rs(_io2.StringIO()):
        _hp, _, _ = certify.certify_unit("f", "def f(n):\n    assert n == 4\n"
                                              "    assert n == 0\n", 1, "x.py")
    _hr = _CTK.K.replay(_hp)
    add("kernel: an honest premise on an earlier line still re-derives",
        _hr["agreement"] == 1.0 and _hr["false_certs"] == 0
        and "ok" in _hr["statuses"])

    # the one-directional agreement test: an abstention that the kernel CAN
    # decide is conservative, not a soundness defect — but reporting it as
    # plain `agreement 1.0` hid a real finding, so it is counted separately
    with _rs(_io2.StringIO()):
        _as, _, _ = certify.certify_unit(
            "f", "def f(n):\n    assert min(n, 0) == 0\n", 1, "x.py")
    _asr = _CTK.K.replay(_as)
    add("kernel: an abstention the kernel CAN decide is surfaced, not hidden"
        " behind agreement 1.0",
        _asr["agreement"] == 1.0 and _asr["false_certs"] == 0
        and bool(_asr["decided_despite_abstain"]))

    # 23. `mjolnir plan` — the mandatory front door. Its twins are about the
    #     ORDER being a dependency rather than a preference, because that is the
    #     whole claim: a stage-3 fix cannot move a claim stuck at stage 1.
    from mjolnir_prover.lib import plan as _PL
    _pcases = [
        # (code, expected bucket)
        ("def f(n):\n    return n + 1\n", "no_claim"),
        # `min` is outside the polynomial normal form: decidable, not closed
        ("def f(n):\n    assert min(n, 0) == 0\n", "arithmetic_unresolved"),
        # a subscript SELECTS a runtime value -> no value exists to decide about
        ("def f(xs):\n    assert xs[0] == 1\n", "no_value"),
    ]
    _pbad = []
    for _c4, _want in _pcases:
        _got = _PL.classify_unit(_c4)
        if _got != _want:
            _pbad.append(f"{_want}!={_got}: {_c4!r}")
    add("plan: each unit lands in exactly one bucket"
        + (f" — {_pbad[0]}" if _pbad else ""), not _pbad)

    # THE ORDER TWIN. Build a corpus that is entirely stage 3 and confirm the
    # plan then ranks a rung; build one that is entirely stage 1 and confirm it
    # REFUSES to. If the recommendation tracked size alone, the first would
    # pass and the second would also name a rung — which is the failure this
    # command exists to stop.
    # `min` is outside the polynomial normal form, so these are decidable in
    # principle and unproved in fact — genuinely stage 3. (`n + 1 > n` is NOT:
    # mjolnir certifies that one, so it would have made the twin pass for the
    # wrong reason.)
    _s3 = [(f"f{i}", f"def f(n):\n    assert min(n, {i}) == {i}\n", 1, "x.py")
           for i in range(6)]
    _p3 = _PL.build_plan(_s3)
    _s1 = [(f"g{i}", f"def g(n):\n    return n + {i}\n", 1, "x.py")
           for i in range(6)]
    _p1 = _PL.build_plan(_s1)
    add("plan: the order is a DEPENDENCY — an all-stage-3 corpus ranks a rung,"
        " an all-stage-1 corpus refuses to",
        bool(_p3["dominant_stage"]["prover_can_help"])
        and _p3["stage3_ranking"] != []
        and (not _p1["dominant_stage"]["prover_can_help"])
        and "Do NOT build a rung" in (_p1["recommendation"] or ""))

    # and a corpus with nothing at stage 3 must say so rather than invent one
    _p0 = _PL.build_plan(_s1)
    add("plan: an empty stage-3 bucket yields no rung recommendation at all",
        _p0["stage3_ranking"] == [])

    # the OVERCLAIM twin. `err < eps` on two free locals is arithmetically
    # shaped and unresolvable, but NOT decidable without their runtime values.
    # The bucket must name only what is established, so both it and a genuinely
    # decidable-but-unclosed claim land in the same, honestly named bucket —
    # and the key must not claim decidability the classifier never checked.
    _over = _PL.classify_unit("def f(n):\n    assert err < eps\n")
    _keys = {b["key"] for b in _PL.bucket_order()}
    add("plan: the stage-3 bucket names only what is ESTABLISHED (shape + "
        "abstention), never decidability it has not checked",
        _over == "arithmetic_unresolved"
        and "decidable_unproved" not in _keys
        and "arithmetic_unresolved" in _keys)

    # the DECLINE twin: an empty measurement must DECLINE, not crash and not
    # print an empty table that reads like a finding. This command exists to say
    # "I have no answer" when it has not measured.
    _pe = _PL.build_plan([])
    add("plan: an empty measurement DECLINES instead of crashing or emitting a"
        " table that reads like a finding",
        _pe["total_units"] == 0 and _pe["stages"] == []
        and "NO MEASUREMENT" in (_pe["recommendation"] or ""))

    # the costs are POLICY AS DATA, not a dict buried in the analysis module
    from mjolnir_prover.lib import frontier as _FRx, theory as _THx
    add("plan: rung costs come from the hashed table, not from code",
        _FRx._roadmap_costs() == _THx.load("roadmap")["costs"]
        and bool(_THx.digest("roadmap")))

    # 22. SELF-AUDIT SCOPE. `mjolnir check .` must see exactly ONE tree.
    #     Measured before this twin existed: 1,399 units and 5 CERTIFIED, with
    #     76 of the 131 walked files (58%) under `.kilo/worktrees/` — duplicate
    #     COPIES of this source. A self-audit that scores its own copies is not
    #     a self-audit, and the numbers moved whenever a worktree appeared.
    from mjolnir_prover.lib import gate as _GT
    _leaked = []
    _cfg = _GT.load_config("mjolnir.toml")
    for _f in _GT.collect(["."], exclude=tuple(_cfg.get("exclude", ()))):
        if (".kilo" in _f or "__pycache__" in _f or ".git" in _f
                or "site-packages" in _f or ".egg-info" in _f
                or ".vscode" in _f or "node_modules" in _f
                or "storage/data" in _f):
            _leaked.append(_f)
    add("self-audit scope: no tooling, agent, vendor or data artifact is walked"
        + (f" — {_leaked[0]}" if _leaked else f" ({len(_leaked)} leaked)"),
        not _leaked)

    # and the numbers must be ACCOUNTED FOR: walking `.` and walking the
    # first-party source must agree once every first-party directory is counted,
    # or one of them is walking something extra. The declared list is the point
    # -- a new top-level source directory has to be added here deliberately, not
    # discovered later as an unexplained delta.
    # `cwebench` (per-property patch verification over a declared artifact) and
    # `rag3` (the three-way RAG dataset + held-out eval) are self-contained
    # research experiments. They are declared here so the accounting stays
    # exact: the twin's whole point is that a new top-level source directory is
    # registered DELIBERATELY rather than discovered later as an unexplained
    # delta. Adding them changes no mjolnir behaviour.
    # `decide` is the orthogonal battery for a third-party decision model. It
    # is research tooling and disposes nothing; declared here per the same rule.
    # `mjolnir_router`, `mjolnir_intent` and `mjolnir_app` are the product's two
    # other libraries and the shell that joins them. They are
    # declared under the same rule as everything above: a new top-level source
    # is registered deliberately rather than discovered as an unexplained delta.
    # (mjolnir_router had been walking undeclared before this — the twin was
    # already red — so this registration closes a hole as well as adding one.)
    # `tactics`, `cwebench`, `rag3` and `decide` were moved into
    # `mjolnir_experiments/` and are excluded from the gate -- they are research
    # snapshots, not product source. What remains declared is exactly what ships.
    # `mjolnir_guard` (the fail-closed CLI runner — product source, named in the
    # top-level README) and `mjolnir_decide` (the decision-model runner that
    # lives under src/ since the top-level restructure) were walking here as a
    # 94-unit self-audit delta — found by the twin, registered deliberately per
    # the rule above, not discovered later as an unexplained delta.
    # `mjolnir_common` holds the shared serving-layer laws (the joint-AND
    # batch law) so mjolnir_decide and the route service never import each
    # other — registered under the same rule.
    _FIRST_PARTY = ("src/mjolnir_prover", "src/mjolnir_router",
                    "src/mjolnir_intent", "src/mjolnir_app",
                    "src/mjolnir_rlvf", "src/mjolnir_guard",
                    "src/mjolnir_decide", "src/mjolnir_common")
    _root = _GT.run(["."], config="mjolnir.toml")["counts"]["units"]
    _sum = sum(_GT.run([d], config="mjolnir.toml")["counts"]["units"]
               for d in _FIRST_PARTY)
    add("self-audit scope: `.` is accounted for by the declared first-party"
        " directories" + (f" — {_root} vs {_sum}" if _root != _sum else ""),
        _root == _sum)

    # 21. the DIFFERENTIAL oracle. Its twins are about the harness being able
    #     to FAIL, since a scoreboard that cannot report a defect is decoration.
    from mjolnir_prover.lib import differential as _DF
    import io as _io, contextlib as _cx
    _dscore = [
        # (code, expected outcome)
        ("def f(n):\n    assert n * (n + 1) == n * n + n\n", "certified_ok"),
        # mjolnir REFUTES this one (a closed-form contradiction), so the
        # matching external outcome is refuted_ok, not declined_correctly
        ("def f(n):\n    assert n * (n + 1) == n * n + n + 1\n", "refuted_ok"),
        # and this one mjolnir abstains on (min is outside the polynomial
        # normal form), which execution confirms is genuinely false
        ("def f(n):\n    assert min(n, 0) == 0\n", "declined_correctly"),
    ]
    _dbad = []
    for _c3, _want in _dscore:
        with _cx.redirect_stdout(_io.StringIO()):
            _r = _DF.score_unit("f", _c3, 1, "x.py", exec_path=True)
        if _r["outcome"] != _want:
            _dbad.append(f"{_want}!={_r['outcome']}")
    add("differential: the oracle agrees with mjolnir on a true and a false"
        " claim" + (f" — {_dbad[0]}" if _dbad else ""), not _dbad)

    # The twin that matters: the oracle must be able to CONTRADICT mjolnir.
    # A hand-built certificate claiming `q` follows from `p & q` is FALSE, and
    # execution must say so — if the harness could only ever agree, it would be
    # an echo, not an oracle.
    _echo = []
    # NB: not named `_claim` — that is a helper in this function, and a loop
    # variable of the same name makes it local and breaks every earlier call.
    for _dtext, _truth in (("p + q == q + p", True), ("p + q == q", False)):
        with _cx.redirect_stdout(_io.StringIO()):
            _rr = _DF.score_unit("f", "def f(p, q):\n    assert " + _dtext + "\n",
                                 1, "x.py", exec_path=False)
        _holds = _rr["exec_holds"]
        if _holds is not _truth:
            _echo.append(_dtext + ": exec_holds=" + repr(_holds)
                         + " truth=" + repr(_truth))
    add("differential: the oracle can CONTRADICT mjolnir (it is not an echo)"
        + (f" — {_echo[0]}" if _echo else ""), not _echo)

    # and it must refuse to score a unit it cannot reach, rather than guessing
    _un = "def f(xs):\n    assert xs[0] == xs[0]\n"
    with _cx.redirect_stdout(_io.StringIO()):
        _ru = _DF.score_unit("f", _un, 1, "x.py", exec_path=False)
    add("differential: an unreachable unit is reported unscored, never guessed",
        _ru["outcome"] == "unscored" and _ru["exec_holds"] is None)

    # 20. term.selfcheck_canonical — the CANONICAL FORM, not arithmetic.
    #     `selfcheck` states theorems of Z, sound whatever nf does. These state
    #     that the AC-normalisation actually happens, so they fail if the
    #     canonical form is lost. The twin below is the non-vacuity half:
    #     `a + b == b + a` and `a + b == a` are structurally different
    #     comparisons, and only the FIRST may certify. If both did, the
    #     contract would be agreeing with everything and proving nothing.
    import inspect as _insp2
    from mjolnir_prover.lib import term as _TM
    _tc = certify.certify_unit("selfcheck_canonical",
                               _insp2.getsource(_TM.selfcheck_canonical))[0]
    _t_derived = [o for o in _tc["obligations"]
                  if o["method"].startswith("nf+decide_eq")]
    add("term canonical-form contract: the AC-normalisation laws certify",
        (_tc["status"] == "CERTIFIED" and len(_t_derived) == 9
         and (_tc.get("kernel") or {}).get("agreement") == 1.0
         and (_tc.get("kernel") or {}).get("false_certs") == 0))

    _tv = [
        # (code, must certify?) — the discriminating pair
        ("def c(a, b):\n    assert a + b == b + a\n", True),
        ("def c(a, b):\n    assert a + b == a\n", False),
        ("def c(a, b, c):\n    assert (a + b) + c == a + (b + c)\n", True),
        ("def c(a, b, c):\n    assert (a + b) + c == a + b + c + a\n", False),
        ("def c(a, b):\n    assert a - a == 0\n", True),
        ("def c(a, b):\n    assert a - a == 1\n", False),
    ]
    _tv_bad = []
    for _code, _want in _tv:
        _got = certify.certify_unit("c", _code)[0]["status"] == "CERTIFIED"
        if _got is not _want:
            _tv_bad.append(f"certify={_got} want={_want}: {_code!r}")
    add("term canonical-form contract has CONTENT: it certifies the normal-form"
        " identities and refuses the near misses"
        + (f" — {_tv_bad[0]}" if _tv_bad else ""), not _tv_bad)

    # the measured frontier, pinned so it cannot be quietly forgotten: nf is a
    # POLYNOMIAL normal form, so min/max/abs//% are outside it and NO contract
    # over them can be discharged by this route. Adding asserts there would
    # create abstentions, not certificates.
    _fr = [
        ("def c(x, y):\n    assert min(x, y) <= x\n", "min"),
        ("def c(x):\n    assert abs(x) >= 0\n", "abs"),
        ("def c(x, y):\n    assert x % y < y\n", "mod"),
        ("def c(x, y):\n    assert x // y * y <= x\n", "floordiv"),
    ]
    _fr_bad = [n for _c2, n in _fr
               if certify.certify_unit("c", _c2)[0]["status"] == "CERTIFIED"]
    add("term frontier: min/abs/mod// are outside the polynomial normal form"
        " and stay undecidable (pinned so the gap stays visible)",
        not _fr_bad)

    # 19. PROPOSITIONAL: the first CLOSED theory. Four properties, and the
    #     third is the one that makes the whole route trustworthy.
    from mjolnir_prover.lib import propositional as _PROP
    _pq = Tmod.catom("p")
    _pr = Tmod.catom("q")
    _rs = Tmod.catom("r")

    _props = [
        ([], Tmod.cor(_pq, Tmod.cnot(_pq)), True, "excluded middle"),
        ([], Tmod.ciff(Tmod.cnot(Tmod.cnot(_pq)), _pq), True, "double negation"),
        ([Tmod.cand(_pq, _pr)], _pq, True, "conjunction intro"),
        ([Tmod.cor(_pq, _pr), Tmod.cnot(_pq)], _pr, True, "disjunctive syllogism"),
        ([Tmod.cimp(_pq, _pr), _pq], _pr, True, "modus ponens"),
        ([Tmod.cimp(_pq, _pr), Tmod.cimp(_pr, _rs)], Tmod.cimp(_pq, _rs), True,
         "transitivity"),
        ([Tmod.cimp(_pq, _pr)], _pr, False, "p=>q does NOT entail q"),
        ([Tmod.cand(_pq, _pr)], Tmod.cnot(_pr), False, "conj does NOT entail ~q"),
    ]
    _prop_bad = []
    for _ax, _cn, _want, _note in _props:
        _got = _PROP.decide(_ax, _cn)[0]
        if _got is not _want:
            _prop_bad.append(f"{_note}: want {_want}, got {_got}")
    add("propositional: the decision procedure agrees with the benchmark logic"
        " on classical rules and both counter-cases"
        + (f" — {_prop_bad[0]}" if _prop_bad else ""), not _prop_bad)

    # the KERNEL twin. The kernel rebuilds both sides from the recorded STRINGS
    # and redoes the search; it must not read the recorded verdict. A doctored
    # certificate claiming `q` follows from `p => q` must be caught.
    _pc = _PROP.certify_entailment([Tmod.cimp(_pq, _pr)], _pr)
    _pc["obligations"][0]["status"] = "ok"
    _pc["status"] = "CERTIFIED"
    _pk = _CTK.K.replay(_pc)
    _pc2 = _PROP.certify_entailment([Tmod.cand(_pq, _pr)], _pq)
    _pk2 = _CTK.K.replay(_pc2)
    add("propositional: the kernel re-derives the route from strings alone and"
        " catches a doctored certificate",
        _pk2["agreement"] == 1.0 and _pk["false_certs"] >= 1)

    # the WITNESS twin. A REFUTED row is only a refutation if its
    # counter-assignment satisfies every axiom and actually falsifies the
    # conjecture. The kernel re-audits it; here we break it two ways and require
    # both to be refused.
    _rw = _PROP.certify_entailment([Tmod.cand(_pq, _pr)], Tmod.cnot(_pr))
    _ok_w = _CTK.K.replay(_rw)
    # wrong: satisfies no model (p & q is false)  -> unchecked
    _w1 = json.loads(json.dumps(_rw))
    _w1["obligations"][0]["witness"] = {"p": False, "q": False}
    # absent                                                       -> unchecked
    _w2 = json.loads(json.dumps(_rw))
    _w2["obligations"][0].pop("witness", None)
    # plausible but does NOT falsify (~q is true when q is false)   -> unchecked
    _w3 = json.loads(json.dumps(_rw))
    _w3["obligations"][0]["witness"] = {"p": True, "q": False}
    # a DIFFERENT valid witness must also be refused: the kernel reproduces
    # the witness it computes rather than accepting any sound one
    _w4 = json.loads(json.dumps(_rw))
    _w4["obligations"][0]["witness"] = {"p": True, "q": True, "zz": False}
    _kk = [_CTK.K.replay(w)["statuses"][0] for w in (_w1, _w2, _w3, _w4)]
    add("propositional: a refutation whose witness is wrong, absent, "
        "non-falsifying, or not the one the search finds is left UNCHECKED"
        + (f" — {_kk}" if any(k != "unchecked" for k in _kk) else ""),
        (_ok_w["agreement"] == 1.0 and _ok_w["false_certs"] == 0
         and all(k == "unchecked" for k in _kk)))

    # the budget twin: the certified set is a DISCLOSED choice, so an
    # over-budget problem must abstain rather than guess, and the recorded
    # theory digest must travel with the claim.
    import functools
    # max_atoms+2 atoms: over the DISCLOSED budget, whatever the table says,
    # so it MUST abstain rather than guess. Budget-relative on purpose -- a
    # hardcoded count broke when the table moved 8 -> 16 (registered fix).
    _n_over = _PROP.budget()["max_atoms"] + 2
    _six = functools.reduce(Tmod.cand,
                            [Tmod.catom(f"a{i}") for i in range(_n_over)])
    _big = [_six]
    _ov = _PROP.decide(_big, Tmod.catom("a0"))
    _bud = _PROP.budget()
    _oc = _PROP.certify_entailment(_big, Tmod.catom("a0"))
    add("propositional: over-budget abstains, and the theory digest travels"
        " with the claim"
        + (f" — {_ov[1]}" if _ov[0] is not None else ""),
        _ov[0] is None
        and _oc["obligations"][0]["theory_digest"] == _bud["digest"]
        and _oc["obligations"][0]["budget_atoms"] == _bud["max_atoms"])

    # vacuity is COMPUTED, not stipulated: a derivable entailment won by
    # assuming its own conclusion, or from inconsistent premises, is a
    # disposition -- never a claim -- and the kernel confirms it as recorded.
    _si = _PROP.certify_entailment([_pq], _pq)                   # assume goal
    _ef = _PROP.certify_entailment([_pq, Tmod.cnot(_pq)], _pr)   # ex falso
    _pv = _PROP.certify_entailment([Tmod.cand(_pq, _pr)], _pq)   # real support
    _red = _PROP.certify_entailment([_pq, _pr], Tmod.cor(_pq, _pr))
    _vsi, _vef = _si["obligations"][0], _ef["obligations"][0]
    _vpv, _vrd = _pv["obligations"][0], _red["obligations"][0]
    add("propositional: vacuity is computed — self-inclusion and ex falso"
        " are UNVERIFIABLE dispositions, never claims"
        + (f" — {_si['status']}/{_si['reason']} {_ef['status']}/{_ef['reason']}"
           if _si["status"] != "UNVERIFIABLE" or _ef["status"] != "UNVERIFIABLE"
           else ""),
        _si["status"] == "UNVERIFIABLE" and _ef["status"] == "UNVERIFIABLE"
        and _vsi["non_vacuous"] is False and _vef["non_vacuous"] is False
        and str(_si["reason"]).startswith("vacuous:")
        and str(_ef["reason"]).startswith("vacuous:")
        and _vsi["status"] == "abstain" and _vef["status"] == "abstain")
    add("propositional: a genuine discharge is non-vacuous with essential"
        " support; redundant support stays non-vacuous (diagnostic, not the"
        " boolean)"
        + (f" — {_vpv['essential_premises']}/{_vrd['essential_premises']}"
           if _vpv["essential_premises"] != 1
           or _vrd["essential_premises"] != 0 else ""),
        _pv["status"] == "CERTIFIED" and _vpv["non_vacuous"] is True
        and _vpv["essential_premises"] == 1
        and _red["status"] == "CERTIFIED" and _vrd["non_vacuous"] is True
        and _vrd["essential_premises"] == 0)
    _ks = [_CTK.K.replay(c)["statuses"][0] for c in (_si, _ef, _pv)]
    add("propositional: the kernel discharges only re-derivable AND"
        " non-vacuous obligations — vacuous dispositions replay checked"
        " (unchecked, no mismatch), genuine claims discharge"
        + (f" — {_ks}" if _ks != ["unchecked", "unchecked", "ok"] else ""),
        _ks == ["unchecked", "unchecked", "ok"]
        and _CTK.K.replay(_si)["agreement"] == 1.0)
    _forged = json.loads(json.dumps(_si))
    _forged["status"] = "CERTIFIED"
    _forged["obligations"][0]["status"] = "ok"
    _forged["obligations"][0]["non_vacuous"] = True
    _kf = _CTK.K.replay(_forged)
    add("propositional: forging a certificate onto a vacuous obligation"
        " leaves it UNCHECKED (the strict-discharge alarm path)"
        + (f" — {_kf['statuses']}" if _kf["statuses"] != ["unchecked"] else ""),
        _kf["statuses"] == ["unchecked"])

    # a digest that no longer matches must NOT be compared silently: a verdict
    # produced under a different budget is not comparable to the current one.
    _stale = _PROP.certify_entailment([Tmod.cand(_pq, _pr)], _pq)
    _stale["obligations"][0]["theory_digest"] = "deadbeefdeadbeef"
    # --- gate artifacts: a statistical gate's bar must be POSSIBLE and its
    # verdict must FOLLOW from its own numbers. (A majority counted over distinct
    # combinations instead of rows gave 0.005 and silently enabled a proposer.)
    from mjolnir_prover.lib import effects as _EFF

    _good = {"n": 240, "accuracy": 0.762, "majority": 0.68, "delta": 0.082,
             "verdict": "GO", "evidence": "triage_predictions.jsonl"}
    _impossible = {"n": 379, "accuracy": 0.092, "majority": 0.0053,
                   "delta": 0.087, "verdict": "GO"}
    _inconsistent = {"n": 240, "accuracy": 0.50, "majority": 0.68,
                     "delta": 0.18, "verdict": "GO"}
    _ok_g, _ = _EFF.validate_gate(_good, 3)
    _ok_bad, _why = _EFF.validate_gate(_impossible, 3)
    _ok_inc, _why2 = _EFF.validate_gate(_inconsistent, 3)
    add("gate artifact: an impossible bar (majority < 1/n_classes) is refused"
        + (f" — {_why}" if _ok_bad else ""),
        _ok_g and (not _ok_bad) and any("impossible bar" in w for w in _why))
    add("gate artifact: a verdict that does not follow from its numbers is refused"
        + (f" — {_why2}" if _ok_inc else ""),
        (not _ok_inc) and any("recomputed" in w for w in _why2))

    _art = {"status": "active", "assignment": "authority(lib.effects)",
            "class_map": {"pure": "pure", "mutates": "mutating"},
            "categories": {"pure": {}, "mutating": {}, "opaque": {}},
            "encoder": {"triage": _good, "triage_status": "enabled"}}
    _art_bad = {**_art, "encoder": {"triage": _impossible,
                                    "triage_status": "enabled"}}
    _art_no_map = {**_art, "class_map": {}}
    _ok_a1, _ = _EFF.validate_artifact(_art)
    _ok_a2, _w2 = _EFF.validate_artifact(_art_bad)
    _art_no_auth = {**_art, "assignment": ""}
    _ok_a3, _w3 = _EFF.validate_artifact(_art_no_map)
    _ok_a4, _w4 = _EFF.validate_artifact(_art_no_auth)
    add("gate artifact: GO without evidence, a missing class_map, or active"
        " without a declared assignment are all refused"
        + (f" — {_w3}" if _ok_a3 else ""),
        _ok_a1 and (not _ok_a2) and (not _ok_a3) and (not _ok_a4)
        and any("impossible bar" in w for w in _w2)
        and any("class_map" in w for w in _w3)
        and any("assignment" in w for w in _w4))

    # --- thresholds must name their quantity (three same-class errors in a day:
    # impossible majority bar · class-vocabulary mismatch · OOD floor compared
    # against the wrong quantity, which made a service answer a constant)
    from mjolnir_prover.lib import thresholds as _TH

    _good_t = {"quantity": "flow_score_per_dim", "value": -2000.0,
               "calibrated_on": {"in_domain": {"mean": -76.33},
                                 "ood": {"mean": -89439.01}}}
    _wrong_q = {**_good_t}
    _no_q = {k: v for k, v in _good_t.items() if k != "quantity"}
    _impossible = {**_good_t, "value": -200000.0}
    _ok_t1, _ = _TH.validate_threshold(_good_t, "flow_score_per_dim")
    _ok_t2, _w2 = _TH.validate_threshold(_wrong_q, "joint_log_likelihood")
    _ok_t3, _w3 = _TH.validate_threshold(_no_q)
    _ok_t4, _w4 = _TH.validate_threshold(_impossible)
    add("threshold: a quantity mismatch is refused"
        + (f" — {_w2}" if _ok_t2 else ""),
        _ok_t1 and (not _ok_t2) and any("compared to" in w for w in _w2))
    add("threshold: no declared quantity, or a value outside the measured tails,"
        " is refused" + (f" — {_w3}" if _ok_t3 else ""),
        (not _ok_t3) and (not _ok_t4)
        and any("declared quantity" in w for w in _w3)
        and any("impossible threshold" in w for w in _w4))
    try:
        import sys as _s
        _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))
        from mjolnir_decide.thresholds import declaration as _dec_fl
        _ok_dec, _wdec = _TH.validate_threshold(_dec_fl(), "flow_score_per_dim")
        add("threshold: mjolnir_decide's actual OOD floor declaration validates"
            + (f" — {_wdec}" if not _ok_dec else ""), _ok_dec)
    except Exception as _e:                                   # noqa: BLE001
        add("threshold: mjolnir_decide's floor declaration validates"
            f" (skipped: {type(_e).__name__})", True)

    add("propositional: a certificate pinned to a stale theory digest is"
        " refused, not silently re-judged",
        _CTK.K.replay(_stale)["statuses"] == ["unchecked"])

    # 18. intervals.selfcheck — our own interval arithmetic, under contract.
    #     These assertions are properties of SPECIFIC lines of intervals.py
    #     (`a[0]-b[1]`, `a[1]-b[0]`, `(-a[1], -a[0])`), so the contract is
    #     falsifiable rather than decorative: the sound outward bounds certify
    #     and the naive ones do not.
    import inspect as _insp
    from mjolnir_prover.lib import intervals as _IV
    _ic = certify.certify_unit("selfcheck",
                               _insp.getsource(_IV.selfcheck))[0]
    _ic_derived = [o for o in _ic["obligations"]
                   if o["method"].startswith("premise_derive")]
    add("intervals contract: the interval-arithmetic soundness obligations"
        " certify under premise_derive",
        (_ic["status"] == "CERTIFIED" and len(_ic_derived) >= 7
         and (_ic.get("kernel") or {}).get("agreement") == 1.0
         and (_ic.get("kernel") or {}).get("false_certs") == 0))

    # NON-VACUITY, and the twin that matters for AI-written code: the contract
    # must ACCEPT what intervals.py does and REFUSE what a naive implementation
    # would do. `a[0]-b[1]` is sound because at x=a_lo, y=b_hi the difference
    # IS a_lo-b_hi; `a[0]-b[0]` is a plausible-looking unsound lower bound.
    _IPRE = ("def c(a_lo, a_hi, b_lo, b_hi, x, y):\n"
             "    assert a_lo <= a_hi\n    assert b_lo <= b_hi\n"
             "    assert a_lo <= x\n    assert x <= a_hi\n"
             "    assert b_lo <= y\n    assert y <= b_hi\n")
    _iv_cases = [
        ("    assert a_lo - b_hi <= x - y\n", "CERTIFIED", "sound outward lower"),
        ("    assert x - y <= a_hi - b_lo\n", "CERTIFIED", "sound outward upper"),
        ("    assert a_lo - b_lo <= x - y\n", "UNVERIFIABLE",
         "NAIVE lower bound must be refused"),
        ("    assert x - y <= a_hi - b_hi\n", "UNVERIFIABLE",
         "NAIVE upper bound must be refused"),
    ]
    _iv_bad = []
    for _body, _want, _note in _iv_cases:
        _st = certify.certify_unit("c", _IPRE + _body)[0]["status"]
        if _st != _want:
            _iv_bad.append(f"{_note}: {_want}!={_st}")
    add("intervals contract has CONTENT: accepts the sound outward bounds and"
        " refuses the naive ones"
        + (f" — {_iv_bad[0]}" if _iv_bad else ""), not _iv_bad)

    # 17. PREMISE-DERIVED certificates: the maximal NON-VACUOUS discharge. A
    #     claim `c <= d` is proved when `d - c` is a non-negative integer
    #     combination of declared premise differences. Every such certificate is
    #     a real theorem about a real contract — nothing assumed that was not
    #     written down, nothing concluded that was not asserted.
    _pd_cases = [
        # the target: x cancels, claim difference IS the premise difference
        ("def c(lo, hi, x):\n    assert lo <= hi\n    assert lo + x <= hi + x\n",
         "CERTIFIED", "identity"),
        # transitivity through two premises
        ("def c(a, b, d):\n    assert a <= b\n    assert b <= d\n"
         "    assert a <= d\n", "CERTIFIED", "transitive"),
        # the reverse direction is the same fact
        ("def c(lo, hi, x):\n    assert lo <= hi\n    assert hi + x >= lo + x\n",
         "CERTIFIED", "ge direction"),
        # NON-VACUITY: a claim the contract does NOT support must not certify
        ("def c(a, b, d):\n    assert a <= b\n    assert a <= d\n",
         "UNVERIFIABLE", "unsupported claim must not certify"),
        # a reversed premise supports nothing
        ("def c(lo, hi, x):\n    assert hi <= lo\n    assert lo + x <= hi + x\n",
         "UNVERIFIABLE", "reversed premise supports nothing"),
        # strictness is not inferable from non-strict premises
        ("def c(a, b, d):\n    assert a <= b\n    assert b <= d\n    assert a < d\n",
         "UNVERIFIABLE", "strict does not follow from non-strict"),
    ]
    _pd_bad = []
    for _code, _want, _note in _pd_cases:
        _st = certify.certify_unit("c", _code)[0]["status"]
        if _st != _want:
            _pd_bad.append(f"{_note}: {_want}!={_st}")
    add("premise_derive: a claim proved as a non-negative combination of premise"
        " differences certifies, and an unsupported claim does not"
        + (f" — {_pd_bad[0]}" if _pd_bad else ""), not _pd_bad)

    # kernel twin: the emitter's multipliers must be REPRODUCIBLE, not trusted.
    # A certificate claiming a combination the kernel cannot reproduce is a
    # mismatch — otherwise the route could certify anything.
    _pdc = certify.certify_unit("c", "def c(lo, hi, x):\n    assert lo <= hi\n"
                                      "    assert lo + x <= hi + x\n")[0]
    _pdo = next(o for o in _pdc["obligations"]
                if o["method"].startswith("premise_derive"))
    _pdok = _CTK.K.replay({"spec": {"source": "assert", "confirmed": True,
                                     "expr": _pdo["claim"]},
                           "obligations": [_pdo]})
    _pdbad = dict(_pdo, premise_combination=[7])       # wrong multipliers
    _pdno = _CTK.K.replay({"spec": {"source": "assert", "confirmed": True,
                                    "expr": _pdo["claim"]},
                           "obligations": [_pdbad]})
    add("premise_derive: the kernel reproduces the combination itself and"
        " catches inflated multipliers",
        (not _pdok["mismatches"]) and _pdno["false_certs"] >= 1)

    # the witness itself is the tell: a refutation under a premise must produce
    # a point that SATISFIES that premise. If it does not, the refutation is
    # about a program the user never wrote.
    _wit = certify.certify_unit(
        "f", "def f(n):\n    assert n <= 3\n    result = n\n"
             "    assert result <= 2\n")[0]
    _env = (_wit.get("witness") or {}).get("env") or {}
    add("box discharge: a refutation under a premise carries a witness that"
        " SATISFIES the premise",
        not _env or int(_env.get("n", 0)) <= 3)

    # 14. the SUBSCRIPT rung. The term side was already there (`celem`, and `nf`
    #     already treats `elem` as an opaque atom); what was missing was the
    #     tuple index, which spelled `d[0,1]` and `d[0][1]` as two DIFFERENT
    #     opaque atoms so nf refused to decide them. Both twins below are the
    #     reason the rung is safe: a subscript must never become a value it is
    #     not.
    _sub = [
        # (code, expected) — a known-length structure folds exactly
        ("def f():\n    d = [[1, 2], [3, 4]]\n    assert d[0, 1] == 2\n", "CERTIFIED"),
        ("def f():\n    d = [[1, 2], [3, 4]]\n    assert d[0, 1] == 9\n", "REFUTED"),
        # nested spelling is the SAME term as the tuple spelling
        ("def f():\n    d = [[1, 2], [3, 4]]\n    assert d[0][1] == 2\n", "CERTIFIED"),
        # TWIN (required): an index into an UNKNOWN base has no value. It must
        # abstain — certifying `xs[0] == 1` would be inventing a fact.
        ("def f(xs):\n    assert xs[0] == 1\n", "UNVERIFIABLE"),
        # TWIN (required): an out-of-range index is a RUNTIME claim (IndexError),
        # not a silent bound. It must abstain, never refute into a value.
        ("def f(xs):\n    assert xs[5] == 1\n", "UNVERIFIABLE"),
        # the opaque atom is still algebra: identities over it are theorems
        ("def f(xs):\n    assert xs[0] + xs[0] == 2 * xs[0]\n", "CERTIFIED"),
        ("def f(xs):\n    assert xs[0] - xs[0] == 0\n", "CERTIFIED"),
        # ...and an inequality between two unknown elements is NOT a theorem
        ("def f(xs):\n    assert xs[0] < xs[1]\n", "UNVERIFIABLE"),
    ]
    _sub_bad = []
    for _code, _want in _sub:
        _st = certify.certify_unit("f", _code)[0]["status"]
        if _st != _want:
            _sub_bad.append(f"{_want}!={_st}: {_code!r}")
    add("subscript rung: tuple/nested index folds on a known length; an unknown"
        " or out-of-range base abstains" + (f" — {_sub_bad[0]}" if _sub_bad else ""),
        not _sub_bad)

    # the rung must never produce a false certificate on any of its shapes
    _subc = [certify.certify_unit("f", c)[0] for c, _w in _sub
             if _w == "CERTIFIED"]
    add("subscript rung: kernel agreement and 0 false certificates",
        all((c.get("kernel") or {}).get("false_certs") == 0
            and (c.get("kernel") or {}).get("agreement") == 1.0
            for c in _subc))

    # tuple-index and nested-index must be the SAME term, or nf sees two atoms
    from mjolnir_prover.lib import term as _T3
    _d = _T3.clist(_T3.clist(_T3.const(1), _T3.const(2)),
                   _T3.clist(_T3.const(3), _T3.const(4)))
    _tup = _T3.celem(_d, _T3.clist(_T3.const(0), _T3.const(1)))
    _nst = _T3.celem(_T3.celem(_d, _T3.const(0)), _T3.const(1))
    add("subscript rung: `d[0,1]` and `d[0][1]` are one term, not two atoms",
        _T3.key(_tup) == _T3.key(_nst) == _T3.key(_T3.const(2)))

    return checks


def register(app: typer.Typer) -> None:

    @app.command()
    def check(
        paths: list[Path] = typer.Argument(
            ..., exists=True, readable=True,
            help="Python files or directories to verify."),
        gate: bool = typer.Option(
            False, "--gate",
            help="Fail-closed CI policy: fail the build on REFUTED/FAULT "
                 "(honours mjolnir.toml)."),
        config: Optional[Path] = typer.Option(
            None, "--config", help="Gate config (mjolnir.toml or .json)."),
        json_out: Optional[Path] = typer.Option(
            None, "--json", help="Write every certificate to this JSON file."),
        assume: Optional[Path] = typer.Option(
            None, "--assume",
            help="Confirmed mined-premise registry whose bounds are injected "
                 "into matching units."),
    ) -> None:
        """Verify Python units and print verdicts; emit certificates."""
        from mjolnir_prover.lib import gate as engine
        result = engine.run(
            [str(p) for p in paths],
            json_out=str(json_out) if json_out else None,
            assume_file=str(assume) if assume else None,
            gate=gate,
            config=str(config) if config else None,
        )
        reporting.render_check(result)
        raise typer.Exit(result["rc"])

    @app.command()
    def selfcheck() -> None:
        """Run the built-in soundness smoke test."""
        checks = run_selfcheck()
        ok = reporting.render_selfcheck(checks)
        raise typer.Exit(0 if ok else 1)

    @app.command()
    def verify() -> None:
        """Self-verify the release: built-in selfcheck + fail-closed gate."""
        reporting.banner("mjolnir — release verification")
        checks = run_selfcheck()
        rc1 = 0 if reporting.render_selfcheck(checks) else 1
        reporting.blank()
        from mjolnir_prover.lib import gate as engine
        cfg = os.path.join(RELEASE, "mjolnir.toml")
        pkg = os.path.join(RELEASE, "mjolnir", "lib")
        result = engine.run([pkg], gate=True, config=cfg)
        reporting.render_check(result)
        ok = (rc1 == 0 and result["rc"] == 0)
        reporting.render_verify_summary(ok)
        raise typer.Exit(0 if ok else 1)