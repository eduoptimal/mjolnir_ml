"""The @dsl compiler's laws: fail-closed authoring, one IR, -O immunity."""
import ast
import os
import subprocess
import sys

import pytest

import mjolnir_ml
from mjolnir_ml import DslError, dsl, rule
from mjolnir_ml.policy import MODEL_FACTS

import policies as P


def test_compiles_one_ir_shape():
    pol = P.FraudPolicy.policy()
    assert pol.name == "FraudPolicy"
    assert pol.digest.startswith("sha256:")
    kinds = [s.kind for s in pol.statements]
    assert kinds == ["premise", "premise", "rule", "release", "rule", "rule"]
    codes = [s.code for s in pol.statements]
    assert codes == ["amount_positive", "limit_positive", "hard_deny",
                     "auto_release", "escalate_mid", "escalate_uncertain"]
    # requires come from the lambda params, not from runtime introspection
    auto = pol.statement("auto_release")
    assert auto.requires == ("p_positive", "conf")
    assert auto.verdict == "release"
    assert auto.predicate == "p_positive < 0.05 and conf >= 0.95"


def test_digest_is_stable_and_content_addressed():
    a = P.FraudPolicy.policy().digest
    # recompiling the same source yields the same digest
    dsl(P.FraudPolicy)
    assert P.FraudPolicy.policy().digest == a
    # a different policy is a different digest
    assert P.CertifiedPolicy.policy().digest != a


def test_model_facts_are_reserved():
    with pytest.raises(DslError):
        @dsl
        class Bad(mjolnir_ml.DecisionPolicyBase):
            facts = ("pred",)

            assert rule("r", when=lambda pred: pred > 0, verdict="deny")


def test_method_form():
    pol = P.FraudPolicy.policy()
    st = pol.statement("escalate_uncertain")
    assert st.requires == ("conf",)
    assert st.predicate == "conf < 0.6"
    assert st.verdict == "ask"


def test_premise_needs_a_code():
    with pytest.raises(DslError, match="needs a code"):
        dsl(P.PremiseWithoutCode)


def test_attribute_access_rejected():
    with pytest.raises(DslError, match="attribute access"):
        dsl(P.BadAttributePolicy)


def test_call_rejected():
    with pytest.raises(DslError, match="calls are not allowed"):
        dsl(P.BadCallPolicy)


def test_free_name_rejected():
    with pytest.raises(DslError, match="not a declared fact"):
        dsl(P.BadFreeNamePolicy)


def test_forbidden_verdicts():
    with pytest.raises(DslError, match="abstain is"):
        dsl(P.ForbiddenVerdictPolicy)


def test_duplicate_codes():
    with pytest.raises(DslError, match="duplicate code"):
        dsl(P.DuplicateCodePolicy)


def test_undeclared_fact_with_facts_declared():
    with pytest.raises(DslError, match="neither declared facts"):
        dsl(P.UndeclaredFactPolicy)


def test_source_unavailable_fails_closed():
    src = (
        "from mjolnir_ml import dsl, rule, DENY, DecisionPolicyBase\n"
        "class Dynamic(DecisionPolicyBase):\n"
        "    facts = ('x',)\n"
        "    assert rule('r', when=lambda x: x > 0, verdict=DENY)\n"
        "dsl(Dynamic)\n"
    )
    with pytest.raises(DslError, match="source unavailable"):
        exec(compile(src, "<dynamic>", "exec"), {})


def test_optimized_interpreter_keeps_policy():
    """`python -O` strips asserts; source harvest must not care."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([here, root]))
    code = (
        "import policies as P\n"
        "print(P.FraudPolicy.policy().digest)\n"
        "print(len(P.FraudPolicy.policy().statements))\n"
    )
    out = subprocess.run([sys.executable, "-O", "-c", code], env=env,
                         capture_output=True, text=True, check=True).stdout
    lines = out.split()
    assert lines[0] == P.FraudPolicy.policy().digest
    assert lines[1] == str(len(P.FraudPolicy.policy().statements))


def test_policy_without_statements_rejected():
    with pytest.raises(DslError, match="no statements"):
        @dsl
        class Empty(mjolnir_ml.DecisionPolicyBase):
            facts = ("x",)


def test_unknown_statement_rejected():
    with pytest.raises(DslError, match="assignable"):
        @dsl
        class Weird(mjolnir_ml.DecisionPolicyBase):
            facts = ("x",)
            THRESH = 0.5


def test_ipython_cell_source_fallback(monkeypatch, tmp_path):
    """Notebook cells have no file; the decorator may read the cell history."""
    import sys as _sys
    import types
    cell = (
        "from mjolnir_ml import DecisionPolicyBase, rule, DENY, dsl\n"
        "class CellPolicy(DecisionPolicyBase):\n"
        "    facts = ('x',)\n"
        "    assert rule('r', when=lambda x: x > 0, verdict=DENY)\n"
        "dsl(CellPolicy)\n"
    )
    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: types.SimpleNamespace(
        user_ns={"In": [cell]})
    monkeypatch.setitem(_sys.modules, "IPython", fake)
    ns = {"__name__": "fake_cell"}
    exec(compile(cell, "<cell>", "exec"), ns)
    pol = ns["CellPolicy"].policy()
    assert [s.code for s in pol.statements] == ["r"]
