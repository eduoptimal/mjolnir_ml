"""The live session: journal, monitoring, containment, retraction, change control."""
import pytest

from mjolnir_ml import (ApprovalAuthority, ChangeSet, DecisionModel, Journal,
                        LiveSession, Policy)
from mjolnir_ml.vendors import wire

import policies as P


class XClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def _policy(deny_hazard=False):
    stmts = []
    if deny_hazard:
        stmts.append({"kind": "rule", "code": "deny_hazard",
                      "requires": ["x"], "predicate": "x < 0.1",
                      "verdict": "deny"})
    stmts += [
        {"kind": "rule", "code": "ask_uncertified", "requires": ["certified"],
         "predicate": "not certified", "verdict": "ask"},
        {"kind": "release", "code": "release_certified",
         "requires": ["certified"], "predicate": "certified",
         "verdict": "release"},
    ]
    return Policy({"schema": Policy.SCHEMA, "name": "LivePolicy",
                   "facts": [], "statements": stmts})


def _release_all_safe():
    return Policy({"schema": Policy.SCHEMA, "name": "LoosenPolicy", "facts": [],
                   "statements": [{"kind": "release", "code": "release_safe",
                                   "requires": ["x"], "predicate": "x >= 0.1",
                                   "verdict": "release"}]})


def _dm(policy=None, adaptive=False):
    from mjolnir_ml import HazardLedger
    xs = [0.01 + 0.98 * i / 39 for i in range(40)]
    dm = DecisionModel(XClf(), policy or _policy(), features=("x",),
                       regime="off")
    dm.calibrate([{"x": x} for x in xs],
                 [1 if x >= 0.5 else 0 for x in xs], alpha=0.10,
                 adaptive=adaptive)
    return dm


def _ledger():
    from mjolnir_ml import HazardLedger
    return HazardLedger().add("low_x", lambda r: r["x"] < 0.1, "spec §1")


ROWS = [{"x": 0.02}, {"x": 0.50}, {"x": 0.98}]
YS = [0, 1, 1]


def _session(tmp_path=None, adaptive=False, verifier=None):
    j = Journal(tmp_path / "j.jsonl") if tmp_path else Journal()
    s = LiveSession(_dm(adaptive=adaptive), journal=j, ledger=_ledger(),
                    approval_verifier=verifier)
    for r, y in zip(ROWS, YS):
        d = s.decide(r, batch=1)
        s.observe(d.certificate_id, y)
    return s


def test_journal_hash_chain_and_tamper_detection():
    j = Journal()
    j.append({"kind": "decision", "n": 1})
    j.append({"kind": "outcome", "y": 0})
    assert j.verify() is True
    j._entries[0]["payload"]["n"] = 2
    assert j.verify() is False


def test_decide_observe_status():
    s = _session()
    st = s.status()
    assert st["n"] == 3
    assert st["verdicts"][wire.RELEASE] == 2
    assert st["verdicts"][wire.ASK] == 1
    assert st["coverage"] == pytest.approx(2 / 3)
    assert st["released_hazards"] == 1          # x=0.02 is a hazard release
    assert st["hazard_precision"] == pytest.approx(0.5)
    assert st["mae"] == pytest.approx(0.0)
    assert s.journal.verify() is True


def test_tick_is_deterministic_and_flags_the_release():
    s = _session()
    a = [d.as_dict() for d in s.tick()]
    b = [d.as_dict() for d in s.tick()]
    assert a == b
    hazard = next(d for d in s.tick() if d.metric == "hazard_release")
    assert hazard.alarm and hazard.severity == "critical"
    coverage = next(d for d in s.tick() if d.metric == "coverage")
    assert coverage.n == 3 and "insufficient n" in coverage.detail


def test_auto_act_contains_and_containment_only_tightens():
    s = _session()
    actions = s.auto_act(s.tick(), batch=1)
    assert len(actions) == 1 and s.containment is not None
    d = s.decide({"x": 0.02}, batch=2)
    assert d.verdict == wire.ASK
    assert any(b.startswith("contained:") for b in d.basis)
    assert d.certificate_id != ""
    assert s.status()["contained"] is not None


def test_uncontain_requires_external_approval(tmp_path):
    authority = ApprovalAuthority(b"demo")
    s = _session(tmp_path / "a", verifier=authority.verify)
    s.contain("test")
    c = s.containment
    with pytest.raises(PermissionError):
        s.uncontain(authority.sign("wrong"))
    s.uncontain(authority.sign(c.change_digest))
    assert s.containment is None
    d = s.decide({"x": 0.02}, batch=2)
    assert d.verdict == wire.RELEASE


def test_retraction_lists_and_does_not_duplicate():
    s = _session()
    policy_digest = s.model.policy.digest
    r1 = s.retract(policy_digest=policy_digest, reason="faulty version")
    assert len(r1.certificate_ids) == 3
    r2 = s.retract(policy_digest=policy_digest, reason="again")
    assert r2.certificate_ids == ()
    assert len(s.affected_certificates(policy_digest=policy_digest)) == 3


def _evaluate_repair(s):
    change = ChangeSet(lever="contract", reason="deny the low-x hazard",
                       policy=_policy(deny_hazard=True), name="repair")
    oracle = lambda row, y: (row["x"] >= 0.1, True)      # noqa: E731
    return s.evaluate(change, oracle=oracle), change, oracle


def test_change_evaluation_pareto_and_promotion_without_approval():
    s = _session()
    ev, change, oracle = _evaluate_repair(s)
    assert ev.may_loosen is False
    assert ev.before["released_hazards"] == 1
    assert ev.after["released_hazards"] == 0
    assert ev.after["precision"] > ev.before["precision"]
    assert ev.passed is True
    p = s.promote(change, oracle=oracle)
    assert p.approved is True
    assert p.after["policy_digest"] != p.before["policy_digest"]
    assert s.model.policy.digest == change.policy.digest
    assert len(s.versions) == 2


def test_loosening_change_requires_approval_and_is_refused_without_it():
    authority = ApprovalAuthority(b"demo")
    s = _session(verifier=authority.verify)
    change = ChangeSet(lever="contract", reason="release more safe rows",
                       policy=_release_all_safe(), name="loosen")
    oracle = lambda row, y: (row["x"] >= 0.1, True)      # noqa: E731
    ev = s.evaluate(change, oracle=oracle)
    assert ev.may_loosen is True and ev.added_releases == 1
    blocked = s.promote(change, oracle=oracle)
    assert blocked.approved is False and "approval required" in blocked.reason

    approval = authority.sign(change.digest(s.model.features))
    promoted = s.promote(change, oracle=oracle, approval=approval)
    assert promoted.approved is True
    assert s.model.policy.digest == change.policy.digest


def test_promotion_journal_and_versions_are_appended():
    s = _session()
    ev, change, oracle = _evaluate_repair(s)
    s.promote(change, oracle=oracle)
    changes = s.journal.payloads("change")
    assert changes and changes[-1]["promoted"] is True
    assert changes[-1]["change_digest"] == ev.change_digest
    assert changes[-1]["after"]["policy_digest"] == change.policy.digest
