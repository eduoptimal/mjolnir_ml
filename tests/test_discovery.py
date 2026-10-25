"""The discovery session: unlabeled candidates, an independent oracle, the
gates (tests, twin, census movement), the union-only ledger, containment, and
the certificate of the move."""
import copy

import pytest

from mjolnir_ml import (ApprovalAuthority, Candidate, Census,
                        DiscoveryDetector, DiscoveryError, DiscoverySession,
                        KnowledgeRecord,
                        certify_admission, verify_admission_certificate)
from mjolnir_ml.discovery import (CENSUS_VERDICTS, CERTIFIED, DECLINED,
                                  DECLARED, REFUTED, _admission_id)

# ------------------------------------------------------------------ corpus --
C0 = {
    "xs[0] < 3": DECLINED,
    "xs[1] < 3": DECLINED,
    "len(xs) > 0": DECLINED,
    "n * 0.5 + n * 0.5 == n": CERTIFIED,
    "commutative(a, b)": DECLARED,
    "x / 0 == 1": REFUTED,
}
REASONS = {
    "xs[0] < 3": "subscript",
    "xs[1] < 3": "subscript",
    "len(xs) > 0": "call",
    "n * 0.5 + n * 0.5 == n": "float",
    "commutative(a, b)": "boolop",
    "x / 0 == 1": "arithmetic",
}
MINE = "mine.subscript-v1"
RING = {"xs[0] < 3", "xs[1] < 3", "ys[0] == 1"}


def subscript_rung(claim):
    """Independent oracle: decides the subscript class; everything else keeps
    its stored verdict (or DECLINED when the corpus does not know it)."""
    if claim in RING:
        return CERTIFIED
    return C0.get(claim, DECLINED)


def refuting_rung(claim):
    if claim in RING:
        return CERTIFIED
    if claim == "len(xs) > 0":
        return REFUTED
    return C0.get(claim, DECLINED)


def loosening_rung(claim):
    if claim in RING or claim == "x / 0 == 1":
        return CERTIFIED
    return C0.get(claim, DECLINED)


def tightening_rung(claim):
    if claim in RING:
        return CERTIFIED
    if claim == "n * 0.5 + n * 0.5 == n":
        return DECLINED
    return C0.get(claim, DECLINED)


class SameProducer:
    discovery_id = MINE

    def __call__(self, claim):
        return CERTIFIED


def sub_cand(**kw):
    base = dict(kind="rung", shape="V[N] < N",
                claims=("xs[0] < 3", "xs[1] < 3"),
                proposed_by=MINE, reason="subscript", domain="python",
                population=109,
                tests=(("ys[0] == 1", CERTIFIED),),
                twin=(("xs[99] < 3", DECLINED),),
                evidence={"repo": "example"})
    base.update(kw)
    return Candidate(**base)


def _session(*, verifier=None, corpus=None, reasons=REASONS, journal=None):
    return DiscoverySession(dict(C0) if corpus is None else corpus,
                            reasons=reasons, approval_verifier=verifier,
                            journal=journal)


# ------------------------------------------------------------------- census --
def test_census_is_exactly_four_numbers_never_a_scalar():
    c = Census.from_verdicts([CERTIFIED, CERTIFIED, DECLINED])
    assert c.as_dict() == {"certified": 2, "declared": 0, "declined": 1,
                           "refuted": 0}
    assert c.total == 3
    assert set(c.as_dict()) == set(CENSUS_VERDICTS)
    assert set(c.as_dict()) == {"certified", "declared", "declined", "refuted"}
    assert not any("score" in k or "rate" in k or "quality" in k
                   for k in c.as_dict())
    assert c.delta(Census(certified=1)) == {"certified": 1, "declared": 0,
                                            "declined": 1, "refuted": 0}
    with pytest.raises(DiscoveryError, match="unknown census verdict"):
        Census.from_verdicts(["trusted"])


# ----------------------------------------------------------------- propose --
def test_candidates_are_unlabeled_and_propose_stamps_provenance():
    def mine():
        return [Candidate(kind="rung", shape="V == V", claims=("a == a",))]

    (c,) = _session().propose(mine)
    assert c.proposed_by.endswith(".mine")
    assert c.candidate_id.startswith("sha256:")

    with pytest.raises(DiscoveryError, match="unlabeled"):
        Candidate(kind="rung", shape="V == V", claims=("a == a",),
                  evidence={"expect": CERTIFIED})
    with pytest.raises(DiscoveryError, match="at least one claim"):
        Candidate(kind="rung", shape="s", claims=())


def test_propose_refuses_non_candidates():
    s = _session()
    with pytest.raises(DiscoveryError, match="not a Candidate"):
        s.propose(["not a candidate"])


def test_evaluation_requires_provenance_and_corpus_membership():
    s = _session()
    bare = Candidate(kind="rung", shape="V[N] < N",
                     claims=("xs[0] < 3", "xs[1] < 3"),
                     tests=(("ys[0] == 1", CERTIFIED),),
                     twin=(("xs[99] < 3", DECLINED),))
    with pytest.raises(DiscoveryError, match="provenance"):
        s.evaluate(bare, subscript_rung)
    with pytest.raises(DiscoveryError, match="outside the corpus"):
        s.evaluate(sub_cand(claims=("never-seen",)), subscript_rung)
    with pytest.raises(DiscoveryError, match="fail-closed"):
        s.evaluate(sub_cand(), lambda claim: "trusted")


def test_propose_accepts_one_candidate_or_a_batch():
    def one():
        return Candidate(kind="rung", shape="V == V", claims=("a == a",))

    def many():
        return (Candidate(kind="rung", shape="V == V", claims=("a == a",)),
                Candidate(kind="rung", shape="V < V", claims=("b < b",)))

    s = _session()
    assert len(s.propose(one)) == 1
    assert len(s.propose(many)) == 2


# ---------------------------------------------------------------- evaluate --
def test_evaluate_moves_the_census_and_journals_provenance_first():
    s = _session()
    ev = s.evaluate(sub_cand(), subscript_rung)
    assert ev.passed and ev.may_loosen is False
    # the census is scoped to the evaluated claims, not the whole corpus
    assert ev.n == 2
    assert ev.before["declined"] == 2 and ev.after["declined"] == 0
    assert ev.after["certified"] == ev.before["certified"] + 2
    kinds = [p["kind"] for p in s.journal.payloads()]
    assert kinds == ["candidate", "evaluation"]     # provenance before verdicts
    gate = next(g for g in ev.gates if g.name == "targeted_decline_moved")
    assert gate.passed and gate.value == {"before": 2, "after": 0}


def test_declared_target_without_reason_data_fails_closed():
    s = _session(reasons=None)
    ev = s.evaluate(sub_cand(), subscript_rung)
    gate = next(g for g in ev.gates if g.name == "targeted_decline_moved")
    assert not gate.passed and "decline-reason data" in gate.detail


def test_self_certification_is_refused_journaled_and_contains():
    s = _session()
    with pytest.raises(PermissionError, match="self-certification"):
        s.evaluate(sub_cand(), SameProducer())
    refusals = s.journal.payloads("refusal")
    assert refusals[-1]["reason"] == "self_certification"
    det = next(d for d in s.tick() if d.metric == "refusal_self_cert")
    assert det.alarm and det.severity == "critical"
    acts = s.auto_act(s.tick())
    assert len(acts) == 1 and s.containment is not None
    adm = s.promote(sub_cand(), subscript_rung)
    assert adm.approved is False and adm.reason.startswith("contained:")


def test_twin_and_test_battery_failures_refuse_admission():
    no_tests = sub_cand(tests=())
    s1 = _session()
    adm = s1.promote(no_tests, subscript_rung)
    assert adm.approved is False and "labelled_tests" in adm.reason

    bad_twin = sub_cand(twin=(("xs[0] < 3", DECLINED),))   # rung certifies it
    s2 = _session()
    adm2 = s2.promote(bad_twin, subscript_rung)
    assert adm2.approved is False and "twin_holds" in adm2.reason
    assert s2.corpus["xs[0] < 3"] == DECLINED        # corpus untouched
    assert s2.ledger.active == {} and s2.versions == []


def test_no_new_refuted_gate_treats_a_new_refutation_as_a_defect():
    c = sub_cand(claims=("xs[0] < 3", "len(xs) > 0"))
    s = _session()
    adm = s.promote(c, refuting_rung)
    assert adm.approved is False and "no_new_refuted" in adm.reason
    det = next(d for d in s.tick() if d.metric == "refuted_growth")
    assert det.alarm and det.severity == "critical"
    assert s.containment is None     # detections do not act by themselves


def test_decline_growth_warns_and_never_contains():
    c = sub_cand(claims=("commutative(a, b)",), reason="",
                 tests=(("commutative(a, b)", DECLINED),),
                 twin=(("x / 0 == 1", REFUTED),))
    s = _session()
    adm = s.promote(c, lambda claim: DECLINED)
    assert adm.approved is False and "certified_gained" in adm.reason
    det = next(d for d in s.tick() if d.metric == "declined_growth")
    assert det.alarm and det.severity == "warning"
    assert s.auto_act(s.tick()) == [] and s.containment is None


# ------------------------------------------------------ the promotion laws --
def test_admission_applies_and_is_free_for_the_frontier_move():
    s = _session()
    adm = s.promote(sub_cand(), subscript_rung)
    assert adm.approved and adm.reason == "admitted"
    assert s.corpus["xs[0] < 3"] == CERTIFIED
    assert s.corpus["xs[1] < 3"] == CERTIFIED
    assert len(s.ledger.active) == 1 and len(s.versions) == 1
    assert s.ledger.has(adm.candidate_id)
    adms = s.journal.payloads("admission")
    assert adms[-1]["promoted"] is True


def test_refuted_to_certified_needs_external_approval():
    c = sub_cand(claims=("xs[0] < 3", "xs[1] < 3", "x / 0 == 1"))

    s = _session()                       # no verifier held at all
    ev = s.evaluate(c, loosening_rung)
    assert ev.may_loosen is True and ev.flipped_refuted == 1
    adm = s.promote(c, loosening_rung)
    assert adm.approved is False and "approval required" in adm.reason
    assert s.corpus["x / 0 == 1"] == REFUTED      # nothing loosened

    auth = ApprovalAuthority(b"discovery")
    s2 = _session(verifier=auth.verify)
    ev2 = s2.evaluate(c, loosening_rung)
    blocked = s2.promote(c, loosening_rung, approval=auth.sign("wrong"))
    assert blocked.approved is False and "approval required" in blocked.reason
    ok = s2.promote(c, loosening_rung,
                    approval=auth.sign(ev2.change_digest))
    assert ok.approved is True
    assert s2.corpus["x / 0 == 1"] == CERTIFIED


def test_tightening_a_certified_claim_is_free():
    c = sub_cand(claims=("xs[0] < 3", "xs[1] < 3", "n * 0.5 + n * 0.5 == n"))
    s = _session()                       # no verifier, no approval offered
    adm = s.promote(c, tightening_rung)
    assert adm.approved is True
    ev = adm.evaluation
    assert ev.tightened == 1 and ev.may_loosen is False
    assert s.corpus["n * 0.5 + n * 0.5 == n"] == DECLINED


def test_contained_session_refuses_admissions_until_signed_uncontain():
    auth = ApprovalAuthority(b"discovery")
    s = _session(verifier=auth.verify)
    c = sub_cand(claims=("xs[0] < 3", "len(xs) > 0"))
    s.promote(c, refuting_rung)                    # critical: new refuted
    s.auto_act(s.tick())
    assert s.containment is not None
    blocked = s.promote(sub_cand(), subscript_rung)
    assert blocked.approved is False
    assert blocked.reason.startswith("contained:")

    with pytest.raises(PermissionError, match="approval"):
        s.uncontain(auth.sign("wrong"))
    s.uncontain(auth.sign(s.containment.change_digest))
    assert s.containment is None
    # epoch: the old refuted_growth evaluation does not re-alarm
    det = next(d for d in s.tick() if d.metric == "refuted_growth")
    assert det.alarm is False
    assert s.promote(sub_cand(), subscript_rung).approved is True


# ------------------------------------------------------- the union-only law --
def test_knowledge_ledger_is_union_only_and_retirement_reverts():
    s = _session()
    adm = s.promote(sub_cand(), subscript_rung)
    cid = adm.candidate_id
    led = s.ledger
    with pytest.raises(DiscoveryError, match="never reused"):
        led.add(led.get(cid))
    dup = s.promote(sub_cand(), subscript_rung)
    assert dup.approved is False and "already admitted" in dup.reason

    with pytest.raises(ValueError, match="refutation"):
        led.retire(cid, "")
    payload = s.retire(cid, "the rung certified a claim the kernel refutes")
    assert payload["reverted"] == 2
    assert s.corpus["xs[0] < 3"] == DECLINED       # reverted to pre-admission
    assert led.active == {} and cid in led.retired
    stale = KnowledgeRecord(knowledge_id=cid, kind="rung", shape="V[N] < N",
                            domain="", reason="", claims=(), before={}, after={},
                            before_verdicts={}, after_verdicts={},
                            oracle_id="", admitted_at="")
    with pytest.raises(DiscoveryError, match="never reused"):
        led.add(stale)
    with pytest.raises(KeyError):
        led.retire(cid, "already retired")
    assert s.status()["ledger"]["retired"] == 1
    assert s.journal.payloads("retraction")


# --------------------------------------------------------------- monitoring --
def test_tick_is_deterministic_from_the_journal():
    s = _session()
    s.promote(sub_cand(), subscript_rung)
    a = [d.as_dict() for d in s.tick()]
    b = [d.as_dict() for d in s.tick()]
    assert a == b
    assert all(not d.alarm for d in s.tick())


def test_status_reports_frontier_census_and_queue():
    s = _session()
    st = s.status()
    assert st["census"] == {"certified": 1, "declared": 1, "declined": 3,
                            "refuted": 1}
    assert st["frontier"] == {"subscript": 2, "call": 1}
    assert st["candidates"]["proposed"] == 0
    s.propose([sub_cand()])
    assert s.status()["candidates"]["queue"]     # proposed, not yet admitted
    adm = s.promote(sub_cand(), subscript_rung)
    assert adm.approved
    st2 = s.status()
    assert st2["candidates"]["admitted"] == 1
    assert st2["candidates"]["queue"] == []
    assert st2["frontier"] == {"call": 1}        # subscript class moved
    assert st2["census"]["declined"] == 1


def test_journal_is_tamper_evident():
    s = _session()
    s.propose([sub_cand()])
    s.evaluate(sub_cand(), subscript_rung)
    assert s.journal.verify() is True
    s.journal._entries[0]["payload"]["candidate"]["proposed_by"] = "x"
    assert s.journal.verify() is False


def test_constructor_fails_closed():
    with pytest.raises(DiscoveryError, match="not a census verdict"):
        DiscoverySession({"a": "trusted"})
    with pytest.raises(DiscoveryError, match="outside the corpus"):
        DiscoverySession({}, reasons={"z": "rung"})
    with pytest.raises(DiscoveryError, match="unknown detector metric"):
        DiscoverySession({}, detectors=[
            DiscoveryDetector("x", "lever", "not_a_metric")])
    with pytest.raises(TypeError, match="corpus"):
        DiscoverySession(["not", "a", "corpus"])


# -------------------------------------------------------------- certificate --
KEY = b"discovery-key"


def _admitted():
    s = _session()
    adm = s.promote(sub_cand(), subscript_rung)
    assert adm.approved
    return s, adm


def test_certificate_sign_verify_and_reject_tampering():
    _, adm = _admitted()
    cert = certify_admission(adm, key=KEY)
    assert verify_admission_certificate(cert, key=KEY) is True

    tampered = copy.deepcopy(cert)
    tampered["after"]["refuted"] = 9
    with pytest.raises(DiscoveryError, match="signature"):
        verify_admission_certificate(tampered, key=KEY)
    with pytest.raises(DiscoveryError, match="signature"):
        verify_admission_certificate(cert, key=b"wrong")
    with pytest.raises(DiscoveryError, match="unsigned"):
        verify_admission_certificate(certify_admission(adm),
                                     require_signature=True)


def test_certificate_binding_and_replay_are_independent_checks():
    _, adm = _admitted()
    unsigned = certify_admission(adm)

    bound = copy.deepcopy(unsigned)
    bound["after"]["certified"] = 99
    with pytest.raises(DiscoveryError, match="binding"):
        verify_admission_certificate(bound)

    replay = copy.deepcopy(unsigned)
    replay["after"]["refuted"] = replay["before"]["refuted"] + 5
    body = {k: v for k, v in replay.items()
            if k not in ("admission_id", "issued_at", "signature")}
    replay["admission_id"] = _admission_id(body)      # id now consistent
    with pytest.raises(DiscoveryError, match="replay"):
        verify_admission_certificate(replay)


def test_only_an_admitted_move_certifies():
    s, _ = _admitted()
    refused = s.promote(sub_cand(), subscript_rung)   # already admitted
    assert refused.approved is False
    with pytest.raises(DiscoveryError, match="admitted move"):
        certify_admission(refused)
