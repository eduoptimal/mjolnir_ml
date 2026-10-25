"""The causal layer: do-calculus claims, identification strategies, placebos,
and the event study over a journal series."""
import random

import pytest

from mjolnir_ml import (ApprovalAuthority, CausalClaim, CausalError,
                        CausalSession, Series, Strategy, do, event_study)
from mjolnir_ml.causal import CausalOracle
from mjolnir_ml.causal import series_from_journal as T_series
from mjolnir_ml.discovery import CERTIFIED, DECLINED, REFUTED


# ------------------------------------------------------------------ the SCM --
def scm(n=6000, seed=7, g_bad_violates=True, ynoise=0.30, u_y=0.8):
    """One declared SCM. Columns: z (measured confounder of x1), g (instrument
    for x2), g_bad (an invalid instrument: relevance + exclusion violation),
    a,b (collider at y), w (pure cause of y), p (placebo treatment), x1, x2,
    y, y2 (replication outcome), y_null (placebo outcome). `u` is the unmeasured
    confounder of x2 and is NOT in the table."""
    rng = random.Random(seed)
    rng2 = random.Random(seed + 1000)   # auxiliary draws never disturb the SCM
    rows = []
    for _ in range(n):
        u = rng.gauss(0, 1)
        z = 1 if rng.random() < 0.5 else 0
        g = 1 if rng.random() < 0.5 else 0
        g_bad = 1 if rng.random() < 0.5 else 0
        a, b, w = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        p = 1 if rng.random() < 0.5 else 0
        x1 = 1 if (1.6 * z + rng.gauss(0, 1.2)) > 0.8 else 0
        x2 = 1 if (1.4 * g + (0.6 if g_bad_violates else 0.0) * g_bad
                   + 1.0 * u + rng.gauss(0, 0.8)) > 0.6 else 0
        y = (1.0 * x1 + 0.5 * x2 + 1.8 * z + u_y * u
             + 0.9 * a + 0.9 * b + 0.7 * w
             + (0.8 if g_bad_violates else 0.0) * g_bad
             + rng.gauss(0, ynoise))
        y2 = 0.5 * x1 + 1.8 * z + 0.7 * w + rng2.gauss(0, 0.4)
        y_null = rng2.gauss(0, 1)
        rows.append({"z": z, "g": g, "g_bad": g_bad, "a": a, "b": b, "w": w,
                     "p": p, "x1": x1, "x2": x2, "y": y, "y2": y2,
                     "y_null": y_null})
    return rows


ASSUME_OBS = ("no unmeasured confounding for the declared adjustment set",)
ASSUME_IV = ("exclusion restriction", "monotonicity", "independence of g")
ASSUME_CI = ("linear-Gaussian", "faithfulness")


def claims():
    return [
        CausalClaim(kind="association", outcome="y", treatment="x1",
                    arms=(do("x1", 1), do("x1", 0)),
                    criterion={"min": 0.8, "max": 1.2}, label="association"),
        CausalClaim(kind="ate", outcome="y", treatment="x1",
                    arms=(do("x1", 1), do("x1", 0)),
                    criterion={"min": 0.8, "max": 1.2},
                    assumptions=ASSUME_OBS, label="ate_x1"),
        CausalClaim(kind="ate", outcome="y", treatment="x2",
                    arms=(do("x2", 1), do("x2", 0)),
                    criterion={"min": 0.2, "max": 0.8},
                    assumptions=ASSUME_OBS, label="ate_x2"),
        CausalClaim(kind="late", outcome="y", treatment="x2",
                    arms=(do("x2", 1), do("x2", 0)), instrument="g",
                    criterion={"min": 0.2, "max": 0.8},
                    assumptions=ASSUME_IV, label="late_x2"),
        CausalClaim(kind="ate", outcome="y", treatment="p",
                    arms=(do("p", 1), do("p", 0)),
                    criterion={"min": -0.25, "max": 0.25},
                    assumptions=("p is randomized",), label="placebo_treatment"),
        CausalClaim(kind="ate", outcome="y_null", treatment="x1",
                    arms=(do("x1", 1), do("x1", 0)),
                    criterion={"min": -0.25, "max": 0.25},
                    assumptions=("y_null is a placebo outcome",),
                    label="placebo_outcome"),
        CausalClaim(kind="ate", outcome="y_null", treatment="x2",
                    arms=(do("x2", 1), do("x2", 0)),
                    criterion={"min": -0.25, "max": 0.25},
                    assumptions=("y_null is a placebo outcome",),
                    label="placebo_iv"),
        CausalClaim(kind="independence", left="a", right="b",
                    criterion={"min": -1, "max": 1}, assumptions=ASSUME_CI,
                    label="a ⟂ b"),
        CausalClaim(kind="independence", left="a", right="b", conditions=("y",),
                    criterion={"min": -1, "max": 1}, assumptions=ASSUME_CI,
                    label="a ⟂ b | y"),
        CausalClaim(kind="independence", left="w", right="x2",
                    criterion={"min": -1, "max": 1}, assumptions=ASSUME_CI,
                    label="w ⟂ x2"),
        CausalClaim(kind="independence", left="w", right="x2",
                    conditions=("y",),
                    criterion={"min": -1, "max": 1}, assumptions=ASSUME_CI,
                    label="w ⟂ x2 | y"),
        CausalClaim(kind="independence", left="g", right="z",
                    criterion={"min": -1, "max": 1}, assumptions=ASSUME_CI,
                    label="g ⟂ z"),
    ]


def by_label(cs, label):
    return next(c for c in cs.claims.values() if c.label == label)


def session(**kw):
    return CausalSession(claims(), table=scm(), **kw)


# ---------------------------------------------------------------- primitives --
def test_do_operator_and_claim_ids_are_canonical():
    assert str(do("x", 1)) == "do(x=1)"
    with pytest.raises(CausalError):
        do("", 1)
    ate = CausalClaim(kind="ate", outcome="y", treatment="x",
                      arms=(do("x", 1), do("x", 0)),
                      criterion={"min": 0.9, "max": 1.1},
                      assumptions=("a",))
    assert ate.id == "ATE[y|do(x=1), do(x=0)] ∈ [0.9, 1.1]"
    ind = CausalClaim(kind="independence", left="a", right="b",
                      conditions=("y",), criterion={"min": -1, "max": 1},
                      assumptions=("linear-Gaussian",))
    assert ind.id == "a ⟂ b | y ∈ [-1, 1]"
    with pytest.raises(CausalError, match="DECLARE"):
        CausalClaim(kind="ate", outcome="y", treatment="x",
                    arms=(do("x", 1), do("x", 0)),
                    criterion={"min": 0, "max": 1})
    with pytest.raises(CausalError, match="acceptance region"):
        CausalClaim(kind="association", outcome="y", treatment="x",
                    arms=(do("x", 1), do("x", 0)), criterion={})


# ------------------------------------------------------------------ the naive --
def test_association_is_biased_and_causal_claims_start_unidentified():
    cs = session()
    assert cs.corpus[by_label(cs, "ate_x1").id] == DECLINED
    assert cs.corpus[by_label(cs, "late_x2").id] == DECLINED
    assoc = by_label(cs, "association")
    assert cs.corpus[assoc.id] == REFUTED        # confounded association ≠ 1.0
    frontier = cs.identification_frontier()
    assert frontier.get("unidentified:no-strategy", 0) >= 4


# ------------------------------------------------------------------ backdoor --
def test_backdoor_identifies_the_effect_the_association_could_not():
    cs = session()
    claim = by_label(cs, "ate_x1")
    strat = Strategy("backdoor", adjust=("z",))
    oracle = cs.strategy(strat)
    assert oracle(claim.id) == CERTIFIED
    est = oracle.estimate(claim.id)
    assert 0.8 <= est.lo and est.hi <= 1.2 and est.lo <= 1.0 <= est.hi
    assert est.detail["cells"] == 2


def test_backdoor_positivity_violation_is_declined_not_certified():
    rows = scm(n=400, seed=8)
    for r in rows:                                 # one z cell loses its controls
        if r["z"] == 0:
            r["x1"] = 1
    cs = CausalSession(claims(), table=rows)
    cs.register([by_label(cs, "ate_x1")])
    oracle = cs.strategy(Strategy("backdoor", adjust=("z",), min_cell=20))
    claim = by_label(cs, "ate_x1")
    assert oracle(claim.id) == DECLINED
    assert oracle.reasons[claim.id] == "positivity_violation"


def test_missing_assumptions_decline_even_a_perfect_strategy():
    with pytest.raises(CausalError, match="DECLARE"):
        # the claim refuses to exist undeclared — the gate is in the type
        CausalClaim(kind="ate", outcome="y", treatment="x1",
                    arms=(do("x1", 1), do("x1", 0)),
                    criterion={"min": 0.9, "max": 1.1})
    with pytest.raises(CausalError, match="DECLARE"):
        CausalClaim(kind="independence", left="a", right="b",
                    criterion={"min": -1, "max": 1}, assumptions=())


# -------------------------------------------------------------------- paired --
def paired_table(n=400, seed=21, effect=0.25):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        base = rng.gauss(0, 0.5)
        rows.append({"pair": i, "arm": 1, "y": base + effect + rng.gauss(0, 0.1)})
        rows.append({"pair": i, "arm": 0, "y": base + rng.gauss(0, 0.1)})
    return rows


def test_paired_contrast_recovers_the_within_unit_effect_and_its_placebo():
    claim = CausalClaim(kind="ate", outcome="y", treatment="arm",
                        arms=(do("arm", 1), do("arm", 0)),
                        criterion={"min": 0.2, "max": 0.3},
                        assumptions=("each pair appears under both arms",))
    cs = CausalSession([claim], table=paired_table())
    oracle = cs.strategy(Strategy("paired", pair="pair"))
    assert oracle(claim.id) == CERTIFIED
    est = oracle.estimate(claim.id)
    assert 0.2 <= est.value <= 0.3 and est.detail["pairs"] == 400

    placebo = CausalClaim(kind="ate", outcome="y_null", treatment="arm",
                          arms=(do("arm", 1), do("arm", 0)),
                          criterion={"min": -0.05, "max": 0.05},
                          assumptions=("each pair appears under both arms",))
    rows = paired_table()
    for r in rows:
        r["y_null"] = 0.0
    cs2 = CausalSession([placebo], table=rows)
    o2 = cs2.strategy(Strategy("paired", pair="pair"))
    assert o2(placebo.id) == CERTIFIED
    assert o2.estimate(placebo.id).lo == 0.0


def test_paired_positivity_requires_both_arms_in_every_pair():
    rows = paired_table(n=40)
    rows = [r for r in rows if not (r["pair"] == 3 and r["arm"] == 0)]
    claim = CausalClaim(kind="ate", outcome="y", treatment="arm",
                        arms=(do("arm", 1), do("arm", 0)),
                        criterion={"min": 0.2, "max": 0.3},
                        assumptions=("each pair appears under both arms",))
    cs = CausalSession([claim], table=rows)
    oracle = cs.strategy(Strategy("paired", pair="pair"))
    assert oracle(claim.id) == DECLINED
    assert oracle.reasons[claim.id] == "positivity_violation"


# ----------------------------------------------------------------------- IV --
def test_iv_recovers_the_effect_when_the_confounder_is_unmeasured():
    cs = session()
    claim = by_label(cs, "late_x2")
    oracle = cs.strategy(Strategy("iv", instrument="g"))
    assert oracle(claim.id) == CERTIFIED
    est = oracle.estimate(claim.id)
    assert 0.2 <= est.value <= 0.8
    assert est.detail["first_stage_t"] > 3.16


def test_iv_relevance_gate_refuses_a_weak_instrument():
    cs = session()
    claim = by_label(cs, "ate_x2")
    oracle = cs.strategy(Strategy("iv", instrument="z"))   # z does not move x2
    assert oracle(claim.id) == DECLINED
    assert oracle.reasons[claim.id] == "weak_instrument"


def test_legacy_invalid_instrument_is_refuted_and_repair_needs_approval(tmp_path):
    cs = session()
    claim = by_label(cs, "late_x2")
    assert cs.corpus[claim.id] == DECLINED        # record was the naive pass

    # a prior study used g_bad (relevance + exclusion violation) and REFUTED it
    cs2 = CausalSession(claims(), table=scm(),
                        legacy={by_label(session(), "late_x2").id:
                                Strategy("iv", instrument="g_bad")})
    assert cs2.corpus[claim.id] == REFUTED
    assert cs2._reasons[claim.id] == "prior_study:iv(g_bad)"

    auth = ApprovalAuthority(b"causal")
    cs3 = CausalSession(claims(), table=scm(),
                        legacy={claim.id: Strategy("iv", instrument="g_bad")},
                        approval_verifier=auth.verify)
    strat = Strategy("iv", instrument="g")
    cap = cs3.candidate(strategy=strat, targets=[claim],
                        controls=[by_label(cs3, "placebo_iv")],
                        tests=[(by_label(cs3, "ate_x2"), CERTIFIED)])
    ev = cs3.evaluate(cap, cs3.strategy(strat))
    assert ev.may_loosen is True and ev.flipped_refuted == 1
    blocked = cs3.promote(cap, cs3.strategy(strat))
    assert blocked.approved is False and "approval required" in blocked.reason
    ok = cs3.promote(cap, cs3.strategy(strat),
                     approval=auth.sign(ev.change_digest))
    assert ok.approved is True


# ---------------------------------------------------------------------- DiD --
def did_panel(n=1600, seed=11, parallel=True, effect=1.0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        grp = 1 if rng.random() < 0.5 else 0
        period = rng.choice([0, 1, 2, 3])
        trend = 0.5 * period + ((1.1 - 0.5) * period if (not parallel and grp) else 0)
        y = (rng.gauss(0, 0.6) + trend + 1.0 * grp
             + effect * (grp and period >= 3) + rng.gauss(0, 0.3))
        rows.append({"grp": grp, "period": period, "y": y})
    return rows


def att_claim():
    return CausalClaim(kind="att", outcome="y", group="grp", time="period",
                       post_from=3, criterion={"min": 0.7, "max": 1.3},
                       assumptions=("no anticipation",))


def test_did_recovers_att_and_tests_parallel_trends():
    cs = CausalSession([att_claim()], table=did_panel())
    claim = cs.claim(att_claim().id)
    oracle = cs.strategy(Strategy("did"))
    assert oracle(claim.id) == CERTIFIED
    assert 0.7 <= oracle.estimate(claim.id).value <= 1.3

    bad = CausalSession([att_claim()], table=did_panel(parallel=False))
    oracle2 = bad.strategy(Strategy("did"))
    assert oracle2(claim.id) == DECLINED
    assert oracle2.reasons[claim.id] == "assumption_violated:parallel_trends"


# ---------------------------------------------------------------------- RDD --
def rdd_data(n=900, seed=13, jump=2.0, cov_jump=0.9):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r = rng.uniform(-1, 1)
        side = 1 if r >= 0 else 0
        rows.append({"r": r,
                     "cov": 0.3 + cov_jump * side + 0.2 * r,
                     "y": 0.5 + jump * side + 0.8 * r + rng.gauss(0, 0.3)})
    return rows


def rdd_claim():
    return CausalClaim(kind="rdd", outcome="y", running="r", cutoff=0.0,
                       criterion={"min": 1.8, "max": 2.2},
                       assumptions=("continuity of potential outcomes",))


def test_rdd_local_linear_and_covariate_continuity():
    cs = CausalSession([rdd_claim()], table=rdd_data())
    claim = cs.claim(rdd_claim().id)
    oracle = cs.strategy(Strategy("rdd", bandwidth=0.4))
    assert oracle(claim.id) == CERTIFIED
    assert 1.8 <= oracle.estimate(claim.id).value <= 2.2

    bad = cs.strategy(Strategy("rdd", bandwidth=0.4, covariates=("cov",)))
    assert bad(claim.id) == DECLINED
    assert bad.reasons[claim.id] == "assumption_violated:covariate_discontinuity"


# ----------------------------------------------------------- independence --
def test_conditioning_on_a_collider_is_refuted():
    cs = session()
    strat = Strategy("ci", alpha=0.05, mde=0.12)
    oracle = cs.strategy(strat)
    ab = by_label(cs, "a ⟂ b")
    cond = by_label(cs, "a ⟂ b | y")
    assert oracle(ab.id) == CERTIFIED
    assert oracle(cond.id) == REFUTED
    assert oracle.estimate(ab.id).detail["detectable_r"] <= 0.12

    cs.register([ab, cond])
    cap = cs.candidate(strategy=strat, targets=[ab],
                       twin=[(cond, REFUTED)],
                       tests=[(by_label(cs, "g ⟂ z"), CERTIFIED)])
    assert cs.promote(cap, cs.strategy(strat)).approved is True


def test_underpowered_independence_is_declined_not_certified():
    tiny = scm(n=40, seed=9)
    cs = CausalSession(claims(), table=tiny)
    claim = by_label(cs, "a ⟂ b")
    oracle = cs.strategy(Strategy("ci", alpha=0.05, mde=0.05))
    assert oracle(claim.id) == DECLINED
    assert oracle.reasons[claim.id] == "underpowered"


# ------------------------------------------------------- discovery over claims --
def test_capability_moves_the_frontier_and_records_assumptions():
    cs = session()
    claim = by_label(cs, "ate_x1")
    strat = Strategy("backdoor", adjust=("z",))
    cap = cs.candidate(strategy=strat, targets=[claim],
                       controls=[by_label(cs, "placebo_treatment"),
                                 by_label(cs, "placebo_outcome")],
                       tests=[(CausalClaim(
                           kind="ate", outcome="y2", treatment="x1",
                           arms=(do("x1", 1), do("x1", 0)),
                           criterion={"min": 0.4, "max": 0.6},
                           assumptions=ASSUME_OBS, label="replication_y2"),
                           CERTIFIED)])
    adm = cs.promote(cap, cs.strategy(strat))
    assert adm.approved is True
    assert cs.corpus[claim.id] == CERTIFIED
    assert "no unmeasured confounding for the declared adjustment set" \
        in cs.assumptions()
    assert cs.ledger.has(adm.candidate_id)


def test_a_strategy_that_manufactures_refutations_is_contained():
    cs = session()
    claim = by_label(cs, "ate_x2")
    bad = Strategy("backdoor", adjust=())        # ignores the unmeasured u
    cap = cs.candidate(strategy=bad, targets=[claim],
                       controls=[by_label(cs, "placebo_treatment")])
    adm = cs.promote(cap, cs.strategy(bad))
    assert adm.approved is False and "no_new_refuted" in adm.reason
    det = next(d for d in cs.tick() if d.metric == "refuted_growth")
    assert det.alarm and det.severity == "critical"
    assert len(cs.auto_act(cs.tick())) == 1 and cs.containment is not None


def test_placebo_twins_invert_the_expectation_and_still_gate():
    cs = session()
    claim = by_label(cs, "ate_x1")
    strat = Strategy("backdoor", adjust=("z",))
    # a placebo claim the oracle REFUTES must fail the candidate
    broken_placebo = CausalClaim(kind="ate", outcome="y", treatment="x1",
                                 arms=(do("x1", 1), do("x1", 0)),
                                 criterion={"min": -0.05, "max": 0.05},
                                 assumptions=("declared as a placebo",),
                                 label="bad_placebo")
    cap = cs.candidate(strategy=strat, targets=[claim],
                       controls=[broken_placebo])
    adm = cs.promote(cap, cs.strategy(strat))
    assert adm.approved is False and "twin_holds" in adm.reason


def test_markov_class_reports_what_is_not_identified():
    cs = session()
    strat = Strategy("ci", alpha=0.05, mde=0.12)
    ab = by_label(cs, "a ⟂ b")
    cond = by_label(cs, "a ⟂ b | y")
    cap = cs.candidate(strategy=strat, targets=[ab], twin=[(cond, REFUTED)],
                       tests=[(by_label(cs, "g ⟂ z"), CERTIFIED)])
    cs.promote(cap, cs.strategy(strat))
    mc = cs.markov_class(Strategy("ci", alpha=0.05, mde=0.12))
    assert ab.id in mc["separated"]
    assert cond.id in mc["dependent"]
    assert {"at": "y", "left": "a", "right": "b"} in mc["colliders"]
    assert any("orientation" in u for u in mc["unidentified"])


# -------------------------------------------------------------------- tier 1 --
def hazard_series(drop_at=20, drop=0.20, noise=0.02, seed=3, ramp=0.0,
                  n_before=20, n_after=10):
    rng = random.Random(seed)
    points = []
    for i in range(n_before + n_after):
        base = 0.30 + ramp * i
        val = base if i < drop_at else base - drop
        points.append((i, max(0.0, min(1.0, val + rng.gauss(0, noise)))))
    return Series("hazard_release_rate", tuple(points))


def its_claims():
    return [
        CausalClaim(kind="intervention_effect", outcome="hazard_release_rate",
                    event_index=20, criterion={"max": -0.10},
                    assumptions=("no simultaneous shock",), label="containment"),
        CausalClaim(kind="intervention_effect", outcome="hazard_release_rate",
                    event_index=20, criterion={"max": -0.12},
                    assumptions=("no simultaneous shock",), label="severity"),
        CausalClaim(kind="intervention_effect", outcome="hazard_release_rate",
                    event_index=8, criterion={"min": -0.10, "max": 0.10},
                    assumptions=("placebo date",), label="placebo_date"),
    ]


def test_event_study_finds_the_injection_effect_and_its_placebo():
    out = event_study(hazard_series(), 20, pre=6, post=6,
                      criterion={"max": -0.10})
    assert out["verdict"] == CERTIFIED
    assert out["value"] <= -0.15
    assert out["detail"]["placebo_effect"] is None or \
        abs(out["detail"]["placebo_effect"]) < 0.10
    assert out["detail"]["pre_slope_ci"][0] <= 0 <= out["detail"]["pre_slope_ci"][1]


def test_event_study_refuses_a_series_with_a_pre_trend():
    out = event_study(hazard_series(ramp=0.02), 20, pre=6, post=6)
    assert out["verdict"] == DECLINED
    assert out["reason"] == "assumption_violated:pre_trend"


def test_its_capability_promotes_with_a_placebo_date_twin():
    cs = CausalSession(its_claims(), series=hazard_series())
    target = by_label(cs, "containment")
    strat = Strategy("its", pre=6, post=6, placebo_offset=6)
    cap = cs.candidate(strategy=strat, targets=[target],
                       controls=[by_label(cs, "placebo_date")],
                       tests=[(by_label(cs, "severity"), CERTIFIED)])
    adm = cs.promote(cap, cs.strategy(strat))
    assert adm.approved is True
    assert cs.corpus[target.id] == CERTIFIED


def test_series_from_journal_aggregates_to_batches():
    from mjolnir_ml import Journal
    j = Journal()
    for b in range(6):
        for _ in range(10):
            hazard = (b % 2 == 0)
            j.append({"kind": "decision", "batch": b,
                      "disposition": {"verdict": "release",
                                      "certificate_id": f"{b}-{_}"},
                      "hazard_codes": ["h"] if hazard else []})
    raw = T_series(j, per="decision")
    batched = T_series(j, per="batch")
    assert len(raw.points) == 60 and len(batched.points) == 6
    assert batched.metric == "hazard_release_rate"
    assert batched.at(0) == 1.0 and batched.at(1) == 0.0
    with pytest.raises(Exception):
        T_series(j, per="nonsense")


def test_insufficient_window_and_vacuous_claims_are_declined():
    claim = CausalClaim(kind="intervention_effect", outcome="m",
                        event_index=2, criterion={"min": -1, "max": 1},
                        assumptions=("a",))
    oracle = CausalOracle([claim], series=hazard_series()).strategy(
        Strategy("its", pre=6, post=6))
    assert oracle(claim.id) == DECLINED
    assert "positivity" in oracle.reasons[claim.id]

    vac = CausalClaim(kind="ate", outcome="y", treatment="x1",
                      arms=(do("x1", 1), do("x1", 0)),
                      criterion={"min": float("-inf"), "max": float("inf")},
                      assumptions=("a",))
    cs = session()
    cs.register([vac])
    oracle2 = cs.strategy(Strategy("backdoor", adjust=("z",)))
    assert oracle2(vac.id) == DECLINED
    assert oracle2.reasons[vac.id] == "vacuous_region"


def test_strategy_mismatch_is_declined():
    cs = session()
    claim = by_label(cs, "late_x2")
    oracle = cs.strategy(Strategy("did"))
    assert oracle(claim.id) == DECLINED
    assert oracle.reasons[claim.id] == "strategy_mismatch"


def test_determinism_of_bootstrap_intervals():
    cs = session()
    claim = by_label(cs, "late_x2")
    oa = cs.strategy(Strategy("iv", instrument="g", seed=5))
    oa(claim.id)
    a = oa.estimate(claim.id)
    ob = cs.strategy(Strategy("iv", instrument="g", seed=5))
    ob(claim.id)
    b = ob.estimate(claim.id)
    assert a.as_dict() == b.as_dict()
