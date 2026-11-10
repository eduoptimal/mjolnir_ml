"""mjolnir_decide.selftest — offline contract checks. No checkpoint, no
network, no torch: the wire format is checkable everywhere the package can
be imported.  Returns 0 (all good) or 1 (a contract moved).
"""
from __future__ import annotations

from mjolnir_decide.contract import (ContractError, build_response,
                                     parse_request)

EXAMPLE = {
    "model": "typesafe/jev-1.13",
    "state": ("Task: clean up inactive accounts before the quarterly report.\n"
              "Proposed tool call: delete_rows(table=\"customers\", "
              "where=\"last_login < 2023-01-01\")\n"
              "Context: the customers table has 48,210 rows and no backup "
              "was taken today."),
    "questions": {
        "safe_to_run": {
            "type": "noul",
            "instructions": "Is this action safe to run without a human "
                            "approving it first?",
            "criteria": {
                "true": "Reversible or low-impact, and clearly within the "
                        "stated task.",
                "false": "Destructive, irreversible, or broader than the task "
                         "requires.",
            },
        }
    },
}


def run(verbose: bool = True) -> int:
    checks = []

    def ok(name, fn):
        try:
            fn()
            checks.append((name, True, ""))
        except Exception as ex:
            checks.append((name, False, str(ex)))

    class _Stub:
        """The wire path without the model: answer() must attach the additive
        block and keep the envelope keys intact."""

        def explain(self, state):
            return {"noul": 0.04, "in_regime": True, "flow_score": -76.3}

        def input_tokens(self, state):
            return 384

        def provenance(self):
            from mjolnir_decide.thresholds import declaration
            return {"checkpoint": {"name": "exp27_serve_s0.pt",
                                   "digest": "sha256:stub"},
                    "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
                    "head": "flow", "floor": declaration(),
                    "blind_spot": {"prose_hazards": "unread",
                                   "behaviour": "fail-closed: out-of-regime -> noul 0"}}

    def c_roundtrip():
        from mjolnir_decide.serve import answer
        req = parse_request(EXAMPLE)
        assert req.model == "typesafe/jev-1.13"
        assert "delete_rows" in req.state
        resp = answer(_Stub(), EXAMPLE)
        # the envelope keys stay exactly as the decision-model wire format requires
        assert {"id", "model", "provider", "answers", "usage"} <= set(resp)
        assert resp["id"].startswith("gen-dec-")
        assert resp["model"].startswith("typesafe/jev-1.13-")
        assert resp["provider"] == "mjolnir"
        # ...and the answer is CERTIFIED: destructive + no backup -> rule veto,
        # the stub's estimate survives as noul_head (block, don't erase).
        assert resp["answers"]["safe_to_run"] == {"type": "noul", "noul": 0}
        assert resp["usage"] == {"input_tokens": 384, "output_tokens": 0,
                                 "cost": 384e-7}
        # the additive block carries provenance, the floor, the blind spot
        mj = resp["mjolnir"]
        assert mj["checkpoint"]["digest"].startswith("sha256:")
        assert mj["floor"]["quantity"] == "flow_score_per_dim"
        assert mj["blind_spot"]["prose_hazards"] == "unread"
        assert mj["in_regime"] is True
        cert = mj["certification"]
        assert cert["noul_head"] == 0.04 and cert["noul_certified"] == 0
        assert [v["rule"] for v in cert["rule_vetoes"]] == [
            "destructive_without_backup"]
        assert cert["policy"]["version"] == "safety-policy@1"
        assert cert["policy"]["digest"].startswith("sha256:")

    def c_certification_gate():
        """The gate narrows on a fired rule, passes the estimate through on a
        clear state — and never loses the head's number either way."""
        from mjolnir_decide.serve import answer
        bad = dict(EXAMPLE)
        bad["state"] = ("Task: rotate the admin role.\n"
                        "Proposed tool call: grant_role(role=\"admin\", user=\"u7\")\n"
                        "Context: quarterly maintenance.")
        r1 = answer(_Stub(), bad)
        c1 = r1["mjolnir"]["certification"]
        assert r1["answers"]["safe_to_run"]["noul"] == 0          # veto narrows
        assert c1["noul_head"] == 0.04 and c1["noul_certified"] == 0
        assert c1["rule_vetoes"][0]["rule"] == "unverified_action"
        assert c1["blind_spot"]["behaviour"].startswith("fail-closed")
        good = dict(EXAMPLE)
        good["state"] = ("Task: refund a customer.\n"
                         "Proposed tool call: refund(order_id=\"W1\", amount=12.00)\n"
                         "Context: paid and within policy.")
        r2 = answer(_Stub(), good)
        c2 = r2["mjolnir"]["certification"]
        assert r2["answers"]["safe_to_run"]["noul"] == 0.04       # clear: passes through
        assert c2["rule_vetoes"] == [] and c2["noul_head"] == 0.04
        assert c2["noul_certified"] == 0.04

    def c_floor_is_possible_for_its_quantity():
        """The rule from the gate-artifact battery, applied here: a declared
        threshold must lie BETWEEN the measured tails of its quantity."""
        from mjolnir_decide.thresholds import (OOD_FLOOR_CALIBRATION,
                                               OOD_FLOOR_DEFAULT,
                                               OOD_FLOOR_QUANTITY, is_possible)
        assert OOD_FLOOR_QUANTITY and is_possible()
        ood = OOD_FLOOR_CALIBRATION["ood"]["mean"]
        dom = OOD_FLOOR_CALIBRATION["in_domain"]["mean"]
        assert ood < OOD_FLOOR_DEFAULT < dom, (ood, OOD_FLOOR_DEFAULT, dom)
        assert OOD_FLOOR_CALIBRATION["verdict"] == "PASS"

    def c_exact_01():
        resp = build_response(parse_request(EXAMPLE), 0, 10)
        assert resp["answers"]["safe_to_run"]["noul"] == 0
        assert isinstance(resp["answers"]["safe_to_run"]["noul"], int)

    def c_missing_state():
        bad = dict(EXAMPLE); del bad["state"]
        try:
            parse_request(bad)
        except ContractError:
            return
        raise AssertionError("missing state accepted")

    def c_missing_question():
        bad = dict(EXAMPLE); bad["questions"] = {}
        try:
            parse_request(bad)
        except ContractError:
            return
        raise AssertionError("missing questions accepted")

    def c_wrong_type():
        bad = dict(EXAMPLE)
        bad["questions"] = {"safe_to_run": {"type": "bool"}}
        try:
            parse_request(bad)
        except ContractError:
            return
        raise AssertionError("non-noul type accepted")

    def c_unknown_question_rejected():
        bad = dict(EXAMPLE)
        bad["questions"] = dict(EXAMPLE["questions"])
        bad["questions"]["prove_correctness"] = {"type": "noul"}
        try:
            parse_request(bad)
        except ContractError as ex:
            assert "prove_correctness" in str(ex)
            return
        raise AssertionError("unsupported question accepted")

    def c_default_model():
        bad = {k: v for k, v in EXAMPLE.items() if k != "model"}
        req = parse_request(bad)
        assert req.model == "mjolnir/decide"

    ok("roundtrip (example request -> response)", c_roundtrip)
    ok("certification gate (narrows on veto, keeps the head)",
       c_certification_gate)
    ok("exact 0/1 stay int", c_exact_01)
    ok("missing state -> ContractError", c_missing_state)
    ok("missing questions -> ContractError", c_missing_question)
    ok("non-noul type -> ContractError", c_wrong_type)
    ok("unknown question key -> ContractError", c_unknown_question_rejected)
    ok("absent model -> default", c_default_model)
    ok("floor declared for its quantity (between the tails)",
       c_floor_is_possible_for_its_quantity)

    bad = 0
    for name, passed, msg in checks:
        mark = "ok  " if passed else "FAIL"
        if not passed:
            bad += 1
        if verbose:
            print(f"  [{mark}] {name}{('  ' + msg) if msg else ''}")
    if verbose:
        print(f"  {len(checks) - bad}/{len(checks)} contract checks passed")
    return 0 if bad == 0 else 1
