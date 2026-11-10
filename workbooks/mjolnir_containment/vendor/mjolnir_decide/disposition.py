"""Project a decide answer into the disposition record.

This head never disposes: it produces an ESTIMATE. `verdict` below is derived
under a DECLARED policy threshold and says so; `ASK` covers "not released
under policy", "out of regime" and "rule-vetoed", with the stage distinguishing
them. No DENY is ever produced — refusing to auto-run is the caller's decision
to make.

The stage vocabulary (each stage names the gate that fired):
    estimated-in-regime    the estimate cleared the caller threshold
    estimated-out-of-regime the OOD floor failed the answer closed
    rule-veto              safety-policy@1 fired; the release is narrowed to
                           the certified noul, the raw estimate stays visible

`estimate.noul` always reports the RAW estimator value (the certification
block's `noul_head`), while `answers`/`legacy` carry the certified value —
the record says what was believed AND what was allowed, never one number
pretending to be both.
"""
from __future__ import annotations


def to_disposition(resp: dict, threshold: float = 0.8) -> dict:
    ans = (resp.get("answers") or {}).get("safe_to_run") or {}
    noul = ans.get("noul")                 # certified: answers is post-policy
    mj = resp.get("mjolnir") or {}
    in_regime = bool(mj.get("in_regime", True))
    cert = mj.get("certification")
    vetoed = bool((cert or {}).get("rule_vetoes"))
    released = isinstance(noul, (int, float)) and float(noul) >= threshold and in_regime
    stage = ("rule-veto" if vetoed else
             "estimated-in-regime" if in_regime else
             "estimated-out-of-regime")
    out = {
        "kind": "estimate",
        "verdict": "ALLOW" if released else "ASK",
        "stage": stage,
        "estimate": {"noul": (cert or {}).get("noul_head", noul)},
        "policy": {"name": f"noul>={threshold}", "threshold": threshold,
                   "owner": "caller-declared: this head never disposes"},
        "floor": mj.get("floor"),
        "blind_spot": mj.get("blind_spot"),
        "provenance": mj.get("checkpoint"),
        "legacy": {"noul": noul},
    }
    if cert:
        out["certification"] = cert
    return out
