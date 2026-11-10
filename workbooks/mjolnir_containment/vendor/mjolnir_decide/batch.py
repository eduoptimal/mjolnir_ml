"""mjolnir_decide.batch — multi-call states, aggregated with the joint-AND law.

The contract question stays ONE number (safe_to_run); what changes is the
STATE: a plan may carry several "Proposed tool call:" lines. Each call is
scored on its own (one call per state keeps every score in-regime — the
trained template is one call), and the batch answer is the joint-AND:

    noul_batch = min(noul_i)          (never prod: steps are not independent)
    any step out of regime -> 0.0     (the regime gate is joint too)

    The response shape is UNCHANGED for single-call states apart from the
    additive `certification` block, which travels on EVERY answer: the
    estimator's value (`noul_head`), the policy's narrowed value
    (`noul_certified`), the versioned policy digest and any fired rule
    vetoes (see mjolnir_decide.certify).  Certification can only lower the
    headline: rules narrow the estimate, never widen it.

    For plans the additive `mjolnir.batch` block reports the per-step values
    so the escalation queue can order STEPS, not a blob; each step carries
    its raw `noul_head` beside the certified `noul`.  Splitting is
    deterministic line work on the closed template — no model, no guessing;
    a state whose call lines cannot be split cleanly is answered as one
    state (the floor still fails it closed).
"""
from __future__ import annotations

import re

from mjolnir_common.batch import min_noul

from mjolnir_decide.certify import certification_block, certify

_CALL_RE = re.compile(r"^Proposed tool call:\s*\S.*$", re.MULTILINE)


def _exact(x):
    """exact 0/1 as int — the contract rounding rule, applied to the min."""
    return int(x) if x in (0.0, 1.0) else x


def _step(certified, head, ex, vetoes) -> dict:
    step = {"noul": certified, "noul_head": head,
            "in_regime": ex["in_regime"], "flow_score": ex.get("flow_score")}
    if vetoes:
        step["rule_vetoes"] = vetoes
    return step


def split_state(state: str) -> list:
    """One state per "Proposed tool call:" line; Task/Context travel with all.

    Single call (or none) -> [state] unchanged: the serve path stays
    byte-identical for the common case.
    """
    calls = _CALL_RE.findall(state or "")
    if len(calls) <= 1:
        return [state]
    head = (state or "").split("Proposed tool call:", 1)[0].rstrip("\n")
    tail = ""
    lines = (state or "").splitlines()
    last_call = max(i for i, ln in enumerate(lines)
                    if ln.startswith("Proposed tool call:"))
    rest = [ln for ln in lines[last_call + 1:] if ln.strip()]
    if rest:
        tail = "\n".join(rest)
    states = []
    for call in calls:
        s = f"{head}\n{call}"
        if tail:
            s += f"\n{tail}"
        states.append(s)
    return states


def answer_batch(model, parsed, build_response, model_provenance):
    """Score a (possibly multi-call) state; joint-AND over the steps, then
    narrow through safety-policy@1 (certification can only lower noul).

    Returns the same envelope build_response always produced; the additive
    block gains `batch` only when the state actually split, and always
    carries `certification` — both numbers, the policy, the fired vetoes.
    """
    states = split_state(parsed.state)
    if len(states) == 1:
        ex = model.explain(parsed.state)
        vetoes = certify(parsed.state)
        head = ex["noul"]
        certified = 0 if vetoes else head
        extra = dict(model_provenance())
        extra["in_regime"] = ex["in_regime"]
        extra["flow_score"] = ex.get("flow_score")
        extra["certification"] = certification_block(head, certified, vetoes)
        return build_response(parsed, certified,
                              model.input_tokens(parsed.state),
                              extra=extra), [ex]

    exs = [model.explain(s) for s in states]
    step_vetoes = [certify(s) for s in states]
    head_nouls = [e["noul"] for e in exs]
    cert_nouls = [0 if v else n for v, n in zip(step_vetoes, head_nouls)]
    regimes = [e["in_regime"] for e in exs]
    extra = dict(model_provenance())
    extra["in_regime"] = all(regimes)
    extra["flow_score"] = None            # no single density for a plan
    extra["batch"] = {
        "n_steps": len(states),
        "aggregate": "min(noul_i); prod would assume independence",
        "steps": [_step(c, h, e, v) for c, h, e, v
                  in zip(cert_nouls, head_nouls, exs, step_vetoes)],
    }
    fired = [{**v, "step": i} for i, vs in enumerate(step_vetoes) for v in vs]
    cert_hl = _exact(min_noul(cert_nouls, regimes))
    extra["certification"] = certification_block(
        _exact(min_noul(head_nouls, regimes)), cert_hl, fired)
    tokens = sum(model.input_tokens(s) for s in states)
    return build_response(parsed, cert_hl, tokens, extra=extra), exs
