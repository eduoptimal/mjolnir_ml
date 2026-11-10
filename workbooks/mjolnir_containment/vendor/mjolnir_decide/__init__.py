"""mjolnir_decide — calibrated decision proposals under the noul contract.

Answers one question about an action -- "is this safe to run without a human
approving it first?" -- as a calibrated probability (noul, 0-1), in the
OpenRouter decision-model wire format (JSON request in, JSON response out).

    from mjolnir_decide import Model, parse_request, build_response
    m = Model()                                  # generative (flow) head
    resp = build_response(parse_request(req), m.noul(req["state"]))

    # or over HTTP (OpenRouter-submittable shape):
    python -m mjolnir_decide serve --http 8080

HEADS (runtime parameter --head / MJOLNIR_DECIDE_HEAD):
    flow   generative (conditional NSPNF density ratio)  DEFAULT
    full   discriminative, text + exact features
    text   discriminative, text only
    rules  discriminative, exact features only
    gauss  generative 0-param Gaussian density ratio

STANDING RULE (framework mandate): this package PROPOSES.  A number here may
drive routing or rank an escalation queue; it never substitutes for a
verdict.  The critical decider — binary, with a named stage — is
mjolnir_guard; a decide noul never overrides a guard 0.
"""
from mjolnir_decide.contract import (DEFAULT_MODEL, PROVIDER, ContractError,
                                     build_response, parse_request)
from mjolnir_decide.model import HEADS, DEFAULT_HEAD, Model, parse_feats, \
    resolve_checkpoint

__version__ = "0.1.0"
__all__ = [
    "Model", "HEADS", "DEFAULT_HEAD", "parse_feats", "resolve_checkpoint",
    "parse_request", "build_response", "ContractError",
    "PROVIDER", "DEFAULT_MODEL", "__version__",
]
