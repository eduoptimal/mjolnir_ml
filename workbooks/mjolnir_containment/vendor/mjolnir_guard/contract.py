"""mjolnir_guard.contract — the wire format, both directions.

Identical envelope to mjolnir_decide so both are submittable to OpenRouter;
the `model` field is the only thing a caller needs to tell them apart.
Differences that belong to THIS package:

    - `noul` is binary (0 or 1, as int).  Guard disposes; it does not
      estimate.
    - the response carries a `decision` block (stage, authority verdict and
      reasons, reader verdict) — the audit trail.  Strip it at the gateway
      if a strict OpenRouter passthrough requires the minimal envelope.

Fail-closed rules are the same shape as decide: state and
questions.safe_to_run (type noul) required; any other question key is
rejected, never partially answered.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import date

PROVIDER = "mjolnir"
DEFAULT_MODEL = "mjolnir/guard"
TOKEN_COST = 1e-7
ANSWERED = "safe_to_run"


class ContractError(ValueError):
    """The request cannot be answered under the contract (fail-closed)."""


@dataclass(frozen=True)
class DecisionRequest:
    model: str
    state: str
    questions: dict = field(default_factory=dict)


def parse_request(req: dict) -> DecisionRequest:
    if not isinstance(req, dict):
        raise ContractError("request must be a JSON object")
    state = req.get("state")
    if not isinstance(state, str) or not state.strip():
        raise ContractError("'state' is required and must be a non-empty string")
    questions = req.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ContractError("'questions' is required and must be a non-empty object")
    extra = [k for k in questions if k != ANSWERED]
    if extra:
        raise ContractError(
            f"unsupported question(s) {extra}: guard answers "
            f"'{ANSWERED}' only")
    q = questions.get(ANSWERED)
    if not isinstance(q, dict):
        raise ContractError(f"questions.{ANSWERED} is required")
    if q.get("type") != "noul":
        raise ContractError(
            f"questions.{ANSWERED}.type must be 'noul', got {q.get('type')!r}")
    model = req.get("model") or DEFAULT_MODEL
    if not isinstance(model, str):
        raise ContractError("'model' must be a string when present")
    return DecisionRequest(model=model, state=state, questions=questions)


def build_response(req: DecisionRequest, noul: int, input_tokens: int,
                   decision: dict | None = None) -> dict:
    if noul not in (0, 1):
        raise ContractError(f"guard noul must be 0 or 1, got {noul!r}")
    resp = {
        "id": f"gen-guard-{int(time.time())}-{secrets.token_hex(9)}",
        "model": f"{req.model}-{date.today().strftime('%Y%m%d')}",
        "provider": PROVIDER,
        "answers": {ANSWERED: {"type": "noul", "noul": int(noul)}},
        "usage": {"input_tokens": input_tokens, "output_tokens": 0,
                  "cost": round(input_tokens * TOKEN_COST, 10)},
    }
    if decision is not None:
        resp["decision"] = decision
    return resp
