"""mjolnir_decide.contract — the wire format, both directions.

REQUEST (what OpenRouter forwards from a decision-model caller):
    {"model": "<provider>/<model>",
     "state": "<task + tool call + context>",
     "questions": {"safe_to_run": {"type": "noul", ...}}}

RESPONSE (what OpenRouter expects back):
    {"id": "gen-dec-<ts>-<hex>",
     "model": "<request model>-<yyyymmdd>",
     "provider": "mjolnir",
     "answers": {"safe_to_run": {"type": "noul", "noul": p}},
     "usage": {"input_tokens": n, "output_tokens": 0,
               "cost": n * 1e-7},
     "mjolnir": {           // additive mandate block, includes
       ...,                 //   "certification": the safety-policy gate —
                            //   noul_head (the estimate), noul_certified
                            //   (= answers.noul, narrowed not widened),
                            //   the policy digest, any fired rule vetoes and
                            //   the rule layer's declared blind spot}}

CONTRACT RULES (all fail-closed):
    - `state` and `questions.safe_to_run` are required; the question's
      `type` must be "noul".
    - `instructions` / `criteria` are accepted and IGNORED: the trained
      template is the mode (the battery measured free-form criteria as the
      failure regime).  Semantics are fixed at training, not per request.
    - No other question key is answerable here — a request carrying one is
      rejected, never partially answered.
    - `answers.safe_to_run.noul` is the CERTIFIED value: safety-policy@1
      (mjolnir_decide.certify) may narrow the estimator's answer to 0 on a
      fired rule, never raise it; the raw estimate stays in
      `mjolnir.certification.noul_head`.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from datetime import date

PROVIDER = "mjolnir"
DEFAULT_MODEL = "mjolnir/decide"
TOKEN_COST = 1e-7          # cost per input token, contract-visible pricing
ANSWERED = "safe_to_run"   # the one question this model was trained for


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
            f"unsupported question(s) {extra}: this model answers "
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


def build_response(req: DecisionRequest, noul: float | int,
                   input_tokens: int, extra: dict | None = None) -> dict:
    """The OpenRouter envelope, plus an additive `mjolnir` block.

    The envelope keys stay exactly as the decision-model wire format requires;
    provenance, the floor declaration and the blind spot travel beside them
    (precedent: mjolnir_guard's `decision` block).
    """
    out = {
        "id": f"gen-dec-{int(time.time())}-{secrets.token_hex(9)}",
        "model": f"{req.model}-{date.today().strftime('%Y%m%d')}",
        "provider": PROVIDER,
        "answers": {ANSWERED: {"type": "noul", "noul": noul}},
        "usage": {"input_tokens": input_tokens, "output_tokens": 0,
                  "cost": round(input_tokens * TOKEN_COST, 10)},
    }
    if extra:
        out["mjolnir"] = extra
    return out
