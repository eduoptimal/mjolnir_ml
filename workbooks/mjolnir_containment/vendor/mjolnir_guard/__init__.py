"""mjolnir_guard — the critical decider.  May this action run without a
human, and if not, named why.

ONE FUNCTION.  Same OpenRouter wire envelope as mjolnir_decide, but the
semantics are binary and it is the only one of the two that disposes:

    authority  (mjolnir_router actions policy — exact, rules-as-data, free)
      ESCALATE / ABSTAIN  -> noul 0 + the rule or missing fact
      ROUTE               -> veto lane below (the reader may only TIGHTEN)
    veto-reader (trained qwen, temp 0, the battery's exact templates)
      ROUTE               -> noul 1, stage "released"
      anything else       -> noul 0, stage "vetoed-by-reader"

Reading the answer:
    noul 1  -> two independent keys agreed; the decision block names them
    noul 0  -> decision.stage says who refused: blocked-by-authority:<rule>,
               abstained-by-authority:<missing fact or unknown verb>,
               vetoed-by-reader:<verdict line>, reader-error (fail-closed)

Limits (its own, not mjolnir_decide's): the authority's vocabulary is
closed — an action it cannot classify ABSTAINS to a human, by design; the
reader is coupled to one checkpoint/template regime and costs ~0.5 s when
it fires; states with no parseable `Proposed tool call:` line are NOT
decidable here and fail closed.

mjolnir_decide proposes numbers and never disposes.  A decide noul never
overrides a guard 0; a guard 1 never depends on decide existing.
"""
from mjolnir_guard.contract import (DEFAULT_MODEL, PROVIDER, ContractError,
                                    build_response, parse_request)
from mjolnir_guard.dialect import DIALECT, extract_action, to_tau
from mjolnir_guard.guard import STAGES, Guard

__version__ = "0.1.0"
__all__ = [
    "Guard", "STAGES", "DIALECT", "extract_action", "to_tau",
    "parse_request", "build_response", "ContractError",
    "PROVIDER", "DEFAULT_MODEL", "__version__",
]
