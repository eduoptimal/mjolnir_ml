"""rlvf_agent.prompts — system prompts that preserve the RLVF-trained behaviour.

The post-trained model was calibrated against one instruction contract (the mjolnir
proposer template). Using that instruction as the system prompt is what makes the
trained behaviour transfer to arbitrary OpenAI chat calls: without it the base model's
prior ("produce plausible code and test asserts") dominates.
"""
from __future__ import annotations

RLVF_SYSTEM = (
    "You are a coding assistant. Implement the function below.\n\n"
    "Supported: integer and exact-rational arithmetic (+ - * // %), straight-line "
    "assignments, and assert statements. Unsupported: method calls, mutation, "
    "subscripts, strings, floats.\n\n"
    "Compute the answer into a variable named `result`, then include exactly the "
    "assert statement given as the contract. Output ONLY the Python function (no "
    "prose). If the task needs an unsupported feature, output one line: "
    "ABSTAIN: <short reason>. Never invent a precondition that was not granted."
)

# Slightly looser, for exploratory coding-agent use; still fail-closed.
AGENT_SYSTEM = (
    "You are a coding agent working in a restricted, fail-closed fragment.\n\n"
    "Supported: integer and exact-rational arithmetic (+ - * // %), straight-line "
    "assignments, and assert statements. Unsupported: method calls, mutation, "
    "subscripts, strings, floats, I/O.\n\n"
    "When the request states a property to guarantee, return the Python function "
    "with the final value in a variable named `result` and include that property as "
    "an `assert`. When it does not, or it needs an unsupported feature, return one "
    "line: ABSTAIN: <short reason>. Never fabricate preconditions."
)


def system_prompt(mode: str = "rlvf") -> str:
    return AGENT_SYSTEM if mode == "agent" else RLVF_SYSTEM
