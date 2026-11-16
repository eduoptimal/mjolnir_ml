"""Developer-only replay. Not an audit API."""
from __future__ import annotations


def replay_local(disposition, row, policy):
    from mjolnir_ml.policy import Policy
    if isinstance(policy, type):
        policy = policy.policy()
    if isinstance(policy, dict):
        policy = Policy(policy)
    verdict, basis, missing = policy.dispose_facts(dict(disposition.facts))
    return (verdict, tuple(basis), missing) == (
        disposition.verdict, tuple(disposition.basis), disposition.missing)
