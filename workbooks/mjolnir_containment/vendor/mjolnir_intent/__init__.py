"""mjolnir_intent — intent-contract check.

A prompt is a specification.  Mjolnir checks whether the response (and diff)
that came back still honours what the prompt committed to, and whether the
prompt itself was determined enough to build without asking.

    from mjolnir_intent import audit
    report = audit("set the timeout to 30 seconds",
                   "Set timeout to 30 minutes")
    report["verdict"]      # 'BUILD' or 'ASK'
    report["gate"]         # blocking findings (units, negation)
    report["advisory"]     # report-only findings (conservation)

See README.md for the measured behaviour of each check.
"""
from mjolnir_intent.checks import GATE_CHECKS, audit, attribute, check_conservation, \
    check_negation, check_unit_adjacency, detect_ambiguity

__version__ = "0.1.0"
__all__ = [
    "audit", "check_unit_adjacency", "check_negation", "check_conservation",
    "detect_ambiguity", "attribute", "GATE_CHECKS", "__version__",
]
