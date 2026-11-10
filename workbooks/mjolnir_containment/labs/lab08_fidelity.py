"""Lab 8: the unit swap that the toolchain misses and the gate blocks."""

from __future__ import annotations

import sys

from _harness import emit, parse_args

PROMPT = "wait 30 seconds before retrying"
DIFF = "wait 30 minutes before retrying"


def toolchain() -> dict[str, bool]:
    return {"compiler": True, "type_checker": True, "linter": True, "tests": True}


def fidelity_gate(prompt: str, diff: str) -> tuple[bool, str]:
    prompt_unit = "seconds" if "seconds" in prompt else "unknown"
    diff_unit = "minutes" if "minutes" in diff else "unknown"
    if prompt_unit != diff_unit:
        return False, f"unit changed: prompt says {prompt_unit}, got {diff_unit}"
    return True, "commitment preserved"


def main() -> None:
    args = parse_args(__doc__)
    checks = toolchain()
    passed, reason = fidelity_gate(PROMPT, DIFF)
    emit(
        "lab08_fidelity",
        {
            "toolchain": checks,
            "toolchain_passes": all(checks.values()),
            "gate_passes": passed,
            "gate_reason": reason,
            "exit_code": 1 if not passed else 0,
            "undecidable_commitment": {
                "segment": "retry semantics under clock skew",
                "class": "UNVERIFIABLE",
                "decider": "a registered clock-skew model and a byte-exact transcript",
            },
        },
        args.replay,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
