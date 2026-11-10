"""Lab 12: assemble the boundary ledger from the twelve capabilities."""

from __future__ import annotations

from _harness import emit, parse_args

CAPABILITIES = (
    ("Reactive containment", "authority", "on-fragment", "CERTIFIED", "guard trace", "Rice"),
    ("Belief-space planning", "belief updater", "declared graph", "CERTIFIED", "belief trace", "Rice"),
    ("Horizon soundness", "horizon guard", "within depth", "CERTIFIED", "lookahead certificate", "Rice"),
    ("Two-key release", "authority + reader", "both route", "CERTIFIED", "stage trace", "Rice"),
    ("Calibrated head", "density head", "above floor", "HONEST_ABSTAIN", "density record", "Chow"),
    ("Claim verification", "claim ledger", "re-derives", "CERTIFIED", "checker output", "Rice"),
    ("Decline training", "RLVF gates", "registered gates", "CERTIFIED", "Pareto record", "Rice"),
    ("Commitment fidelity", "intent checker", "decidable segments", "HONEST_ABSTAIN", "gate output", "Rice"),
    ("Reachability", "certifiability checker", "C1-C6", "CERTIFIED", "condition record", "Rice"),
    ("Characterization", "matched-weight control", "pinned ladder", "CERTIFIED", "matched-weight record", "none"),
    ("Cost", "contour accountant", "regime-qualified cells", "CERTIFIED", "contour record", "Chow"),
    ("Boundary itself", "ledger assembler", "twelve capabilities", "CERTIFIED", "this ledger", "Godel"),
)

EXTRA = (
    "Reader deployment freeze",
    "change gate",
    "signed change set",
    "CERTIFIED",
    "approval record",
    "Rice",
)


def row(values: tuple[str, ...]) -> dict[str, str]:
    capability, authority, fragment, verdict, certificate, wall = values
    return {
        "capability": capability,
        "authority": authority,
        "fragment": fragment,
        "verdict": verdict,
        "certificate": certificate,
        "wall": wall,
        "claim": "[A]" if wall == "Godel" else "[P]",
        "refusal": "no aggregate safety verdict",
    }


def main() -> None:
    args = parse_args(__doc__)
    rows = [row(values) for values in CAPABILITIES] + [row(EXTRA)]
    emit(
        "lab12_ledger",
        {
            "rows": rows,
            "every_row_disposed": all(item["verdict"] in {"CERTIFIED", "HONEST_ABSTAIN", "REFUTED", "DECLARED"} for item in rows),
            "aggregate_safety_verdict": None,
            "resolution_ceiling": "0/2,148",
            "walls": {"Chow": 2, "Rice": 9, "Godel": 1},
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
