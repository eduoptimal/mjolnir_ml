"""Lab 2: the degenerate controls.

A scalar compresses the pair. ALWAYS and NEVER are the two controls that show
what the compression erases: NEVER reports vacuous precision, never 1.0.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (ASK, DENY, RELEASE, degenerate_controls, omega_pair)


def main() -> None:
    args = parse_args(__doc__)
    verdicts = [RELEASE, RELEASE, DENY, ASK, RELEASE, DENY]
    safe = [True, False, False, True, True, False]
    decidable = [True] * len(verdicts)

    pair = omega_pair(verdicts, safe, decidable)
    controls = degenerate_controls(safe, decidable)

    def scalar(claim_correct, claims, n):
        # A naive trust score: 1.0 when it never claims (the vacuous case).
        return 1.0 if claims == 0 else claim_correct / claims

    scalar_scores = {
        "policy": scalar(pair.n_cert, pair.n_claim, pair.n),
        "ALWAYS": scalar(controls["ALWAYS"]["n_cert"],
                         controls["ALWAYS"]["n_claim"],
                         controls["ALWAYS"]["n"]),
        "NEVER": scalar(controls["NEVER"]["n_cert"],
                        controls["NEVER"]["n_claim"],
                        controls["NEVER"]["n"]),
    }
    safest_by_scalar = max(scalar_scores, key=scalar_scores.get)

    emit(
        "lab02_pair",
        {
            "pair": pair.as_dict(),
            "controls": controls,
            "scalar_scores": scalar_scores,
            "scalar_ranks_never_as_safe": safest_by_scalar == "NEVER",
            "scalar_safest": safest_by_scalar,
            "denominators": {
                "n": pair.n,
                "n_claim": pair.n_claim,
                "n_decidable": pair.n_A,
                "abstention_rate": round(
                    1.0 - pair.n_claim / pair.n, 6) if pair.n else None,
            },
            "never_precision_is_vacuous": controls["NEVER"]["prec"] is None,
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
