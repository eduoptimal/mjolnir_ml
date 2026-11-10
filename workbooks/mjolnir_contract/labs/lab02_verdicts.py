"""Lab 2: four verdicts and the review queue.

A two-verdict function has no state for refusal. A four-verdict policy names
the basis of every outcome, including a missing required fact.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (ASK, DENY, RELEASE, DecisionModel,
                        DecisionPolicyBase, dsl, rule)


@dsl
class FourVerdicts(DecisionPolicyBase):
    facts = ("score",)

    assert lambda score: score >= 0.0, "score_nonnegative"

    assert rule("deny_high",
                when=lambda p_positive: p_positive > 0.80,
                verdict=DENY)
    assert rule("ask_mid",
                when=lambda p_positive: p_positive > 0.20,
                verdict=ASK)
    assert rule("release_low",
                when=lambda p_positive: p_positive <= 0.20,
                verdict=RELEASE)


class ScoreClf:
    """p_positive is the stated score; the estimator only carries it."""

    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def two_verdict(p: float) -> str:
    return "approve" if p >= 0.5 else "decline"


def main() -> None:
    args = parse_args(__doc__)
    dm = DecisionModel(ScoreClf(), FourVerdicts, features=("score",),
                       regime="off")

    scored = [0.05, 0.15, 0.35, 0.55, 0.85, 0.95]
    rows = [{"score": p} for p in scored]
    rows.append({"other": 1.0})              # missing required fact

    two = {two_verdict(p) for p in scored}
    four = [dm.dispose(row) for row in rows]
    four_census = {}
    for d in four:
        four_census[d.verdict] = four_census.get(d.verdict, 0) + 1

    emit(
        "lab02_verdicts",
        {
            "two_verdict_states": sorted(two),
            "two_verdict_abstentions": 0,
            "two_verdict_asks": 0,
            "four_verdict_census": four_census,
            "four_verdict_bases": [list(d.basis) for d in four],
            "every_row_labelled": all(d.verdict and d.basis for d in four),
            "missing_fact_row": {
                "verdict": four[-1].verdict,
                "missing": four[-1].missing,
                "basis": list(four[-1].basis),
            },
            "review_queue": four_census.get(ASK, 0),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
