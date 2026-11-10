"""Lab 4: the observability ceiling.

Completeness is measured against the institution's declared hazards, not
against labels. Removing an observable rule costs recall and precision; a
hidden hazard can never be caught by the policy and remains at the ceiling.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import ASK, RELEASE, HazardLedger, measure_completeness


class Disp:
    def __init__(self, verdict):
        self.verdict = verdict


def rows():
    out = []
    for i in range(80):
        out.append({"amount": 100.0 + i, "flag": 0})
    for i in range(20):
        out.append({"amount": 5000.0 + i, "flag": 0})
    for i in range(10):
        out.append({"amount": 900.0 + i, "flag": 1})
    return out


def ledger():
    return (HazardLedger()
            .add("large_transfer", lambda row: row["amount"] > 1000.0,
                 "policy:large-transfer", observable=True)
            .add("hidden_fraud", lambda row: row["flag"] == 1,
                 "audit:hidden-fraud", observable=False))


def full_dispositions(data):
    return [Disp(ASK) if row["amount"] > 1000.0 or row["flag"] == 1
            else Disp(RELEASE) for row in data]


def gap_dispositions(data):
    # The observable rule is removed; the hidden hazard cannot be seen.
    return [Disp(ASK) if row["flag"] == 1 else Disp(RELEASE)
            for row in data]


def main() -> None:
    args = parse_args(__doc__)
    data = rows()
    led = ledger()
    full = measure_completeness(full_dispositions(data), data, led)
    gap = measure_completeness(gap_dispositions(data), data, led)

    emit(
        "lab04_completeness",
        {
            "ledger_version": led.version,
            "ledger_digest": led.digest,
            "full": {
                "precision": full["precision"],
                "recall": full["recall"],
                "n_hazard": full["n_hazard"],
                "n_false_certificate": full["n_false_certificate"],
            },
            "gap": {
                "precision": gap["precision"],
                "recall": gap["recall"],
                "n_hazard": gap["n_hazard"],
                "n_false_certificate": gap["n_false_certificate"],
            },
            "observability_ceiling": full["ceiling"],
            "gap_costs_recall": gap["recall"] < full["recall"],
            "hidden_at_ceiling": gap["n_false_certificate"] > 0,
            "per_hazard": gap["per_hazard"],
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
