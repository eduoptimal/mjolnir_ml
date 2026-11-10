"""Lab 7: the injected hazard.

The journal preserves sequence; status projects it; containment records the
response. One released hazard raises the hazard detector and the automatic
action contains the session.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (DecisionModel, DecisionPolicyBase, HazardLedger,
                        Journal, LiveSession, RELEASE, dsl, rule)


@dsl
class LivePolicy(DecisionPolicyBase):
    facts = ("score",)

    assert rule("release_low",
                when=lambda p_positive: p_positive < 0.90,
                verdict=RELEASE)


class ScoreClf:
    classes_ = [0, 1]

    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in X]

    def predict_proba(self, X):
        return [[1.0 - row[0], row[0]] for row in X]


def batches():
    out = []
    for b in range(6):
        rows = [{"score": 0.10 + 0.01 * i, "flag": 0} for i in range(12)]
        if b == 3:
            rows[5] = {"score": 0.50, "flag": 1}
        out.append(rows)
    return out


def main() -> None:
    args = parse_args(__doc__)
    dm = DecisionModel(ScoreClf(), LivePolicy, features=("score",),
                       regime="off")
    ledger = (HazardLedger()
              .add("unsafe_release", lambda row: row.get("flag") == 1,
                   "audit:unsafe-release", observable=True))
    journal = Journal(None)
    live = LiveSession(dm, journal=journal, ledger=ledger)

    first_alarm_batch = None
    containment_batch = None
    for b, rows in enumerate(batches()):
        for row in rows:
            live.decide(row, batch=b)
        detections = live.tick()
        alarmed = [d for d in detections if d.alarm]
        if alarmed and first_alarm_batch is None:
            first_alarm_batch = b
            actions = live.auto_act(detections, batch=b)
            if actions:
                containment_batch = b

    status = live.status()
    emit(
        "lab07_live",
        {
            "first_alarm_batch": first_alarm_batch,
            "containment_batch": containment_batch,
            "detector": "hazard_release",
            "contained": live.containment is not None,
            "journal_verifies": journal.verify(),
            "journal_entries": len(journal),
            "journal_head": journal.entries()[-1]["hash"],
            "status": {
                "verdicts": status["verdicts"],
                "claim_rate": status["claim_rate"],
                "review_queue": status.get("review_queue"),
                "contained": status.get("contained"),
            },
            "alarm_precedes_containment": (
                first_alarm_batch is not None
                and containment_batch is not None
                and first_alarm_batch == containment_batch),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
