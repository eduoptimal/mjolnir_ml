"""Lab 5: compile, digest, break.

The compiler hashes the canonical IR, not the source bytes. A semantic edit
changes the digest; a recompile does not. Unknown facts and duplicate codes
fail at compile time. A rule that cannot fire is reported by `feasibility`.
"""

from __future__ import annotations

from _harness import emit, parse_args

from mjolnir_ml import (ASK, DENY, RELEASE, DecisionModel, DecisionPolicyBase,
                        DslError, dsl, feasibility, rule)


@dsl
class FiveRulePolicy(DecisionPolicyBase):
    facts = ("amount", "limit")

    assert lambda amount: amount > 0, "amount_positive"

    assert rule("deny_over",
                when=lambda amount, limit: amount > limit,
                verdict=DENY)
    assert rule("ask_mid",
                when=lambda amount, limit: amount > limit * 0.5,
                verdict=ASK)
    assert rule("release_small",
                when=lambda amount, limit: amount <= limit * 0.1,
                verdict=RELEASE)
    assert rule("release_exact",
                when=lambda amount, limit: amount == limit * 0.2,
                verdict=RELEASE)


@dsl
class FiveRuleEdited(DecisionPolicyBase):
    facts = ("amount", "limit")

    assert lambda amount: amount > 0, "amount_positive"

    assert rule("deny_over",
                when=lambda amount, limit: amount > limit * 1.1,
                verdict=DENY)
    assert rule("ask_mid",
                when=lambda amount, limit: amount > limit * 0.5,
                verdict=ASK)
    assert rule("release_small",
                when=lambda amount, limit: amount <= limit * 0.1,
                verdict=RELEASE)
    assert rule("release_exact",
                when=lambda amount, limit: amount == limit * 0.2,
                verdict=RELEASE)


class UnknownFactPolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("r", when=lambda danger: danger > 0, verdict=DENY)


class DuplicateCodePolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("same", when=lambda x: x > 0, verdict=DENY)
    assert rule("same", when=lambda x: x < 0, verdict=ASK)


@dsl
class DeadRulePolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("never_fires", when=lambda x: x < 0, verdict=DENY)


class ZeroReg:
    def get_params(self, deep=False):
        return {}

    def predict(self, X):
        return [0.0 for _ in X]


def compile_error(cls):
    try:
        dsl(cls)
    except DslError as ex:
        return str(ex)
    return None


def main() -> None:
    args = parse_args(__doc__)
    base = FiveRulePolicy.policy()
    again = dsl(FiveRulePolicy).policy()
    edited = FiveRuleEdited.policy()

    dm = DecisionModel(ZeroReg(), DeadRulePolicy, features=("x",),
                       regime="off")
    report = feasibility(dm, [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])

    emit(
        "lab05_compile",
        {
            "n_statements": len(base.statements),
            "digest": base.digest,
            "recompile_digest": again.digest,
            "digest_stable_on_recompile": again.digest == base.digest,
            "edited_digest": edited.digest,
            "digest_changes_on_semantic_edit": edited.digest != base.digest,
            "unknown_fact_error": compile_error(UnknownFactPolicy),
            "duplicate_code_error": compile_error(DuplicateCodePolicy),
            "feasibility": report.as_dict(),
            "dead_rule_detected": bool(report.dead_rules),
        },
        args.replay,
    )


if __name__ == "__main__":
    main()
