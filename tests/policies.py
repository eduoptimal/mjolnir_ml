"""Fixture policies — module-level so @dsl can read their source (by design)."""
from mjolnir_ml import (ASK, DENY, RELEASE, DecisionPolicyBase, dsl, rule)


@dsl
class FraudPolicy(DecisionPolicyBase):
    """amount/credit_limit are stated facts; p_positive/conf are model facts."""

    facts = ("amount", "credit_limit")

    assert lambda amount: amount > 0, "amount_positive"
    assert lambda credit_limit: credit_limit > 0, "limit_positive"

    assert rule("hard_deny",
                when=lambda amount, credit_limit: amount > credit_limit * 2,
                verdict=DENY)
    assert rule("auto_release",
                when=lambda p_positive, conf: p_positive < 0.05 and conf >= 0.95,
                verdict=RELEASE)
    assert rule("escalate_mid", when=lambda conf: conf < 0.80, verdict=ASK)

    @rule(verdict=ASK)
    def escalate_uncertain(self, conf):
        return conf < 0.60


@dsl
class CertifiedPolicy(DecisionPolicyBase):
    """The high-risk shape: release only on a CERTIFIED low-risk prediction."""

    facts = ("x",)

    assert rule("certified_release",
                when=lambda certified, p_positive: certified and p_positive < 0.20,
                verdict=RELEASE)
    assert rule("ambiguous",
                when=lambda certified: not certified,
                verdict=ASK)


@dsl
class RegressorPolicy(DecisionPolicyBase):
    """Gate an actuation on the whole conformal interval clearing the band."""

    facts = ("x",)

    assert rule("safe_band", when=lambda lo, hi: lo >= 0.0 and hi <= 10.0,
                verdict=RELEASE)
    assert rule("too_high", when=lambda hi: hi > 20.0, verdict=DENY)


@dsl
class OnlyDenyPolicy(DecisionPolicyBase):
    facts = ("amount",)

    assert rule("deny_big", when=lambda amount: amount > 100, verdict=DENY)


@dsl
class UncoveredPolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("never_for_small", when=lambda x: x > 1000, verdict=DENY)


class BadAttributePolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("r", when=lambda x: x.foo > 0, verdict=DENY)


class BadCallPolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("r", when=lambda x: abs(x) > 0, verdict=DENY)


class BadFreeNamePolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("r", when=lambda x: x > GLOBAL_THRESHOLD, verdict=DENY)


class PremiseWithoutCode(DecisionPolicyBase):
    facts = ("x",)

    assert lambda x: x > 0


class ForbiddenVerdictPolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("r", when=lambda x: x > 0, verdict="abstain")


class DuplicateCodePolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("same", when=lambda x: x > 0, verdict=DENY)
    assert rule("same", when=lambda x: x < 0, verdict=ASK)


class UndeclaredFactPolicy(DecisionPolicyBase):
    facts = ("x",)

    assert rule("r", when=lambda danger: danger > 0, verdict=DENY)


@dsl
class NoFactsPolicy(DecisionPolicyBase):
    """No `facts` declaration — names are validated at bind against the schema."""

    assert rule("credit_rule", when=lambda credit: credit > 0, verdict=DENY)


@dsl
class SklearnPolicy(DecisionPolicyBase):
    """No facts declaration — bind validates names against the fitted schema."""

    assert rule("block_high",
                when=lambda p_positive, conf: p_positive >= 0.80 and conf >= 0.80,
                verdict=DENY)
    assert rule("allow_certified",
                when=lambda certified, p_positive: certified and p_positive <= 0.20,
                verdict=RELEASE)
    assert rule("review_middle",
                when=lambda p_positive: p_positive > 0.20 and p_positive < 0.80,
                verdict=ASK)


@dsl
class RiskPolicy(DecisionPolicyBase):
    """Two-track policy over CRC facts; ask only when neither action is certified."""

    facts = ("x",)

    assert rule("release_risk", when=lambda risk_release: risk_release,
                verdict=RELEASE)
    assert rule("deny_risk", when=lambda risk_deny: risk_deny, verdict=DENY)
    assert rule("review_risk",
                when=lambda risk_release, risk_deny:
                not risk_release and not risk_deny, verdict=ASK)


@dsl
class DeadDenyPolicy(DecisionPolicyBase):
    """A coverage policy whose deny rule is unreachable at this threshold."""

    facts = ("x",)

    assert rule("release_low", when=lambda hi: hi <= 10.0, verdict=RELEASE)
    assert rule("deny_high", when=lambda lo: lo > 240.0, verdict=DENY)
    assert rule("review_mid",
                when=lambda lo, hi: lo <= 240.0 and hi > 10.0, verdict=ASK)


@dsl
class RegimePolicy(DecisionPolicyBase):
    """Escalates in-regime rows; the regime floor decides before this runs."""

    facts = ("a", "b")

    assert rule("review", when=lambda a: a == a, verdict=ASK)


@dsl
class DualPolicy(DecisionPolicyBase):
    """A release certified by both the coverage and risk tracks."""

    facts = ("x",)

    assert rule("release_coverage", when=lambda hi: hi <= 10.0,
                verdict=RELEASE)
    assert rule("release_risk", when=lambda risk_release: risk_release,
                verdict=RELEASE)
    assert rule("review", when=lambda hi, risk_release:
                not (hi <= 10.0 or risk_release), verdict=ASK)
