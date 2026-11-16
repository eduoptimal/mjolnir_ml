"""mjolnir_router.core.verdict -- the frozen core. No domain nouns in this file.

WHAT THIS IS
------------
One function. Given a set of declared facts and a list of rules, it returns one of
three verdicts and the reason. Nothing else.

    ROUTE      every fact the verdict depends on is stated, and no obligation
               triggered
    ESCALATE   the stated facts trigger an obligation
    ABSTAIN    a fact a rule depends on is NOT STATED

WHAT IS DELIBERATELY ABSENT
---------------------------
No domain vocabulary. This module does not know what money is, what a record is,
what a verb class is, or what a contract is. Those live in `frontends/`, which
declare their own fact derivations and their own rules and hand them here. That is
the whole architecture claim, and it is checkable: `test_core.py` fails the build
if a domain noun appears in this package.

THE ONE INVARIANT WORTH HAVING A FUNCTION FOR
---------------------------------------------
FAIL CLOSED, ON THE FRONTEND'S BEHALF. A rule declares the facts it depends on:

    rule("money_moves_on_write", requires=("money", "verb_class"), ...)

and if `money` is UNSTATED the core returns ABSTAIN with `missing:money` without
the frontend having written that check. The first version of this policy had the
checker read an absent payment key as "no money moves" -- a decision with no basis
-- and the differential test found it on 141 real actions. Writing the requirement
into the rule signature means that class of defect is now structurally harder to
write than to catch, which is the only reliable way to stop writing it.

THE OTHER DIRECTION, WHICH IS ALSO STRUCTURAL
--------------------------------------------
A rule may not read a fact it did not declare. Reading `facts["amount"]` from a
rule that requires only `("money",)` raises. The point is that a frontend author
cannot accidentally build a verdict that depends on something the core does not
know was missing -- the exact failure this project found five times, in five
components, always in the same shape: a value compared against the wrong namespace.

NEW VERDICT SEMANTICS IS A CORE CHANGE, AND SHOULD BE ARGUED. New vocabulary is a
frontend. That line is the discipline this package exists to hold.
"""
from __future__ import annotations

ROUTE = "ROUTE"
ESCALATE = "ESCALATE"
ABSTAIN = "ABSTAIN"
DENY = "DENY"
VERDICTS = (ROUTE, ESCALATE, DENY, ABSTAIN)

# ARGUED CORE CHANGE: a fourth verdict. The core had three, and adding one is a
# core change on purpose. ESCALATE means "an obligation fired, a human may still
# decide"; DENY means "forbidden, and not a human's to grant". The deployed
# services already expressed both (guard's ASK vs a hard refusal) and had a word
# for neither. No existing rule table uses DENY, so behaviour on every current
# policy is unchanged.
_SEV = {ROUTE: 0, ESCALATE: 1, DENY: 2}

# The one value that means UNSTATED. A frontend may declare a fact and fail to
# supply it; it may not invent a default. `None` is the only way to be silent, and
# only this module is allowed to interpret it.
UNSTATED = None


class Rule:
    """One policy statement.

    `requires`  fact names this rule depends on. UNSTATED in any of them means the
                core abstains rather than running the rule.
    `when`      (facts) -> bool, and it may ONLY touch declared facts.
    `verdict`   ROUTE or ESCALATE for a rule that fires; `None` means "this rule
                exists to abstain", which is how a frontend states that a fact
                could not be derived at all.
    `code`      stable reason code. Shown to operators, asserted in tests.
    """

    __slots__ = ("code", "requires", "when", "verdict", "data")

    def __init__(self, code, requires=(), when=None, verdict=ESCALATE, data=None):
        self.code = code
        self.requires = tuple(requires)
        self.when = when
        self.verdict = verdict
        self.data = data

    def __repr__(self):
        return f"Rule({self.code!r}, requires={self.requires}, verdict={self.verdict})"


class Verdict:
    """The result. Carries the reason, because a verdict without one is unusable.

    `basis` is the list of rule codes that fired; `missing` is the fact name that
    was unstated, when the verdict is an abstention. Both are for a human reading
    a log, and both are asserted in tests, so neither can rot.
    """

    __slots__ = ("value", "basis", "missing", "rule_data")

    def __init__(self, value, basis=(), missing=None, rule_data=None):
        assert value in VERDICTS, value
        self.value = value
        self.basis = list(basis)
        self.missing = missing
        self.rule_data = rule_data or {}

    @property
    def reason(self) -> str:
        if self.value is ABSTAIN or self.missing:
            return f"missing:{self.missing}" if self.missing else "abstained"
        return ", ".join(self.basis) or "no_obligation_triggered"

    def as_dict(self):
        return {"verdict": self.value, "basis": self.basis, "missing": self.missing,
                "reason": self.reason}

    def __eq__(self, other):
        return (isinstance(other, Verdict) and self.value == other.value
                and self.basis == other.basis and self.missing == other.missing)

    def __repr__(self):
        return f"Verdict({self.value}, {self.reason})"


class _Declared(dict):
    """Facts, restricted to what a rule declared it depends on.

    The guard is in the data rather than in the source: reading an undeclared fact
    raises at the point of use, which is faster than inspecting source and works
    for lambdas, partials and closures. A source-scanning version of this check was
    the first implementation and it broke on every lambda defined outside a file --
    a good illustration of a guard that only works where its author tested it.
    """

    __slots__ = ("_declared",)

    def __init__(self, facts, declared):
        super().__init__(facts)
        object.__setattr__(self, "_declared", frozenset(declared))

    def _undeclared(self, key):
        raise KeyError(
            f"rule read fact {key!r}, which it did not declare in `requires` "
            f"(declared: {sorted(self._declared)})")

    def __missing__(self, key):
        self._undeclared(key)

    def __getitem__(self, key):
        # The guard has to be here, not only in __missing__: the fact being present
        # but undeclared is exactly the case, and __missing__ does not see it.
        if key not in self._declared:
            self._undeclared(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key not in self._declared:
            self._undeclared(key)
        return dict.get(self, key, default)


def evaluate(facts: dict, rules, *, guard=True) -> Verdict:
    """facts -> Verdict. THE core function.

    Order is policy: the first rule whose `requires` are all stated and whose
    `when` returns true decides. A rule with unmet requirements does not fire --
    it ABSTAINS, naming the fact, and stops. That is fail-closed and it is why a
    frontend cannot forget to abstain.
    """
    fired, data, sev = [], {}, 0
    for r in rules:
        absent = [f for f in r.requires if facts.get(f, UNSTATED) is UNSTATED]
        if absent:
            return Verdict(ABSTAIN, basis=[f"missing:{a}" for a in absent],
                           missing=absent[0])
        if r.when is None:
            if r.verdict is None:            # exists only to abstain
                return Verdict(ABSTAIN, basis=[r.code], missing=r.requires[0]
                               if r.requires else None)
            continue
        view = _Declared(facts, r.requires) if (guard and r.requires) else facts
        if not r.when(view):
            continue
        if r.verdict is ABSTAIN:
            return Verdict(ABSTAIN, basis=[r.code])
        fired.append(r.code)
        sev = max(sev, _SEV.get(r.verdict, 1))
        if r.data:
            data[r.code] = r.data(view) if callable(r.data) else r.data
    if not fired:
        return Verdict(ROUTE, basis=fired, rule_data=data)
    return Verdict(DENY if sev >= 2 else ESCALATE, basis=fired, rule_data=data)