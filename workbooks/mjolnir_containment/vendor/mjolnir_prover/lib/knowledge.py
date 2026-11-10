"""mjolnir_prover.lib.knowledge — declarative domain knowledge (DATA, not code branches).

The checker engine in `checker.py` is generic; everything domain-specific lives
here as data: unit dimensions + affine maps, ambiguous notational facts,
calendar-unit policy, and NL contradiction/antonym tables. Adding a unit, a
conversion, or a contradiction fact is a data edit — never an engine edit.
"""
from __future__ import annotations
from fractions import Fraction
import re

# --- Units: name -> (dimension, scale, offset) mapping value -> base unit -------
# base value = scale * value + offset
UNITS = {
    "C": ("temperature", Fraction(1), Fraction(0)),
    "F": ("temperature", Fraction(5, 9), Fraction(-160, 9)),   # (F-32)*5/9
    "K": ("temperature", Fraction(1), Fraction(-27315, 100)),
    "m": ("length", Fraction(1), Fraction(0)),
    "km": ("length", Fraction(1000), Fraction(0)),
    "cm": ("length", Fraction(1, 100), Fraction(0)),
    "mm": ("length", Fraction(1, 1000), Fraction(0)),
    "bit": ("information", Fraction(1), Fraction(0)),
    "byte": ("information", Fraction(8), Fraction(0)),
}

# Units whose definition is not unique -> any claim using them abstains.
AMBIGUOUS_UNITS = {
    "kilobyte": "SI 1000 vs binary 1024",
    "kb": "SI 1000 vs binary 1024",
    "megabyte": "SI 10^6 vs binary 2^20",
}

# --- Calendar: compound time units that are not fixed-length -> abstain ---------
AMBIGUOUS_DURATION_UNITS = {"month": "30 days vs calendar month",
                            "year": "365 days vs calendar year"}

# --- Notational ambiguity facts (parsed-claim predicates -> reason) -------------
# `expr` is matched against the normalised claim text (whitespace-insensitive).
AMBIGUOUS_EXPRESSIONS = [
    ("0.1 + 0.2 = 0.3", "float vs exact rational"),
    ("sqrt(x**2) = x", "valid only for x>=0 (domain)"),
    ("abs(x) = x", "valid only for x>=0 (domain)"),
    ("x**0.5 = x", "valid only for x>=0 (domain)"),
    ("0**0 = 1", "0^0 is a convention"),
    ("(a + b)/2", "integer-truncating vs float average"),
]

# --- THE NUMERIC DOMAIN THIS KERNEL DOES NOT HAVE ---------------------------
# THE ARITHMETIC HERE IS EXACT, OVER THE RATIONALS. It has no IEEE-754
# semantics: no infinities, no NaN, no rounding, no signed zero, no overflow.
#
# That gap had a cost, and it was found the way every gap in this project has
# been found -- by meeting real code. Mining `mpmath`, a mature numerical library
# with more tests than most production code, produced eight REFUTED verdicts, all
# in `test_special.py`, all asserting correct IEEE behaviour: `3 > -inf`,
# `inf * -3 == -inf`, `3 / inf == 0`, `(-inf) ** 3 == -inf`. The kernel refuted
# true facts because `-inf` is not a rational, and it REFUTED them rather than
# abstaining, which is the expensive direction: a refutation is a defect signal,
# so a reviewer would have gone looking for a bug in a test suite that has none.
#
# THERE ARE TWO WAYS TO CLOSE THIS AND ONLY ONE OF THEM IS AVAILABLE SAFELY.
#
#   (a) Teach the kernel IEEE-754. Real work, and it would make every existing
#       certificate conditional on a numeric model that is itself unverified.
#       Rounding and overflow are exactly the kind of semantics that must be
#       proven before it is trusted, and nothing here proves it.
#
#   (b) DECLARE THE DOMAIN AND ABSTAIN. The claim states it is about IEEE floats;
#       this kernel does not reason about those, so it abstains with the reason
#       "numeric domain: ieee_float". No new proofs, no new certificates, no
#       change to any arithmetic.
#
# (b) is the whole fix, and it is the direction this project has moved in five
# times now: when a value is outside the namespace, declare it and decline. Not
# "get smarter about the value" -- that is what produced the 361 false
# cross-reference findings and the 80% false-alarm grounding rate.
#
# SAFETY OF THE DIRECTION: this change can only turn REFUTED into UNVERIFIABLE.
# It cannot create a certificate, and it cannot turn an UNVERIFIABLE into a
# REFUTED. That asymmetry is why it is cheap enough to do now, and it is asserted
# by the regression test rather than argued here.
#
# The markers below are DATA, like everything else in this module. A new numeric
# domain is a data edit.
IEEE_FLOAT_MARKERS = (
    # names, as they appear in code. Non-finite values are not in the rationals
    # at all -- there is no fraction equal to infinity -- so a claim naming one is
    # about a different object, not an imprecise version of the same one.
    "inf", "infinity", "nan", "isfinite", "isinf", "isnan", "nans", "infs",
)
FLOAT_DOMAIN_REASON = "numeric_domain:ieee_float"
# Overflowing literals: representable as a Python float only as `inf`, so the
# rational they denote is not the value the program computes.
_OVERFLOW_LITERAL = re.compile(r"\d[eE][+-]?\d{3,}")


def exactly_representable(literal: str) -> bool:
    """True when this decimal denotes a value IEEE-754 double holds EXACTLY.

    This is the whole answer to "mathematically correct or language-correct", and
    it is arithmetic rather than a preference. A double is a dyadic rational: an
    integer divided by a power of two, with at most 53 bits of mantissa. So for a
    decimal whose reduced denominator is a power of two, and whose magnitude is
    inside the double range, a proof over the rationals is ALSO a proof over the
    doubles -- same object, two representations. The two domains coincide there.

    Where they do not coincide is the only place a decision is needed:
      * `0.5`, `0.25`, `2.5`, `1e300`? -- exact in binary, so prove it. (`1e300`
        is 10^300, which carries factors of five, so it is NOT exact.)
      * `0.1`, `0.3`, `0.7` -- not exact; the rational and the double differ, so a
        rational proof says nothing about what the program computes.
      * `inf`, `nan` -- not rational at all.

    The corpus already made this call. `frag_float_oof` and `frag_float_exact`
    certify `n * 0.5 + n * 0.5 == n` under the note "float literal is an exact
    rational" -- a deliberate E5 upgrade, and it is sound for exactly this reason.
    A guard that abstains on every decimal throws those two away, which is why
    this function exists rather than a pattern.
    """
    from fractions import Fraction
    # BOUND THE EXPONENT BEFORE CONSTRUCTING ANYTHING. This check was written
    # after the fact and its first version called `Fraction(literal)` first and
    # asked about the bit length afterwards, so `1e100000000000000000` -- which
    # mpmath's `test_special.py` really does assert on -- made it try to
    # materialise 10^(10^17) and hang. The stall was in the GUARD, not the
    # kernel, and it is the same defect as every other unbounded resource in
    # this project: an instrument trying to be complete and paying for it.
    #
    # A double's exponent spans about -324..308, so an exponent written with more
    # than four digits is out of range by inspection. No arithmetic required, and
    # no way to be surprised by it.
    _m = re.fullmatch(r"\d+[eE]([+-]?\d+)", literal.strip())
    if _m and len(_m.group(1).lstrip("+-")) > 4:
        return False
    if len(literal.strip().lstrip("+-").replace(".", "")) > 320:
        return False                     # a 320-digit mantissa cannot be a double
    try:
        v = Fraction(literal)
    except (ValueError, ZeroDivisionError):
        return False
    d = v.denominator
    if d & (d - 1):                      # not a power of two -> not exact
        return False
    # A double is m * 2^e with |m| <= 2^53 and |value| < 2^1024. So strip the
    # factors of two out of the numerator and require what remains to fit 53
    # bits. `1e300` is 10^300: denominator 1 passes the dyadic test, but its odd
    # part 5^300 is ~700 bits, and a double cannot hold it. The first version of
    # this check got that wrong and would have certified claims about 1e300.
    p = v.numerator
    while p and p % 2 == 0:
        p //= 2
    if p.bit_length() > 53:
        return False
    e = v.numerator.bit_length() - d.bit_length()
    return -1074 <= e <= 1023


def _decimal_literals(text):
    import re
    return re.findall(r"\d+\.\d+|\d+[eE][+-]?\d+", text or "")


def numeric_domain_markers(text: str):
    """-> sorted markers placing `text` outside what the rationals can decide.

    `text` is the CLAIM's own words, which is the right place to look. Nothing is
    inferred from the file it came from: a claim is judged on what it says.
    """
    if not text:
        return []
    found = set()
    low = text.lower()
    for m in IEEE_FLOAT_MARKERS:
        if m in low:
            found.add(m)
    for m in _OVERFLOW_LITERAL.findall(text):
        found.add(f"overflow_literal:{m}")
    for lit in _decimal_literals(text):
        if not exactly_representable(lit):
            found.add(f"inexact_decimal:{lit}")
    return sorted(found)


def ambiguous_expression_reason(text: str):
    """The reason a claim is outside this kernel's reach, or None.

    Wired here at last. `AMBIGUOUS_EXPRESSIONS` was written early and never
    called -- a data table nobody read, which is the failure mode this project
    keeps meeting in a different costume: knowledge that exists, is correct, and
    changes no behaviour.
    """
    if not text:
        return None
    flat = "".join((text or "").split())
    for expr, reason in AMBIGUOUS_EXPRESSIONS:
        if "".join(expr.split()) == flat:
            return reason
    markers = numeric_domain_markers(text)
    if markers:
        return f"{FLOAT_DOMAIN_REASON} ({', '.join(markers)})"
    return None

# --- Natural-language contradiction facts (data-driven exact rules) -------------
# Each fact: conditions on (ctx, claim); verdict; reason; rule id; evidence.
# `ctx_has`/`claim_has` are substrings; `claim_startswith` optional.
NL_FACTS = [
    dict(rule="pronoun_ambiguous", ctx_has="it was scared",
         claim_has_any=("cat was scared", "dog was scared"),
         verdict="UNKNOWN", reason="pronoun 'it' is ambiguous",
         evidence="two possible referents"),
    dict(rule="negation_scope", ctx_has="not all", claim_startswith="no ",
         verdict="REFUTED", reason="'not all' does not mean 'none'",
         evidence="ctx: not all; claim: no"),
    dict(rule="population_mismatch", ctx_has="adults", claim_has="children",
         verdict="REFUTED", reason="population mismatch",
         evidence="ctx: adults; claim: children"),
    dict(rule="contradicts_context", ctx_has="mild nausea", claim_has="no participants",
         verdict="REFUTED", reason="contradicts reported side effects",
         evidence="ctx reports nausea"),
]

# Antonym pairs (both directions); used by the generic lexical-contradiction rule.
ANTONYMS = [("reduces", "increases"), ("lowers", "raises"), ("lowers", "increases"),
            ("improves", "worsens"), ("increases", "decreases"),
            ("higher", "lower"), ("more", "fewer")]

STOPWORDS = {"the", "a", "an", "of", "in", "by", "and", "to", "at", "on", "is", "are",
             "was", "were", "it", "that", "with", "for", "from"}
