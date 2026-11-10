"""mjolnir_prover.lib.certify — THE single verdict authority.

TRUST INVARIANT (the product's core): exactly one component in this system may
assign a verdict — this module. Every certificate it mints is

  1. disposed by the prover against the claim's TYPED READING,
  2. serialized into the certificate artifact,
  3. replayed by the detached checkers (kernel + step-checker; the Lean kernel
     for the Lean arm) which share no prover code, and
  4. fail-closed: any definite prover/checker disagreement is a FAULT, never a
     verdict.

Typed readings — assigned STRUCTURALLY by the miner (AST), never from prover
output. The reading selects which statement the disposal is faithful to:

  theorem        claim over parameters only. The universal reading is the
                 faithful one: CERTIFIED proves the source claim; REFUTED is a
                 source violation.
  strengthening  claim over runtime-assigned locals. The universal reading is
                 a STRICT strengthening of the site claim: CERTIFIED entails
                 the site instance (proved_by_strengthening); REFUTED refutes
                 the strengthening only and says NOTHING about the site —
                 recorded as an honest abstention, NEVER a violation.
  premise        declared runtime guard (E1c). What is certified is the
                 implication `guard holds at runtime -> contract`. A refuted
                 premise demotes to observation (fail-open), never a violation.
  instance       runtime-observed claim; not statically decidable. Abstains
                 without consulting the prover.

A prover CERTIFIED whose obligations are all kind="premise" means the prover
BOUND or ASSUMED the claim (E1c premise shape / R1 substitution), it did not
prove it. For theorem/strengthening readings that is recorded UNVERIFIABLE
(assumed_not_proved). Premises and proofs never share a counter.
"""
from __future__ import annotations

import ast
import os
import re

from . import certificate as C
from . import kernel as K
from . import knowledge as KN
from . import stepcheck as SC
from . import termination as TM

CERTIFIED = "CERTIFIED"
REFUTED = "REFUTED"
UNVERIFIABLE = "UNVERIFIABLE"
FAULT = "FAULT"

_HERE = os.path.dirname(os.path.abspath(__file__))          # <release>/mjolnir_prover/lib
_PKG = os.path.dirname(_HERE)                               # <release>/mjolnir
_RELEASE = os.path.dirname(_PKG)                            # <release>

_BUILTINS = {"assert", "len", "range", "abs", "min", "max", "sum", "int",
             "float", "str", "bool", "sorted", "list", "set", "dict",
             "tuple", "enumerate", "zip", "isinstance", "print"}


# --------------------------------------------------------------- artifact ----
def copy_certificate(cert):
    """Round-trip through JSON text — the checker sees only the artifact."""
    return C.load(C.dump(cert))


def _base_cert(code, predicate, res, status, reason):
    return {
        "schema": C.SCHEMA,
        "unit": {"path": "", "name": "", "kind": "unit",
                 "code_hash": C.sha(code) if code else None, "line": None},
        "spec": {"expr": predicate, "source": "assert", "confirmed": True,
                 "hash": C.sha(predicate)},
        "obligations": (res or {}).get("obligations", []),
        "assumptions": [], "tcb": [],
        "tool": {"name": "mjolnir-certify"},
        "encoder": {"role": "untrusted"},
        "status": status, "reason": reason,
        "witness": (res or {}).get("witness"),
        "trace": (res or {}).get("trace", []),
        "trace_digest": (res or {}).get("trace_digest"),
    }


def _arbiter(cert, audit=True):
    """The detached re-check: step-checker on the trace, kernel replay on the
    serialized artifact. FAULT only on a DEFINITE disagreement — a POSITIVE
    claim (proof or violation) the checkers cannot reproduce. Abstentions
    carry no proof, so there is nothing to audit (audit=False); an unverifiable
    trace behind a POSITIVE claim is an alarm, never a padded verdict."""
    if not audit:
        cert["kernel"] = {"agreement": None, "false_certs": 0, "unchecked": 0}
        return cert
    if cert.get("trace"):
        sc = SC.check_trace(cert["trace"])
        cert["stepcheck"] = {"valid": sc["valid"], "steps": sc["steps"]}
        if not sc["valid"]:
            if cert["status"] in (CERTIFIED, REFUTED):
                cert["status"] = FAULT
                cert["reason"] = f"trace_invalid:{sc['bad'][:3]}"
            else:
                cert["reason"] = (cert["reason"] + "|" if cert["reason"] else "") \
                    + f"trace_unchecked:{sc['bad'][:1]}"
    k = K.replay(copy_certificate(cert))
    if cert["status"] == CERTIFIED and k.get("false_certs", 0) > 0:
        cert["status"] = FAULT
        cert["reason"] = f"kernel_disagreement:{k['mismatches']}"
    cert["kernel"] = {"agreement": k.get("agreement"),
                      "false_certs": k.get("false_certs", 0),
                      "unchecked": k.get("unchecked", 0)}
    return cert


# ------------------------------------------------------- function-level ------
def _proved(res):
    """A CERTIFIED verdict must include at least one NON-premise discharge. A
    verdict built only from declared premises proves nothing about the claim
    (a lone `assert n == 0` is a declared premise, not a proof) and is demoted
    to UNVERIFIABLE/assumed_not_proved — mirroring the theorem guard in
    certify_claim. Fail-closed: no padded CERTIFIED."""
    if res.get("status") != CERTIFIED:
        return res.get("status"), res.get("reason", "")
    if not any(o.get("status") == "ok" and o.get("kind") != "premise"
               for o in res.get("obligations", [])):
        return UNVERIFIABLE, "assumed_not_proved"
    return CERTIFIED, res.get("reason", "")


def certify_unit(name, code, def_line=1, path=".", pre_bounds=None):
    """Dispose one FUNCTION unit. For a unit contract the universal reading is
    the faithful one: REFUTED is a real defect signal (the unit's own assert
    fails on an allowed input)."""
    prof_ok, viol = TM.profile_check(code)
    has_asserts = any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(code))) \
        if prof_ok else False
    if not has_asserts:
        res = dict(status=UNVERIFIABLE, obligations=[],
                   reason="no_assert_spec", location=None)
    else:
        res = C.unit_vc(code, pre_bounds=pre_bounds)
        for o in res["obligations"]:          # segment lines -> real file lines
            if o.get("line"):
                o["line"] = def_line - 1 + o["line"]
        if res.get("location") and res["location"].get("line"):
            res["location"]["line"] = def_line - 1 + res["location"]["line"]
    status, reason = _proved(res)
    cert = _base_cert(code, "assert-claims", res, status, reason)
    cert["spec"] = {"expr": "assert-claims", "source": "assert",
                    "confirmed": True, "hash": C.sha("assert-claims")}
    cert["unit"] = {"path": os.path.relpath(path, _RELEASE), "name": name,
                    "kind": "function", "code_hash": C.sha(code),
                    "line": def_line}
    return _arbiter(cert), prof_ok, viol


# --------------------------------------------------------- claim-level -------
def _param_names(predicate, cap=12):
    """Valid, bounded python identifiers from a predicate (keywords and
    builtins excluded — they cannot be wrapper parameters)."""
    import keyword
    names, seen = [], set()
    for n in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", predicate or ""):
        if n in seen or n in _BUILTINS or keyword.iskeyword(n) or "." in n:
            continue
        seen.add(n)
        names.append(n)
        if len(names) >= cap:
            break
    return names


def _wrapper(predicate):
    return (f"def f({', '.join(_param_names(predicate))}):\n"
            f"    assert {predicate}\n")


def _claim_cert(claim, code, res, status, reason, reading):
    origin = claim.get("origin") or {}
    predicate = claim.get("predicate") or ""
    cert = _base_cert(code, predicate, res, status, reason)
    # provenance: every mined claim IS an assert-backed statement in pinned
    # source (the miner harvests asserts/tests/guards only) — the kernel's
    # provenance gate checks exactly this. The mining metadata travels
    # separately; it is bookkeeping, never a truth input.
    cert["provenance"] = {
        "mined": dict(claim.get("provenance") or {}),
        "origin": dict(origin),
        "tier": claim.get("tier"),
    }
    # runtime-enforcement bookkeeping (never a truth input): an enforced
    # observation is a runtime contract; it can never upgrade a verdict.
    cert["enforced"] = bool((claim.get("provenance") or {}).get("enforced"))
    cert["unit"] = {"path": origin.get("file", ""),
                    "name": claim.get("id", "claim"), "kind": "claim",
                    "code_hash": C.sha(code) if code else None,
                    "line": origin.get("line")}
    cert["reading"] = reading
    return cert


def certify_claim(claim, module_source=None):
    """Dispose one mined claim against its typed reading.

    `claim` is a miner candidate (tier / predicate / canonical_unit / origin /
    provenance). A stored `canonical_unit` (frontend-lowered unit of record)
    is disposed as-is; otherwise the claim's predicate is wrapped into a
    disposable unit.
    """
    tier = claim.get("tier")
    origin = claim.get("origin") or {}
    cu = claim.get("canonical_unit")

    # --- NUMERIC DOMAIN GUARD, BEFORE ANY DISPOSAL ---------------------------
    # This kernel's arithmetic is exact, over the rationals. A claim that names
    # IEEE-754 semantics -- an infinity, a NaN, an overflowing literal, a decimal
    # literal -- is outside that space, and the honest answer is to abstain with
    # the reason rather than to refute a true fact.
    #
    # It has to happen BEFORE disposal, because the disposal is what produced the
    # damage: eight REFUTED verdicts on mpmath's `test_special.py`, all of them
    # correct IEEE assertions about infinity. A refutation is a defect signal, so
    # each one would have sent a reviewer hunting a bug in a test suite with none.
    #
    # The direction is one-way: this can only turn REFUTED into UNVERIFIABLE. It
    # cannot create a certificate and cannot turn an abstention into a refutation,
    # which is exactly why it is safe to do without also teaching the kernel what
    # rounding is.
    _domain = KN.ambiguous_expression_reason(cu or claim.get("predicate") or "")
    if _domain:
        cert = _claim_cert(claim, cu, None, UNVERIFIABLE, _domain, "domain")
        cert["domain"] = {"outside": "rationals", "reason": _domain,
                          "markers": KN.numeric_domain_markers(
                              cu or claim.get("predicate") or "")}
        return cert

    if tier not in ("static", "premise") and not cu:
        # instance claim: runtime observation, not statically decidable
        cert = _claim_cert(claim, None, None, UNVERIFIABLE,
                           "runtime_observation", "instance")
        cert["kernel"] = {"agreement": None, "false_certs": 0, "unchecked": 0}
        return cert

    if tier == "premise":
        if not cu:
            cert = _claim_cert(claim, None, None, UNVERIFIABLE,
                               "no_canonical_unit", "premise")
            cert["kernel"] = {"agreement": None, "false_certs": 0, "unchecked": 0}
            return cert
        code, reading = cu, "premise"
    else:
        reading = "strengthening" if origin.get("runtime_locals") else "theorem"
        code = cu or _wrapper(claim.get("predicate") or "")

    # --- THE ROW'S OWN READING, HONOURED -------------------------------------
    # A claim may arrive with the inputs its reading requires: declared
    # `pre_bounds`, a `module_source` to resolve calls against, a `body`+`guard`
    # pair. Wrapping the predicate in a bare `def f(...): assert ...` discards all
    # of them, so the obligation is asked without its hypotheses.
    #
    # Found by running the frozen corpus: 20 of 85 rows disagreed with their
    # expectations, and six of the eight I checked certified correctly the moment
    # the declared bounds were passed through. The corpus was right and the reader
    # was ignoring it -- the same shape as `AMBIGUOUS_EXPRESSIONS`, which was
    # written correctly and called by nothing.
    _pb = claim.get("pre_bounds")
    if _pb is None and origin.get("pre_bounds"):
        _pb = origin.get("pre_bounds")
    _ms = module_source or claim.get("module_source") or origin.get("module_source")

    try:
        # tier flows into the unit: a premise-tier unit may DECLARE its own
        # last equality (see E1b in certificate.py); theorems never can.
        res = C.unit_vc(code, pre_bounds=_pb, module_source=_ms,
                        tier=claim.get("tier"))
    except TypeError:
        # a caller passing the older two-argument shape still works
        res = C.unit_vc(code, module_source=module_source)
    except Exception as e:
        cert = _claim_cert(claim, code, None, UNVERIFIABLE,
                           f"disposal_error:{type(e).__name__}", reading)
        cert["kernel"] = {"agreement": None, "false_certs": 0, "unchecked": 0}
        return cert

    status, reason = res["status"], res.get("reason", "")
    obligs = res.get("obligations", [])
    if status == CERTIFIED and reading in ("theorem", "strengthening"):
        if not any(o.get("status") == "ok" and o.get("kind") != "premise"
                   for o in obligs):
            status, reason = UNVERIFIABLE, "assumed_not_proved"
        elif reading == "strengthening":
            reason = "proved_by_strengthening"
    elif status == REFUTED and reading == "strengthening":
        status, reason = UNVERIFIABLE, "strengthening_refuted"
    elif status == REFUTED and reading == "premise":
        status, reason = UNVERIFIABLE, "premise_refuted"    # fail-open demote

    return _arbiter(_claim_cert(claim, code, res, status, reason, reading),
                    audit=status in (CERTIFIED, REFUTED))


def unit_status(code, pre_bounds=None, module_source=None):
    """Bare verdict for internal consumers (shape-transfer validation, the
    adversarial battery): prover + kernel arbiter, no artifact retained."""
    try:
        res = C.unit_vc(code, pre_bounds=pre_bounds, module_source=module_source)
    except Exception as e:
        return {"status": UNVERIFIABLE, "reason": f"disposal_error:{type(e).__name__}"}
    status, reason = _proved(res)
    if status == CERTIFIED:
        mini = _base_cert(code, "assert-claims", res, CERTIFIED, "")
        k = K.replay(copy_certificate(mini))
        if k.get("false_certs", 0) > 0:
            return {"status": FAULT,
                    "reason": f"kernel_disagreement:{k['mismatches']}"}
    return {"status": status, "reason": reason}


# ------------------------------------------------------------- lean arm ------
def certify_lean(stmt):
    """Dispose one Lean statement: the IR frontend lowers it to claim IR and
    the independent Lean kernel re-derives the verdict. A kernel REFUTATION of
    a prover-CERTIFIED claim is a FAULT; an unconfirmed method is coverage."""
    from . import frontend
    from . import leankernel as LK

    v = frontend.verdict(stmt, "ir")
    if v == "skip":
        return {"stmt": stmt, "status": UNVERIFIABLE,
                "reason": "skip_not_lowerable", "kernel": None}
    claims = frontend.claims(stmt)
    if claims is None:
        return {"stmt": stmt, "status": v, "reason": "", "kernel": None}
    rows = [dict(kind=c.kind, op=c.op, lhs=c.lhs, rhs=c.rhs,
                 is_premise=c.is_premise) for c in claims]
    kernel = LK.replay_text(rows)
    status = v
    if v == CERTIFIED and kernel == REFUTED:
        status = FAULT
    return {"stmt": stmt, "status": status, "kernel": kernel,
            "claims": len(rows), "reason": ""}


# --------------------------------------------------------- architecture ------
_CHECKER_SELF = {"lib/certify.py", "lib/kernel.py", "lib/stepcheck.py",
                 "lib/leankernel.py",
                 # tests import the checkers they test. A regression suite that
                 # cannot reach the kernel is a suite that proves nothing; it
                 # disposes no claim and is excluded on that ground, not on
                 # convenience.
                 "lib/test_domain.py",
                 # audit-only: re-derives via the kernel to COUNT coverage and
                 # to show the verdict is a function of the certificate. It
                 # disposes no claim and emits no verdict, so importing the
                 # checker here does not create a second writer.
                 "lib/countability.py",
                 # audit-only: measures TWO certified sets and their intersection
                 # to test sigma-additivity (T-NL3). It calls existing decision
                 # procedures and reports set identities; it disposes nothing and
                 # emits no verdict of its own.
                 "lib/composition.py"}
_PROVER_SELF = {"lib/certify.py", "lib/certificate.py", "lib/frontend.py",
                "lib/claim_ir.py"}
_CHECKER_RX = re.compile(
    r"^\s*(?:from|import)\s+[^\n]*\b(kernel|leankernel|stepcheck)\b", re.M)
_PROVER_RX = re.compile(r"\bunit_vc\s*\(")


def check_single_writer(root=None):
    """Structural invariant, machine-checked: the detached checkers are
    imported only by the authority (or are the checkers themselves), and the
    prover is called only inside the authority/prover package. Returns
    (ok, offenders)."""
    offenders = []
    for dirpath, dirs, files in os.walk(root or _PKG):
        dirs[:] = sorted(d for d in dirs
                         if d not in ("__pycache__", "storage", "data"))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, root or _PKG).replace(os.sep, "/")
            src = open(p, encoding="utf-8", errors="replace").read()
            if _CHECKER_RX.search(src) and rel not in _CHECKER_SELF:
                offenders.append(f"checker import outside authority: {rel}")
            if _PROVER_RX.search(src) and rel not in _PROVER_SELF:
                offenders.append(f"prover call outside authority: {rel}")
    return (not offenders), offenders