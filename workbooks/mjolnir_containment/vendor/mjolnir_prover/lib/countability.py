"""mjolnir_prover.lib.countability — the isolation / countability gate (PC-2c).

Two properties, both checked without trusting mjolnir's process:

**Isolation.** The verdict is a *function only of the certificate*. Two tests:

  1. *Structural* — the counting kernel is re-run in a fresh subprocess whose
     import hook admits only the kernel's own re-derivation closure. If the
     verdict needed the gate, the encoder, the registry, the ledger or the
     model, that subprocess would fail to import and the gate says so instead
     of claiming isolation. This is the difference between "our tests say the
     kernel agrees" and "the kernel cannot see anything else".
  2. *Influence* — every top-level certificate field is deleted in turn and the
     derived verdicts recomputed. The surviving fields are the certificate's
     *decisive surface*; everything else is measured inert, not asserted inert.
     Flipping the emitter's own ``status`` field must not move any derived
     verdict — that is the field a laundered verdict would travel through.

**Countability.** ``philosophy.md`` §181 claims the Ω-certifiable algebras are
countable because they are a subset of the recursive theories. This module
makes that concrete where it is checkable: the certificates form a *finite,
canonically ordered, hashed manifest*, and coverage is an integer ratio over
that manifest — the number of obligations the kernel discharges by re-derivation
divided by the number of obligations enumerated. A third party recomputes the
same digest and the same integer with the kernel and nothing else. Coverage is a
count, not a process.

The route table below is the published routing: method -> rung -> decision
procedure. It is data, not a second implementation of the maths; the procedures
are executed by :mod:`mjolnir_prover.lib.kernel`, which is the only verdict authority
for re-derivation. An obligation whose method is not in the table is *coverage
gap*, counted honestly in the denominator — which is what makes the ratio
meaningful rather than flattering.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile

from mjolnir_prover.lib import kernel as K

# The isolated process may import the KERNEL'S OWN re-derivation closure and
# nothing else. Closure is computed from kernel.py's import graph (the same
# machinery the trust pin uses), so it cannot drift from the code. The emitter
# side — certify, gate, certificate, claim_ir, countermodel, intervals, theory,
# shapes, mine, the frontends, the registry — is banned by name: if the verdict
# ever came to depend on it, this subprocess would fail to import and the gate
# would report that instead of claiming isolation.
TCB_CACHE: list[str] = []

_BANNED_EMITTER = ("gate", "certify", "certificate", "claim_ir", "countermodel",
                   "intervals", "theory", "shapes", "mine", "transfer",
                   "frontend", "frontend_java", "frontend_sql", "frontend_lean2",
                   "leankernel", "authority", "countability", "knowledge",
                   "effects", "effects_proposer", "bindings", "review",
                   "termination", "paths", "dsl", "frontend")


def tcb() -> tuple[str, ...]:
    """kernel.py's transitive local-module closure, as module names."""
    if not TCB_CACHE:
        from mjolnir_prover.lib import authority as AU
        TCB_CACHE.extend(sorted(n[:-3] for n in AU.closure(("kernel.py",))))
    return tuple(TCB_CACHE)

# The published route table: method -> (rung, procedure). `rederivable` marks
# routes the kernel discharges from the certificate ALONE (the countable core);
# the rest are recorded in the manifest as declared-but-not-rederived, so the
# denominator stays honest. Derived from the kernel's own replay dispatch.
ROUTES = (
    # method prefix      rung                 procedure                      rederivable
    ("decide_eq",        "algebra",           "term.decide_eq on both sides",  True),
    ("nf+decide_eq",     "algebra",           "term.decide_eq on both sides",  True),
    ("nf>0",             "algebra",           "nf(premise - conclusion) >= 0",  True),
    ("family_shape",     "family",            "structured variant shape check", True),
    ("rewrite+decide_eq", "theory",           "rewrite under cert.theory",     True),
    ("program_exit",     "program",           "program re-derivation",         True),
    ("sign_analysis:",   "sign",              "term.provably_{ge,gt,le,lt}",   True),
    ("interval:",        "box",               "trace digest + step-check",     True),
    ("premise",          "declared",          "recorded assumption",           True),
    ("premise_derive:",  "premise-derived",   "non-negative combo of premise "
                                              "differences",                  True),
    ("no_precondition",  "vacuous",           "conclusion == 'vacuous'",       True),
    # declared, not independently re-derived (counted, not hidden)
    ("linarith-system",  "declared",          "premise system (recorded)",     False),
    ("quad:disc-sign",   "declared",          "discriminant sign (recorded)",  False),
    ("rel:",             "declared",          "prose relation (recorded)",     False),
    ("mod:",             "declared",          "modelled step (recorded)",      False),
    ("out_of_fragment",  "abstain",           "declared out of fragment",      False),
    ("",                 "unknown",           "no route published",            False),
)

_VERSION = "countability/v2"

# A placeholder obligation is not an obligation. `out_of_fragment` is emitted
# once per undecidable VC carrying the raw claim text in `premise` and a "?"
# conclusion: it records WHICH RUNG is missing, and nothing else. Counting these
# in the coverage denominator made the ratio measure its own instrumentation —
# on the release, 2148 of 2217 rows were placeholders, so "coverage" read 2.1%
# when it was really "2.1% of rows aren't placeholders". Placeholders are
# reported as `pending_rungs` (the rung backlog, a genuinely useful integer)
# and excluded from the coverage denominator.
PLACEHOLDERS = ("out_of_fragment",)


def is_placeholder(o: dict) -> bool:
    """True iff this row is an abstention marker, not a dischargeable obligation."""
    m = o.get("method", "")
    return m in PLACEHOLDERS or (m == "" and o.get("conclusion") == "?")


# ------------------------------------------------------------------ routing ---
def route_of(method: str) -> tuple[str, str, bool]:
    """(rung, procedure, rederivable) for an obligation method."""
    for prefix, rung, proc, red in ROUTES:
        if prefix and method.startswith(prefix):
            return rung, proc, red
    return "unknown", "no route published", False


def route_table() -> list[dict]:
    """The routing, as a sorted table a third party can read and diff."""
    return sorted(({"method": p or "*", "rung": r, "procedure": q,
                    "rederivable": d} for p, r, q, d in ROUTES),
                  key=lambda r: r["method"])


# ------------------------------------------------------- certificate field ---
DECISIVE_FIELDS = ("obligations", "program", "spec", "trace", "theory")


def _derived(cert: dict) -> tuple:
    """The derived verdict, reduced to the part that is a decision.

    The emitter's own `status`/`reason` and the audit metadata are excluded by
    construction: comparing this tuple across mutations is exactly the claim
    "the verdict is a function of the certificate's decision content".
    """
    r = K.replay(cert)
    return (tuple(r["statuses"]), r["agreement"], r["false_certs"])


def influence(cert: dict) -> dict:
    """Delete each top-level field in turn; report which fields move the verdict.

    `decisive` = removing the field changes the derived verdict (or breaks it,
    which is a change: an unparseable certificate derives nothing). `inert` =
    removing it changes nothing.
    """
    base = _derived(cert)
    decisive, inert, broke = [], [], []
    for field in sorted(cert.keys()):
        probe = {k: v for k, v in cert.items() if k != field}
        try:
            got = _derived(probe)
        except Exception as e:
            broke.append(f"{field}:{type(e).__name__}")
            continue
        (decisive if got != base else inert).append(field)
    return {"decisive": sorted(decisive), "inert": sorted(inert),
            "unremovable": sorted(broke)}


def isolation_in_process(cert: dict) -> dict:
    """Influence half of the isolation test, in this process."""
    inf = influence(cert)
    # the launder twin: an emitter that stamps `status: CERTIFIED` onto a claim
    # whose obligations do not derive must not move the derived verdicts.
    tampered = dict(cert, status="CERTIFIED", reason="", false_certificates=0)
    laundered = _derived(tampered) == _derived(cert)
    return {"influence": inf,
            "decisive_fields": inf["decisive"],
            "inert_fields": inf["inert"],
            "status_launder_twinned": laundered,
            "isolated": bool(laundered and set(inf["decisive"]) <= set(DECISIVE_FIELDS))}


# ---------------------------------------------------- structural isolation ---
_RUNNER = r'''
import sys, json, importlib.util as _u
sys.path.insert(0, {release!r})
ALLOWED = {allowed!r}
BANNED = {banned!r}
# Resolve every permitted spec BEFORE installing the hook: inside find_spec a
# nested import re-enters the finder and recurses.
_SPECS = {{name: _u.find_spec("mjolnir_prover.lib." + name) for name in ALLOWED}}

class _Gate:
    """Admit ONLY the re-derivation TCB. Anything else is a hard failure — an
    unimportable verdict path is reported, never silently tolerated."""
    _in = False
    def find_spec(self, name, path=None, target=None):
        if _Gate._in:
            return None
        if name.startswith("mjolnir"):
            parts = name.split(".")
            if len(parts) == 3 and parts[1] == "lib" and parts[2] in ALLOWED:
                return _SPECS[parts[2]]
            raise ImportError("isolation: module outside the kernel TCB: " + name)
        if name.split(".")[0] in BANNED:
            raise ImportError("isolation: module outside the kernel TCB: " + name)
        return None

sys.meta_path.insert(0, _Gate())
from mjolnir_prover.lib import kernel as K
_c = json.load(open(sys.argv[1]))
r = K.replay(_c)
print(json.dumps({{"statuses": r["statuses"], "agreement": r["agreement"],
                   "false_certs": r["false_certs"], "mismatches": r["mismatches"]}},
                  sort_keys=True))
'''


def isolated_replay(cert: dict) -> dict:
    """Re-derive `cert` in a fresh process that can only see the TCB.

    The certificate is written alone into an empty directory: no repo, no
    registry, no ledger, no config, no sibling artifacts. Environment is
    scrubbed to the minimum and PYTHONHASHSEED is pinned so the run cannot
    depend on inherited ambient state.
    """
    lib_root = os.path.dirname(os.path.dirname(os.path.abspath(K.__file__)))
    release = os.path.dirname(lib_root)
    with tempfile.TemporaryDirectory(prefix="mj_isolation_") as d:
        cert_path = os.path.join(d, "certificate.json")
        runner_path = os.path.join(d, "runner.py")
        with open(cert_path, "w", encoding="utf-8") as fh:
            json.dump(cert, fh, sort_keys=True)
        with open(runner_path, "w", encoding="utf-8") as fh:
            fh.write(_RUNNER.format(allowed=set(tcb()), banned=tuple(_BANNED_EMITTER),
                                    release=release))
        env = {"PATH": os.environ.get("PATH", ""),
               "PYTHONPATH": release,
               "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", runner_path, cert_path],
                cwd=d, env=env, capture_output=True, text=True, timeout=120)
        except Exception as e:                       # noqa: BLE001
            return {"isolated": False, "error": f"{type(e).__name__}: {e}",
                    "derived": None, "in_process": _derived(cert)}
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:]
            return {"isolated": False,
                    "error": f"subprocess_exit_{proc.returncode}: {tail[0] if tail else ''}",
                    "derived": None, "in_process": _derived(cert)}
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:                       # noqa: BLE001
            return {"isolated": False, "error": f"unparseable_output:{type(e).__name__}",
                    "derived": None, "in_process": _derived(cert)}
        derived = (tuple(out["statuses"]), out["agreement"], out["false_certs"])
        return {"isolated": derived == _derived(cert),
                "error": None,
                "mismatches": out["mismatches"],
                "derived": list(out["statuses"]),
                "tcb": list(tcb()),
                "in_process": _derived(cert)}


def isolation(cert: dict) -> dict:
    """Both halves. `isolated` requires the structural AND influence results."""
    struct = isolated_replay(cert)
    infl = isolation_in_process(cert)
    return {
        "isolated": bool(struct.get("isolated")) and bool(infl["isolated"]),
        "structural": struct,
        "influence": infl,
        "errors": [e for e in (struct.get("error"),) if e],
    }


# ------------------------------------------------------------- manifest ----
def obligation_rows(certs, include_placeholders: bool = True) -> list[dict]:
    """Canonically ordered obligation rows. The manifest is the countable set.

    Placeholder rows (`out_of_fragment`) are included in the MANIFEST — they are
    real evidence, the rung backlog — but excluded from the coverage DENOMINATOR
    by `coverage()`. They are not obligations and must never be scored as if
    they were.
    """
    rows = [r for _c, _o, r in _indexed(certs)]
    if not include_placeholders:
        rows = [r for r in rows if not is_placeholder(
            {"method": r["method"], "conclusion": r["conclusion"]})]
    return rows


def manifest(certs) -> dict:
    """The countable set + its digest. A third party can recompute both.

    Two sets, deliberately separated: `rows` is everything emitted (including
    the rung backlog), `obligations` is the dischargeable subset that the
    coverage ratio is over. Hashing both means a change to either is visible.
    """
    rows = obligation_rows(certs)
    real = obligation_rows(certs, include_placeholders=False)
    body = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "version": _VERSION,
        "schema": "manifest/v2",
        "rows": rows,
        "count": len(rows),
        "obligations": len(real),
        "placeholders": len(rows) - len(real),
        "digest": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "routes": route_table(),
    }


# ------------------------------------------------------------- coverage ----
def coverage(certs) -> dict:
    """Coverage as an integer ratio over the enumerated manifest.

    numerator   = obligations the kernel DECIDES by re-derivation from the
                  certificate alone (derived verdict True or False, not the
                  "cannot re-derive" abstain)
    denominator = every DISCHARGEABLE obligation in the manifest
    Placeholder rows are excluded and reported as `pending_rungs` — they mark
    a missing rung, not a failed discharge.
    The ratio is a count over a finite hashed set, not a process: the manifest
    digest and the two integers are the entire artifact. A false certificate
    found while counting is reported, never absorbed into the numerator.
    """
    per_rung: dict[str, dict[str, int]] = {}
    discharged = total = false_certs = pending = 0
    pending_kinds: dict[str, int] = {}
    for c, o, r in _indexed(certs):
        if is_placeholder(o):
            # not an obligation: it is the rung backlog. Counted, published,
            # and kept OUT of the coverage denominator.
            pending += 1
            pending_kinds[r["method"]] = pending_kinds.get(r["method"], 0) + 1
            continue
        total += 1
        rung = per_rung.setdefault(r["rung"], {"total": 0, "discharged": 0})
        rung["total"] += 1
        derived = K._rederive(c, o, _trace_ok(c))
        if derived is None or not r["rederivable"]:
            continue
        discharged += 1
        rung["discharged"] += 1
        if o.get("status") == "ok" and derived is False:
            false_certs += 1
    return {
        "numerator": discharged,
        "denominator": total,
        "coverage": (discharged / total) if total else None,
        "false_certificates": false_certs,
        "pending_rungs": pending,
        "pending_by_method": dict(sorted(pending_kinds.items())),
        "uncovered_rungs": sorted(r for r, v in per_rung.items()
                                  if v["discharged"] < v["total"]),
        "per_rung": per_rung,
    }


def _trace_ok(cert: dict) -> bool:
    trace = cert.get("trace")
    if not trace:
        return False
    try:
        from mjolnir_prover.lib import trace as TR, stepcheck as SC
        digest_ok = (not cert.get("trace_digest")) or \
            TR.digest(trace) == cert["trace_digest"]
        return digest_ok and bool(SC.check_trace(trace)["valid"])
    except Exception:                               # noqa: BLE001
        return False


def _indexed(certs):
    """(certificate, obligation, row) triples in manifest order."""
    triples = []
    for c in certs:
        for o in c.get("obligations", []):
            rung, proc, red = route_of(o.get("method", ""))
            triples.append((c, o, {
                "unit": c.get("unit", {}).get("path", "") + ":" +
                        str(c.get("unit", {}).get("name", "")),
                "kind": o.get("kind", ""), "method": o.get("method", ""),
                "rung": rung, "procedure": proc, "rederivable": red,
                "premise": o.get("premise", ""),
                "conclusion": o.get("conclusion", "")}))
    triples.sort(key=lambda t: (t[2]["unit"], t[2]["kind"], t[2]["method"],
                                t[2]["premise"], t[2]["conclusion"]))
    return triples
