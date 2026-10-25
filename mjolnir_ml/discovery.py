"""mjolnir_ml.discovery — the discovery loop: journaled, evidence-gated
admission of NEW knowledge over a claim corpus.

The flywheel (`live.py`) answers *may this version keep running* over live
decisions. `DiscoverySession` answers *does this candidate capability earn a
place in the knowledge ledger* over a corpus of claims (a prover registry
export, a decline census, an experiment's fragment — any mapping the caller
supplies). The laws are the flywheel's, re-encoded for claims:

PROPOSER NEVER DISPOSES. A `Candidate` is UNLABELED — its claims carry no
expectation (`mine.py`'s `expect=null`); only the independent oracle assigns
verdicts. An oracle whose identity equals the candidate's `proposed_by` is
REFUSED, journaled as a refusal, and surfaced as a critical alarm: the
certifier may never be the proposer of what it certifies (the CWE lesson —
oracle and subject must be independent).

FOUR NUMBERS, NEVER A SCALAR. The census is (certified, declared, declined,
refuted) over the corpus. Discovery is movement of that census plus the
frontier — the decline classes, counted — never a score. `Census.as_dict()`
has exactly four keys by construction.

TWINS ARE MANDATORY. Every candidate carries labelled `tests` (what the rung
must newly certify) and a `twin` — the near-miss battery that must keep its
verdict. Either absent or failing ⇒ no admission. A capability that cannot
name what must remain undecidable is not understood.

TIGHTENING IS FREE; LOOSENING IS SIGNED. Downgrading a certified claim
(shrinking the certified set) is unconditional. Certifying a previously
REFUTED claim is `may_loosen` and is refused unless an external `Approval`
verifies against the session's verifier — the session never holds a signing
key. `contain()` is unconditional; `uncontain()` needs the signature.

UNION-ONLY LEDGER. `KnowledgeLedger` grows by admission and shrinks only with
a refutation record; ids are never reused; retiring reverts the admitted
claims to their pre-admission verdicts (tightening, always free).

DETERMINISTIC MONITORING. `tick()` recomputes detections from the journal, so
the same journal and the same declared detectors produce the same alarms. The
critical ones are `new_refuted` (a rung that manufactures refutations is a
defect signal, not a discovery) and `self_certification`; either contains the
session until an external approval lifts it. `decline_growth` warns and never
acts.

THE MOVE IS CERTIFIABLE. `certify_admission` pins the move (candidate, before/
after censuses, gates) under HMAC with a binding id recomputed from those
parts, and replays the census-arithmetic gates; the oracle-dependent gates are
labelled `attested_gates` — the certificate says what a third party can and
cannot recompute instead of pretending.

Nothing here imports mjolnir_prover: the loop is the same, the corpus is the
caller's.
"""
from __future__ import annotations

import hashlib
import hmac
from collections import Counter
from dataclasses import dataclass, field, replace

from mjolnir_ml._digest import digest as _digest
from mjolnir_ml.live import Approval, Journal, _canonical, _now, _sha

SCHEMA = "mjolnir.ml.discovery/1"
CERT_SCHEMA = "mjolnir.ml.discovery.certificate/1"
SIG_SCHEME = "hmac-sha256"

# The claim axis. A disposition decides a ROW (release/ask/deny/abstain — the
# wire vocabulary); a census verdict counts a CLAIM. One axis, four words each,
# deliberately different vocabularies so the two can never be conflated.
CERTIFIED = "certified"
DECLARED = "declared"
DECLINED = "declined"
REFUTED = "refuted"
CENSUS_VERDICTS = (CERTIFIED, DECLARED, DECLINED, REFUTED)

REPLAY_GATES = ("no_new_refuted", "certified_gained")


class DiscoveryError(ValueError):
    """The discovery request itself is malformed (not a failed gate)."""


def _who(obj) -> str:
    """Stable identity of a producer: class/func module + qualname, or an
    explicit `discovery_id`. Assign `discovery_id` to distinguish instances —
    two objects sharing an identity are treated as the same producer."""
    pid = getattr(obj, "discovery_id", None)
    if isinstance(pid, str) and pid.strip():
        return pid
    mod = getattr(obj, "__module__", None) or type(obj).__module__
    name = (getattr(obj, "__qualname__", None)
            or getattr(obj, "__name__", None)
            or type(obj).__qualname__)
    return f"{mod}.{name}"


# ------------------------------------------------------------------- census --
@dataclass(frozen=True)
class Census:
    """Four numbers. The only shape a corpus verdict summary may take."""

    certified: int = 0
    declared: int = 0
    declined: int = 0
    refuted: int = 0

    @property
    def total(self) -> int:
        return self.certified + self.declared + self.declined + self.refuted

    @classmethod
    def from_verdicts(cls, verdicts) -> "Census":
        counts = dict.fromkeys(CENSUS_VERDICTS, 0)
        for v in verdicts:
            if v not in counts:
                raise DiscoveryError(f"unknown census verdict {v!r} "
                                     f"(the census has exactly "
                                     f"{list(CENSUS_VERDICTS)})")
            counts[v] += 1
        return cls(**counts)

    def as_dict(self) -> dict:
        """Exactly four keys — this is what status, journals and certificates
        publish. There is deliberately no aggregate accessor besides `total`."""
        return {"certified": self.certified, "declared": self.declared,
                "declined": self.declined, "refuted": self.refuted}

    def delta(self, before: "Census") -> dict:
        return {k: getattr(self, k) - getattr(before, k)
                for k in self.as_dict()}


# ---------------------------------------------------------------- candidate --
def _pairs(seq, what: str) -> tuple:
    out = []
    for item in seq:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise DiscoveryError(f"{what} entries must be (claim, expected)")
        claim, expected = item
        if not isinstance(claim, str) or not claim.strip():
            raise DiscoveryError(f"{what} entries need a non-empty claim")
        if expected not in CENSUS_VERDICTS:
            raise DiscoveryError(f"{what}: {expected!r} is not a census verdict")
        out.append((claim, expected))
    return tuple(out)


@dataclass(frozen=True)
class Candidate:
    """An UNLABELED proposal. Its own claims carry no expectation; labels live
    only in `tests`/`twin` (the batteries) and come from the oracle at
    evaluation time."""
    kind: str                       # rung | lemma | premise | hazard_rule | ...
    shape: str                      # normalized pattern of the class
    claims: tuple                   # the targeted claims (the population sample)
    candidate_id: str = ""          # stamped by propose() as a content address
    proposed_by: str = ""           # producer identity, stamped by propose()
    reason: str = ""                # targeted decline class (frontier target)
    domain: str = ""
    population: int = 0             # frontier count for that class (>= sample)
    tests: tuple = ()               # (claim, expected) — what it must certify
    twin: tuple = ()                # (claim, expected) — what must NOT drift
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "tests", _pairs(self.tests, "tests"))
        object.__setattr__(self, "twin", _pairs(self.twin, "twin"))
        if not self.kind or not self.shape:
            raise DiscoveryError("a candidate needs a kind and a shape")
        if not self.claims:
            raise DiscoveryError("a candidate needs at least one claim")
        if self.evidence.get("expect") is not None:
            raise DiscoveryError(
                "a candidate is unlabeled: evidence may not carry `expect` — "
                "labels come from the oracle, never the proposer")

    def as_dict(self) -> dict:
        return {"candidate_id": self.candidate_id, "kind": self.kind,
                "shape": self.shape, "claims": list(self.claims),
                "proposed_by": self.proposed_by, "reason": self.reason,
                "domain": self.domain, "population": self.population,
                "tests": [list(p) for p in self.tests],
                "twin": [list(p) for p in self.twin],
                "evidence": dict(self.evidence)}


def _candidate_id(c: Candidate) -> str:
    return _digest({"kind": c.kind, "shape": c.shape, "domain": c.domain,
                    "claims": list(c.claims)})


# -------------------------------------------------------------- evaluations --
@dataclass(frozen=True)
class DiscoveryGate:
    name: str
    passed: bool
    value: object
    threshold: object
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "value": self.value,
                "threshold": self.threshold, "detail": self.detail}


@dataclass(frozen=True)
class DiscoveryEvaluation:
    candidate_id: str
    passed: bool
    may_loosen: bool
    flipped_refuted: int
    tightened: int
    gates: tuple
    before: dict
    after: dict
    after_verdicts: dict
    n: int
    claims: tuple
    oracle_id: str
    change_digest: str

    def as_dict(self) -> dict:
        return {"candidate_id": self.candidate_id, "passed": self.passed,
                "may_loosen": self.may_loosen,
                "flipped_refuted": self.flipped_refuted,
                "tightened": self.tightened,
                "gates": [g.as_dict() for g in self.gates],
                "before": self.before, "after": self.after,
                "n": self.n, "claims": list(self.claims),
                "oracle_id": self.oracle_id,
                "change_digest": self.change_digest}

    def failed(self) -> list:
        return [g.name for g in self.gates if not g.passed]


@dataclass(frozen=True)
class Admission:
    approved: bool
    reason: str
    candidate_id: str
    before: dict
    after: dict | None
    evaluation: DiscoveryEvaluation | None
    approval: Approval | None = None
    candidate: Candidate | None = field(default=None, repr=False)


# ------------------------------------------------------------- the ledger --
@dataclass(frozen=True)
class KnowledgeRecord:
    """One admitted capability. Carries both verdict maps so retirement can
    revert the corpus to its pre-admission state (tightening, free)."""
    knowledge_id: str
    kind: str
    shape: str
    domain: str
    reason: str
    claims: tuple
    before: dict
    after: dict
    before_verdicts: dict
    after_verdicts: dict
    oracle_id: str
    admitted_at: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"knowledge_id": self.knowledge_id, "kind": self.kind,
                "shape": self.shape, "domain": self.domain,
                "reason": self.reason, "claims": list(self.claims),
                "before": self.before, "after": self.after,
                "oracle_id": self.oracle_id, "admitted_at": self.admitted_at,
                "evidence": dict(self.evidence)}

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())


class KnowledgeLedger:
    """Union-only knowledge ledger. Ids are never reused; retirement is
    recorded WITH its refutation, so the record of what was believed (and why
    it stopped counting) only grows."""

    SCHEMA = "mjolnir.ml.knowledge-ledger/1"

    def __init__(self, version: str = "knowledge-ledger@1"):
        self.version = version
        self._active: dict[str, KnowledgeRecord] = {}
        self._retired: dict[str, dict] = {}

    def add(self, record: KnowledgeRecord) -> "KnowledgeLedger":
        if not isinstance(record, KnowledgeRecord):
            raise DiscoveryError("add takes a KnowledgeRecord")
        if record.knowledge_id in self._active \
                or record.knowledge_id in self._retired:
            raise DiscoveryError(f"knowledge id {record.knowledge_id!r} is "
                                 f"already in the ledger (ids are never reused)")
        self._active[record.knowledge_id] = record
        return self

    def retire(self, knowledge_id: str, refutation: str) -> KnowledgeRecord:
        if knowledge_id not in self._active:
            raise KeyError(f"knowledge {knowledge_id!r} is not active")
        if not isinstance(refutation, str) or not refutation.strip():
            raise ValueError(f"retiring {knowledge_id!r} needs a refutation "
                             f"record")
        rec = self._active.pop(knowledge_id)
        self._retired[knowledge_id] = {**rec.as_dict(),
                                       "refutation": refutation}
        return rec

    def has(self, knowledge_id: str) -> bool:
        return knowledge_id in self._active or knowledge_id in self._retired

    def get(self, knowledge_id: str) -> KnowledgeRecord | None:
        return self._active.get(knowledge_id)

    @property
    def active(self) -> dict:
        return dict(self._active)

    @property
    def retired(self) -> dict:
        return dict(self._retired)

    @property
    def digest(self) -> str:
        return _digest({"schema": self.SCHEMA, "version": self.version,
                        "active": {k: v.as_dict()
                                   for k, v in sorted(self._active.items())},
                        "retired": dict(sorted(self._retired.items()))})

    def as_dict(self) -> dict:
        return {"schema": self.SCHEMA, "version": self.version,
                "digest": self.digest,
                "active": {k: v.as_dict()
                           for k, v in sorted(self._active.items())},
                "retired": dict(sorted(self._retired.items()))}


# --------------------------------------------------------------- detectors --
@dataclass(frozen=True)
class DiscoveryDetector:
    """A declared monitor over the journal. Same discipline as
    `live.DetectorSpec`: named, with a target and a declared MDE."""
    name: str
    lever: str
    metric: str          # refuted_growth | refusal_self_cert | declined_growth
    severity: str = "critical"
    target: float = 0.0
    threshold: float = 0.0
    min_n: int = 1
    mde: float = 1.0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("name", "lever", "metric", "severity", "target", "threshold",
                 "min_n", "mde")}


@dataclass(frozen=True)
class DiscoveryDetection:
    name: str
    lever: str
    metric: str
    alarm: bool
    severity: str
    stat: float
    threshold: float
    target: float
    n: int
    mde: float
    detail: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("name", "lever", "metric", "alarm", "severity", "stat",
                 "threshold", "target", "n", "mde", "detail")}


def default_discovery_detectors() -> tuple:
    return (
        DiscoveryDetector("new_refuted", "rung", "refuted_growth",
                          severity="critical"),
        DiscoveryDetector("self_certification", "authority",
                          "refusal_self_cert", severity="critical"),
        DiscoveryDetector("decline_growth", "rung", "declined_growth",
                          severity="warning"),
    )


# ------------------------------------------------------------- containment --
@dataclass(frozen=True)
class Containment:
    containment_id: str
    reason: str
    at: str

    @property
    def change_digest(self) -> str:
        return _sha({"uncontain": self.containment_id})

    def as_dict(self) -> dict:
        return {"containment_id": self.containment_id, "reason": self.reason,
                "at": self.at}


# ------------------------------------------------------------------ session --
class DiscoverySession:
    """A journaled discovery loop over a claim corpus. Holds the corpus, the
    ledger and the journal; never holds a signing key."""

    def __init__(self, corpus=None, *, reasons=None, journal=None,
                 ledger=None, approval_verifier=None, detectors=None):
        if corpus is None:
            corpus = {}
        if not hasattr(corpus, "items"):
            raise TypeError("DiscoverySession takes a corpus mapping "
                            "claim -> census verdict")
        self._corpus: dict[str, str] = {}
        for claim, verdict in corpus.items():
            if verdict not in CENSUS_VERDICTS:
                raise DiscoveryError(f"corpus verdict {verdict!r} for "
                                     f"{claim!r} is not a census verdict")
            self._corpus[claim] = verdict
        if reasons is not None:
            for claim in reasons:
                if claim not in self._corpus:
                    raise DiscoveryError(
                        f"decline reason for a claim outside the corpus: "
                        f"{claim!r}")
        self._reasons = dict(reasons) if reasons is not None else None
        self._journal = journal if isinstance(journal, Journal) \
            else Journal(journal)
        self._ledger = ledger if isinstance(ledger, KnowledgeLedger) \
            else KnowledgeLedger()
        self._approval_verifier = approval_verifier
        self._detectors = tuple(detectors) if detectors is not None \
            else default_discovery_detectors()
        for d in self._detectors:
            if d.metric not in ("refuted_growth", "refusal_self_cert",
                                "declined_growth"):
                raise DiscoveryError(
                    f"unknown detector metric {d.metric!r} (fail-closed: "
                    f"a detector that cannot be computed cannot be trusted)")
        self._contained: Containment | None = None
        self._versions: list[dict] = []

    # --------------------------------------------------------------- state --
    @property
    def journal(self) -> Journal:
        return self._journal

    @property
    def ledger(self) -> KnowledgeLedger:
        return self._ledger

    @property
    def corpus(self) -> dict:
        return dict(self._corpus)

    @property
    def containment(self) -> Containment | None:
        return self._contained

    @property
    def detectors(self) -> tuple:
        return self._detectors

    @property
    def versions(self) -> list:
        return list(self._versions)

    # -------------------------------------------------------------- propose --
    def _ensure_proposed(self, candidate: Candidate) -> Candidate:
        """Journal a candidate's provenance before anything evaluates it."""
        if not candidate.candidate_id:
            candidate = replace(candidate, candidate_id=_candidate_id(candidate))
        if not candidate.proposed_by:
            raise DiscoveryError("a candidate must carry `proposed_by` — "
                                 "provenance before evaluation")
        known = {p["candidate"]["candidate_id"]
                 for p in self._journal.payloads("candidate")}
        if candidate.candidate_id not in known:
            self._journal.append(
                {"kind": "candidate", "candidate": candidate.as_dict()})
        return candidate

    def propose(self, proposer) -> list[Candidate]:
        """Run a proposer (callable, or an iterable of candidates) and journal
        the unlabeled candidates it yields. Stamps `proposed_by` and the
        content-addressed id when absent. Refuses labeled candidates."""
        producer = _who(proposer) if callable(proposer) else None
        raw = proposer() if callable(proposer) else proposer
        if isinstance(raw, Candidate):
            items = [raw]
        else:
            try:
                items = list(raw)
            except TypeError as ex:
                raise DiscoveryError(
                    "propose takes a callable returning a Candidate or an "
                    "iterable of candidates") from ex
        out = []
        for cand in items:
            if not isinstance(cand, Candidate):
                raise DiscoveryError(f"proposer yielded {type(cand).__name__}, "
                                     f"not a Candidate")
            if not cand.proposed_by:
                cand = replace(cand, proposed_by=producer or "")
            out.append(self._ensure_proposed(cand))
        return out

    # ------------------------------------------------------------ evaluate --
    def evaluate(self, candidate: Candidate, oracle, *, claims=None,
                 oracle_id: str | None = None) -> DiscoveryEvaluation:
        """Run the INDEPENDENT oracle over the candidate's claims and measure
        the census movement. Read-only except for the journal."""
        if not isinstance(candidate, Candidate):
            raise TypeError("evaluate takes a Candidate")
        if not callable(oracle):
            raise DiscoveryError("evaluate takes a callable oracle")
        candidate = self._ensure_proposed(candidate)
        oid = oracle_id or _who(oracle)
        if oid == candidate.proposed_by:
            self._journal.append({"kind": "refusal",
                                  "reason": "self_certification",
                                  "candidate_id": candidate.candidate_id,
                                  "proposed_by": candidate.proposed_by,
                                  "oracle_id": oid})
            raise PermissionError(
                "self-certification refused: the oracle "
                f"({oid!r}) is the candidate's own proposer — an oracle and "
                f"its subject must be independent")
        target = tuple(claims) if claims is not None else tuple(candidate.claims)
        if not target:
            raise DiscoveryError("no claims to evaluate")
        unknown = [c for c in target if c not in self._corpus]
        if unknown:
            raise DiscoveryError(
                f"claim(s) outside the corpus: {unknown[:3]!r} — the corpus "
                f"is the baseline; nothing is invented for a claim it does "
                f"not hold")

        before_map = {c: self._corpus[c] for c in target}
        before = Census.from_verdicts(before_map.values())
        after_map: dict[str, str] = {}
        for c in target:
            v = oracle(c)
            if v not in CENSUS_VERDICTS:
                raise DiscoveryError(f"oracle returned {v!r} for {c!r} "
                                     f"(fail-closed: not a census verdict)")
            after_map[c] = v
        after = Census.from_verdicts(after_map.values())

        def _scored(pairs):
            results = [(claim, expected, oracle(claim))
                       for claim, expected in pairs]
            return results, sum(1 for _, e, a in results if a == e)

        tests, n_tests_ok = _scored(candidate.tests)
        twins, n_twin_ok = _scored(candidate.twin)
        flipped = sum(1 for c, v in after_map.items()
                      if v == CERTIFIED and before_map[c] == REFUTED)
        tightened = sum(1 for c, v in before_map.items()
                        if v == CERTIFIED and after_map[c] != CERTIFIED)

        gates: list[DiscoveryGate] = [
            DiscoveryGate(
                "labelled_tests",
                len(tests) >= 1 and n_tests_ok == len(tests),
                f"{n_tests_ok}/{len(tests)}", ">=1 and all pass",
                "a capability earns admission with a labelled test, never "
                "without one"),
            DiscoveryGate(
                "twin_holds",
                len(twins) >= 1 and n_twin_ok == len(twins),
                f"{n_twin_ok}/{len(twins)}", ">=1 and all pass",
                "the twin is the battery that fails when the rung is "
                "reverted or overreaches"),
            DiscoveryGate(
                "no_new_refuted",
                after.refuted <= before.refuted,
                {"before": before.refuted, "after": after.refuted},
                "after <= before",
                "a new REFUTED is a defect signal, not a discovery"),
            DiscoveryGate(
                "certified_gained",
                after.certified > before.certified,
                {"before": before.certified, "after": after.certified},
                "after > before",
                "movement is the point: the rung must move the census"),
        ]
        if candidate.reason:
            if self._reasons is None:
                gates.append(DiscoveryGate(
                    "targeted_decline_moved", False, None,
                    "measurable decline class",
                    f"target {candidate.reason!r} declared without "
                    f"decline-reason data — declared but unmeasurable fails, "
                    f"like a required-and-absent fact"))
            else:
                # "still undecided" rather than "still declined": a claim the
                # registry holds as REFUTED is also work the class has not
                # closed, and its certification is guarded by `may_loosen`.
                b_t = sum(1 for c in target
                          if before_map[c] != CERTIFIED
                          and self._reasons.get(c) == candidate.reason)
                a_t = sum(1 for c in target
                          if after_map[c] != CERTIFIED
                          and self._reasons.get(c) == candidate.reason)
                gates.append(DiscoveryGate(
                    "targeted_decline_moved", a_t < b_t,
                    {"before": b_t, "after": a_t}, "after < before",
                    f"the declared target is undecided claims of class "
                    f"{candidate.reason!r}"))

        ev = DiscoveryEvaluation(
            candidate_id=candidate.candidate_id,
            passed=all(g.passed for g in gates),
            may_loosen=flipped > 0,
            flipped_refuted=flipped,
            tightened=tightened,
            gates=tuple(gates),
            before=before.as_dict(),
            after=after.as_dict(),
            after_verdicts=dict(after_map),
            n=len(target),
            claims=target,
            oracle_id=oid,
            change_digest=_sha({"candidate": candidate.candidate_id,
                                "before": before.as_dict(),
                                "after": after.as_dict(),
                                "flipped_refuted": flipped,
                                "n": len(target)}))
        self._journal.append({"kind": "evaluation",
                              "evaluation": ev.as_dict()})
        return ev

    # ------------------------------------------------------------- promote --
    def promote(self, candidate: Candidate, oracle, *, approval: Approval | None = None,
                claims=None, oracle_id: str | None = None) -> Admission:
        """Evidence-gated admission. Refusals are journaled with their reason;
        an approved admission writes the ledger, applies the census movement
        and appends a version."""
        candidate = self._ensure_proposed(candidate)
        ev = self.evaluate(candidate, oracle, claims=claims, oracle_id=oracle_id)

        def refuse(reason: str, appr: Approval | None = None) -> Admission:
            self._journal.append({
                "kind": "admission", "candidate_id": ev.candidate_id,
                "promoted": False, "detail": reason,
                "before": ev.before, "after": None,
                "approval": appr.as_dict() if appr else None,
                "evaluation": ev.as_dict()})
            return Admission(False, reason, ev.candidate_id, ev.before, None,
                             ev, appr, candidate)

        if self._contained is not None:
            return refuse(f"contained:{self._contained.reason}")
        if self._ledger.has(candidate.candidate_id):
            return refuse("already admitted")
        if not ev.passed:
            return refuse("gate failed: " + ", ".join(ev.failed()))
        if ev.may_loosen:
            ok = (approval is not None
                  and self._approval_verifier is not None
                  and self._approval_verifier(approval, ev.change_digest))
            if not ok:
                return refuse(f"approval required: candidate certifies "
                              f"{ev.flipped_refuted} previously REFUTED "
                              f"claim(s)", appr=approval)

        before_map = {c: self._corpus[c] for c in ev.claims}
        record = KnowledgeRecord(
            knowledge_id=candidate.candidate_id,
            kind=candidate.kind, shape=candidate.shape,
            domain=candidate.domain, reason=candidate.reason,
            claims=ev.claims, before=ev.before, after=ev.after,
            before_verdicts=before_map,
            after_verdicts=dict(ev.after_verdicts),
            oracle_id=ev.oracle_id, admitted_at=_now(),
            evidence=dict(candidate.evidence))
        self._ledger.add(record)
        for c, v in ev.after_verdicts.items():
            self._corpus[c] = v
        version = {"candidate_id": candidate.candidate_id,
                   "ledger_digest": self._ledger.digest,
                   "census": ev.after, "at": _now()}
        self._versions.append(version)
        self._journal.append({
            "kind": "admission", "candidate_id": ev.candidate_id,
            "promoted": True, "detail": "admitted",
            "before": ev.before, "after": ev.after,
            "knowledge": record.as_dict(),
            "approval": approval.as_dict() if approval else None,
            "evaluation": ev.as_dict()})
        return Admission(True, "admitted", ev.candidate_id, ev.before,
                         ev.after, ev, approval, candidate)

    # ------------------------------------------------------------- retire --
    def retire(self, knowledge_id: str, refutation: str) -> dict:
        """Union-only retirement: the ledger keeps the record WITH its
        refutation, and the corpus reverts the admitted claims to their
        pre-admission verdicts (shrinking the certified set — always free)."""
        rec = self._ledger.retire(knowledge_id, refutation)
        reverted = 0
        for c, was in rec.before_verdicts.items():
            if self._corpus.get(c) == rec.after_verdicts.get(c):
                self._corpus[c] = was
                reverted += 1
        payload = {"kind": "retraction", "knowledge_id": knowledge_id,
                   "refutation": refutation, "reverted": reverted,
                   "at": _now()}
        self._journal.append(payload)
        return payload

    # ---------------------------------------------------------------- tick --
    def _epoch(self) -> int:
        return max((e["seq"] for e in self._journal.entries()
                    if e["payload"].get("kind") == "uncontainment"),
                   default=-1)

    def tick(self) -> list[DiscoveryDetection]:
        """Deterministic: every detection is recomputed from the journal on
        every call. Epoch-aware — history before the last lifted containment
        does not re-alarm (the flywheel's hazard-detector lesson)."""
        entries = self._journal.entries()
        epoch = self._epoch()
        evals = [e["payload"]["evaluation"] for e in entries
                 if e["seq"] > epoch and e["payload"].get("kind") == "evaluation"]
        refuted_growth = sum(1 for ev in evals
                             if ev["after"]["refuted"] > ev["before"]["refuted"])
        declined_growth = sum(1 for ev in evals
                              if ev["after"]["declined"] > ev["before"]["declined"])
        self_cert = sum(1 for e in entries
                        if e["seq"] > epoch
                        and e["payload"].get("kind") == "refusal"
                        and e["payload"].get("reason") == "self_certification")
        stats = {"refuted_growth": refuted_growth,
                 "refusal_self_cert": self_cert,
                 "declined_growth": declined_growth}
        details = {
            "refuted_growth": f"{refuted_growth} evaluation(s) with "
                              f"after.refuted > before.refuted",
            "refusal_self_cert": f"{self_cert} self-certification attempt(s)",
            "declined_growth": f"{declined_growth} evaluation(s) with "
                               f"after.declined > before.declined",
        }
        out = []
        for spec in self._detectors:
            n = len(evals) if spec.metric != "refusal_self_cert" else len(entries)
            stat = float(stats[spec.metric])
            alarm = n >= spec.min_n and stat > spec.threshold
            out.append(DiscoveryDetection(
                spec.name, spec.lever, spec.metric,
                alarm=alarm,
                severity=(spec.severity if alarm else "info"),
                stat=stat, threshold=spec.threshold, target=spec.target,
                n=n, mde=spec.mde, detail=details[spec.metric]))
        return out

    def auto_act(self, detections) -> list[Containment]:
        """The entire automatic action set: contain on a critical alarm.
        Warnings never act."""
        return [self.contain(d.detail) for d in detections
                if d.alarm and d.severity == "critical"
                and self._contained is None]

    # ---------------------------------------------------------- containment --
    def contain(self, reason: str) -> Containment:
        """UNCONDITIONAL: while contained, admissions are refused."""
        if self._contained is not None:
            return self._contained
        c = Containment(containment_id=_sha({"reason": reason, "at": _now()}),
                        reason=reason, at=_now())
        self._contained = c
        self._journal.append({"kind": "containment", **c.as_dict()})
        return c

    def uncontain(self, approval: Approval, *,
                  reason="approved release from containment") -> None:
        """Lifting containment is loosening: needs an external signature."""
        if self._contained is None:
            raise ValueError("session is not contained")
        ok = (self._approval_verifier is not None
              and self._approval_verifier(approval,
                                          self._contained.change_digest))
        if not ok:
            raise PermissionError(
                "uncontain requires a valid external approval for "
                f"{self._contained.change_digest}")
        self._journal.append({"kind": "uncontainment",
                              "containment_id": self._contained.containment_id,
                              "approval": approval.as_dict(),
                              "reason": reason})
        self._contained = None

    # --------------------------------------------------------------- status --
    def status(self, *, queue: int = 10) -> dict:
        census = Census.from_verdicts(self._corpus.values())
        frontier = Counter()
        if self._reasons is not None:
            for claim, reason in self._reasons.items():
                if self._corpus.get(claim) == DECLINED and reason:
                    frontier[reason] += 1
        proposed = [p["candidate"]["candidate_id"]
                    for p in self._journal.payloads("candidate")]
        admitted = {p["candidate_id"] for p in
                    self._journal.payloads("admission") if p.get("promoted")}
        evaluated = len(self._journal.payloads("evaluation"))
        q = [cid for cid in proposed if cid not in admitted]
        return {
            "schema": SCHEMA,
            "n": len(self._corpus),
            "census": census.as_dict(),
            "frontier": dict(sorted(frontier.items(),
                                    key=lambda kv: (-kv[1], kv[0]))),
            "candidates": {"proposed": len(proposed),
                           "evaluated": evaluated,
                           "admitted": len(admitted),
                           "queue": q[:queue]},
            "ledger": {"version": self._ledger.version,
                       "digest": self._ledger.digest,
                       "active": len(self._ledger.active),
                       "retired": len(self._ledger.retired)},
            "contained": self._contained.as_dict() if self._contained else None,
            "versions": list(self._versions),
        }


# ------------------------------------------------------------- certificate --
def _admission_id(body: dict) -> str:
    return _digest(body)


def _cert_body(cert: dict) -> dict:
    return {k: v for k, v in cert.items()
            if k not in ("admission_id", "issued_at", "signature")}


def _signature(cert: dict, key) -> str:
    body = {k: v for k, v in cert.items() if k != "signature"}
    mac = hmac.new(key.encode() if isinstance(key, str) else bytes(key),
                   _canonical(body), hashlib.sha256)
    return f"{SIG_SCHEME}:{mac.hexdigest()}"


def certify_admission(admission: Admission, *, key=None) -> dict:
    """The certificate of the move: candidate + before/after censuses + gates,
    pinned under a binding id. Only an APPROVED admission certifies — a
    refusal is journaled, not certified."""
    if not isinstance(admission, Admission):
        raise DiscoveryError("certify_admission takes an Admission")
    if not admission.approved or admission.evaluation is None:
        raise DiscoveryError("only an admitted move can be certified")
    if admission.candidate is None:
        raise DiscoveryError("the admission must carry its candidate")
    ev = admission.evaluation
    body = {
        "schema": CERT_SCHEMA,
        "candidate_id": ev.candidate_id,
        "candidate": admission.candidate.as_dict(),
        "before": ev.before,
        "after": ev.after,
        "gates": [{"name": g.name, "passed": g.passed} for g in ev.gates],
        "flipped_refuted": ev.flipped_refuted,
        "tightened": ev.tightened,
        "n": ev.n,
        "oracle_id": ev.oracle_id,
        "approved": admission.approved,
        "reason": admission.reason,
        "replay_gates": list(REPLAY_GATES),
        "attested_gates": [g.name for g in ev.gates
                           if g.name not in REPLAY_GATES],
    }
    cert = {**body, "admission_id": _admission_id(body),
            "issued_at": _now(), "signature": None}
    # The signature covers the whole published certificate minus itself (so
    # issued_at and the binding id are both pinned by it).
    if key is not None:
        cert["signature"] = _signature({k: v for k, v in cert.items()
                                        if k != "signature"}, key)
    return cert


def verify_admission_certificate(cert: dict, *, key=None,
                                 require_signature: bool = False) -> bool:
    """structure -> signature -> binding -> replay (the census-arithmetic
    gates are re-derived; oracle-dependent gates are attested, and the
    certificate says so)."""
    if not isinstance(cert, dict):
        raise DiscoveryError("certificate must be a dict")
    if cert.get("schema") != CERT_SCHEMA:
        raise DiscoveryError(f"unknown certificate schema "
                             f"{cert.get('schema')!r}")
    if not isinstance(cert.get("gates"), list) or not cert["gates"]:
        raise DiscoveryError("certificate must carry its gates")
    for field_name in ("candidate_id", "admission_id", "oracle_id"):
        if not isinstance(cert.get(field_name), str) or not cert[field_name]:
            raise DiscoveryError(f"missing {field_name}")
    for census_name in ("before", "after"):
        c = cert.get(census_name)
        if not isinstance(c, dict) or set(c) != set(CENSUS_VERDICTS):
            raise DiscoveryError(
                f"{census_name} must be a census with exactly "
                f"{list(CENSUS_VERDICTS)}")
    if not isinstance(cert.get("candidate"), dict):
        raise DiscoveryError("missing candidate")
    if cert["candidate"].get("candidate_id") != cert["candidate_id"]:
        raise DiscoveryError("candidate_id disagrees with the pinned candidate")
    for count_field in ("flipped_refuted", "tightened", "n"):
        v = cert.get(count_field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise DiscoveryError(f"{count_field} must be a non-negative int")

    sig = cert.get("signature")
    if sig is not None:
        if key is None:
            raise DiscoveryError("certificate is signed but no key was "
                                 "supplied (fail-closed)")
        if not isinstance(sig, str) or not sig.startswith(SIG_SCHEME + ":"):
            raise DiscoveryError(f"unknown signature scheme {sig!r}")
        expect = _signature({k: v for k, v in cert.items()
                             if k != "signature"}, key)
        if not hmac.compare_digest(sig, expect):
            raise DiscoveryError("signature does not match the certificate")
    elif require_signature:
        raise DiscoveryError("certificate is unsigned")

    if _admission_id(_cert_body(cert)) != cert["admission_id"]:
        raise DiscoveryError(
            "admission_id does not match the pinned parts (binding check "
            "failed)")

    gates = {g["name"]: g for g in cert["gates"]}
    if not all(g.get("passed") for g in cert["gates"]):
        raise DiscoveryError("certificate records a failed gate — an "
                             "admitted move passed every gate")
    derived = {
        "no_new_refuted": cert["after"]["refuted"] <= cert["before"]["refuted"],
        "certified_gained": cert["after"]["certified"] > cert["before"]["certified"],
    }
    for name, want in derived.items():
        if name not in gates:
            raise DiscoveryError(f"certificate does not carry the replayable "
                                 f"gate {name!r}")
        if bool(gates[name]["passed"]) != want:
            raise DiscoveryError(
                f"replay disagrees on {name!r}: the census says {want}, the "
                f"certificate says {gates[name]['passed']}")
    return True


__all__ = ["SCHEMA", "CERT_SCHEMA", "CENSUS_VERDICTS",
           "CERTIFIED", "DECLARED", "DECLINED", "REFUTED",
           "Census", "Candidate", "DiscoveryError",
           "DiscoveryGate", "DiscoveryEvaluation", "Admission",
           "KnowledgeRecord", "KnowledgeLedger",
           "DiscoveryDetector", "DiscoveryDetection",
           "default_discovery_detectors", "Containment",
           "DiscoverySession", "certify_admission",
           "verify_admission_certificate"]
