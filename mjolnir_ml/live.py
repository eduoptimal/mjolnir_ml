"""mjolnir_ml.live — the flywheel: a journaled, read-mostly deployment session.

Three operations are easy to confuse and are kept separate on purpose:

    readiness()   one-shot: may this version deploy?          (readiness.py)
    LiveSession   runtime: decide, observe, monitor, contain  (this module)
    ChangeSet     versioned promotion of policy/model changes  (this module)

THE LAW OF THE FLYWHEEL: **tightening is always allowed; loosening needs an
external signature.** `contain()` is unconditional and may only turn a release
into an ask. A `ChangeSet` that adds a release on any logged row is
`may_loosen` and is refused unless an `Approval` verifies against a verifier
the session holds — the session never holds a signing key.

MONITORING is deterministic: `tick()` re-derives detections from the journal,
so the same journal and the same declared detectors produce the same alarms.
Detectors are sequential CUSUM tests with a declared target, slack, threshold,
minimum n and minimum detectable effect — not repeated 95% interval checks.

WHAT THE FLYWHEEL MAY AUTO-APPLY
    * ACI: `observe()` updates kappa's risk level (declared at calibration
      time). This is the only continuous learner.
    * `contain()` on a released external hazard (critical alarm).
Everything else — recalibration, policy edits, widening the regime, model
swaps — is a `ChangeSet`: evidence-gated, and loosening approval-gated.

JOURNAL AND PRIVACY: decisions store the row (needed for outcomes and model
candidate re-scoring) and the fact table (needed to replay policy candidates
without the model). Certificates publish facts; treat the journal as
sensitive, hash-chain it, and redact at export.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mjolnir_ml.model import DecisionModel, Disposition, _model_digest
from mjolnir_ml.pair import omega_pair
from mjolnir_ml.policy import Policy
from mjolnir_ml.vendors import wire

SCHEMA = "mjolnir.ml.live/1"


# ------------------------------------------------------------------ journal --
def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str, ensure_ascii=False).encode("utf-8")


def _sha(obj) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj)).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Journal:
    """Append-only, hash-chained JSONL. Single writer. Tamper-evident."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._entries: list[dict] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._entries.append(json.loads(line))

    def append(self, payload: dict) -> dict:
        prev = self._entries[-1]["hash"] if self._entries else None
        body = {"seq": len(self._entries), "t": _now(), "prev": prev,
                "payload": payload}
        entry = {**body, "hash": _sha(body)}
        self._entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict]:
        return list(self._entries)

    def payloads(self, kind: str | None = None) -> list[dict]:
        out = [e["payload"] for e in self._entries]
        return out if kind is None else [p for p in out if p.get("kind") == kind]

    def verify(self) -> bool:
        prev = None
        for i, e in enumerate(self._entries):
            if e["seq"] != i or e["prev"] != prev:
                return False
            body = {k: e[k] for k in ("seq", "t", "prev", "payload")}
            if e["hash"] != _sha(body):
                return False
            prev = e["hash"]
        return True

    def __len__(self):
        return len(self._entries)


# ----------------------------------------------------------------- approval --
@dataclass(frozen=True)
class Approval:
    change_digest: str
    approver: str
    issued_at: str
    signature: str

    def as_dict(self) -> dict:
        return {"change_digest": self.change_digest, "approver": self.approver,
                "issued_at": self.issued_at, "signature": self.signature}


class ApprovalAuthority:
    """Demo authority. The SIGNING key lives here, outside the session; the
    session receives only `verify` (a bound method). Production: asymmetric
    signatures and a separate approver system."""

    def __init__(self, key=b"approval-demo-key"):
        self._key = key.encode() if isinstance(key, str) else bytes(key)

    def _mac(self, body: dict) -> str:
        return "hmac-sha256:" + hmac.new(self._key, _canonical(body),
                                         hashlib.sha256).hexdigest()

    def sign(self, change_digest: str, approver: str = "risk-owner",
             issued_at: str | None = None) -> Approval:
        body = {"change_digest": change_digest, "approver": approver,
                "issued_at": issued_at or _now()}
        return Approval(signature=self._mac(body), **body)

    def verify(self, approval: Approval, change_digest: str) -> bool:
        if approval.change_digest != change_digest:
            return False
        body = {"change_digest": approval.change_digest,
                "approver": approval.approver,
                "issued_at": approval.issued_at}
        return hmac.compare_digest(self._mac(body), approval.signature)


# ----------------------------------------------------------------- detectors --
@dataclass(frozen=True)
class DetectorSpec:
    """A declared sequential monitor.

    metric: coverage | abstain | abs_error | hazard_release
    target: in-control value (coverage 1-alpha; abstain max rate; abs_error
            calibration MAE; hazard_release 0)
    slack: tolerated in-control deviation
    threshold: CUSUM h (higher = fewer false alarms)
    min_n: minimum observations before testing
    mde: minimum detectable effect declared with the detector
    """
    name: str
    lever: str
    metric: str
    target: float
    slack: float = 0.02
    threshold: float = 5.0
    min_n: int = 30
    mde: float = 0.05

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("name", "lever", "metric", "target", "slack", "threshold",
                 "min_n", "mde")}


@dataclass(frozen=True)
class Detection:
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


def default_detectors(alpha: float = 0.10, mae_target: float | None = None,
                      max_abstain: float = 0.10) -> tuple:
    ds = [
        DetectorSpec("coverage_low", "index", "coverage", 1.0 - alpha,
                     slack=0.02, threshold=5.0, min_n=30, mde=0.05),
        DetectorSpec("abstain_high", "contract", "abstain", max_abstain,
                     slack=0.01, threshold=6.0, min_n=30, mde=0.05),
        DetectorSpec("hazard_release", "contract", "hazard_release", 0.0,
                     slack=0.0, threshold=0.0, min_n=1, mde=0.0),
    ]
    if mae_target is not None:
        ds.append(DetectorSpec("error_high", "proposer", "abs_error",
                               float(mae_target), slack=1.0, threshold=8.0,
                               min_n=30, mde=2.0))
    return tuple(ds)


# ------------------------------------------------------------- containment --
@dataclass(frozen=True)
class Containment:
    containment_id: str
    reason: str
    batch: object
    at: str

    @property
    def change_digest(self) -> str:
        return _sha({"uncontain": self.containment_id})

    def as_dict(self) -> dict:
        return {"containment_id": self.containment_id, "reason": self.reason,
                "batch": self.batch, "at": self.at}


@dataclass(frozen=True)
class Retraction:
    retraction_id: str
    reason: str
    certificate_ids: tuple
    at: str

    def as_dict(self) -> dict:
        return {"retraction_id": self.retraction_id, "reason": self.reason,
                "certificate_ids": list(self.certificate_ids), "at": self.at}


# ------------------------------------------------------------------ changes --
@dataclass(frozen=True)
class ChangeSet:
    """A versioned candidate. Policy changes replay from stored facts; model
    changes re-score stored rows and MUST carry calibration data."""
    lever: str
    reason: str
    policy: Policy | None = None
    estimator: object | None = None
    calibration_rows: list | None = None
    calibration_y: list | None = None
    name: str = "change"

    def digest(self, features: tuple) -> str:
        return _sha({
            "lever": self.lever, "reason": self.reason, "name": self.name,
            "policy": self.policy.digest if self.policy is not None else None,
            "model": (_model_digest(self.estimator, features)
                      if self.estimator is not None else None),
            "n_cal": len(self.calibration_rows or []),
        })


@dataclass(frozen=True)
class ChangeGate:
    name: str
    passed: bool
    before: object
    after: object

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed,
                "before": self.before, "after": self.after}


@dataclass(frozen=True)
class Evaluation:
    change_digest: str
    passed: bool
    may_loosen: bool
    added_releases: int
    gates: tuple
    before: dict
    after: dict
    n: int
    n_observed: int

    def as_dict(self) -> dict:
        return {"change_digest": self.change_digest, "passed": self.passed,
                "may_loosen": self.may_loosen,
                "added_releases": self.added_releases,
                "gates": [g.as_dict() for g in self.gates],
                "before": self.before, "after": self.after,
                "n": self.n, "n_observed": self.n_observed}


@dataclass(frozen=True)
class Promotion:
    approved: bool
    reason: str
    change_digest: str
    before: dict
    after: dict | None
    evaluation: Evaluation | None
    approval: Approval | None = None
    model: DecisionModel | None = field(default=None, repr=False)


def _oracle_answer(oracle, row, y):
    ans = oracle(row, y)
    if isinstance(ans, tuple) and len(ans) == 2:
        return bool(ans[0]), bool(ans[1])
    return bool(ans), True


def _metrics(verdicts, hazards, outcomes, safe, decidable):
    claims = sum(1 for v in verdicts if v in (wire.RELEASE, wire.DENY_W))
    releases = sum(1 for v in verdicts if v == wire.RELEASE)
    released_haz = sum(1 for v, h in zip(verdicts, hazards)
                       if v == wire.RELEASE and h)
    idx = [i for i, y in enumerate(outcomes) if y is not None]
    pair = (omega_pair([verdicts[i] for i in idx],
                       [safe[i] for i in idx],
                       [decidable[i] for i in idx]) if idx else None)
    return {
        "claims": claims, "releases": releases,
        "released_hazards": released_haz,
        "coverage": claims / len(verdicts) if verdicts else 0.0,
        "precision": pair.prec if pair else None,
        "recall": pair.rec if pair else None,
        "n_observed": len(idx),
    }


def _ge(after, before, tol=0.0):
    if before is None:
        return True
    if after is None:
        return False
    return after >= before - tol


def _gt(after, before):
    if after is None:
        return False
    if before is None:
        return True
    return after > before + 1e-12


class LiveSession:
    """Runtime session. Holds versions and a journal; never signs approvals."""

    def __init__(self, dm: DecisionModel, *, journal=None, ledger=None,
                 certificate_key=None, detectors=None,
                 approval_verifier=None):
        if not isinstance(dm, DecisionModel):
            raise TypeError("LiveSession takes a DecisionModel")
        self._dm = dm
        self._journal = (journal if isinstance(journal, Journal)
                         else Journal(journal))
        self._ledger = ledger
        self._cert_key = certificate_key
        self._detectors = (tuple(detectors) if detectors is not None else
                           default_detectors(
                               alpha=dm.kappa.alpha if dm.kappa else 0.10))
        self._approval_verifier = approval_verifier
        self._contained: Containment | None = None
        self._versions: list[dict] = [self._version(dm)]

    # --------------------------------------------------------------- state --
    def _version(self, dm: DecisionModel) -> dict:
        return {"policy_digest": dm.policy.digest,
                "model_digest": dm.model_digest,
                "calibration_digest": (dm.kappa.digest
                                       if dm.kappa is not None else None)}

    @property
    def model(self) -> DecisionModel:
        return self._dm

    @property
    def journal(self) -> Journal:
        return self._journal

    @property
    def containment(self) -> Containment | None:
        return self._contained

    @property
    def versions(self) -> list[dict]:
        return list(self._versions)

    @property
    def detectors(self) -> tuple:
        return self._detectors

    # -------------------------------------------------------------- decide --
    def decide(self, row, *, batch=None) -> Disposition:
        d = self._dm.dispose(row, context={"batch": batch}
                             if batch is not None else None)
        if self._contained is not None and d.verdict == wire.RELEASE:
            d = d.retag(
                wire.ASK, d.basis + (f"contained:{self._contained.reason}",),
                reason=f"contained:{self._contained.reason}",
                say=f"escalated by containment ({self._contained.reason}); "
                    f"was: {d.say}")
        hazards = self._ledger.hazard_codes(row) if self._ledger else []
        self._journal.append({
            "kind": "decision", "batch": batch, "row": dict(row),
            "disposition": d.as_dict(), "hazard_codes": hazards,
            "containment": (self._contained.containment_id
                            if self._contained else None),
        })
        return d

    # -------------------------------------------------------------- observe --
    def observe(self, certificate_id: str, y_true) -> dict:
        decisions = [p for p in self._journal.payloads("decision")
                     if p["disposition"]["certificate_id"] == certificate_id]
        if not decisions:
            raise KeyError(f"unknown certificate {certificate_id!r}")
        if any(p.get("certificate_id") == certificate_id
               for p in self._journal.payloads("outcome")):
            raise ValueError(
                f"outcome already recorded for {certificate_id!r}")
        row = decisions[-1]["row"]
        self._dm.observe(row, y_true)   # ACI: kappa only; no rule changes
        payload = {"kind": "outcome", "certificate_id": certificate_id,
                   "y": y_true,
                   "calibration_digest": (self._dm.kappa.digest
                                          if self._dm.kappa else None)}
        self._journal.append(payload)
        return payload

    # --------------------------------------------------------------- status --
    def _decisions(self, window_batches: int | None = None) -> list[dict]:
        ds = self._journal.payloads("decision")
        if window_batches is not None:
            batches: list = []
            for p in reversed(ds):
                b = p.get("batch")
                if b not in batches:
                    batches.append(b)
                if len(batches) >= window_batches:
                    break
            keep = set(batches)
            ds = [p for p in ds if p.get("batch") in keep]
        return ds

    def _outcomes(self) -> dict:
        return {p["certificate_id"]: p["y"]
                for p in self._journal.payloads("outcome")}

    def status(self, *, window_batches: int | None = None,
               queue: int = 10) -> dict:
        ds = self._decisions(window_batches)
        outcomes = self._outcomes()
        verdicts = [p["disposition"]["verdict"] for p in ds]
        census = {v: 0 for v in wire.WIRE}
        for v in verdicts:
            census[v] = census.get(v, 0) + 1
        causes = Counter()
        for p in ds:
            d = p["disposition"]
            if d["verdict"] == wire.ABSTAIN_W:
                causes[d["reason"] or "abstained"] += 1
        ys, preds = [], []
        for p in ds:
            cid = p["disposition"]["certificate_id"]
            if cid in outcomes:
                ys.append(outcomes[cid])
                preds.append(p["disposition"].get("prediction") or {})
        cover = [1.0 if ((pred.get("set") is not None and y in pred["set"])
                         or (pred.get("lo") is not None
                             and pred["lo"] <= float(y) <= pred["hi"]))
                 else 0.0
                 for y, pred in zip(ys, preds)
                 if pred.get("set") is not None or pred.get("lo") is not None]
        errors = [abs(float(y) - float(p.get("pred")))
                  for y, p in zip(ys, preds) if p.get("pred") is not None]
        correct = [1.0 if y == p.get("pred") else 0.0
                   for y, p in zip(ys, preds) if p.get("pred") is not None]
        hazards = [bool(p.get("hazard_codes")) for p in ds]
        releases = sum(1 for v in verdicts if v == wire.RELEASE)
        released_haz = sum(1 for v, h in zip(verdicts, hazards)
                           if v == wire.RELEASE and h)
        review = [{"certificate_id": p["disposition"]["certificate_id"],
                   "batch": p.get("batch"),
                   "reason": p["disposition"]["reason"],
                   "say": p["disposition"]["say"]}
                  for p in ds
                  if p["disposition"]["verdict"] == wire.ABSTAIN_W][:queue]
        return {
            "schema": SCHEMA, "n": len(ds),
            "batches": sorted({p.get("batch") for p in ds}, key=str),
            "verdicts": census,
            "claim_rate": sum(1 for v in verdicts
                              if v in (wire.RELEASE, wire.DENY_W)) / len(ds)
            if ds else 0.0,
            "abstain_causes": dict(causes),
            "coverage": (sum(cover) / len(cover)) if cover else None,
            "coverage_n": len(cover),
            "mae": (sum(errors) / len(errors)) if errors else None,
            "accuracy": (sum(correct) / len(correct)) if correct else None,
            "released_hazards": released_haz,
            "hazard_precision": (1 - released_haz / releases
                                 if releases else None),
            "contained": self._contained.as_dict() if self._contained else None,
            "review_queue": review,
            "versions": list(self._versions),
        }

    # ----------------------------------------------------------------- tick --
    def _values(self, metric: str, ds: list[dict], outcomes: dict):
        vals = []
        for p in ds:
            d = p["disposition"]
            pred = d.get("prediction") or {}
            y = outcomes.get(d["certificate_id"])
            if metric == "abstain":
                vals.append(1.0 if d["verdict"] == wire.ABSTAIN_W else 0.0)
            elif metric == "coverage" and y is not None:
                if pred.get("set") is not None:
                    vals.append(1.0 if y in pred["set"] else 0.0)
                elif pred.get("lo") is not None:
                    vals.append(1.0 if pred["lo"] <= float(y) <= pred["hi"]
                                else 0.0)
            elif metric == "abs_error" and y is not None \
                    and pred.get("pred") is not None:
                vals.append(abs(float(y) - float(pred["pred"])))
        return vals

    def tick(self, *, window_batches: int | None = None) -> list[Detection]:
        """Deterministic: recomputed from the journal on every call."""
        ds = self._decisions(window_batches)
        outcomes = self._outcomes()
        out = []
        for spec in self._detectors:
            if spec.metric == "hazard_release":
                # Epoch-aware: a hazard release from before the last lifted
                # containment is history, not a new emergency. Counting it
                # forever would re-contain the session after every repair.
                entries = self._journal.entries()
                epoch = max((e["seq"] for e in entries
                             if e["payload"].get("kind") == "uncontainment"),
                            default=-1)
                hits = 0
                for e in entries:
                    pl = e["payload"]
                    if (e["seq"] > epoch and pl.get("kind") == "decision"
                            and pl["disposition"]["verdict"] == wire.RELEASE
                            and pl.get("hazard_codes")):
                        hits += 1
                out.append(Detection(
                    spec.name, spec.lever, spec.metric, alarm=hits > 0,
                    severity="critical" if hits else "info", stat=float(hits),
                    threshold=0.0, target=0.0, n=len(ds), mde=0.0,
                    detail=f"{hits} release(s) on external hazard rows"))
                continue
            vals = self._values(spec.metric, ds, outcomes)
            n = len(vals)
            if n < spec.min_n:
                out.append(Detection(
                    spec.name, spec.lever, spec.metric, alarm=False,
                    severity="info", stat=0.0, threshold=spec.threshold,
                    target=spec.target, n=n, mde=spec.mde,
                    detail=f"insufficient n ({n} < {spec.min_n})"))
                continue
            s_hi = s_lo = 0.0
            for x in vals:
                s_hi = max(0.0, s_hi + x - (spec.target + spec.slack))
                s_lo = max(0.0, s_lo + (spec.target - spec.slack) - x)
            stat = max(s_hi, s_lo)
            alarm = stat > spec.threshold
            out.append(Detection(
                spec.name, spec.lever, spec.metric, alarm=alarm,
                severity="warning" if alarm else "info", stat=stat,
                threshold=spec.threshold, target=spec.target, n=n,
                mde=spec.mde,
                detail=(f"CUSUM {stat:.2f} vs h={spec.threshold:.2f}; "
                        f"target={spec.target:.3f}, slack={spec.slack:.3f}, "
                        f"n={n}, MDE={spec.mde}")))
        return out

    def auto_act(self, detections, *, batch=None) -> list[Containment]:
        """The entire automatic action set: contain on a released hazard.
        ACI updates already happened in observe()."""
        actions = []
        for det in detections:
            if det.metric == "hazard_release" and det.alarm:
                actions.append(self.contain(det.detail, batch=batch))
        return actions

    # ----------------------------------------------------------- containment --
    def contain(self, reason: str, *, batch=None) -> Containment:
        """UNCONDITIONAL and TIGHTEN-ONLY: releases become asks."""
        if self._contained is not None:
            return self._contained
        body = {"reason": reason, "batch": batch, "at": _now()}
        c = Containment(containment_id=_sha(body), **body)
        self._contained = c
        self._journal.append({"kind": "containment", **c.as_dict()})
        return c

    def uncontain(self, approval: Approval, *,
                  reason="approved release from containment") -> None:
        """Loosening: requires an external signature for THIS containment."""
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

    # ------------------------------------------------------------ retraction --
    def affected_certificates(self, *, model_digest=None,
                              policy_digest=None) -> list[str]:
        out = []
        for p in self._journal.payloads("decision"):
            d = p["disposition"]
            if model_digest is not None and d["model_digest"] != model_digest:
                continue
            if policy_digest is not None \
                    and d["policy_digest"] != policy_digest:
                continue
            out.append(d["certificate_id"])
        return out

    def retract(self, *, model_digest=None, policy_digest=None,
                reason="version retracted") -> Retraction:
        already = {cid for p in self._journal.payloads("retraction")
                   for cid in p["certificate_ids"]}
        ids = tuple(cid for cid in self.affected_certificates(
            model_digest=model_digest, policy_digest=policy_digest)
            if cid not in already)
        body = {"reason": reason, "certificate_ids": list(ids),
                "at": _now(), "model_digest": model_digest,
                "policy_digest": policy_digest}
        r = Retraction(retraction_id=_sha(body), reason=reason,
                       certificate_ids=ids, at=body["at"])
        self._journal.append({"kind": "retraction", **r.as_dict(),
                              "model_digest": model_digest,
                              "policy_digest": policy_digest})
        return r

    # ------------------------------------------------------- change control --
    def evaluate(self, change: ChangeSet, *, oracle, ledger=None,
                 tol: float = 0.0) -> Evaluation:
        if not isinstance(change, ChangeSet):
            raise TypeError("evaluate takes a ChangeSet")
        if change.policy is None and change.estimator is None:
            raise ValueError("a ChangeSet must change a policy or a model")
        ds = self._journal.payloads("decision")
        if not ds:
            raise ValueError("no decisions in the journal to evaluate")
        rows = [p["row"] for p in ds]
        facts = [p["disposition"]["facts"] for p in ds]
        outcomes = self._outcomes()

        cand_dm = None
        if change.estimator is not None:
            cand_dm = self._dm.with_estimator(
                change.estimator,
                calibration_rows=change.calibration_rows,
                calibration_y=change.calibration_y,
                alpha=self._dm.kappa.alpha if self._dm.kappa else None)

        base_v, cand_v = [], []
        for row, f in zip(rows, facts):
            if cand_dm is not None:
                base_v.append(self._dm.dispose(row).verdict)
                cand_v.append(cand_dm.dispose(row).verdict)
            else:
                base_v.append(self._dm.policy.dispose_facts(f)[0])
                cand_v.append((change.policy or self._dm.policy)
                              .dispose_facts(f)[0])

        hz = []
        for p in ds:
            if ledger is not None:
                hz.append(bool(ledger.hits(p["row"])))
            else:
                hz.append(bool(p.get("hazard_codes")))

        y = [outcomes.get(p["disposition"]["certificate_id"]) for p in ds]
        safe, dec = [], []
        for row, yy in zip(rows, y):
            if yy is None:
                safe.append(None)
                dec.append(False)
            else:
                s, d = _oracle_answer(oracle, row, yy)
                safe.append(s)
                dec.append(d)

        before = _metrics(base_v, hz, y, safe, dec)
        after = _metrics(cand_v, hz, y, safe, dec)
        added = sum(1 for b, c in zip(base_v, cand_v)
                    if c == wire.RELEASE and b != wire.RELEASE)
        may_loosen = added > 0
        gates = (
            ChangeGate("hazard_releases_not_increased",
                       after["released_hazards"] <= before["released_hazards"],
                       before["released_hazards"], after["released_hazards"]),
            ChangeGate("precision_non_regression",
                       _ge(after["precision"], before["precision"], tol),
                       before["precision"], after["precision"]),
            ChangeGate("recall_non_regression",
                       _ge(after["recall"], before["recall"], tol),
                       before["recall"], after["recall"]),
            ChangeGate("improves_something",
                       (after["released_hazards"] < before["released_hazards"]
                        or _gt(after["precision"], before["precision"])
                        or _gt(after["recall"], before["recall"])),
                       None, None),
        )
        passed = all(g.passed for g in gates)
        return Evaluation(
            change_digest=change.digest(self._dm.features), passed=passed,
            may_loosen=may_loosen, added_releases=added, gates=gates,
            before=before, after=after, n=len(ds),
            n_observed=before["n_observed"])

    def promote(self, change: ChangeSet, *, oracle, approval: Approval | None = None,
                ledger=None, tol: float = 0.0) -> Promotion:
        ev = self.evaluate(change, oracle=oracle, ledger=ledger, tol=tol)
        before = self._version(self._dm)

        def _record(approved: bool, reason: str, after=None):
            self._journal.append({
                "kind": "change", "change_digest": ev.change_digest,
                "lever": change.lever, "reason": change.reason,
                "promoted": approved, "detail": reason,
                "before": before, "after": after,
                "approval": approval.as_dict() if approval else None,
                "evaluation": ev.as_dict()})

        if not ev.passed:
            reason = ("gate failed: "
                      + ", ".join(g.name for g in ev.gates if not g.passed))
            _record(False, reason)
            return Promotion(False, reason, ev.change_digest, before, None,
                             ev, approval)

        if ev.may_loosen:
            ok = (approval is not None and self._approval_verifier is not None
                  and self._approval_verifier(approval, ev.change_digest))
            if not ok:
                reason = (f"approval required: candidate adds "
                          f"{ev.added_releases} release(s)")
                _record(False, reason)
                return Promotion(False, reason, ev.change_digest, before, None,
                                 ev, approval)

        new_dm = self._dm
        if change.estimator is not None:
            new_dm = new_dm.with_estimator(
                change.estimator,
                calibration_rows=change.calibration_rows,
                calibration_y=change.calibration_y,
                alpha=self._dm.kappa.alpha if self._dm.kappa else None)
        if change.policy is not None:
            new_dm = new_dm.with_policy(change.policy)
        self._dm = new_dm
        after = self._version(new_dm)
        self._versions.append(after)
        _record(True, "promoted", after)
        return Promotion(True, "promoted", ev.change_digest, before, after,
                         ev, approval, new_dm)


__all__ = ["Journal", "LiveSession", "DetectorSpec", "Detection",
           "default_detectors", "Containment", "Retraction", "ChangeSet",
           "ChangeGate", "Evaluation", "Promotion", "Approval",
           "ApprovalAuthority", "SCHEMA"]
