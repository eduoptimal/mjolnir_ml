"""mjolnir_ml.readiness — the one-shot deployment gate. No state.

`readiness(dm, rows, ...)` answers exactly one question: *may this
(estimator, policy, calibration) triple be deployed?* It disposes a held-out
batch against a pinned oracle, scores the Ω pair and (optionally) external
hazards, replays certificates, and returns a `ReadinessReport` with per-gate
verdicts and an evidence bundle.

It mutates nothing: no calibration, no bounds, no policy, no journal. A
deployment is a claim about a *version*, and the report pins the digests of
that version.

    report = readiness(dm, rows, safe=..., decidable=..., ledger=..., key=...)
    report.approved      # bool
    report.gates         # every gate, its value, its threshold
    report.readiness_id  # content address of gates + metrics + digests
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field

from mjolnir_ml.model import DecisionModel
from mjolnir_ml.report import report_data
from mjolnir_ml.vendors import wire


class ReadinessError(ValueError):
    """The readiness request itself is malformed (not a failed gate)."""


@dataclass(frozen=True)
class ReadinessThresholds:
    min_precision: float = 0.95
    min_recall: float = 0.0
    max_overclaim: int = 0
    min_coverage: float = 0.0
    max_abstain_rate: float = 1.0
    min_hazard_precision: float | None = None
    min_hazard_recall: float | None = None
    min_replay_rate: float = 0.99
    require_calibration: bool = True
    require_reachable_rules: bool = False


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    value: object
    threshold: object
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "value": self.value,
                "threshold": self.threshold, "detail": self.detail}


@dataclass(frozen=True)
class ReadinessReport:
    schema: str
    readiness_id: str
    approved: bool
    gates: tuple
    metrics: dict
    digests: dict
    evidence: dict
    created_at: str
    signature: str | None = None

    def as_dict(self) -> dict:
        return {
            "schema": self.schema, "readiness_id": self.readiness_id,
            "approved": self.approved,
            "gates": [g.as_dict() if isinstance(g, Gate) else g
                      for g in self.gates],
            "metrics": self.metrics, "digests": self.digests,
            "evidence": self.evidence, "created_at": self.created_at,
            "signature": self.signature,
        }


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str, ensure_ascii=False).encode("utf-8")


def _digest(obj) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj)).hexdigest()


def _sign(body: dict, key) -> str:
    kb = key.encode() if isinstance(key, str) else bytes(key)
    return "hmac-sha256:" + hmac.new(kb, _canonical(body),
                                     hashlib.sha256).hexdigest()


def readiness(dm: DecisionModel, rows, *, safe, decidable, ledger=None,
              key=None, thresholds: ReadinessThresholds | None = None,
              title: str = "readiness",
              threshold: float | None = None) -> ReadinessReport:
    """Evaluate a deployment candidate. `safe` is the oracle's per-row answer
    (True: release is correct; False: deny is correct; None: undecided) and
    `decidable` marks rows the oracle settles. Without an oracle there is no
    readiness: refusing to ask is not a pass."""
    if not isinstance(dm, DecisionModel):
        raise ReadinessError("readiness takes a DecisionModel")
    rows = list(rows)
    safe, decidable = list(safe), list(decidable)
    if not (len(rows) == len(safe) == len(decidable)) or not rows:
        raise ReadinessError("rows, safe and decidable must be equal-length "
                             "and non-empty")
    th = thresholds or ReadinessThresholds()

    data = report_data(dm, rows, title=title, safe=safe, decidable=decidable,
                       ledger=ledger, key=key, sample=0, threshold=threshold)
    pair = data.get("pair", {}).get("pair", {})
    certs = data.get("certificates")
    haz = data.get("hazards")
    n = len(rows)
    abstain_rate = data["decisions"]["verdicts"].get(wire.ABSTAIN_W, 0) / n
    coverage = data["decisions"]["claim_rate"]

    gates: list[Gate] = []
    if th.require_reachable_rules:
        dead = data.get("feasibility", {}).get("dead_rules", [])
        gates.append(Gate(
            "rules_reachable", not dead, list(dead), "no dead rules",
            "a declared rule that cannot fire is a contract defect; "
            "see the FEASIBILITY section"))
    gates.append(Gate("calibration_present",
                      dm.kappa is not None if th.require_calibration else True,
                      dm.kappa is not None, "required" if th.require_calibration
                      else "optional",
                      "unknown is not zero: a policy needing conformal facts "
                      "abstains without calibration"))
    over = pair.get("n_overclaim", 0)
    gates.append(Gate("overclaim", over <= th.max_overclaim, over,
                      th.max_overclaim,
                      "claims on oracle-undecidable rows"))
    prec = pair.get("prec")
    gates.append(Gate("precision", prec is not None
                      and prec >= th.min_precision, prec, th.min_precision,
                      "certified claims / claims"))
    rec = pair.get("rec")
    gates.append(Gate("recall", rec is not None and rec >= th.min_recall,
                      rec, th.min_recall, "certified claims / decidable rows"))
    gates.append(Gate("coverage", coverage >= th.min_coverage, coverage,
                      th.min_coverage, "claims / rows"))
    gates.append(Gate("abstain_rate", abstain_rate <= th.max_abstain_rate,
                      abstain_rate, th.max_abstain_rate,
                      "a deployment that abstains on everything is not one"))
    if haz is not None:
        if th.min_hazard_precision is not None:
            gates.append(Gate("hazard_precision",
                              haz["precision"] is not None
                              and haz["precision"] >= th.min_hazard_precision,
                              haz["precision"], th.min_hazard_precision,
                              "1 - releases on external hazard rows / releases"))
        if th.min_hazard_recall is not None:
            gates.append(Gate("hazard_recall",
                              haz["recall"] is not None
                              and haz["recall"] >= th.min_hazard_recall,
                              haz["recall"], th.min_hazard_recall,
                              "hazard rows not released / hazard rows"))
    if certs is not None:
        rate = certs["verified"] / certs["n"] if certs["n"] else 0.0
        gates.append(Gate("certificate_replay", rate >= th.min_replay_rate,
                          rate, th.min_replay_rate,
                          "signed certificates that verify with the key"))

    approved = all(g.passed for g in gates)
    digests = {
        "policy": dm.policy.digest,
        "model": dm.model_digest,
        "calibration": dm.kappa.digest if dm.kappa is not None else None,
        "alpha": dm.kappa.alpha if dm.kappa is not None else None,
        "ledger": ledger.digest if ledger is not None else None,
    }
    metrics = {
        "n": n, "verdicts": data["decisions"]["verdicts"],
        "abstain_causes": data["decisions"]["abstain_causes"],
        "claim_rate": coverage, "abstain_rate": abstain_rate,
        "pair": pair, "hazards": haz, "predictions": data["predictions"],
        "certificates": ({k: certs[k] for k in ("n", "verified",
                                               "replayable")}
                         if certs else None),
    }
    body = {"schema": "mjolnir.ml.readiness/1", "title": title,
            "approved": approved, "gates": [g.as_dict() for g in gates],
            "metrics": metrics, "digests": digests}
    readiness_id = _digest(body)
    signature = _sign(body, key) if key is not None else None
    report = ReadinessReport(
        schema="mjolnir.ml.readiness/1", readiness_id=readiness_id,
        approved=approved, gates=tuple(gates), metrics=metrics,
        digests=digests, evidence=data,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        signature=signature)
    return report


__all__ = ["readiness", "ReadinessReport", "ReadinessThresholds", "Gate",
           "ReadinessError"]
