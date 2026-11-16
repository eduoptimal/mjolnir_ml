"""Ω-SR (Omega Selective Release).

A release is pinned against a policy certified on calibration data from an
independent truth oracle: release when ``P(supported | evidence) >= tau``;
otherwise abstain. ``tau`` is selected so selective risk is at most ``alpha``.
When clusters are supplied, the bound resamples evidence clusters instead of
rows. The module cannot enforce oracle independence; that is a deployment
invariant.

The module is dependency-free. It accepts calibrated probabilities; it does not
fit a language model. Use ``release_claim`` to plug in a natural-language scorer.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from mjolnir_ml.release.pair import omega_pair
from mjolnir_ml.calibration.risk import crc_threshold
from mjolnir_ml.core import wire

ABSTENTION_SCHEMA = "mjolnir.ml.abstention/1"


class AbstentionError(ValueError):
    """The release calibrator refused (malformed or insufficient data)."""


@dataclass(frozen=True)
class AbstentionCertificate:
    schema: str
    alpha: float
    delta: float
    tau: float | None
    coverage: float | None
    risk: float | None
    upper: float | None
    n: int
    n_clusters: int | None
    mode: str

    def as_dict(self) -> dict:
        return {
            "schema": self.schema, "alpha": self.alpha, "delta": self.delta,
            "tau": self.tau, "coverage": self.coverage, "risk": self.risk,
            "upper": self.upper, "n": self.n,
            "n_clusters": self.n_clusters, "mode": self.mode,
        }


@dataclass(frozen=True)
class ReleaseDecision:
    verdict: str
    probability: float
    tau: float | None
    cluster: str | None = None
    claim: str | None = None

    @property
    def released(self) -> bool:
        return self.verdict == wire.RELEASE

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "probability": self.probability,
                "tau": self.tau, "cluster": self.cluster,
                "claim": self.claim}


def _check(scores, outcomes, clusters=None):
    scores = [float(s) for s in scores]
    outcomes = [int(y) for y in outcomes]
    if len(scores) != len(outcomes) or not scores:
        raise AbstentionError("scores and outcomes must be equal-length and non-empty")
    if any(not math.isfinite(s) for s in scores):
        raise AbstentionError("scores must be finite")
    if any(s < 0.0 or s > 1.0 for s in scores):
        raise AbstentionError("scores must be probabilities in [0,1]")
    if any(y not in (0, 1) for y in outcomes):
        raise AbstentionError("outcomes must be 0/1")
    if clusters is not None:
        clusters = list(clusters)
        if len(clusters) != len(scores):
            raise AbstentionError("clusters must align with scores")
    return scores, outcomes, clusters


def _cluster_upper(scores, outcomes, clusters, tau, delta=0.05,
                   draws=500, seed=0):
    selected = [i for i, s in enumerate(scores) if s >= tau]
    if not selected:
        return 1.0
    groups = {}
    for i in selected:
        groups.setdefault(clusters[i], []).append(outcomes[i])
    if len(groups) < 2:
        return 1.0
    keys = list(groups)
    rng = random.Random(seed)
    risks = []
    for _ in range(draws):
        loss = n = 0
        for key in rng.choices(keys, k=len(keys)):
            vals = groups[key]
            loss += sum(1 - y for y in vals)
            n += len(vals)
        if n:
            risks.append(loss / n)
    if not risks:
        return 1.0
    risks.sort()
    return risks[min(len(risks) - 1, int((1.0 - delta) * len(risks)))]


class RiskControlledReleaseGate:
    """Release claims only under a calibrated selective-risk budget.

    This is the independent Ω-SR release gate. It accepts a calibrated
    probability ``P(supported | claim, evidence)`` and releases only when that
    probability clears a threshold selected on held-out calibration data.

    ``fit`` chooses the largest coverage threshold whose selective-risk upper
    bound is at most ``alpha``. Without clusters it uses conformal risk control
    under exchangeability. With clusters it resamples evidence clusters instead
    of rows. ``decide`` returns ``release`` or ``abstain``.

    The gate does not fit or call a language model, determine truth, prove that
    evidence entails a claim, supply L3 authority, or enforce oracle
    independence. The caller must provide calibrated scores, independent
    calibration and audit labels, and evidence clusters. If no threshold meets
    ``alpha``, ``tau`` is ``None`` and every decision abstains (fail closed).
    """

    def __init__(self, alpha: float = 0.10, delta: float = 0.05,
                 draws: int = 500, seed: int = 0):
        if not (0.0 < float(alpha) < 1.0):
            raise AbstentionError("alpha must be in (0,1)")
        if not (0.0 < float(delta) < 1.0):
            raise AbstentionError("delta must be in (0,1)")
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.draws = int(draws)
        self.seed = int(seed)
        self._cert: AbstentionCertificate | None = None

    @property
    def certificate(self) -> AbstentionCertificate:
        if self._cert is None:
            raise AbstentionError("call fit() before certificate")
        return self._cert

    @property
    def tau(self) -> float | None:
        return self.certificate.tau

    def fit(self, scores, outcomes, clusters=None):
        scores, outcomes, clusters = _check(scores, outcomes, clusters)
        if clusters is None:
            tau = crc_threshold(scores, [1 - y for y in outcomes],
                                alpha=self.alpha, side="high")
            if tau is None:
                self._cert = AbstentionCertificate(
                    ABSTENTION_SCHEMA, self.alpha, self.delta, None,
                    None, None, 1.0, len(scores), None, "iid")
                return self
            selected = [i for i, s in enumerate(scores) if s >= tau]
            risk = sum(1 - outcomes[i] for i in selected) / len(selected)
            self._cert = AbstentionCertificate(
                ABSTENTION_SCHEMA, self.alpha, self.delta, tau,
                len(selected) / len(scores), risk, None, len(scores), None,
                "iid")
            return self

        best = None
        best_upper = 1.0
        for tau in sorted(set(scores), reverse=True):
            selected = [i for i, s in enumerate(scores) if s >= tau]
            cov = len(selected) / len(scores)
            upper = _cluster_upper(scores, outcomes, clusters, tau,
                                   self.delta, self.draws, self.seed)
            if upper <= self.alpha and (best is None or cov > best[1]):
                best = (tau, cov, selected)
                best_upper = upper
        if best is None:
            self._cert = AbstentionCertificate(
                ABSTENTION_SCHEMA, self.alpha, self.delta, None,
                None, None, 1.0, len(scores), len(set(clusters)),
                "cluster")
            return self
        tau, cov, selected = best
        risk = sum(1 - outcomes[i] for i in selected) / len(selected)
        self._cert = AbstentionCertificate(
            ABSTENTION_SCHEMA, self.alpha, self.delta, tau, cov, risk,
            best_upper, len(scores), len(set(clusters)), "cluster")
        return self

    def decide(self, probability: float, cluster: str | None = None,
               claim: str | None = None) -> ReleaseDecision:
        p = float(probability)
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise AbstentionError("probability must be finite and in [0,1]")
        tau = self.tau
        verdict = (wire.RELEASE if tau is not None and p >= tau
                   else wire.ABSTAIN_W)
        return ReleaseDecision(verdict, p, tau, cluster, claim)

    def decide_many(self, probabilities, clusters=None) -> list:
        probabilities = list(probabilities)
        if clusters is not None:
            clusters = list(clusters)
            if len(clusters) != len(probabilities):
                raise AbstentionError("clusters must align with probabilities")
        return [self.decide(p, None if clusters is None else clusters[i])
                for i, p in enumerate(probabilities)]

    def release_claim(self, claim: str, evidence: str, scorer, *,
                      cluster: str | None = None) -> ReleaseDecision:
        """Apply the gate to a natural-language claim.

        ``scorer(claim, evidence)`` must return a calibrated
        ``P(supported | claim, evidence)``. The gate does not inspect the
        scorer or validate its independence; the caller owns that boundary.
        """
        p = scorer(claim, evidence)
        return self.decide(p, cluster=cluster, claim=claim)

    def audit(self, scores, outcomes, clusters=None) -> dict:
        scores, outcomes, clusters = _check(scores, outcomes, clusters)
        decisions = self.decide_many(scores, clusters)
        released = [d.released for d in decisions]
        n = len(scores)
        k = sum(1 for r in released if r)
        losses = sum(1 - y for y, r in zip(outcomes, released) if r)
        risk = losses / k if k else None
        if clusters is not None and k:
            upper = _cluster_upper(scores, outcomes, clusters, self.tau,
                                   self.delta, self.draws, self.seed)
        else:
            upper = None
        pair = omega_pair([d.verdict for d in decisions], outcomes,
                          [True] * n)
        return {
            "certificate": self.certificate.as_dict(),
            "coverage": k / n,
            "n_released": k,
            "n_abstained": n - k,
            "abstention_rate": 1 - k / n,
            "selective_risk": risk,
            "risk_upper": upper,
            "pair": pair.as_dict(),
            "decisions": [d.as_dict() for d in decisions],
        }

    def audit_decisions(self, decisions, outcomes) -> dict:
        outcomes = list(outcomes)
        if len(decisions) != len(outcomes):
            raise AbstentionError("decisions and outcomes must align")
        released = [d.released for d in decisions]
        n = len(outcomes)
        k = sum(1 for r in released if r)
        losses = sum(1 - y for y, r in zip(outcomes, released) if r)
        pair = omega_pair([d.verdict for d in decisions], outcomes,
                          [True] * n)
        return {
            "coverage": k / n,
            "n_released": k,
            "n_abstained": n - k,
            "selective_risk": losses / k if k else None,
            "pair": pair.as_dict(),
        }


__all__ = ["ABSTENTION_SCHEMA", "AbstentionCertificate", "AbstentionError",
           "ReleaseDecision", "RiskControlledReleaseGate"]
