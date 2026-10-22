"""mjolnir_ml.pair — the Ω-certifiability pair for decision dispositions.

Definition 1 of the companion paper, instantiated for decision models:

    claim   disposition committed an operational outcome (release | deny).
            ask/abstain are non-claims — the reject option, not a promise.
    A       the oracle settles the row (decidable).
    cert    the claim matches the oracle's required side:
              release certified  iff  safe (the action may run)
              deny     certified  iff  not safe (the refusal is right)
    overclaim  a claim on a row the oracle has NOT settled.

    prec = #cert / #claim            (vacuous: 0 claims -> ⊥, never 1)
    rec  = #cert / #A                (0 if A empty)

Two axes, read together; no scalar summary survives degenerate policies
(companion: omega_constructibility — the registered naive sum was falsified).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from mjolnir_ml.model import Disposition
from mjolnir_ml.vendors import wire


@dataclass(frozen=True)
class Pair:
    prec: float | None          # None = vacuous (⊥) — never 1
    rec: float
    n_claim: int
    n_cert: int
    n_A: int
    n: int
    n_overclaim: int            # claims on rows the oracle has not settled
    n_missed: int               # A rows left unclaimed (ask/abstain)

    def as_dict(self) -> dict:
        return {"prec": self.prec, "rec": self.rec, "n_claim": self.n_claim,
                "n_cert": self.n_cert, "n_A": self.n_A, "n": self.n,
                "n_overclaim": self.n_overclaim, "n_missed": self.n_missed}


def omega_pair(dispositions: list, safe: list, decidable: list) -> Pair:
    """(prec, rec) of a batch against a pinned oracle.

    dispositions  Disposition objects (or wire verdict strings)
    safe          ground truth per row: True (release is right) /
                  False (deny is right) / None (oracle does not settle)
    decidable     bool per row — redundant with `safe is not None` when the
                  oracle speaks via None; pass explicitly when both exist.
    Lengths must match; mismatch is a contract error (fail closed loudly).
    """
    n = len(dispositions)
    if not (len(safe) == len(decidable) == n):
        raise ValueError(f"length mismatch: {n} dispositions, "
                         f"{len(safe)} safe, {len(decidable)} decidable")
    claim = cert = over = missed = A = 0
    for d, s, dec in zip(dispositions, safe, decidable):
        v = d.verdict if isinstance(d, Disposition) else d
        is_claim = v in (wire.RELEASE, wire.DENY_W)
        if dec:
            A += 1
        if is_claim:
            claim += 1
            if not dec:
                over += 1
            elif (v == wire.RELEASE) == bool(s):
                cert += 1
        elif dec:
            missed += 1
    prec = (cert / claim) if claim else None
    rec = (cert / A) if A else 0.0
    return Pair(prec=prec, rec=rec, n_claim=claim, n_cert=cert, n_A=A, n=n,
                n_overclaim=over, n_missed=missed)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval — the paper's CI on proportions. (None, None)
    where undefined (n = 0 or vacuous precision)."""
    if n == 0 or (k is None):
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def audit(dispositions: list, safe: list, decidable: list) -> dict:
    """The pair + Wilson intervals + the verdict census — one report."""
    p = omega_pair(dispositions, safe, decidable)
    counts = {v: 0 for v in wire.WIRE}
    for d in dispositions:
        v = d.verdict if isinstance(d, Disposition) else d
        counts[v] = counts.get(v, 0) + 1
    return {
        "pair": p.as_dict(),
        "prec_ci": wilson(p.n_cert, p.n_claim),
        "rec_ci": wilson(p.n_cert, p.n_A),
        "verdicts": counts,
        "overclaim_rate": (p.n_overclaim / p.n_claim) if p.n_claim else None,
    }


def degenerate_controls(safe: list, decidable: list) -> dict:
    """The constructibility table, computed for this batch: ALWAYS claims every
    row and certifies none; NEVER claims nothing and reports (⊥, 0).

    The point of printing it beside a real policy: the operational scalar (a
    low claim rate, a low error rate) ranks NEVER safest, while the pair shows
    vacuous precision as ⊥, never 1.
    """
    n = len(safe)
    always = [wire.RELEASE] * n
    never = [wire.ABSTAIN_W] * n
    return {"ALWAYS": omega_pair(always, safe, decidable).as_dict(),
            "NEVER": omega_pair(never, safe, decidable).as_dict()}


def audit_model(dm, rows, safe: list, decidable: list) -> dict:
    """Dispose a batch and audit it against the pinned oracle."""
    disps = [dm.dispose(r) for r in rows]
    report = audit(disps, safe, decidable)
    report["degenerate"] = degenerate_controls(safe, decidable)
    return report


__all__ = ["Pair", "omega_pair", "wilson", "audit", "audit_model",
           "degenerate_controls"]
