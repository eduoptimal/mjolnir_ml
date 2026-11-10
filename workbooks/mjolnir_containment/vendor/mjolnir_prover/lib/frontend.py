"""mjolnir_prover.lib.frontend — the Lean frontend dispatch.

DEFAULT (and only pipeline in the trusted path) is the IR frontend:
`frontend_lean2.lower_theorem` -> claim_ir.Claim -> `claim_ir.dispose_claims`.
Structured claims, no string round-trip.

The old regex frontend (`mjolnir_prover.legacy.frontend_lean`) is RETIRED. It is no
longer imported here at module load and no longer used by the gate: it lives in
`mjolnir_prover.legacy` for archaeology and can still be requested explicitly for a
one-off differential (`verdict(stmt, "regex")`), but nothing default depends on
it.

Nothing here is in the proof TCB — this module only routes a theorem to a
verdict; the trusted floor is term.py + intervals.py.
"""
from __future__ import annotations

from .frontend_lean2 import lower_theorem as _lower_ir
from .claim_ir import dispose_claims as _dispose_ir

DEFAULT = "ir"
FRONTENDS = ("ir", "regex")          # "regex" = retired oracle, lazy-loaded


def verdict(stmt, frontend=DEFAULT, meta=("lean", "lean", "src", 0)):
    """Return the frontend's verdict for one theorem: CERTIFIED / REFUTED /
    UNVERIFIABLE, or 'skip' when the frontend cannot lower it."""
    if frontend == "ir":
        lr = _lower_ir(stmt)
        if lr is None:
            return "skip"
        claims, meta = lr
        u = _dispose_ir(claims)[0]
        alt = meta.get("alt_claims")
        if u == "UNVERIFIABLE" and alt:
            # floor-division lowering stalled; the exact-rational lowering is a
            # sound fallback (both are consequences of the same premises).
            u2 = _dispose_ir(alt)[0]
            if u2 != "UNVERIFIABLE":
                return u2
        return u
    if frontend == "regex":
        # retired oracle: not shipped in the release (kept in dev only)
        try:
            from mjolnir_prover.legacy.frontend_lean import mine_theorem as _mine_regex
        except Exception:
            raise ValueError("the regex frontend is retired and not shipped in "
                             "this release; use frontend='ir'")
        from . import certificate as _C
        rel, repo, commit, idx = meta
        m = _mine_regex(stmt, rel, repo, commit, idx)
        return "skip" if not m else _C.unit_vc(m["canonical_unit"])["status"]
    raise ValueError(f"unknown frontend: {frontend!r} (have {FRONTENDS})")


def claims(stmt, frontend="ir"):
    """The lowerable claim list (IR), or None. Only the IR frontend has an IR."""
    if frontend != "ir":
        raise ValueError("only the IR frontend produces claims")
    lr = _lower_ir(stmt)
    return None if lr is None else lr[0]
