"""mjolnir_ml.hazards — an EXTERNAL hazard oracle + completeness measurement.

`CERTIFIABILITY.md` (beta_release) turns "is the policy complete?" — which is
not solvable — into a measured, bounded number:

    H          an external hazard oracle: predicates over rows, computed from
               evidence outside the policy (domain geometry, a regulatory
               hazard list, an independent simulator). NEVER the policy itself.
    ledger     the declared hazard set, versioned and digested, union-only:
               retirement requires a refutation record, so completeness is
               non-decreasing under legitimate growth.
    precision  1 − (releases on hazard rows) / releases  — no false certificate
    recall     (hazard rows not released) / hazard rows  — hazards caught
    coverage   releases / rows
    ceiling    the share of hazard rows whose hazard is OBSERVABLE in the
               declared representation. Hazards outside σ(X) are at the Chow
               ceiling: no policy over those inputs can catch them, and recall
               above the ceiling means the "hidden" bit leaked into X.

The measurement is deliberately a pair, never a scalar: an always-abstain
policy has recall 0 (honest) and a perfect-looking error rate (dishonest); the
pair plus coverage keeps both visible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mjolnir_ml.pair import wilson
from mjolnir_ml.vendors import wire


@dataclass(frozen=True)
class Hazard:
    code: str
    predicate: object                 # row -> bool, external to the policy
    provenance: str
    observable: bool = True           # False: not a function of the model inputs


class HazardLedger:
    """Union-only hazard ledger. Versioned, digested, refutation-recorded."""

    SCHEMA = "mjolnir.ml.hazard-ledger/1"

    def __init__(self, version: str = "hazard-ledger@1"):
        self.version = version
        self._active: dict = {}
        self._retired: dict = {}

    # ------------------------------------------------------------ authoring --
    def add(self, code: str, predicate, provenance: str,
            observable: bool = True) -> "HazardLedger":
        if not isinstance(code, str) or not code.isidentifier():
            raise ValueError(f"hazard code {code!r} must be an identifier")
        if not callable(predicate):
            raise ValueError(f"hazard {code!r}: predicate must be callable")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError(f"hazard {code!r}: provenance is required "
                             f"(an unreferenced hazard is an opinion)")
        if code in self._active or code in self._retired:
            raise ValueError(f"hazard {code!r} already in the ledger "
                             f"(codes are never reused)")
        self._active[code] = Hazard(code, predicate, provenance, observable)
        return self

    def retire(self, code: str, refutation: str) -> "HazardLedger":
        """Union-only growth: removal is recorded WITH its refutation."""
        if code not in self._active:
            raise KeyError(f"hazard {code!r} is not active")
        if not isinstance(refutation, str) or not refutation.strip():
            raise ValueError(f"retiring {code!r} needs a refutation record")
        self._retired[code] = {**self._active.pop(code).__dict__,
                               "refutation": refutation}
        return self

    # --------------------------------------------------------------- query --
    def hits(self, row) -> list:
        """Active hazards the row satisfies (hidden ones included)."""
        out = []
        for h in self._active.values():
            try:
                if h.predicate(row):
                    out.append(h)
            except Exception as ex:                      # noqa: BLE001
                raise RuntimeError(
                    f"hazard {h.code!r} raised on a row: "
                    f"{type(ex).__name__}: {ex}") from ex
        return out

    def hazard_codes(self, row) -> list:
        return [h.code for h in self.hits(row)]

    @property
    def digest(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True,
                          separators=(",", ":"), default=repr).encode()
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    def as_dict(self) -> dict:
        return {"schema": self.SCHEMA, "version": self.version,
                "active": {c: {"provenance": h.provenance,
                               "observable": h.observable}
                           for c, h in self._active.items()},
                "retired": {c: {"provenance": r["provenance"],
                                "refutation": r["refutation"]}
                            for c, r in self._retired.items()}}


def measure_completeness(dispositions, rows, ledger: HazardLedger) -> dict:
    """Measure a batch of dispositions against the external hazard ledger.

    A RELEASE on a hazard row is a false certificate (the only unsafe
    direction here); ask/deny/abstain all count as "caught". Precision is the
    certificate-soundness axis, recall the completeness axis, coverage the
    price. Everything is reported per hazard code as well as pooled.
    """
    if len(dispositions) != len(rows):
        raise ValueError(f"{len(dispositions)} dispositions for {len(rows)} rows")
    n = len(rows)
    n_rel = n_haz = n_caught = n_false = n_haz_obs = 0
    n_hidden_only = n_caught_obs = n_caught_hidden = 0
    per = {}
    for disp, row in zip(dispositions, rows):
        v = disp.verdict if hasattr(disp, "verdict") else disp
        hits = ledger.hits(row)
        released = v == wire.RELEASE
        if released:
            n_rel += 1
        for h in hits:
            d = per.setdefault(h.code, {"rows": 0, "released": 0,
                                        "observable": h.observable})
            d["rows"] += 1
            if released:
                d["released"] += 1
        if hits:
            n_haz += 1
            observable = [h for h in hits if h.observable]
            if observable:
                n_haz_obs += 1
                if not released:
                    n_caught_obs += 1
            else:
                n_hidden_only += 1
                if not released:
                    n_caught_hidden += 1
            if released:
                n_false += 1
            else:
                n_caught += 1
    prec = (1.0 - n_false / n_rel) if n_rel else None
    rec = (n_caught / n_haz) if n_haz else None
    ceiling = (n_haz_obs / n_haz) if n_haz else None
    return {
        "schema": "mjolnir.ml.completeness/1",
        "ledger_version": ledger.version, "ledger_digest": ledger.digest,
        "n": n, "n_release": n_rel, "n_hazard": n_haz,
        "n_caught": n_caught, "n_false_certificate": n_false,
        "precision": prec, "recall": rec,
        "coverage": n_rel / n if n else 0.0,
        "precision_ci": wilson(n_rel - n_false, n_rel) if n_rel else (None, None),
        "recall_ci": wilson(n_caught, n_haz) if n_haz else (None, None),
        # The independence ceiling: the share of hazard rows with at least one
        # OBSERVABLE hit. A hazard outside sigma(X) cannot be TARGETED; it can
        # still be caught incidentally when it correlates with X, so
        # `within_ceiling` is guaranteed only under independence (the paper's
        # hidden-bit construction makes it exact).
        "ceiling": ceiling,
        "within_ceiling": (rec is None or ceiling is None
                           or rec <= ceiling + 1e-9),
        "n_observable_hazard": n_haz_obs,
        "n_hidden_only": n_hidden_only,
        "recall_observable": (n_caught_obs / n_haz_obs) if n_haz_obs else None,
        "recall_hidden_only": (n_caught_hidden / n_hidden_only)
                              if n_hidden_only else None,
        "per_hazard": per,
    }


__all__ = ["Hazard", "HazardLedger", "measure_completeness"]
