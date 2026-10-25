"""mjolnir_ml._digest — content addressing shared by every calibrated index.

Digests are over canonical JSON (sorted keys, no whitespace) so a table and
its digest cannot drift. Estimator digests include the fitted-state attributes
we can read generically; they are audit aids, not a substitute for pinning the
artifact itself.
"""
from __future__ import annotations

import hashlib
import json


def digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=repr, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def estimator_digest(estimator) -> str:
    """Deterministic class + params + readable fitted-state digest."""
    parts = [type(estimator).__module__ + "." + type(estimator).__name__]
    if hasattr(estimator, "get_params"):
        try:
            params = estimator.get_params(deep=False)
            parts.append(json.dumps({k: repr(v)
                                     for k, v in sorted(params.items())},
                                    sort_keys=True))
        except Exception:                                   # noqa: BLE001
            parts.append(repr(estimator))
    else:
        parts.append(repr(estimator))
    for attr in ("classes_", "coef_", "intercept_", "n_features_in_",
                 "feature_importances_"):
        val = getattr(estimator, attr, None)
        if val is not None:
            parts.append(f"{attr}={val!r}")
    return digest(parts)


__all__ = ["digest", "estimator_digest"]
