"""storage.config — resolve the active Store.

Today: LocalStore at `MJ_STORAGE` (default: this directory). Tomorrow: the same
interface over an object bucket. Nothing else in mjolnir changes when that flips.

Backends (see mjolnir_prover/storage/README.md) implement `Store` and are
selected here; the read path must stay deterministic and offline-verifiable.
"""
from __future__ import annotations
import os
from functools import lru_cache
from .interface import Store
from .local import LocalStore

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.environ.get("MJ_STORAGE", os.path.join(_HERE, "data"))


@lru_cache(maxsize=None)
def get_store(root: str | None = None) -> Store:
    """Return the active store. `store://s3/...` style URIs are reserved for a
    future cloud backend; anything else is a filesystem path."""
    loc = root or DEFAULT_ROOT
    if loc.startswith("store://"):
        raise NotImplementedError(
            f"cloud (bucket) backend not wired yet: {loc!r} "
            f"(implement CloudStore in mjolnir_prover/storage/cloud.py)")
    if loc.startswith("db://"):
        raise NotImplementedError(
            f"centralized DB backend not wired yet: {loc!r} "
            f"(implement DbStore in mjolnir_prover/storage/db.py)")
    return LocalStore(loc)
