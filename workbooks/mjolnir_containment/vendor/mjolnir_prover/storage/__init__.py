"""mjolnir_prover.storage — the persistence layer (a MODULE inside the package).

mjolnir generates artifacts (mined candidates, confirmation ledgers, corpora,
theory tables, logs, certificates). Persistence is abstracted behind `Store`:

  now   : JSON files on local disk (`data/`) — development / local library
  next  : the same keys synced to a Google Cloud Storage bucket
  later : a client to a centralized database/service

Only the backend changes; callers use `get_store()` and never touch files
directly. `mjolnir_prover.lib.paths` resolves DATA locations onto the same root, so
the engine is backend-agnostic.
"""
from .config import get_store
from .interface import Store

__all__ = ["get_store", "Store"]
