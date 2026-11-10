"""mjolnir_prover/storage/db.py — PLANNED centralized-DB client backend.

The end state: mjolnir as a local library **client** against a central service.
Implement `DbStore(Store)` here (keys become rows/objects; append_jsonl becomes
an append/insert; list becomes a query). Select it in `config.get_store` and set
`MJ_STORAGE=db://<dsn>`. Keep the read path deterministic and offline-cacheable;
the registry stays append-only.
"""
from __future__ import annotations
from .interface import Store


class DbStore(Store):  # pragma: no cover - placeholder
    def __init__(self, dsn: str):
        raise NotImplementedError("centralized DB client planned; see docstring")
