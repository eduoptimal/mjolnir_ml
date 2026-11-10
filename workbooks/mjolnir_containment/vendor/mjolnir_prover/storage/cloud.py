"""storage.cloud — PLANNED object-bucket backend (S3/GCS/Blob).

Implement `CloudStore(Store)` here with the same method set as `LocalStore`,
LISTS with pagination, and **append semantics** for ledgers (write-ahead object
or per-run segment). Then select it in `config.get_store` and set
`MJ_STORAGE=store://s3/bucket/prefix`. No engine change is required.
"""
from __future__ import annotations
from .interface import Store


class CloudStore(Store):  # pragma: no cover - placeholder
    def __init__(self, uri: str):
        raise NotImplementedError("cloud backend planned; see module docstring")
