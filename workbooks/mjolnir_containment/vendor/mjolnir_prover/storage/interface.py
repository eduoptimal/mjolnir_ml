"""storage.interface — the abstract data-plane contract.

The registry, corpus, theories, models and logs are DATA, addressed by a
namespaced relative key (e.g. "registry/confirmations.jsonl"). Any backend that
satisfies `Store` can host them: LocalStore (today), S3Store/GCSStore (planned).
Code never depends on the backend — only on this interface.

Key discipline:
  - keys are POSIX-style relative paths, never absolute, never "../";
  - reads are byte-exact; JSONL helpers preserve append order (append-only
    ledgers must never be rewritten in place by a whole-file write);
  - a backend MUST be deterministic and offline-checkable for the read path.
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Iterable, Iterator


class Store(ABC):
    """Abstract, namespaced, append-capable object store."""

    # ---- identity ----
    @property
    @abstractmethod
    def root(self) -> str:
        """Human-readable location of this store (for reports/provenance)."""

    # ---- bytes ----
    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def append_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """Sorted keys under prefix (deterministic order)."""

    # ---- jsonl (append-only registries/ledgers) ----
    def read_jsonl(self, key: str) -> list[dict]:
        if not self.exists(key):
            return []
        out = []
        for line in self.read_bytes(key).splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def write_jsonl(self, key: str, rows: Iterable[dict]) -> None:
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        self.write_bytes(key, data.encode("utf-8"))

    def append_jsonl(self, key: str, row: dict) -> None:
        self.append_bytes(key, (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))

    # ---- json ----
    def read_json(self, key: str):
        return json.loads(self.read_bytes(key)) if self.exists(key) else None

    def write_json(self, key: str, obj) -> None:
        self.write_bytes(key, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

    # ---- iteration ----
    def iter_jsonl(self, key: str) -> Iterator[dict]:
        for row in self.read_jsonl(key):
            yield row
