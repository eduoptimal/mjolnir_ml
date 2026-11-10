"""storage.local — LocalStore: the filesystem backend (today's default)."""
from __future__ import annotations
import os
from .interface import Store


def _safe_key(key: str) -> str:
    k = key.replace("\\", "/").strip("/")
    if not k or k.startswith("/") or ".." in k.split("/"):
        raise ValueError(f"unsafe storage key: {key!r}")
    return k


class LocalStore(Store):
    def __init__(self, root: str):
        self._root = os.path.abspath(root)

    @property
    def root(self) -> str:
        return self._root

    def _path(self, key: str) -> str:
        return os.path.join(self._root, _safe_key(key))

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def read_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as fh:
            return fh.read()

    def write_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p) or self._root, exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as fh:          # atomic replace
            fh.write(data)
        os.replace(tmp, p)

    def append_bytes(self, key: str, data: bytes) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p) or self._root, exist_ok=True)
        with open(p, "ab") as fh:
            fh.write(data)

    def list(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self._root
        if not os.path.isdir(base):
            return []
        out = []
        for dp, _dirs, fs in os.walk(base):
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), self._root)
                out.append(rel.replace(os.sep, "/"))
        return sorted(out)
