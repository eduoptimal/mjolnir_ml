"""mjolnir_prover.lib.paths — the single place that resolves DATA locations.

Code never hardcodes a data path. Everything durable lives in the detached
`mjolnir_prover.storage` module (JSON on disk now; a bucket / centralized DB later),
resolved here so the same code runs locally or as a client against a service.

Default data root:  <package>/storage/data/{registry,corpus,theories,models,logs,out}
Env overrides: MJ_STORAGE (data root), MJ_VENDOR (external test corpora).
"""
from __future__ import annotations
import os

_HERE = os.path.dirname(os.path.abspath(__file__))          # <release>/mjolnir_prover/lib
PKG = os.path.dirname(_HERE)                                # <release>/mjolnir
RELEASE_ROOT = os.path.dirname(PKG)                         # <release>
STORAGE = os.environ.get("MJ_STORAGE", os.path.join(PKG, "storage", "data"))
VENDOR = os.environ.get("MJ_VENDOR", os.path.join(STORAGE, "repos"))


def storage(*parts: str) -> str:
    return os.path.join(STORAGE, *parts)


def registry(*parts: str) -> str:
    return storage("registry", *parts)


def corpus(*parts: str) -> str:
    return storage("corpus", *parts)


def theories(*parts: str) -> str:
    return storage("theories", *parts)


def models(*parts: str) -> str:
    return storage("models", *parts)


def logs(*parts: str) -> str:
    return storage("logs", *parts)


def out(*parts: str) -> str:
    return storage("out", *parts)
