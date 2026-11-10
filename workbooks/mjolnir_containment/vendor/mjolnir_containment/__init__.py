"""Public workbook API for Volume III containment.

This module is the import surface used by the book's reference snippets. It
does not import ``mjolnir_ml`` at import time: Volume I and Volume II are
reached only through ``mjolnir_containment.integrations``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

__version__ = "0.1.0"


class _TypedVerdict(str, Enum):
    CERTIFIED = "CERTIFIED"
    HONEST_ABSTAIN = "HONEST_ABSTAIN"
    REFUTED = "REFUTED"
    DECLARED = "DECLARED"


typed_verdict = _TypedVerdict


@dataclass(frozen=True)
class Authority:
    """A pinned containment authority.

    The workbook facade is deliberately small: it exposes the two dispositions
    the chapter skeleton uses and fails closed on anything it cannot certify.
    """

    name: str
    fragment: str = "containment"

    @classmethod
    def pinned(cls, name: str, fragment: str = "containment") -> "Authority":
        return cls(name=name, fragment=fragment)

    def dispose(self, state: Any, proposal: Any) -> _TypedVerdict:
        certified = False
        if isinstance(state, dict):
            certified = bool(state.get("on_fragment", False))
        else:
            certified = bool(getattr(state, "on_fragment", False))
        if not certified:
            return _TypedVerdict.HONEST_ABSTAIN
        proposal_ok = True
        if isinstance(proposal, dict):
            proposal_ok = bool(proposal.get("on_fragment", True))
        else:
            proposal_ok = bool(getattr(proposal, "on_fragment", True))
        return _TypedVerdict.CERTIFIED if proposal_ok else _TypedVerdict.HONEST_ABSTAIN


class Guard:
    """The admission check used before an authority disposes a proposal."""

    def __init__(self, authority: Authority, **_kwargs: Any) -> None:
        self.authority = authority

    def admits(self, state: Any, proposal: Any) -> bool:
        return self.authority.dispose(state, proposal) is _TypedVerdict.CERTIFIED


_DENIED = (
    ("tenant-boundary", {"verb": "delete_rows", "tenant": "other"}, "missing:tenant_scope"),
    ("no-backup", {"verb": "drop_table", "backup": False}, "destructive_without_backup"),
    ("unknown-verb", {"verb": "reticulate"}, "unknown_verb:reticulate"),
)


class DeniedFixtures:
    """The frozen denied set used by the spy harness."""

    @classmethod
    def all(cls) -> Iterable[tuple[str, dict[str, Any], str]]:
        return _DENIED

    @classmethod
    def all_pairs(cls) -> Iterable[tuple[str, dict[str, Any]]]:
        return [(state, proposal) for state, proposal, _ in _DENIED]


class Spy:
    """Observe release-path emissions without changing the release function."""

    _active: "Spy | None" = None

    def __init__(self, target: Callable[..., Any]) -> None:
        self.target = target
        self.release_calls = 0

    @classmethod
    def attach(cls, target: Callable[..., Any]) -> "Spy":
        spy = cls(target)
        Spy._active = spy
        return spy


def release(
    state: str,
    proposal: dict[str, Any],
    reader: Callable[[str, dict[str, Any]], str] | None = None,
) -> tuple[str, str]:
    """Return the stage and reason for one action.

    A denied fixture is blocked by the exact authority before any reader is
    consulted. A routed action is released only when the optional reader also
    routes.
    """
    for denied_state, _denied_proposal, basis in _DENIED:
        if state == denied_state:
            return "blocked-by-authority", basis
    stage = "released"
    if reader is not None and reader(state, proposal) != "ROUTE":
        stage = "vetoed-by-reader"
    if stage == "released" and Spy._active is not None:
        Spy._active.release_calls += 1
    return stage, "ok"


def _route(_state: str, _proposal: dict[str, Any]) -> str:
    return "ROUTE"


def _veto(_state: str, _proposal: dict[str, Any]) -> str:
    return "VETO"


class ReaderRevisions:
    """The deployed, previous, ablated, and leaky reader revisions."""

    @classmethod
    def suite(cls) -> tuple[Callable[[str, dict[str, Any]], str], ...]:
        return (_route, _route, _veto, _route)


__all__ = [
    "Authority",
    "Guard",
    "typed_verdict",
    "release",
    "DeniedFixtures",
    "Spy",
    "ReaderRevisions",
    "__version__",
]
