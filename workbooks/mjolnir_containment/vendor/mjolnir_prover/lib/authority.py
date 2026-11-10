"""mjolnir_prover.lib.authority — the authority identity (single source of truth).

Verdicts are defined by the authority's source, so every product keys to the same
fingerprint: the audit pins it in a report, and RLVF keys its runs/datasets to it.
There is exactly one definition; import it, never recompute it elsewhere.

Read-only: hashes the authority's source files. It does not touch the verdict path.
"""
from __future__ import annotations

import ast
import hashlib
import os

from mjolnir_prover.lib import certify

# Every module in the transitive import closure of the entry points. The
# fingerprint is the trust pin: if a change to any of these could move a
# verdict, the pin must move with it.
#
# This list was four files (certify, certificate, kernel, stepcheck) — the
# *orchestration* — while the *maths* lived outside it: term.py (decide_eq,
# provably_*), intervals.py (the box computation that produces refutations),
# theory.py (the pinned tables), claim_ir.py (obligation shaping), and
# countermodel.py (the witness writer). A weakened decide_eq would have moved
# no pin and invalidated no run, which is the exact property the pin exists to
# provide. Over-pinning is the cheap failure (an occasional re-run);
# under-pinning is the dangerous one (silent staleness). So: pin the closure.
#
# `verify.py` asserts closure(FILES) == FILES, so this cannot silently reopen.
FILES = (
    "certify.py",
    "certificate.py",
    "claim_ir.py",
    "countermodel.py",
    "frontend.py",
    "frontend_lean2.py",
    "intervals.py",
    "kernel.py",
    # in the verdict closure since the numeric-domain guard: certify reads it
    # before disposal, so a change to the markers moves verdicts and must move
    # the pin.
    "knowledge.py",
    "leankernel.py",
    "paths.py",
    "stepcheck.py",
    "propositional.py",      # the closed theory: exact entailment the kernel
                               # re-derives. Pinned because a verdict that
                               # depends on it must not be movable.
    # the soft / disclosure chain. All four sit ON the certificate path, so
    # none may be movable: a certificate whose soundness rests on unpinned
    # code is the same error as the trust pin itself being rewritable.
    "disclosure.py",         # the S-Omega predicate (0.06ms, no model)
    "router.py",             # L2 as a ROUTER: may route or abstain, never dispose
    "effects.py",            # E6 DISPOSER. Was unpinned while its encoder-side
                              # partner was guarded: the module that decides
                              # pure/mutating/opaque could move a verdict with
                              # no pin movement. Found by the fail-closed band
                              # work in mjolnir_router/ (2026-10-03): the opaque
                              # scan missed calls in non-Expr position, issuing
                              # false `pure` certificates on 14.5% of top-level
                              # functions in TheAlgorithmsPython.
    "effects_proposer.py",   # the encoder slot. Pinned so the proposer/gate
                              # split cannot be moved by editing the proposer.
    "frontend_table.py",     # stage 1+2 for machine-readable sources
    "compose.py",            # Suite B, intra-function composition
    "term.py",
    "termination.py",
    "theory.py",
    "trace.py",
)

ENTRY = ("certify.py", "certificate.py", "kernel.py", "stepcheck.py")


def _local_modules(path: str) -> set:
    """Module names imported by `path`, resolving relative (`from . import x`)."""
    out = set()
    with open(path, "r") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                if node.module:
                    out.add(node.module.split(".")[-1])
                for alias in node.names:
                    out.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[-1])
    return out


def closure(entry=None) -> set:
    """Transitive local-module closure of the verdict path, as filenames."""
    lib = os.path.dirname(certify.__file__)
    seen, stack = set(), list(entry or ENTRY)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        path = os.path.join(lib, name)
        if not os.path.exists(path):
            continue
        for mod in _local_modules(path):
            cand = mod + ".py"
            if os.path.exists(os.path.join(lib, cand)) and cand not in seen:
                stack.append(cand)
    return seen


def fingerprint() -> str:
    """sha256 over the authority's source files (order fixed by FILES)."""
    lib = os.path.dirname(certify.__file__)
    h = hashlib.sha256()
    for name in FILES:
        with open(os.path.join(lib, name), "rb") as f:
            h.update(name.encode())
            h.update(f.read())
    return h.hexdigest()
