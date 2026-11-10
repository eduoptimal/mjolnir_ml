"""mjolnir_guard.guard — the pipeline.  Authority first, reader second,
no-loosen by construction.

Order (the whole design):
    1. structural parse     -> no tool call?  abstain (0), nothing to gate
    2. dialect map          -> unknown verb?  abstain (0), human decides
    3. authority            -> ESCALATE/ABSTAIN?  0 + the rule/missing fact
                               ROUTE?              to the veto lane
    4. veto-reader          -> ROUTE?  1 (released)
                               else    0 (vetoed-by-reader)
                               error   0 (reader-error, fail-closed)

The reader never sees a non-ROUTE, and never upgrades one.  With the reader
off (MJOLNIR_GUARD_READER=off / --reader off) the pipeline degrades to the
authority alone — releases then carry stage "released-authority-only", so a
report can always tell which keys signed.
"""
from __future__ import annotations

import json
import os
import time

from mjolnir_router.core import ABSTAIN, ESCALATE, ROUTE
from mjolnir_router.frontends.actions_tau_bench.policy import check_action

from mjolnir_guard.dialect import extract_action, to_tau
from mjolnir_guard.reader import (MlxReader, parse_verdict, render_prompt,
                                  state_context)

STAGES = ("released", "released-authority-only", "vetoed-by-reader",
          "blocked-by-authority", "abstained-by-authority", "reader-error")


class Guard:
    """reader: None (authority-only) | "on" (default: env, then MlxReader) |
    any object with .generate(prompt) -> (text, usage) — tests inject stubs."""

    def __init__(self, reader="on", reader_path: str | None = None):
        if reader is None or (isinstance(reader, str)
                              and reader.strip().lower() in ("off", "none")):
            self.reader = None
        elif reader == "on":
            env = os.environ.get("MJOLNIR_GUARD_READER", "on").lower()
            self.reader = None if env in ("off", "none") else MlxReader(
                reader_path)
        else:
            self.reader = reader

    def answer(self, state: str) -> tuple[int, dict]:
        """-> (noul in {0,1}, decision audit block)."""
        t0 = time.time()

        def done(noul, **extra):
            return int(noul), {**extra,
                               "seconds": round(time.time() - t0, 3)}

        action = extract_action(state)
        if not action:
            return done(0, stage="abstained-by-authority",
                        reason="no_action_parsed")
        verb = to_tau(action)
        if not verb:
            return done(0, stage="abstained-by-authority",
                        reason=f"unknown_verb:{action['name']}")
        verdict, reasons = check_action(verb, action["args"], state=state)
        auth = {"dialect_verb": action["name"], "authority_verb": verb,
                "verdict": verdict, "reasons": list(reasons)}
        if verdict == ESCALATE:
            return done(0, stage="blocked-by-authority", authority=auth)
        if verdict == ABSTAIN or verdict != ROUTE:
            return done(0, stage="abstained-by-authority", authority=auth)

        if self.reader is None:
            return done(1, stage="released-authority-only", authority=auth)

        action_json = json.dumps({"name": action["name"],
                                  "args": action.get("args") or {}},
                                 sort_keys=True)
        surface, prompt = render_prompt(action_json, state_context(state))
        try:
            text, usage = self.reader.generate(prompt)
        except Exception as ex:
            return done(0, stage="reader-error", authority=auth,
                        error=str(ex))
        rv = parse_verdict(text)
        read = {"surface": surface, "verdict": rv,
                "line": (text or "").strip().splitlines()[0][:200]
                if (text or "").strip() else ""}
        if rv == ROUTE:
            return done(1, stage="released", authority=auth, reader=read)
        return done(0, stage="vetoed-by-reader", authority=auth, reader=read)
