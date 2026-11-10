"""mjolnir_intent.cli — the command line entry point.

Three verbs, plain flags, exit codes a script can trust (Typer, the product's
one CLI framework):

    mjolnir intent check  -p PROMPT -r RESPONSE [-d DIFF] [--context DIR] [--json]
    mjolnir intent session FILE.md [--json]
    mjolnir intent selftest


PROMPT/RESPONSE accept inline text or a file path (if the value names an
existing file, it is read).

EXIT CODES (check)
    0  BUILD — no blocking findings
    1  gate finding — a claim in the prompt was violated (units / negation)
    2  ASK — the prompt itself is under-determined; clarify before building
    3  usage error

Advisory findings (conservation) are reported but never change the exit code;
pass --strict to make them block as well.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjolnir_intent import checks                     # noqa: E402
from mjolnir_intent import selftest as _selftest      # noqa: E402

GATE = checks.GATE_CHECKS
VERDICT_HELP = {
    0: "BUILD — the prompt's claims are honoured",
    1: "BLOCKED — a committed claim from the prompt was violated",
    2: "ASK — the prompt is under-determined; clarify before building",
}


import typer
def _read(value: str) -> str:
    """Inline text, or file contents when the value names an existing file."""
    if os.path.isfile(value):
        with open(value, encoding="utf-8", errors="ignore") as f:
            return f.read()
    return value


def _fmt_finding(f: dict) -> str:
    c = f["check"]
    if c == "unit_adjacency":
        return (f"unit changed: prompt says {f['surface']!r} "
                f"({f['prompt_unit']}), got {f['got_unit']} ({f['dimension']})")
    if c == "unit_dropped":
        return (f"unit dropped: prompt says {f['surface']!r} "
                f"({f['prompt_unit']}), no unit in the response")
    if c == "negation_added":
        return f"negation violated: {f['token']!r} was added despite being forbidden"
    if c == "entity_lost":
        return f"committed name lost: {f['entity']!r} never appears downstream"
    if c == "number_lost":
        return f"committed number lost: {f['number']!r} never appears downstream"
    if c == "negation_prose_unverifiable":
        return (f"cannot verify from prose: {f['token']!r} is mentioned — "
                f"re-run with --diff to check exactly")
    if c == "claim_refuted":
        return (f"lowered claim refuted by the authority: {f['source']!r} "
                f"({f.get('why', '')})")
    if c == "commitment_unmaterialised":
        return (f"commitment never materialised downstream: "
                f"{f['artifact']!r} — {f.get('reason', '')}")
    return json.dumps(f, sort_keys=True)


def _print_report(rep: dict, prompt: str) -> int:
    gate, adv, unver = rep["gate"], rep["advisory"], rep["unverifiable"]

    print("mjolnir — intent contract check")
    print("=" * 62)
    print(f"PROMPT   {prompt.strip()[:200].replace(chr(10), ' ')}"
          f"{'...' if len(prompt.strip()) > 200 else ''}")
    print(f"VERDICT  {rep['verdict']}")

    if rep["verdict"] == "ASK":
        print()
        print("ASK — clarify the prompt first:")
        for r in rep["ambiguity"]:
            print(f"  ? {r.get('ask', r['kind'])}")

    if gate:
        print()
        print(f"BLOCKING FINDINGS ({len(gate)})")
        for f in gate:
            print(f"  ! {_fmt_finding(f)}")
    if adv:
        print()
        print(f"advisory ({len(adv)}) — report only, does not block")
        for f in adv:
            print(f"  - {_fmt_finding(f)}")
    if unver:
        print()
        print(f"unverifiable ({len(unver)}) — needs a diff, not counted")
        for f in unver:
            print(f"  ? {_fmt_finding(f)}")
    if not gate and not adv and not unver:
        print()
        print("no findings — every committed claim survived")

    if rep.get("lowered"):
        low = rep["lowered"]
        print()
        print(f"lowered {low['n_claims']} claim(s) -> "
              f"authority {low['unit']}:")
        for v in low["verdicts"]:
            print(f"  - {v['source']} -> {v['verdict']} ({v['why']})")
        for s in low["sinks"]:
            print(f"  - sink {s['kind']}: {s.get('ask') or s.get('reason')}")
        if low.get("halt"):
            print(f"  - HALT {low['halt']['reason']}")

    if rep["attribution"]:
        print()
        print(f"attribution  {rep['attribution']}")
    n = rep["n_claims"]
    print(f"prompt carries {n['numbers_with_units']} numbers-with-units, "
          f"{n['bare_numbers']} bare numbers, {n['entities']} entities, "
          f"{n['negations']} negations")

    if gate:
        return 1
    if rep["verdict"] == "ASK":
        return 2
    return 0


def _lower_strict(prompt: str, response: str, diff: str | None) -> dict | None:
    """Contractual Lowering (`frontend_nl`): lower the prompt into CLOSED claims
    over T_Q and dispose them with the authority. Returns the serialisable
    summary, or None when the prover is unavailable (intent stays zero-dep by
    default; strict is the opt-in that buys the second authority).

    The proposal is parsed EXACTLY: numbers on added diff lines and numeric
    JSON fields in the response. Artifacts are bound to config keys by shared
    tokens (rule-based, single-candidate only -- never fuzzy). A REFUTED claim
    blocks (gate); an unmaterialised commitment is advisory, promoted by --strict.
    """
    import re as _re
    try:
        from mjolnir_prover.lib import frontend_nl
    except Exception:                                          # noqa: BLE001
        return None
    config = {}
    for line in (diff or "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            m = _re.search(r"([A-Za-z_][\w]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)", line)
            if m and m.group(1) not in config:
                config[m.group(1)] = (float(m.group(2)) if "." in m.group(2)
                                      else int(m.group(2)))
    try:
        obj = json.loads(response) if response else None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    config.setdefault(k, v)
    except Exception:                                          # noqa: BLE001
        pass
    bind = {}
    keys = [(k, {t for t in k.lower().split("_") if len(t) > 2}) for k in config]
    for art in frontend_nl.entities_of(prompt):
        at = {t for t in art.lower().split("_") if len(t) > 2}
        cand = [k for k, kt in keys if at and at & kt]
        if len(cand) == 1:
            bind[art] = cand[0]
    r = frontend_nl.lower_prompt(prompt, {"config": config,
                                          "diff": diff or "",
                                          "tool_calls": []},
                                 bind, dialect=None)
    ir = r["ir"]
    return {"unit": ir["unit"], "n_claims": len(ir["claims"]),
            "verdicts": ir["verdicts"], "sinks": ir["sinks"],
            "lowered": ir["lowered"], "halt": r["halt"],
            "_gate": [v for v in ir["verdicts"] if v["verdict"] == "REFUTED"],
            "_advisory": [s for s in ir["sinks"]
                          if s["kind"] in ("missing_proposal_value",
                                           "missing_proposal_action")]}


def _do_check(prompt: str, response: str, diff: str | None,
              context: str | None, json_out: bool, strict: bool) -> int:
    prompt = _read(prompt)
    response = _read(response) if response else ""
    diff = _read(diff) if diff else None
    if context:
        if os.path.isdir(context):
            lines = []
            for root, _dirs, files in os.walk(context):
                for fn in files:
                    lines.append(os.path.relpath(os.path.join(root, fn), context))
            context = "\n".join(lines)
        else:
            context = _read(context)

    rep = checks.audit(prompt, response, diff=diff, context=context)

    if strict:
        # Contractual Lowering: a second, independent authority (the prover)
        # over the prompt's numeric core, before advisory is promoted to gate.
        low = _lower_strict(prompt, response, diff)
        if low:
            for v in low.pop("_gate"):
                rep["gate"].append({"check": "claim_refuted",
                                    "source": v["source"], "why": v["why"]})
            for s in low.pop("_advisory"):
                rep["advisory"].append({"check": "commitment_unmaterialised",
                                        "artifact": s.get("artifact")
                                        or s.get("verb") or "",
                                        "reason": s.get("reason", "")})
            rep["lowered"] = low
        rep["gate"] = rep["gate"] + rep["advisory"]
        rep["advisory"] = []

    if json_out:
        print(json.dumps(rep, indent=1, sort_keys=True))
        if rep["gate"]:
            return 1
        return 2 if rep["verdict"] == "ASK" else 0

    code = _print_report(rep, prompt)
    if strict and rep["gate"] and code == 0:
        code = 1
    return code


def session_report(path: str) -> dict:
    """Transcript -> counts and examples. The data; `_do_session` formats it.

    Extracted so the simple surface (`tasks.py`) and this CLI read a transcript
    the same way. Two parsers for one format is how a count differs between two
    outputs of the same file, which is the parity test this product needs.
    """
    import re
    if not os.path.isfile(path):
        return {"error": f"{path} not found"}
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    blocks = re.split(r"^## (User|Assistant)\s*$", txt, flags=re.M)
    turns, cur = [], None
    for i in range(1, len(blocks) - 1, 2):
        k = blocks[i].strip().lower()
        body = blocks[i + 1].strip()
        if k == "user":
            if cur:
                turns.append(cur)
            cur = {"user": body, "asst": []}
        elif cur is not None:
            cur["asst"].append(body)
    if cur:
        turns.append(cur)
    turns = [{"user": t["user"], "asst": "\n".join(t["asst"])} for t in turns]

    from collections import Counter
    gate_t, adv_t = 0, 0
    per, verdicts = Counter(), Counter()
    examples = []
    for i, t in enumerate(turns):
        r = checks.audit(t["user"], t["asst"])
        verdicts[r["verdict"]] += 1
        for f in r["failures"]:
            per[f["check"]] += 1
        if r["gate"]:
            gate_t += 1
            if len(examples) < 5:
                examples.append((i, r["gate"][0]))
        if r["advisory"]:
            adv_t += 1
    return dict(turns=len(turns), verdicts=dict(verdicts),
                gate_turns=gate_t, advisory_turns=adv_t,
                findings=dict(per), examples=examples)


def _do_session(path: str, json_out: bool) -> int:
    """Audit every user turn of a `## User` / `## Assistant` markdown transcript."""
    out = session_report(path)
    if "error" in out:
        print(f"error: {out['error']}", file=sys.stderr)
        return 3

    if json_out:
        print(json.dumps(out, indent=1, sort_keys=True))
        return 1 if out["gate_turns"] else 0

    n = max(out["turns"], 1)
    print(f"mjolnir — transcript audit  {path}")
    print("=" * 62)
    print(f"turns                 {n}")
    print(f"verdicts              {out['verdicts']}")
    print(f"gate findings on      {out['gate_turns']}/{n} turns = {100*out['gate_turns']/n:.0f}%")
    print(f"advisory findings on  {out['advisory_turns']}/{n} turns = {100*out['advisory_turns']/n:.0f}%")
    print()
    print("findings by check:")
    from collections import Counter as _C
    for k, v in _C(out["findings"]).most_common():
        kind = "gate" if k in GATE else "advisory"
        print(f"  {k:<32} {v:>4}  ({kind})")
    if out["examples"]:
        print()
        print("first blocking examples:")
        for i, f in out["examples"]:
            print(f"  turn {i}: {_fmt_finding(f)}")
    return 1 if out["gate_turns"] else 0


# ---------------------------------------------------------------- the Typer app
# Typer is the one CLI framework in this product -- kernel, router, intent and the
# wrapper all speak it. The argparse version of this file is gone rather than kept
# beside it: two frameworks in one product is exactly the fallback the mandate
# forbids, and a half-migrated command surface is where exit codes drift.

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Prompt → response contract check.")


@app.command("check")
def check_cmd(
    prompt: str = typer.Option(..., "-p", "--prompt",
                               help="the request (inline text or a file path)"),
    response: str = typer.Option("", "-r", "--response",
                                 help="what was built (inline text or a file path)"),
    diff: str = typer.Option(None, "-d", "--diff",
                             help="unified diff; unlocks exact forbidden-thing checking"),
    context: str = typer.Option(None, "--context",
                                help="a directory of existing files"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable report"),
    strict: bool = typer.Option(False, "--strict",
                                help="advisory findings block too"),
) -> None:
    """Check one prompt/response pair. Exit: 0=BUILD 1=blocked 2=ASK 3=usage."""
    raise typer.Exit(_do_check(prompt, response, diff, context, json_out, strict))


@app.command("session")
def session_cmd(
    file: str = typer.Argument(..., help="a `## User` / `## Assistant` transcript"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable report"),
) -> None:
    """Audit every user turn of a chat transcript."""
    raise typer.Exit(_do_session(file, json_out))


@app.command("selftest")
def selftest_cmd() -> None:
    """Prove the checks still behave (16 labelled cases)."""
    raise typer.Exit(_selftest.main())


def main(argv: list[str] | None = None) -> int:
    """Console-script and programmatic entry. Returns the exit code; never exits.

    Tests call this and assert on the code, so swallowing `SystemExit` is part of
    the contract, not an accident: the wrapper and the kernel both delegate here
    the same way.
    """
    import click
    try:
        # standalone_mode=False: click RETURNS the exit code of a `typer.Exit`
        # instead of raising SystemExit for it. The first version of this wrapper
        # ignored that return value and reported 0 for every verdict, including
        # `blocked` -- a gate that always says "go" is worse than no gate, and it
        # happened silently because the printed report was still correct.
        rv = app(args=argv, standalone_mode=False)
        return int(rv) if isinstance(rv, int) else 0
    except (typer.Exit, click.exceptions.Exit) as exc:
        return int(getattr(exc, "exit_code", 1))
    except click.UsageError as exc:
        print(exc.format_message(), file=sys.stderr)
        return 3
    except SystemExit as exc:                      # --help / usage
        return int(exc.code or 0)
    except Exception as exc:
        # typer >= 0.27 vendors its own click (`typer._click.core`): its
        # UsageError/MissingParameter NO LONGER subclass click's, so the
        # clause above misses them and a missing `--prompt` would escape as
        # a traceback instead of the documented usage code 3. The contract
        # is ours, not typer's: match the click-lineage error NAMES across
        # any vendoring, keep every other exception loud.
        _usage_names = {"UsageError", "MissingParameter", "BadParameter",
                        "NoSuchOption", "NoArgsIsAllowedError"}
        if {c.__name__ for c in type(exc).__mro__} & _usage_names:
            _msg = (exc.format_message() if hasattr(exc, "format_message")
                    else str(exc))
            print(_msg, file=sys.stderr)
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
