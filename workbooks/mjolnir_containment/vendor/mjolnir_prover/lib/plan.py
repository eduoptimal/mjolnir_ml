"""mjolnir_prover.lib.plan — the mandatory front door for "what should we build?".

WHY THIS EXISTS. That question was asked repeatedly and got a different answer
each time, because it was being answered from intuition and from instance counts
on a particular corpus. Both are corpus-dependent, and instance counts measure
corpus composition rather than capability — which is how `boolop` came to be
written off on a corpus of 1 and is the entire content of a 274-problem
benchmark.

So the question is made well-posed instead. A claim is undecided for exactly one
of three reasons, and they form a **fixed order**:

    stage 1  no_claim           the unit asserts nothing          -> write the claim
    stage 2  no_value           an operand has no value to decide  -> SCOPE LIMIT
    stage 3  decidable_unproved decidable, prover lacks the op    -> extend nf

The ORDER IS A DEPENDENCY, not a preference: a stage-3 fix cannot move a claim
stuck at stage 1, whatever its size. The corpus changes how BIG each stage is;
it never changes which stage comes next. That is the part that is invariant, and
it is what makes the answer the same whoever asks.

This module therefore refuses to produce a rung recommendation without measuring
the buckets first. There is no code path that answers from intuition.
"""
from __future__ import annotations

import ast
import collections

from . import certify
from . import frontier as FR
from . import theory as TH

# The only judgement input: relative cost of each fix. Declared in
# theories/roadmap.json, not here, so it can be argued with and hashed.
def costs() -> dict:
    return dict(TH.load("roadmap")["costs"])


def bucket_order() -> list[dict]:
    return list(TH.load("roadmap")["bucket_order"])


# ------------------------------------------------------------- the buckets ---
def classify_unit(code: str) -> str:
    """Which stage blocks this unit? One label, decided by inspection.

      no_claim              — the unit contains no assertion at all
      no_value              — the assertion SELECTS a runtime value (a
                             subscript, an attribute, a call), so it is about
                             execution rather than about the program text
      arithmetic_unresolved — arithmetic over parameters and locals that the
                             prover abstained on

    The third key is named for what is ESTABLISHED, not for what is hoped:
    that the claim is arithmetic-shaped and the prover abstained. It is NOT
    named `decidable_unproved` because decidability is not established here.
    `err < eps` on two free locals is arithmetically shaped and unresolvable,
    yet not decidable without their runtime values — the classifier cannot
    tell it from `min(n, 0) == 0`, which is decidable. Claiming otherwise would
    be an unproven epistemic claim stated as fact.

    Honest limitation, stated rather than hidden: this is a SYNTACTIC triage.
    `err < eps` on two free locals reads as stage 3 but is not decidable
    without knowing their runtime values; the classifier cannot tell that from
    the text. It is a routing heuristic for prioritisation, not a decision
    procedure — deciding is still `certify`.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "no_claim"
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return "no_claim"
    asserts = [st for st in fn.body if isinstance(st, ast.Assert)]
    if not asserts:
        return "no_claim"
    # The distinction is not "does it mention a parameter" — `n + 1 > n` does,
    # and it is a theorem over all n. It is whether the assertion SELECTS a
    # value: a subscript, an attribute or a call reads one specific runtime
    # value that only exists once the program has run. Those are stage 2.
    # Everything else is arithmetic over parameters and locals — stage 3.
    for st in asserts:
        for n in ast.walk(st):
            if isinstance(n, (ast.Subscript, ast.Attribute)):
                return "no_value"
            if isinstance(n, ast.Call):
                f2 = n.func
                nm = f2.id if isinstance(f2, ast.Name) else getattr(f2, "attr", "")
                if nm not in ("len", "min", "max", "abs", "sum", "round",
                              "int", "float", "bool"):
                    return "no_value"
    return "arithmetic_unresolved"


def _buckets_from_units(units, progress=None) -> dict:
    """Walk units once, bucketing them, and cross-tabulating stage 3 by rung."""
    counts = collections.Counter()
    stage3 = collections.Counter()          # rung -> addressable instances
    stage3_claims = collections.defaultdict(list)
    for i, (name, code, line, path) in enumerate(units):
        b = classify_unit(code)
        counts[b] += 1
        if b == "arithmetic_unresolved":
            with _quiet():
                cert, _, _ = certify.certify_unit(name, code, line, path)
            # Which RUNG is missing is a property of the CLAIM's construct, not
            # of the obligation method the prover happened to emit — `min(n,0)
            # == 0` is discharged-or-not as `decide_eq`, but the gap is a CALL.
            # `frontier.classify_claim` already answers that question and is
            # twin-tested, so the plan reuses it rather than inventing a second
            # taxonomy that could drift from the first.
            rung = None
            try:
                last = [st for st in ast.parse(code).body
                        if isinstance(st, ast.FunctionDef)][0]
                asserts = [st for st in last.body if isinstance(st, ast.Assert)]
                if asserts:
                    # (class, rung) — the RUNG is what the costs table prices
                    _cls, rung = FR.classify_claim(ast.unparse(asserts[-1].test))
            except Exception:
                rung = None
            if rung and rung != "unknown":
                stage3[rung] += 1
                if len(stage3_claims[rung]) < 3:
                    stage3_claims[rung].append((path, line, name))
        if progress and i % 1000 == 0:
            progress(i, dict(counts))
    return {"counts": counts, "stage3_by_rung": stage3,
            "stage3_examples": dict(stage3_claims)}


class _quiet:
    def __enter__(self):
        import contextlib, io
        self._c = contextlib.redirect_stdout(io.StringIO())
        return self._c.__enter__()

    def __exit__(self, *a):
        return self._c.__exit__(*a)


# ------------------------------------------------------------- the report ---
def build_plan(units, progress=None) -> dict:
    """The whole answer: measured buckets, then ONE derived order."""
    m = _buckets_from_units(units, progress=progress)
    counts = m["counts"]
    total = sum(counts.values())
    if not total:
        # Nothing was measured. A command whose whole point is "measure, or
        # decline" must DECLINE here rather than crash or emit an empty table
        # that reads like a finding.
        return {
            "total_units": 0, "counts": {}, "stages": [], "stage3_ranking": [],
            "dominant_stage": None,
            "recommendation": (
                "NO MEASUREMENT: no units were found under the given paths "
                "after the exclude list. `mjolnir check` reads "
                "mjolnir.toml, so a path under storage/data or another "
                "excluded directory is invisible here by design. Point at "
                "product source, or use a config that does not exclude it."),
            "order_note": TH.load("roadmap")[
                "order_is_dependency_not_preference"],
            "roadmap_digest": TH.digest("roadmap"),
        }
    order = bucket_order()
    cst = costs()

    max_counts = max(counts.values()) if counts else 0
    stages = []
    for i, b in enumerate(order, start=1):
        n = counts.get(b["key"], 0)
        stages.append({
            "stage": i,
            "key": b["key"],
            "instances": n,
            "share": round(100.0 * n / total, 1),
            "fix": b["fix"],
            "prover_can_help": b["prover_can_help"],
            "dominant": n == max_counts and n > 0,
        })

    # ranking WITHIN stage 3 only, and only over instances a fix could move
    s3 = m["stage3_by_rung"]
    ranking = []
    for rung, n in s3.items():
        cost = cst.get(rung)
        ranking.append({"rung": rung, "addressable": n,
                        "cost": cost,
                        # A rung absent from the costs table is UNCOSTED, not
                        # free. It is ranked last and says so, rather than
                        # silently inheriting a default.
                        "instances_per_cost": (n / cost) if cost else None,
                        "uncosted": cost is None,
                        "examples": m["stage3_examples"].get(rung, [])})
    ranking.sort(key=lambda r: (-(r["instances_per_cost"] or 0), r["rung"]))

    dominant = max(stages, key=lambda s: s["instances"]) if stages else None
    recommendation = None
    if dominant and dominant["instances"] == 0:
        recommendation = ("Nothing to build: no unit is blocked by a prover "
                          "gap. Every claim is either unstated or has no "
                          "value. Adding capability cannot move this corpus.")
    elif dominant and not dominant["prover_can_help"]:
        recommendation = (
            f"Do NOT build a rung. {dominant['instances']} of {total} units "
            f"({dominant['share']}%) are at stage {dominant['stage']} "
            f"({dominant['key']}): {dominant['fix']}. A prover improvement "
            f"cannot move them.")
    elif ranking and ranking[0]["instances_per_cost"] is None:
        recommendation = (
            f"{len(ranking)} stage-3 rung(s) have no declared cost in "
            f"roadmap.json, so they cannot be ranked. Declare a cost rather "
            f"than guessing one here.")
    elif ranking:
        top = ranking[0]
        recommendation = (
            f"RANKED: stage 3 is the binding constraint ({ranking[0]['addressable']}"
            f" addressable). Build '{top['rung']}' first "
            f"({top['instances_per_cost']:.1f} addressable per unit cost).")

    return {
        "total_units": total,
        "counts": dict(counts),
        "stages": stages,
        "stage3_ranking": ranking,
        "dominant_stage": dominant,
        "recommendation": recommendation,
        "order_note": TH.load("roadmap")["order_is_dependency_not_preference"],
        "roadmap_digest": TH.digest("roadmap"),
    }


def units_from_paths(paths):
    """(name, code, line, path) for every unit under `paths`, minus artifacts."""
    import os
    from . import gate as G
    cfg = G.load_config("mjolnir.toml")
    excl = tuple(cfg.get("exclude", ()))
    out = []
    for p in paths:
        p = str(p)
        files = [p] if os.path.isfile(p) else None
        if files is None:
            files = []
            for root, dirs, fs in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d != "__pycache__")
                files += [os.path.join(root, f) for f in sorted(fs)
                          if f.endswith(".py")]
        for f in files:
            for name, code, line in G.units_of(f):
                out.append((name, code, line, f))
    return G.collect([str(p) for p in paths], exclude=excl), out

# ------------------------------------------------------------- the scope ---
def scope_counts(units) -> dict:
    """Measured, per scope class. The table PROMISES; this verifies the promise
    is still true of the code, so the two cannot drift apart."""
    n = collections.Counter()
    for name, code, line, path in units:
        try:
            fn = next((x for x in ast.parse(code).body
                       if isinstance(x, ast.FunctionDef)), None)
        except SyntaxError:
            n["unparseable"] += 1
            continue
        if fn is None:
            n["no claim: not a function"] += 1
            continue
        asserts = [st for st in fn.body if isinstance(st, ast.Assert)]
        if not asserts:
            n["declines: units that state no claim"] += 1
            continue
        # NOTE: a unit that takes parameters is NOT separately classified as
        # "takes fixtures". Syntactically a pytest fixture and a contract's
        # quantified parameter are identical, and mjolnir treats them the same
        # way — the fragment decides a claim over parameters because it proves
        # it for ALL values, not because it knows any value. The decline is on
        # VALUES, and that is the predicate below.
        selects = False
        nonpoly = False
        for st in asserts:
            for node in ast.walk(st):
                if isinstance(node, (ast.Subscript, ast.Attribute)):
                    selects = True
                if isinstance(node, ast.Call):
                    f2 = node.func
                    nm = (f2.id if isinstance(f2, ast.Name)
                          else getattr(f2, "attr", ""))
                    if nm in ("min", "max", "abs"):
                        nonpoly = True
                    elif nm not in ("len", "sum", "round", "int", "float",
                                    "bool", "sorted"):
                        selects = True
        if nonpoly:
            n["declines: non-polynomial arithmetic"] += 1
        elif selects:
            n["declines: claims that select a runtime value"] += 1
        else:
            n["decides"] += 1
    return dict(n)


def scope_report(units) -> dict:
    """The measured scope, beside the promised one."""
    tbl = TH.load("scope")
    counts = scope_counts(units)
    total = sum(counts.values()) or 1
    return {
        "total_units": total,
        "counts": counts,
        "decided": counts.get("decides", 0),
        "decided_share": round(100.0 * counts.get("decides", 0) / total, 1),
        "declined": {k: v for k, v in counts.items() if k.startswith("declines")},
        "declared_decides": tbl["decides"],
        "declared_declines": tbl["declines"],
        "on_parameters_and_fixtures": tbl["on_parameters_and_fixtures"],
        "not_a_scope_limit": tbl["not_a_scope_limit"],
        "scope_digest": TH.digest("scope"),
    }
