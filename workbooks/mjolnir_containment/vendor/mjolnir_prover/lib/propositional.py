"""mjolnir_prover.lib.propositional — exact propositional entailment (the TCB semantics).

Propositional entailment is DECIDABLE. That is why this module exists and why it
is different in kind from the arithmetic fragment: a bounded truth table is a
COMPLETE decision procedure for what it can size, so abstaining past the budget
is a *coverage* limit (C3), not a capability gap (C4). This is the first closed
theory in the stack.

Honest scope limits, stated rather than implied:

  * Purely propositional. No quantifiers: ILTP's 2,670 first-order problems
    carry `!`/`?`, which this module rejects by abstaining rather than
    approximating. First-order entailment is undecidable, so those are a THEORY
    gap, not a budget question.
  * Exhaustive by assignment, so cost is 2^|atoms|. The budget is TABLE POLICY
    (theories/propositional.json), pinned and hashed, so the certified set is a
    disclosed choice rather than a machine accident. The same discipline the
    PC-2c gate established for coverage.
  * Nothing is decided by sampling, truncation, or a timeout. A budget overrun
    returns None — abstain — never "probably true".

Entailment here is the standard reading:

    axioms ⊨ conjecture   iff   for every assignment satisfying all axioms,
                                the conjecture holds

A counter-assignment is reported as the witness, so a REFUTED verdict carries
the actual falsifying valuation rather than a bare assertion, and the kernel
re-checks that valuation against every axiom before accepting the refutation.

Implementation note: the search is a BITSET truth table, not an assignment
loop. A formula's truth table is held in a 2^n-bit integer and the DAG is
evaluated once with bitwise ops, because a per-assignment walk costs
2^|atoms| * nodes. Two earlier versions — the sequential walk, and an
`id`-memoised one — are gone: they measured 94s and 95s on the corpus against
0.06s for this, and none of the 0.06s seconds are on a path that decides
anything the bitset path does not.
"""
from __future__ import annotations

from . import term as T


# ------------------------------------------------------------- evaluation ---
def atoms(t) -> list[str]:
    """Atom names of a formula, sorted — the enumeration order that makes the
    witness deterministic across runs and machines."""
    return sorted(T.props_of(t))


# ------------------------------------------------------------- the budget ---
def budget() -> dict:
    """The disclosed decision budget. DATA, pinned and hashed (theory.py)."""
    from . import theory as TH
    tbl = TH.load("propositional")
    return {"max_atoms": int(tbl["max_atoms"]),
            "max_assignments": int(tbl["max_assignments"]),
            "digest": TH.digest("propositional")}


# ------------------------------------------- bitset truth table (the fast
# path, and the same arithmetic) --------------------------------------------
# The walk above is O(2^|atoms| * nodes): every assignment re-walks the whole
# formula, which measured at ~95s for the ILTP corpus. Every one of those walks
# recomputes the SAME Boolean function on different inputs, so they can all be
# done at once: represent a formula's truth table as a 2^n-bit integer and
# evaluate the DAG ONCE with bitwise ops.
#
# Exactness is unchanged — this is the same semantics on the same
# assignments, just not one at a time. The assignment enumeration is identical,
# so the witness for a refutation is the same point the slow path would find
# (lowest such bit).
def _var_mask(k: int, n: int) -> int:
    """Truth table of the k-th variable over 2^n assignments (bit i = value)."""
    total = 1 << n
    block = 1 << k                      # runs of 2^k zeros then 2^k ones
    period = block << 1
    m = 0
    for start in range(block, total, period):
        m |= ((1 << block) - 1) << start
    return m


def truth_table_masks(axioms, conjecture, names) -> tuple[int, int]:
    """(model_mask, conjecture_mask) as 2^|names| bit integers."""
    n = len(names)
    idx = {nm: i for i, nm in enumerate(names)}
    memo = {}

    def tt(node):
        hit = memo.get(id(node))
        if hit is not None:
            return hit
        k = node[0]
        if k == "p":
            val = _var_mask(idx[node[1]], n)
        elif k == "pnot":
            val = ~tt(node[1]) & _FULL[n]
        elif k == "pand":
            val = tt(node[1]) & tt(node[2])
        elif k == "por":
            val = tt(node[1]) | tt(node[2])
        elif k == "c":
            val = _FULL[n] if node[1] else 0
        else:
            raise ValueError(f"prop_out_of_model:{k}")
        memo[id(node)] = val
        return val

    model = _FULL[n]
    for a in axioms:
        model &= tt(a)
    return model, tt(conjecture)


_FULL: dict = {}


def _full_mask(n: int) -> int:
    got = _FULL.get(n)
    if got is None:
        got = (1 << (1 << n)) - 1
        _FULL[n] = got
    return got


def _decode(names, i: int) -> dict:
    """Assignment index i -> {atom: value}. Bit k is the k-th SORTED atom, so
    the witness is the same point the sequential scan finds first."""
    return {nm: bool((i >> k) & 1) for k, nm in enumerate(names)}


def decide(axioms, conjecture) -> tuple[bool | None, str, dict]:
    """THE decision entry point. (verdict, reason, meta)

    Verdict True = entailed (no assignment satisfies all axioms and
    falsifies the conjecture), False = refuted with a counter-assignment
    witness, None = abstain (over budget, or outside the theory).
    """
    bud = budget()
    names = sorted(T.props_of(conjecture).union(
        *[T.props_of(a) for a in axioms]))
    n = len(names)
    meta = {"atoms": n, "axioms": len(axioms), "budget": bud["max_atoms"],
            "theory_digest": bud["digest"], "assignments": 1 << n}
    if n > bud["max_atoms"] or (1 << n) > bud["max_assignments"]:
        meta["assignments"] = 1 << n
        return None, (f"atom_budget_exceeded:{n}>{bud['max_atoms']}"
                      if n > bud["max_atoms"] else
                      f"assignment_budget_exceeded:{1<<n}>{bud['max_assignments']}"), meta
    _FULL[n] = _full_mask(n)
    model, conj = truth_table_masks(axioms, conjecture, names)
    bad = model & ~conj & _FULL[n]
    if bad:
        i = (bad & -bad).bit_length() - 1
        return False, "counter_assignment", {**meta, "witness": _decode(names, i)}
    return True, "exhaustive_entailment", meta


# ------------------------------------------------------- the certificate ----
def vacuity(axioms, conjecture, *, essentials_limit: int = 64) -> dict:
    """The COMPUTED non-vacuity analysis of an entailment obligation.

    House definition (omega-certifiability, §3): a certificate is NON-VACUOUS
    when it discharges at least one obligation that is not itself an assumed
    premise. Computed here so the kernel and the experiments dispose on it,
    rather than the shape guards stipulating it in prose:

      self_inclusion      the conclusion IS one of the axioms -- a reflexive
                          "proof" that assumes what was to be shown;
      premise_consistent  False when the axioms are unsatisfiable: every
                          conclusion then follows (ex falso) and the
                          certificate attests to the premises being broken,
                          not to the conjecture's world;
      non_vacuous         := (not self_inclusion) and premise_consistent;
      essential_premises  how many axioms the entailment actually needs
                          (dropping one destroys entailment). A DIAGNOSTIC,
                          not part of the boolean: redundant support is
                          still support. None when the axiom set exceeds
                          `essentials_limit` -- the boolean stays cheap;
      axiom_independent   the conclusion is a tautology (holds from no
                          premises). A DIAGNOSTIC: true and non-vacuous,
                          though the axioms do no work.

    Acknowledged under-approximation: a premise classically EQUIVALENT to the
    conclusion (e.g. `p <-> c` proving `c`) is not caught by structural
    membership. This predicate never calls a genuine discharge vacuous; it can
    only miss exotic reflexive encodings.
    """
    self_inclusion = any(a == conjecture for a in axioms)
    falsum = ("c", 0)
    cons_verdict, _cr, _cm = decide(axioms, falsum)
    premise_consistent = cons_verdict is not True
    essential = None
    if len(axioms) <= essentials_limit:
        essential = 0
        for i in range(len(axioms)):
            verdict, _r, _m = decide(axioms[:i] + axioms[i + 1:], conjecture)
            # `is not True` (not `is False`): a subset that cannot be
            # re-decided counts as essential -- conservative toward
            # non-vacuity.
            if verdict is not True:
                essential += 1
    taut_verdict, _tr, _tm = decide([], conjecture)
    return {
        "self_inclusion": self_inclusion,
        "premise_consistent": premise_consistent,
        "essential_premises": essential,
        "essentials_limit": essentials_limit,
        "axiom_independent": taut_verdict is True,
        "non_vacuous": (not self_inclusion) and premise_consistent,
    }


def certify_entailment(axioms, conjecture, *, unit=None,
                         external_label=None) -> dict:
    """A certificate in the house shape, for the kernel to re-derive.

    The obligation records the axioms and the conjecture as STRINGS, not as
    pre-evaluated booleans: the kernel must rebuild them with `term.pparse` and
    redo the search. A certificate carrying its own answer would verify nothing.

    `theory_digest` is recorded so a verdict cannot be silently compared across
    two different budgets — changing `max_atoms` changes which problems are
    decided, and that choice must travel with the claim.
    """
    import datetime as _dt
    verdict, reason, meta = decide(axioms, conjecture)
    bud = budget()
    status = ("CERTIFIED" if verdict is True else
              "REFUTED" if verdict is False else "UNVERIFIABLE")
    # Vacuity is computed, not stipulated. A derivable entailment won by
    # assuming its own conclusion (self-inclusion) or from inconsistent
    # premises (ex falso) is NOT a claim: it is emitted as UNVERIFIABLE with a
    # visible `vacuous:*` reason, so the pair can never count it and the
    # degenerate-authority guard holds on this route too.
    vac = vacuity(axioms, conjecture) if verdict is not None else None
    if vac is not None and not vac["non_vacuous"]:
        reason = ("vacuous:self_inclusion" if vac["self_inclusion"]
                  else "vacuous:premise_inconsistent")
        status = "UNVERIFIABLE"
    ob_status = ("abstain" if status == "UNVERIFIABLE" else
                 "ok" if verdict is True else "fail")
    obligation = dict(
        kind="prop", method="tptp:entail", status=ob_status,
        premise=" & ".join(T.tstr(a) for a in axioms),
        conclusion=T.tstr(conjecture),
        axioms=[T.tstr(a) for a in axioms],
        conjecture=T.tstr(conjecture),
        reason=reason,
        non_vacuous=(vac or {}).get("non_vacuous"),
        self_inclusion=(vac or {}).get("self_inclusion"),
        premise_consistent=(vac or {}).get("premise_consistent"),
        essential_premises=(vac or {}).get("essential_premises"),
        axiom_independent=(vac or {}).get("axiom_independent"),
        vacuity=vac,
        theory_digest=bud["digest"],
        atoms=meta.get("atoms", 0),
        assignments=meta.get("assignments", 0),
        budget_atoms=bud["max_atoms"],
    )
    if verdict is False:
        # the counter-assignment IS the refutation: without it a REFUTED row is
        # an assertion, and the house rule is that refutations carry a witness
        obligation["witness"] = meta.get("witness")
    return {
        "schema": "cert/v2",
        # The axioms ARE asserted premises of the problem, so the provenance is
        # the `assert` class the kernel re-checks. The benchmark is not a
        # runtime-enforced contract (unlike the SQL frontend's schema checks) —
        # what makes these decidable is that the LOGIC is decidable, and what
        # makes the ground truth external is ILTP's own label.
        "spec": {"source": "assert", "confirmed": True,
                 "expr": T.tstr(conjecture),
                 "logic": "propositional",
                 "external_label": external_label},
        "tcb": {"theory": "propositional", "digest": bud["digest"]},
        "unit": unit or {"path": "", "name": ""},
        "status": status,
        "reason": reason,
        "assumptions": [T.tstr(a) for a in axioms],
        "tool": "propositional.decide",
        "encoder": "frontend_tptp.parse_formula",
        "obligations": [obligation],
        "witness": (dict(kind="assignment", env=meta["witness"])
                    if verdict is False else None),
        "vacuity": vac,
    }
