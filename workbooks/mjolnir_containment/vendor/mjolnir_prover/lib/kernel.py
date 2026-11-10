"""mjolnir_prover.lib.kernel — independent replay kernel (L0–L8).

A third party re-checks a certificate.json with ONLY this file + the term algebra.
The kernel imports zero synthesizer / encoder / search / sympy dependencies: it
never trusts the emitter's verdict. It re-derives or re-discharges every
obligation from the certificate alone and reports agreement.

Trust base (TCB of the kernel): term.py canonicalization + nf + decide_eq +
rewrite. Nothing else.

Replay rules per obligation method:
  nf+decide_eq / decide_eq   re-derive: decide_eq(premise, conclusion) as terms
  nf>0                       nf(premise - conclusion) must be a positive constant
  family_shape               variant shape check against structured meta
  rewrite+decide_eq          rewrite both premise sides under cert["theory"]
  no_precondition            vacuous; verified as recorded
Program-backed certificates (straight-line units) are additionally re-derived
from the stored program steps and compared obligation-by-obligation.
"""
from __future__ import annotations
import json
from . import term as T


def _term(s: str):
    return T.parse(s)


def _eq_sides(s: str):
    t = _term(s)
    if t[0] != "eq":
        raise ValueError("not_an_equation")
    return t[1], t[2]


def _holds(formula, env: dict) -> bool:
    """Truth of ONE propositional formula under ONE assignment.

    Single-point, so the sequential walk is right and cheap here: this runs once
    per certificate to audit the counter-assignment witness, not once per
    assignment. The BULK search is bitset-valued in propositional.py.
    """
    stack = [(formula, False)]
    done = []
    while stack:
        node, ready = stack.pop()
        k = node[0]
        if not ready:
            stack.append((node, True))
            # Only inner nodes have children. Pushing `node[1:]` for a leaf
            # would push the atom NAME as if it were a subformula, and the next
            # pop would index a string.
            if k not in ("p", "c"):
                for kid in reversed(node[1:]):
                    stack.append((kid, False))
            continue
        if k == "p":
            if node[1] not in env:
                raise ValueError(f"prop_unbound:{node[1]}")
            done.append(bool(env[node[1]]))
        elif k == "pnot":
            done.append(not done.pop())
        elif k in ("pand", "por"):
            right, left = done.pop(), done.pop()
            done.append((left and right) if k == "pand" else (left or right))
        elif k == "c":
            done.append(bool(node[1]))
        else:
            raise ValueError(f"prop_out_of_model:{k}")
    return done[-1]



# Methods whose verdict is asserted BY DECLARATION rather than re-derived from
# the claim. Each one is a potential laundering route for a mislabelled
# obligation, so each is guarded rather than trusted.
_DECLARATION_METHODS = ("premise", "program_exit", "no_precondition")




_INF = float("inf")


def _provably_any(rel: str, k_l, k_r, b) -> bool | None:
    """`rel-box` / `rel-box-refute` decision, shared by both kernel routes."""
    op = rel
    table = {
        "le": lambda: T.provably_ge(k_l, k_r, b),
        "ge": lambda: T.provably_ge(k_r, k_l, b),
        "lt": lambda: T.provably_gt(k_l, k_r, b),
        "gt": lambda: T.provably_gt(k_r, k_l, b),
    }
    f = table.get(op)
    if f is None:
        return None
    return f()

def _affine_residue(premise: str, conclusion: str):
    """`(A % m == r) |- (B % m == s)` for AFFINE A, B  ->  (residue_of_B, s).

    -> None when the strings are not that shape, which is the honest answer: the
    kernel does not guess a reading it was not offered.

    Implemented with `Fraction` and a hand-rolled affine parser rather than the
    term layer, on purpose. This is an INDEPENDENT re-derivation -- re-using the
    parser whose output it is checking would make the check circular, which is the
    mistake the whole `differential` discipline exists to prevent.
    """
    import re
    from fractions import Fraction as _F

    def split_mod(s):
        m = re.match(r"^\s*(.+?)\s*%\s*(\d+)\s*==\s*(-?\d+)\s*$", s or "")
        return (m.group(1).strip(), int(m.group(2)), int(m.group(3))) if m else None

    p, c = split_mod(premise), split_mod(conclusion)
    if not p or not c or p[1] != c[1]:
        return None
    _expr, mod, claimed = c

    # affine in ONE variable: a*n + c
    var = None
    coef, const = _F(0), _F(0)
    # parentheses carry no information for an affine form; leaving them in splits
    # `(3 * n - 7)` into the nonsense tokens `(3`, `*`, `n`, `-7)`.
    _flat = _expr.replace("(", "").replace(")", "").replace(" ", "")
    for tok in re.finditer(r"[+-]?[^+-]+", _flat):
        term = tok.group(0)
        sign = _F(-1) if term.startswith("-") else _F(1)
        term = term.lstrip("+-")
        if not term:
            continue
        if "*" in term:
            parts = term.split("*")
            nums = [x for x in parts if re.fullmatch(r"\d+(?:\.\d+)?", x)]
            names = [x for x in parts if not re.fullmatch(r"\d+(?:\.\d+)?", x)]
            if len(names) != 1 or len(nums) != 1:
                return None                   # not affine in one variable
            if var is None:
                var = names[0]
            elif var != names[0]:
                return None                   # two variables: not this route
            coef += sign * _F(nums[0])
        else:
            if re.fullmatch(r"\d+(?:\.\d+)?", term):
                const += sign * _F(term)
            else:
                if var is None:
                    var = term
                elif var != term:
                    return None
                coef += sign
    if var is None or mod <= 0:
        return None
    # substitute n = r (the residue class representative) -- p[2] is r
    residue = (coef * _F(p[2]) + const) % mod
    return (int(residue) if residue.denominator == 1 else residue, claimed)


def _rederive(cert: dict, o: dict, trace_ok: bool = False) -> bool:
    """Independently re-discharge ONE obligation. Returns the derived verdict."""
    m = o.get("method", "")
    prem, con = o.get("premise", ""), o.get("conclusion", "")
    try:
        if m in ("nf+decide_eq", "decide_eq"):
            return bool(T.decide_eq(_term(prem), _term(con)))
        if m == "nf+decide_ne":
            # the negation of a decided equality. None (undecided) must stay
            # None -- `not None` would read as a verified claim.
            _d = T.decide_eq(_term(prem), _term(con))
            return None if _d is None else (not _d)
        if m == "nullness":
            # closed-literal nullness: re-parse the recorded claim for its
            # operator, re-evaluate against the recorded resolved literal.
            # The literal is the emitter's slice record -- the same trust
            # level as `rel:*`, which re-decides over its own stored lhs/rhs.
            import ast as _astn
            try:
                _tree = _astn.parse(str(o.get("claim") or ""),
                                    mode="eval").body
                if not (isinstance(_tree, _astn.Compare)
                        and len(_tree.ops) == 1
                        and isinstance(_tree.ops[0],
                                       (_astn.Is, _astn.IsNot))):
                    return None
                if o.get("lhs") is None:
                    return None
                _val = _astn.literal_eval(str(o.get("lhs")))
            except Exception:
                return None
            if isinstance(_tree.ops[0], _astn.IsNot):
                return _val is not None
            return _val is None
        if m == "nf>0":
            d = T.csub(_term(prem), _term(con))
            nd = T.nf(d)
            if nd is None:
                return None                      # cannot re-derive (not a refute)
            return nd.get((), 0) >= 1 and all(c >= 0 for c in nd.values())
        if m == "family_shape":
            meta = o.get("meta", {})
            # variant must be linear in the counter with slope <= -1 and no rest
            nd = T.nf(_term(con))
            if nd is None:
                return None
            rest = {k: v for k, v in nd.items() if k != () and len(k) != 1}
            slope = sum(v for k, v in nd.items()
                        if len(k) == 1 and meta.get("counter") in k)
            const = nd.get((), 0)
            guard = o.get("premise", "")
            return (not rest and slope == -1 and const == 0
                    and guard == f"ne({meta.get('counter')}, {meta.get('bound')})"
                    and meta.get("assumptions") == [f"{meta.get('bound')} >= 0",
                                                    f"{meta.get('counter')} <= {meta.get('bound')}"])
        if m == "rewrite+decide_eq":
            theory = [(_term(l), _term(r)) for l, r in cert.get("theory", [])]
            l, r = _eq_sides(prem)
            return bool(T.decide_eq(T.rewrite(l, theory), T.rewrite(r, theory)))
        if m == "program_exit":
            # The authoritative check is the program-backed re-derivation, which
            # is not reachable from the certificate alone. Same MISNAMING
            # exposure as `premise`: asserted by declaration, therefore a
            # laundering channel for a relabelled abstention. Guarded the same
            # way — a declaration may not sit on the claim's own line.
            _last_pe = max([int(x.get("line") or 0)
                            for x in cert.get("obligations", [])] or [0])
            if int(o.get("line") or 0) >= _last_pe:
                return None
            return True
        if m.startswith("mod:"):
            # E-mod: a linear expression is constant on a residue class.
            #
            # Found by the arbiter catching its own blind spot. The obligation is
            # `n % 7 == 2 |- (3n-7) % 7 == 6`, which is TRUE, and the emitter
            # recorded a LEMMA step for it -- but `_rederive` had no branch for
            # `mod:*`, so the obligation came back UNCHECKED, and because the
            # step-checker has no branch for the lemma either, the TRACE failed
            # validation and the whole certificate was published as FAULT.
            #
            # Two missing readers, one symptom. The same failure this project has
            # now hit three times in a week: correct data, no code that reads it.
            #
            # The re-derivation is exact and needs no theory table. If
            # `n = r + m*k` then an AFFINE expression `a*n + c` satisfies
            # `a*(r + m*k) + c = a*r + c + (a*m)*k`, and `a*m*k = 0 (mod m)`. So
            # the expression's residue is `a*r + c (mod m)` for EVERY `k` -- the
            # constancy is unconditional, not a condition on `a`. Compare that
            # with the claimed residue and the obligation is decided.
            _r = _affine_residue(o.get("premise") or "", o.get("conclusion") or "")
            if _r is None:
                return None                      # not the shape this route claims
            got, claimed = _r
            return got == claimed
        if m.startswith("rel:"):
            # The box discharge — mjolnir's MAIN arithmetic route. It had NO
            # kernel branch at all, so every `rel:rel-box` obligation returned
            # None and, because None added no mismatch, scored agreement 1.0
            # while being verified nowhere. A pre-existing twin asserting
            # "kernel agreement" was passing vacuously.
            #
            # Re-decide from the certificate's OWN record: the canonical
            # post-substitution sides and the box that was used. The premises
            # that produced the box are separately recorded obligations, so the
            # chain premise -> box -> verdict is fully re-derived.
            parts = m.split(":")
            why = parts[1]
            # The relation is part of the ROUTE, not an emitter choice: without
            # it the kernel cannot know whether to ask provably_ge or
            # provably_gt, and it was silently returning None.
            relop = parts[2] if len(parts) > 2 else None
            if why not in ("rel-box", "rel-box-refute") or relop not in (
                    "le", "ge", "lt", "gt"):
                return None
            k_l, k_r = _term(o.get("lhs") or ""), _term(o.get("rhs") or "")
            if k_l is None or k_r is None:
                return None
            # The artifact stores `None` for an unbounded side (JSON has no
            # infinity); the primitives want the sentinel. Refusing to guess
            # here would be the safe move, but the sentinel is exact and the
            # shape is checked, so the reconstruction is lossless.
            raw = o.get("bounds") or {}
            b = {}
            for v, pair in raw.items():
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    return None
                lo, hi = pair
                # `None` means unbounded on that side; the artifact cannot hold
                # an infinity, so the sentinel is restored here.
                b[v] = (-_INF if lo is None else lo,
                        _INF if hi is None else hi)   # pair form: the BOX
            try:
                # The box decision was made by INTERVAL arithmetic over the
                # declared box (interval_eval on facts_iv), so the kernel
                # re-derives it the same way. Using the polynomial
                # `provably_*` route here would be a different procedure and
                # would abstain on cases the box settles — an honest route, but
                # not a re-derivation of THIS one.
                from mjolnir_prover.lib import intervals as _IV
                env = _IV.env_from_bounds(
                    {v: (pair[0], pair[1]) for v, pair in b.items()})
                if _IV.decide_rel(k_l, k_r, relop, env):
                    return True
                # NOT PROVEN IS NOT DISPROVEN. This is the whole defect, and it
                # was found by the arbiter catching itself.
                #
                # `frag_rel_box_hold` asserts `x >= 0, x <= 2 |- 3x+1 <= 2x+4`.
                # That is TRUE (it is `x <= 3`). Interval arithmetic over the
                # declared box gives lhs in [1,7] and rhs in [4,8]: the comparison
                # is INCONCLUSIVE, because intervals treat x as independent on the
                # two sides and the correlation is exactly what the claim rests on.
                # The emitter settled it by a linear argument it did not record.
                #
                # `decide_rel` returns False for "cannot show", `replay` reads
                # False as "the emitter certified something untrue", and the
                # arbiter published FAULT: kernel_disagreement['false_certificate']
                # on a claim that is true. A soundness alarm that cries wolf on a
                # true claim is worse than no alarm, because it teaches the reader
                # to ignore the channel that matters.
                #
                # So the two cases are separated, and only the second is an alarm:
                #   intervals DISPROVE the relation  -> False, a real refutation
                #   intervals are merely INCONCLUSIVE -> None, i.e. unchecked
                lo_l, hi_l = _IV.interval_eval(k_l, env)
                lo_r, hi_r = _IV.interval_eval(k_r, env)
                disproved = {
                    "le": lo_l > hi_r, "ge": lo_r > hi_l,
                    "lt": lo_l >= hi_r, "gt": lo_r >= hi_l,
                }.get(relop)
                return False if disproved else None
            except Exception:
                return None

        if m.startswith("sign_analysis:"):
            # obligations store premise=r-side, conclusion=l-side of the claim;
            # the artifact carries the bounds (premises) used at check time
            rel = m.split(":", 1)[1]
            k_l, k_r = _term(prem), _term(con)
            b = o.get("bounds")
            return {"le": lambda: T.provably_ge(k_l, k_r, b),   # l <= r: r-l>=0
                    "ge": lambda: T.provably_ge(k_r, k_l, b),   # l >= r: l-r>=0
                    "lt": lambda: T.provably_gt(k_l, k_r, b),   # l <  r: r-l>0
                    "gt": lambda: T.provably_gt(k_r, k_l, b)}[rel]()
        if m.startswith("interval:"):
            # The claim is in terms of COMPUTED locals; the derivation chain
            # (slice -> box arithmetic -> decision) lives in the trace, which
            # the independent step-checker validates step-by-step. The kernel
            # verifies: trace digest + steps + final decision-step verdict.
            if not trace_ok:
                return None                      # trace unavailable -> unchecked
            return bool(_interval_trace_verdict(o, cert.get("trace")))
        if m.startswith("premise_contradiction:"):
            # Re-derive independently from the OBLIGATION alone: the recorded
            # premise and the claim must disagree on a shared LHS with two
            # different numeric constants. The kernel does not consult the
            # theory table — it re-checks the logic, which is what makes this a
            # re-derivation rather than a replay of the emitter's opinion.
            # If the emitter ever ships this route, an unverifiable refutation
            # is impossible by construction.
            reason = m.split(":", 1)[1]
            pre = str(o.get("premise_obligation", "")).strip()
            lhs = str(o.get("lhs", "")).strip()
            r_claim = str(o.get("rhs_claim", "")).strip()
            r_pre = str(o.get("rhs_premise", "")).strip()
            import re as _re
            eq = _re.compile(r"^([^=<>!]+)==(.*)$")
            num = _re.compile(r"^-?\d+(\.\d+)?$")
            pl = eq.match(pre)
            if not (pl and lhs and num.match(r_claim) and num.match(r_pre)):
                return None
            p_lhs = pl.group(1).strip()
            return bool(p_lhs == lhs and pl.group(2).strip() == r_pre
                        and r_pre != r_claim and reason)
        if m.startswith("premise_derive:"):
            # Independent re-derivation: rebuild the premise difference polys
            # from the OBLIGATION, recompute the claim difference from lhs/rhs,
            # and check the recorded combination actually reproduces it. The
            # kernel does not trust the emitter's multipliers: it recomputes
            # sum(k_i * p_i) == target with exact rational arithmetic. A
            # certificate that claims a premise combination it cannot reproduce
            # is a mismatch, not a pass.
            from fractions import Fraction as _FQ
            # The orientation is part of the ROUTE, not an emitter choice: the
            # target difference is `hi - lo`, and which side of the claim is
            # `hi` depends on the relation. Encoding it in the method lets the
            # kernel rebuild the target itself rather than trust a recorded one.
            _sign = m.split(":", 1)[1]
            diffs = o.get("premise_differences")
            combo = o.get("premise_combination")
            lhs, rhs = o.get("lhs"), o.get("rhs")
            if not (isinstance(diffs, list) and isinstance(combo, list)
                    and lhs and rhs):
                return None
            if len(combo) != len(diffs):
                return False          # malformed: refuses rather than trusts
            polys = []
            for d in diffs:
                try:
                    polys.append({tuple(k.split()): _FQ(v) for k, v in d.items()})
                except Exception:
                    return None
            try:
                if _sign in ("ge", "gt"):
                    target = T.nf(T.csub(T.parse(lhs), T.parse(rhs)))
                else:
                    target = T.nf(T.csub(T.parse(rhs), T.parse(lhs)))
            except Exception:
                return None
            if target is None:
                return None
            acc = {}
            for k, p in zip(combo, polys):
                for atom, coeff in p.items():
                    acc[atom] = acc.get(atom, _FQ(0)) + k * coeff
            return all(_FQ(c) == acc.get(a, _FQ(0))
                       for a, c in target.items())
        if m == "tptp:entail":
            # Independent re-derivation of a propositional verdict. The kernel
            # rebuilds BOTH sides from the recorded strings with term.pparse and
            # redoes the exhaustive search itself. It deliberately does not read
            # the recorded verdict, the recorded witness, or the recorded atom
            # count: a certificate that carries its own answer verifies nothing.
            from fractions import Fraction as _FQF
            import mjolnir_prover.lib.propositional as _PROP
            ax_s, cn_s = o.get("axioms"), o.get("conjecture")
            if not (isinstance(ax_s, list) and isinstance(cn_s, str)):
                return None
            try:
                axioms = [T.pparse(a) for a in ax_s]
                conjecture = T.pparse(cn_s)
            except Exception:
                return None
            # The budget is policy, so the recorded digest must match the one
            # in force NOW. Otherwise a verdict produced under a different
            # max_atoms would be compared against the current one silently.
            if o.get("theory_digest") != _PROP.budget()["digest"]:
                return None
            verdict, reason, meta = _PROP.decide(axioms, conjecture)
            if verdict is None:
                return None                     # abstain here too
            # Independently re-check the recorded counter-assignment: it must
            # actually satisfy every axiom and actually falsify the conjecture.
            # A REFUTED row whose witness is wrong is not a refutation, and
            # this is the only place that gets checked.
            if verdict is False:
                # NOTE: these return None (=> "unchecked"), not False. A
                # refutation obligation is STORED as "fail", so a False here
                # would AGREE with it and the bad witness would pass in
                # silence. None is the visible "I cannot confirm this".
                w = o.get("witness")
                # The empty valuation IS the witness when the obligation has
                # no atoms at all (a closed constant: the sole assignment is
                # {}).  Found by the ILTP protocol run: SYN916+1 is a 0-atom
                # CounterSatisfiable refutation whose witness is correctly {}
                # and was being returned as "unchecked" by the blanket
                # `not w` guard. The guard still rejects a missing or blank
                # witness on every obligation that HAS atoms.
                if not isinstance(w, dict) or (not w
                                               and meta.get("atoms", -1) != 0):
                    return None                 # a refutation without a witness
                env = {k: bool(v) for k, v in w.items()}
                try:
                    if not all(_holds(a, env) for a in axioms):
                        return None             # not even a model of the axioms
                    if _holds(conjecture, env):
                        return None             # does not actually falsify
                except Exception:
                    return None
                # The recorded witness must be the one THIS re-derivation
                # found. Checking only that it is *plausible* accepted a
                # certificate whose witness was invented: the search still
                # returned False, so a wrong witness agreed by accident and the
                # mismatch went unnoticed. The witness is part of the artifact,
                # so it is reproduced, not merely sanity-checked.
                if meta.get("witness") != env:
                    return None                 # not the witness we computed
            # Vacuity is computed here too, never trusted: a DISCHARGE requires
            # re-derivability AND non-vacuity. A derivable but vacuous
            # obligation (self-inclusion, or ex falso from inconsistent
            # premises) is left UNCHECKED -- the visible "I cannot confirm
            # this" -- so a claiming certificate that is vacuous alarms
            # downstream, while an honest vacuity-abstention (stored status
            # "abstain", reason "vacuous:*") stays clean at agreement 1.0.
            # essentials_limit=0: the kernel needs the boolean, not the
            # drop-one diagnostic, so the TCB pays no extra search.
            if not _PROP.vacuity(axioms, conjecture,
                                 essentials_limit=0)["non_vacuous"]:
                return None
            return bool(verdict)
        if m in _DECLARATION_METHODS:
            # Declared assumption — part of the artifact; the auditor sees
            # exactly what was assumed (provenance: registry/program assert).
            #
            # MISNAMING GUARD. A method that asserts truth BY DECLARATION is
            # trivially "re-derivable", which makes it a laundering channel: an
            # abstaining claim whose method was rewritten to `premise` was
            # accepted as a certificate (measured: 0 mismatches, 0 false certs).
            # The kernel re-derived the semantics of the LABEL, never whether
            # the label was the RIGHT one.
            #
            # The fix: a declaration may never BE the claim under certification.
            # `spec.expr` is the literal placeholder "assert-claims", so the
            # claim is identified by POSITION instead: the claim is the LAST
            # assert, and certificate.py's `_is_last` keeps that last assert out
            # of the premise tier. So a declaration must sit on a strictly
            # earlier line than the last obligation in the certificate.
            # Verified against 591 honest certificates: 6 premise obligations,
            # every one on an earlier line than the claim.
            _last = max([int(x.get("line") or 0)
                         for x in cert.get("obligations", [])] or [0])
            _here = int(o.get("line") or 0)
            if _here >= _last:
                return None                 # => "unchecked", visibly not verified
            return True
        if m == "no_precondition":
            return con == "vacuous"
    except Exception:
        return None                              # cannot re-derive (not a refute)
    return None                                  # unsupported method -> unchecked


def replay(cert: dict) -> dict:
    """Re-discharge every obligation from the certificate alone.

    Returns {agreement, statuses, mismatches, false_certs, discharged, reason}.
    agreement = 1.0 iff every stored obligation's verdict matches the
    independently re-derived one AND provenance re-checks clean.
    """
    mismatches = []
    prog = cert.get("program")
    spec = cert["spec"]
    trace_ok = False
    if cert.get("trace") is not None:
        from . import trace as TR, stepcheck as SC
        digest_ok = (not cert.get("trace_digest")) or \
            TR.digest(cert["trace"]) == cert["trace_digest"]
        sc = SC.check_trace(cert["trace"])
        trace_ok = digest_ok and sc["valid"]
        if not digest_ok:
            mismatches.append("trace_digest_mismatch")
        if not sc["valid"]:
            mismatches.append(f"trace_invalid:{sc['bad'][:3]}")

    # provenance re-check (never trust the emitter's status)
    if spec.get("source") not in ("assert", "type_hint") or not spec.get("confirmed"):
        if cert.get("status") == "CERTIFIED":
            mismatches.append("provenance_violation")

    statuses = []
    decided_despite_abstain = []
    obs = cert.get("obligations", [])
    for o in obs:
        derived = _rederive(cert, o, trace_ok)
        if o["status"] == "ok" and derived is False:
            mismatches.append("false_certificate")
        elif o["status"] != "ok" and derived is True:
            mismatches.append(f"status_mismatch:{o['kind']}")
        elif o["status"] == "abstain" and derived is False:
            # The kernel RE-DERIVED a definite verdict where the emitter
            # abstained. That is not a soundness defect — abstention is
            # conservative and claims nothing — but reporting it as
            # `agreement 1.0` hid a real finding: `min(n,0)==0` is REFUTED by
            # the kernel while the certificate says "undecidable". The
            # asymmetry is deliberate (only false certificates break
            # agreement), so it is surfaced as its own count instead.
            decided_despite_abstain.append(f"{o['kind']}:{o.get('method')}")
        statuses.append("ok" if derived is True else
                        ("fail" if derived is False else "unchecked"))

    # program-backed re-derivation (straight-line units): recompute the post
    if prog is not None and obs:
        try:
            steps = [(s[0], s[1], _term(s[2])) for s in prog["steps"]]
            if any(k != "assign" for k, _n, _t in steps):
                mismatches.append("out_of_fragment")
            else:
                env = {}
                for _k, name, rhs in steps:
                    env[name] = rhs
                spec_expr = _term(spec["expr"])
                if spec_expr[0] != "eq":
                    mismatches.append("spec_not_eq")
                else:
                    outs = prog["outputs"]
                    derived = [T.sub_all(env.get(n, T.var(n)), env) for n in outs]
                    stored = obs[-1]
                    if T.key(_term(stored["conclusion"])) != T.key(derived[0]):
                        mismatches.append("conclusion_mismatch")
                    ok = T.decide_eq(spec_expr[2], derived[0])
                    if stored["status"] != ("ok" if ok else "fail"):
                        mismatches.append(f"status_mismatch:{stored['kind']}")
        except Exception as e:
            mismatches.append(f"replay_error:{type(e).__name__}")

    # A certificate that CLAIMS a verdict must have every obligation actually
    # re-derived. An "unchecked" obligation contributes no mismatch — derived is
    # None, which trips neither branch above — so a relabelled abstention
    # scored agreement 1.0 while every obligation was unverified. That is
    # laundering with the agreement number still showing 1.0, which is worse
    # than a plain false certificate because it looks clean.
    # Conservative abstention (cert says UNVERIFIABLE) is unaffected.
    if cert.get("status") == "CERTIFIED" and "unchecked" in statuses:
        mismatches.append("certified_with_unverified_obligations")

    false_certs = sum(1 for m in mismatches
                      if m in ("false_certificate", "provenance_violation"))
    agreement = 1.0 if not mismatches else 0.0
    unchecked = sum(1 for s in statuses if s == "unchecked")
    return dict(agreement=agreement, statuses=statuses, mismatches=mismatches,
                false_certs=false_certs, unchecked=unchecked, discharged=len(obs),
                decided_despite_abstain=decided_despite_abstain,
                reason=cert.get("reason", ""))


def _interval_trace_verdict(o, trace):
    """Return the TRACE's proven verdict for this obligation's relation — the
    comparison against the stored status happens in the caller, so a trace
    that proves the opposite of a recorded verdict is a detectable mismatch."""
    if not trace:
        return None
    rel = o["method"].split(":", 1)[1]
    dec = [s for s in trace if s["rule"].startswith(f"interval-{rel}")]
    return dec[-1]["out"] if dec else None


def replay_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return replay(json.load(f))
