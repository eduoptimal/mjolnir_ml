"""mjolnir_prover.lib.mine — scalable directive/claim miner for trusted repos.

Framework role: the miner is a PROPOSER. It harvests claim-shaped constructs
from reference code and emits UNLABELED candidates (expect=null) with full
provenance chains. It never confirms, never labels, never verifies. The
registry's confirmed tier is filled only by the confirmation pass (oracle +
human) — separate concern, separate run.

Candidate kinds (each maps to a tier):
  vc_claim     assert with arithmetic Eq/ineq       -> tier "static"  (gate can try)
  guard        `if <pred>: raise` guard clause      -> tier "premise" (requires-shaped)
  runtime_cap  assert/cap on computed runtime value  -> tier "monitor" (runtime observer)
  test_pair    test-derived input->output pairs      -> tier "spec"    (spec source: test)

Scalability: pure stdlib ast, streaming over files, per-file time budget,
per-file candidate cap, deterministic ordering (sorted paths, crc32 ids),
no heavy imports (no torch/sympy here). Works on any repo layout.
"""
from __future__ import annotations
import ast, os, time, zlib

from . import certificate as C


def _cid(repo, rel, line, kind):
    return "m::" + zlib.crc32(f"{repo}|{rel}|{line}|{kind}".encode()).to_bytes(4, "big").hex()


def _cmp_pred(test):
    """Extract a canonical predicate string from a Compare node, or None."""
    try:
        return ast.unparse(test)
    except Exception:
        return None


def _arith_only(node):
    """True iff the expression is pure +-/* //% int arithmetic on names."""
    try:
        C._ast2term(node)
        return True
    except ValueError:
        return False


def _form_of(pred):
    """FORM axis of the claim ontology: the construct family of a predicate.

    Structural, exclusive and stable (typing only — never a truth input): the
    rung ladder is addressed through this axis. Precedence is fixed so adding
    classes can never reclassify existing claims (orthogonality invariant)."""
    try:
        test = ast.parse(pred or "", mode="eval").body
    except (SyntaxError, ValueError):
        return "other"
    if isinstance(test, ast.BoolOp):
        return ("conjunction" if isinstance(test.op, ast.And) else "disjunction")
    if isinstance(test, ast.Call):
        if isinstance(test.func, ast.Name) and test.func.id == "isinstance":
            return "type-check"                 # assert isinstance(x, T)
        return "call-value"                     # truthy call assert
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return "truthiness"
    op = type(test.ops[0])
    sides = [test.left] + list(test.comparators)

    def _walk_shallow(nodes):
        for n in nodes:
            yield n
            if isinstance(n, ast.Call):
                yield n.func
                for a in n.args:
                    yield a
                for kw in n.keywords:
                    yield kw.value
            elif isinstance(n, (ast.Attribute, ast.Subscript)):
                yield n.value

    if op in (ast.Is, ast.IsNot):
        return "identity"
    if op in (ast.In, ast.NotIn):
        return "membership"
    if any(isinstance(e, ast.Call) and isinstance(e.func, ast.Name)
           and e.func.id == "isinstance" for e in _walk_shallow(sides)):
        return "type-check"
    if any(isinstance(e, ast.Call) and isinstance(e.func, ast.Name)
           and e.func.id == "len" for e in _walk_shallow(sides)):
        return "length"
    if any(isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute)
           for e in _walk_shallow(sides)):
        return "string-method"
    if any(isinstance(e, ast.Call) for e in _walk_shallow(sides)):
        return "call-value"
    if any(isinstance(e, ast.Attribute) for e in _walk_shallow(sides)):
        return "attr-value"
    if any(isinstance(e, ast.Subscript) for e in _walk_shallow(sides)):
        return "subscript-value"
    if op in (ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq):
        return "arithmetic"
    return "comparison"


def _classify_assert(test):
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    l_arith, r_arith = _arith_only(test.left), all(_arith_only(c) for c in test.comparators)
    if l_arith and r_arith:
        # Static-theorem candidates need BOTH sides to be compound arithmetic
        # (BinOp/Constant). A bare Name on either side is an opaque runtime
        # value: the claim is consistency-at-runtime, not a universal theorem.
        atomic_l = isinstance(test.left, ast.Name)
        atomic_r = len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Name)
        # opaque VALUES (names OR literal collections) compared with Eq:
        # the claim's truth depends on runtime assignment/mutation context
        # (e.g. xs == [sorted...] after sort(xs)) -> runtime claim, not theorem
        opaque_l = isinstance(test.left, (ast.Name, ast.List, ast.Tuple)) or (
            isinstance(test.left, ast.Constant) and isinstance(test.left.value, str))
        opaque_r = len(test.comparators) == 1 and (
            isinstance(test.comparators[0], (ast.Name, ast.List, ast.Tuple)) or (
            isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, str)))
        if isinstance(test.ops[0], ast.Eq) and (opaque_l or opaque_r):
            # identical literal content is still exact (decidable); different
            # content on an opaque side is a runtime claim
            if not (ast.dump(test.left) == ast.dump(test.comparators[0])):
                return "trivial"
        if atomic_l or atomic_r:
            return "trivial"          # runtime claim -> monitor tier (not a theorem)
        if isinstance(test.ops[0], ast.Eq) \
                and ast.dump(test.left) == ast.dump(test.comparators[0]):
            return "trivial"          # e == e: reflexive, zero information
        return "static"
    return "monitor"


def _unit_assert(node):
    """Recognize unittest/pytest assertion methods inside tests:
    self.assertEqual(a, b) -> a == b, assertLess -> a < b, assertTrue -> bool.
    Returns (predicate, kind) or None. Deterministic, AST-only."""
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return None
    fn = node.value.func
    if not isinstance(fn, ast.Attribute):
        return None
    op = {"assertEqual": ast.Eq, "assertNotEqual": ast.NotEq,
          "assertLess": ast.Lt, "assertLessEqual": ast.LtE,
          "assertGreater": ast.Gt, "assertGreaterEqual": ast.GtE}.get(fn.attr)
    if op is not None and len(node.value.args) >= 2:
        cmp_node = ast.Compare(left=node.value.args[0], ops=[op()],
                               comparators=[node.value.args[1]])
        return (_cmp_pred(cmp_node), _classify_assert(cmp_node))
    if fn.attr in ("assertTrue", "assertFalse") and node.value.args:
        return (_cmp_pred(node.value.args[0]), "monitor")
    return None


def _assigned_names(fn):
    """Names ASSIGNED inside fn (excluding parameters) — i.e. runtime locals."""
    params = {a.arg for a in getattr(fn.args, "args", [])}
    params |= {a.arg for a in getattr(fn.args, "kwonlyargs", [])}
    s = set()
    for n in ast.walk(fn):
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, ast.AugAssign):
            tgts = [n.target]
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            tgts = [n.target]
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            tgts = [n.optional_vars]
        for t in tgts:
            for x in ast.walk(t):
                if isinstance(x, ast.Name):
                    s.add(x.id)
    return s - params


def _local_scopes(tree):
    """Map id(assert/unittest-call node) -> enclosing function's assigned set."""
    m = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            a = _assigned_names(fn)
            for c in ast.walk(fn):
                if isinstance(c, ast.Assert) or (
                        isinstance(c, ast.Expr) and isinstance(c.value, ast.Call)):
                    m.setdefault(id(c), a)
    return m


def mine_file(path, rel, repo, commit, budget_s=2.0, cap=200):
    """Mine one file. Returns list of candidate dicts (unlabeled)."""
    t0 = time.time()
    out = []
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except (SyntaxError, OSError):
        return out
    is_test = ("test" in os.path.basename(rel).lower()
               or "test" in rel.lower().split(os.sep)[-2:-1])
    scopes = _local_scopes(tree)

    # parent map: an assert's unit must carry its PRECONDITIONS. `intervals.py`
    # states six box-membership asserts before each claim; extracting the claim
    # alone wrapped it as a bare theorem over free params, and the authority
    # correctly refuted a conditional statement whose condition the EXTRACTION
    # dropped — seven false refutations of true code, found by running mjolnir
    # on itself. Context = the contiguous run of sibling asserts directly in
    # the enclosing function body (statements, so comments don't break it; any
    # non-assert statement does — an if/for block is not a straight-line
    # prefix, and guessing past it would be the same overreach in reverse).
    _parent = {}
    for _p in ast.walk(tree):
        for _c in ast.iter_child_nodes(_p):
            _parent[id(_c)] = _p

    def _precondition_unit(assert_node):
        """The function's leading assert-run + this assert as one unit, or
        None when there is no such context (single asserts keep the bare
        wrapper — no behavior change) or the assert is not top-level in its
        function."""
        fn = _parent.get(id(assert_node))
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        body = fn.body
        idx = next((i for i, s in enumerate(body) if s is assert_node), None)
        if idx is None:
            return None                    # nested: not a straight-line prefix
        run = []
        j = idx - 1
        while j >= 0 and isinstance(body[j], ast.Assert):
            run.append(body[j])
            j -= 1
        if not run:
            return None                    # no preconditions: bare claim stands
        run.reverse()
        lines = [f"def {fn.name}({ast.unparse(fn.args)}):"]
        for st in run + [assert_node]:
            lines.append("    " + ast.unparse(st))
        return "\n".join(lines) + "\n"

    def _runtime(node, pred):
        """The runtime locals (assigned in the enclosing function) that the
        predicate references, or empty. Such a claim is consistency-at-runtime,
        not a universally-quantified theorem — a DETERMINISTIC AST decision,
        independent of any proposer/encoder."""
        return sorted(set(_names(pred)) & scopes.get(id(node), set()))

    for node in ast.walk(tree):
        if time.time() - t0 > budget_s or len(out) >= cap:
            break
        # ---- unittest/pytest assertion methods (the test-harvest tier) -----
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            ua = _unit_assert(node)
            if ua is not None:
                pred, kind = ua
                if kind and pred:
                    # runtime-assigned locals do NOT demote the tier: the claim
                    # stays theorem-shaped (strengthening marker); the verdict
                    # authority chooses the faithful reading.
                    rl = _runtime(node, pred) if kind == "static" else []
                    origin = dict(repo=repo, commit=commit, file=rel,
                                  line=node.lineno, construct="test_assert",
                                  is_test=is_test)
                    if rl:
                        origin["runtime_locals"] = rl
                    out.append(dict(
                        id=_cid(repo, rel, node.lineno, "test_assert"),
                        family="vc" if kind == "static" else "monitor",
                        tier="static" if kind == "static" else "monitor",
                        form=_form_of(pred),
                        predicate=pred, expect=None, origin=origin,
                        provenance=dict(source="mined_test", confirmed=False)))
                continue
        # ---- asserts ------------------------------------------------------
        if isinstance(node, ast.Assert):
            kind = _classify_assert(node.test)
            pred = _cmp_pred(node.test)
            if kind and pred:
                # runtime-assigned locals do NOT demote the tier: the claim
                # stays theorem-shaped (strengthening marker); the verdict
                # authority chooses the faithful reading.
                rl = _runtime(node, pred) if kind == "static" else []
                origin = dict(repo=repo, commit=commit, file=rel,
                              line=node.lineno, construct="assert",
                              is_test=is_test)
                if rl:
                    origin["runtime_locals"] = rl
                cand = dict(
                    id=_cid(repo, rel, node.lineno, "assert"),
                    family="vc" if kind == "static" else "monitor",
                    tier="static" if kind == "static" else "monitor",
                    form=_form_of(pred),
                    predicate=pred, expect=None, origin=origin,
                    provenance=dict(source="mined", confirmed=False))
                # preconditions travel with the claim (theorem-tier only:
                # monitor class stays observation-only by design — a unit on
                # a runtime-observed value would invite the same overreach).
                if kind == "static":
                    _cu = _precondition_unit(node)
                    if _cu:
                        cand["canonical_unit"] = _cu
                out.append(cand)
        # ---- guard clauses: `if <pred>: raise ...` ------------------------
        elif isinstance(node, ast.If):
            body = node.body
            raises = any(isinstance(s, ast.Raise) for s in body)
            if raises:
                pred = _cmp_pred(node.test)
                if pred:
                    # negation-aware: `if not X: raise` declares premise X
                    neg = isinstance(node.test, ast.UnaryOp) and \
                        isinstance(node.test.op, ast.Not)
                    # R1 premise lowering: the canonical unit is the declared
                    # premise (`not <guard>`) lowered exactly — comparison
                    # negation flips the operator, `not X` unwraps to X. No
                    # exact lowering -> NO canonical unit (honest abstention).
                    out.append(dict(
                        id=_cid(repo, rel, node.lineno, "guard"),
                        family="guard", tier="premise",
                        form=_form_of(pred),
                        predicate=("NOT " + pred) if neg else pred,
                        expect=None,
                        canonical_unit=_premise_unit(node.test),
                        origin=dict(repo=repo, commit=commit, file=rel,
                                    line=node.lineno, construct="guard_raise"),
                        provenance=dict(source="mined", confirmed=False)))
        # ---- range/loop bounds as caps -------------------------------------
        elif isinstance(node, ast.For) and isinstance(node.iter, ast.Call) \
                and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            for arg in node.iter.args:
                if isinstance(arg, ast.Name) or isinstance(arg, ast.BinOp):
                    p = _cmp_pred(arg)
                    if p:
                        bound = f"{p} >= 0"
                        unit = None
                        try:
                            code = (f"def f({', '.join(_names(bound))}):\n"
                                    f"    assert {bound}\n")
                            compile(code, "<bound>", "exec")
                            unit = code
                        except SyntaxError:
                            unit = None
                        out.append(dict(
                            id=_cid(repo, rel, node.lineno, "bound"),
                            family="bound", tier="premise",
                            predicate=bound, form="arithmetic", annotation="range bound",
                            expect=None, canonical_unit=unit,
                            origin=dict(repo=repo, commit=commit, file=rel,
                                        line=node.lineno, construct="range_bound"),
                            provenance=dict(source="mined", confirmed=False)))
    return out


def _premise_unit(test):
    """R1 premise lowering: the canonical unit declared by a guard.

    `if <guard>: raise` declares NOT <guard> on every path past the guard.
    Exact lowering only: comparison negation flips the operator (< <-> >=,
    > <-> <=); `if not X: raise` declares X directly. Anything else gets NO
    canonical unit — an honest abstention, never an approximate premise."""
    try:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            e = test.operand
        elif isinstance(test, ast.Compare) and len(test.ops) == 1:
            flip = {ast.Lt: ast.GtE, ast.GtE: ast.Lt,
                    ast.Gt: ast.LtE, ast.LtE: ast.Gt}
            op = flip.get(type(test.ops[0]))
            if op is None:
                return None
            e = ast.Compare(test.left, [op()], list(test.comparators))
        else:
            return None
        expr = ast.unparse(e)
        code = (f"def f({', '.join(_names(expr))}):\n"
                f"    assert {expr}\n")
        compile(code, "<premise>", "exec")
        return code
    except Exception:
        return None


# Bounds for the DETERMINISTIC frontends (.sql/.java). Deliberately not `cap`:
# those files are parsed, not walked under a time budget, so a small number here
# would be a policy about how much of someone's schema to read.
_DETERMINISTIC_FILE_CAP = 5000
_DETERMINISTIC_REPO_CAP = 50_000


def mine_repo(root, repo_name=None, commit=None, budget_per_file=2.0, cap=200,
              ignore=None):
    """Mine a repo tree. Deterministic: sorted paths, stable ids.

    The miner is a PROPOSER: it emits structural candidates only and never
    calls the prover. Verdicts are assigned exclusively by
    mjolnir_prover.lib.certify (the single verdict authority).

    `ignore` is an optional predicate (abs path -> bool) from the caller's
    scope file — one ignore contract for every surface, resolved at the app
    layer where the file lives."""
    import subprocess
    if commit is None:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                    capture_output=True, text=True).stdout.strip()[:12]
        except Exception:
            commit = "unknown"
    repo_name = repo_name or os.path.basename(root)
    cands = []
    sql_files = []          # (full, rel, text): mined after the corpus index
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs
            if d not in (".git", "__pycache__", ".venv")
            and not (ignore and ignore(os.path.join(dirpath, d))))
        for f in sorted(files):
            full = os.path.join(dirpath, f)
            if ignore and ignore(full):
                continue
            rel = os.path.relpath(full, root)
            ext = os.path.splitext(f)[1].lower()
            if ext == ".py":
                cands.extend(mine_file(full, rel, repo_name, commit,
                                       budget_s=budget_per_file, cap=cap))
            elif ext in (".sql", ".java"):
                # THE FRONTENDS EXISTED AND WERE NEVER CALLED, which is why
                # BANKING_SQL and TheAlgorithmsJava both reported zero claims --
                # a Java and a SQL tree walked by a Python-only miner is an empty
                # tree.
                #
                # AND THE FIRST VERSION OF THIS WIRING TRUNCATED THEM SILENTLY.
                # It reused `cap` -- a per-file TIME budget for AST walking -- as
                # a slice, plus a `cap * 4` global stop. On BANKING_SQL that
                # stopped at ~240 candidates out of ~3,400, and because
                # `mine_sql_text` emits a table's declarations BEFORE its
                # theorems, the slice dropped the money claims first: the DDL has
                # 159 scale>0 columns (NUMERIC(19,4), NUMBER(*,2)) and 5
                # certificates came out, all scale=0. A deterministic frontend
                # has no time to budget, so these bounds are only a guard against
                # a pathological file -- and the cap is large enough that it
                # never becomes a policy.
                if len(cands) >= _DETERMINISTIC_REPO_CAP:
                    continue
                try:
                    text = open(full, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                if ext == ".sql":
                    # Deferred: NOT NULL contracts and write paths routinely
                    # live in different files, and the corpus index must be
                    # built once over the whole tree -- identically to
                    # `mine_sql_repo` (harvest/transfer) or the same input
                    # yields different claim ids on different surfaces.
                    sql_files.append((full, rel, text))
                    continue
                try:
                    from . import frontend_java as FJ
                    got = FJ.mine_java_source(text, rel, repo_name, commit)
                except Exception:  # noqa: BLE001
                    continue          # one malformed file must not kill the run
                cands.extend(got[:_DETERMINISTIC_FILE_CAP])
    if sql_files:
        from . import frontend_sql as FS
        index = FS.sql_index([(rel, text) for _f, rel, text in sql_files])
        for full, rel, text in sql_files:
            if len(cands) >= _DETERMINISTIC_REPO_CAP:
                break
            try:
                got = FS.mine_sql_text(text, rel, repo_name, commit,
                                       index=index)
            except Exception:  # noqa: BLE001
                continue
            cands.extend(got[:_DETERMINISTIC_FILE_CAP])
    return cands


def _names(pred):
    """Valid, sanitized python identifiers from a predicate (bounded)."""
    import re
    names, seen = [], set()
    for n in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", pred):
        if n in seen or n == "assert" or "." in n:
            continue
        seen.add(n)
        names.append(n)
        if len(names) >= 12:                  # bounded wrapper signature
            break
    return names
