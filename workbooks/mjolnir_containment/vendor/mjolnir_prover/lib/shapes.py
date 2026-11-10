"""mjolnir_prover.lib.shapes — shape collapse + confirmation ledger for mined candidates.

Trust discipline (safety.md, registry section): humans confirm SHAPES (rules),
instances inherit. The confirmation ledger is append-only JSONL: every
promotion records shape id, rule, evidence (oracle labels, provenance counts),
and the human signature. The gate reads ONLY confirmed entries; the candidate
queue stays fail-open.

Shape function: normalize a predicate into a pattern (numbers -> N, names -> V,
repeated ops folded) keyed by construct. Deterministic, versioned (shape v1).
"""
from __future__ import annotations
import json, re, hashlib
from . import paths as _paths

_NNUM = re.compile(r"\b(0x[0-9a-fA-F]+|\d+\.?\d*)\b")
_NAME = re.compile(r"\b[A-Za-z_][A-Za-z_0-9.]*\b")
_SHAPE_V = "shape-v1"


def _norm(pred: str) -> str:
    """Single-pass normalization: numbers -> N, names -> V (numbers first,
    and the N placeholder is protected from the name pass)."""
    p = _NNUM.sub("\x00", pred)          # sentinel the name regex cannot match
    p = _NAME.sub("V", p)
    p = p.replace("\x00", "N")
    p = re.sub(r"(V)( \+ V)+", r"\1+...", p)
    return p


def shape_id(cand) -> str:
    key = f"{cand['origin']['construct']}|{cand['tier']}|{_norm(cand['predicate'])}"
    return "sh_" + hashlib.sha1(f"{_SHAPE_V}|{key}".encode()).hexdigest()[:10]


def collapse(cands) -> dict:
    """Group candidates by shape id. Returns {shape_id: {shape, members, oracle}}."""
    groups = {}
    for c in cands:
        sid = shape_id(c)
        g = groups.setdefault(sid, dict(sid=sid, shape=_shape_text(c),
                                        construct=c["origin"]["construct"],
                                        tier=c["tier"], members=[], oracle=codes()))
        g["members"].append(c)
        o = (c.get("oracle") or {}).get("oracle")
        if o:
            g["oracle"][o] = g["oracle"].get(o, 0) + 1
    return groups


def codes():
    return {"CERTIFIED": 0, "REFUTED": 0, "UNVERIFIABLE": 0}


def _shape_text(cand):
    return _norm(cand["predicate"])


# --------------------------------------------------------------- ledger -------
def load_ledger(path) -> dict:
    """Append-only confirmation ledger: shape_id -> last record."""
    out = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[r["shape_id"]] = r
    return out


def _signing_key():
    """Identity for confirmations. Resolution order:
    1. MJOLNIR_SIGNING_KEY env (hex or key-id)
    2. mjolnir_prover/registry/signing_key (git-tracked file)
    No key -> confirmations are REJECTED (fail-closed: no anonymous promotion)."""
    env = os.environ.get("MJOLNIR_SIGNING_KEY")
    if env:
        return env.strip()
    keyfile = _paths.registry("signing_key")
    if os.path.exists(keyfile):
        return open(keyfile, encoding="utf-8").read().strip()
    return None


def _sig(rec: dict) -> str:
    payload = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{_signing_key()}|{payload}".encode()).hexdigest()


def sign(rec: dict) -> dict:
    """Attach signature to a confirmation record (fail-closed identity)."""
    body = {k: v for k, v in rec.items() if k != "sig"}
    rec["sig"] = _sig(body)
    return rec


def verify(rec: dict) -> bool:
    body = {k: v for k, v in rec.items() if k != "sig"}
    return rec.get("sig") == _sig(body)


_DECISIONS = ("confirmed", "rejected", "enforced", "revoked")


def confirm(groups, sid, decision, by, note=""):
    """Build one append-only, SIGNED confirmation record.

    Fail-closed identity: requires a signing key (env or registry file). The
    signature binds every field — tampering with a ledger line invalidates it.

    Decision vocabulary is strict:
      confirmed — PROOF-backed sign-off (a certificate exists or the decision
                  drives premise injection); never stamps an observation.
      rejected  — veto; the shape stays fail-open.
      enforced  — RUNTIME-ENFORCEMENT sign-off for an observation-class shape:
                  the claim is a runtime contract the watchdog enforces. It is
                  bookkeeping for the enforcement plane — it can NEVER upgrade
                  a verdict (evidence migrates observation -> proof only via a
                  disposal rung through the verdict authority).
      revoked   — UNSIGN. The ledger is append-only and immutable: nothing is
                  deleted. A revocation appends a new signed record that
                  supersedes the prior decision (latest same-authority decision
                  wins); the full signature history remains for audit.
    """
    if decision not in _DECISIONS:
        raise ValueError(
            f"unknown decision: {decision!r} (have {_DECISIONS})")
    if not _signing_key():
        raise SystemExit(
            "confirm: no signing key — set MJOLNIR_SIGNING_KEY or write "
            "mjolnir_prover/registry/signing_key (identity is required; anonymous "
            "promotion is not allowed)")
    g = groups[sid]
    ev = dict(members=len(g["members"]),
              oracle=dict(g["oracle"]),
              origins=sorted({f"{m['origin']['repo']}:{m['origin']['file']}:{m['origin']['line']}"
                              for m in g["members"]})[:5],
              repos=sorted({m["origin"]["repo"] for m in g["members"]}))
    # persist a representative predicate (a SHAPE, not source code) so generative
    # transfer can recover the schema template without reloading the mined files
    rep = min(g["members"], key=lambda m: (
        m["origin"].get("repo", ""), m["origin"].get("file", ""),
        m["origin"].get("line", 0)))
    return sign(dict(ts=_now(), shape_id=sid, shape=g["shape"],
                     construct=g["construct"], tier=g["tier"], decision=decision,
                     by=by, identity=_identity(), note=note, evidence=ev,
                     sample=bare_pred(rep["predicate"]),
                     shape_version=_SHAPE_V))


def _identity():
    key = _signing_key()
    return "k_" + hashlib.sha1(key.encode()).hexdigest()[:8]


_HUMAN = 0          # precedence: human-signed decisions outrank machine records
_MACHINE = 1


def load_ledger_verified(path) -> dict:
    """Load ledger, verifying signatures. Keyed view: HUMAN decisions (with an
    evidence bundle) are authoritative — machine transfer records can add
    evidence for a shape but can never shadow or override a human decision."""
    out, bad = {}, []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not verify(r):
                    bad.append(i + 1)
                    continue
                pri = _HUMAN if "evidence" in r else _MACHINE
                cur = out.get(r["shape_id"])
                if cur is None or pri < cur[0]:
                    out[r["shape_id"]] = (pri, r)
                elif pri == cur[0]:
                    # same authority class: the LATEST signed decision wins.
                    # The ledger stays append-only (nothing is erased); a later
                    # record is a signed change — e.g. revoked supersedes
                    # confirmed, and the whole history remains for audit.
                    out[r["shape_id"]] = (pri, r)
    if bad:
        raise SystemExit(f"ledger {path}: TAMPERED lines {bad} — audit before use")
    return {k: v[1] for k, v in out.items()}


# --------------------------------------------------------- knowledge transfer -
_KEYWORDS = {"and", "or", "not", "in", "is", "True", "False", "None",
             "if", "else", "assert", "lambda"}
_ID_TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_NUM_TOK = re.compile(r"\d+\.?\d*")
_TOK = re.compile(r"(?P<id>[A-Za-z_][A-Za-z_0-9]*)|(?P<num>\d+\.?\d*)")


_ASSERT_RX = re.compile(r"^\s*assert\s+(?P<expr>.*?)\s*:?\s*$", re.S)


def bare_pred(pred: str) -> str:
    """The predicate as a bare EXPRESSION: miner candidates are inconsistent —
    some carry a leading `assert ` (e.g. Java/SQL guards), some do not. Normalize
    at the boundary so a schema's template and the oracle wrapper agree."""
    m = _ASSERT_RX.match(pred or "")
    return (m.group("expr") if m else (pred or "")).strip()


def _template(pred: str):
    """Faithful positional template of a predicate: identifiers -> {v0},{v1}...
    and numeric literals -> {n0},{n1}... by first occurrence (single pass, so
    placeholders are never re-scanned). Returns (template, meta) where meta =
    {"vars": [names], "nums": [literals]}. Keywords/booleans stay literal."""
    vars_, nums = [], []

    def rep(m):
        if m.group("id"):
            n = m.group("id")
            if n in _KEYWORDS:
                return n
            if n not in vars_:
                vars_.append(n)
            return "{v%d}" % vars_.index(n)
        n = m.group("num")
        if n not in nums:
            nums.append(n)
        return "{n%d}" % nums.index(n)

    return _TOK.sub(rep, bare_pred(pred)), {"vars": vars_, "nums": nums}


def instantiate(template: str, sub: dict) -> str | None:
    """Substitute {v0}/{n0} placeholders. Returns None if any placeholder is
    left unbound (an incomplete instance is never adopted)."""
    out = template
    for k, v in sub.items():
        out = out.replace("{%s}" % k, str(v))
    return None if ("{v" in out or "{n" in out) else out


def environment(cands, limit: int = 64):
    """The target's typed environment as symbol/literal pools, derived from its
    candidates (deterministic, sorted, bounded). Callers may override."""
    syms, nums = [], []
    for c in cands:
        p = c.get("predicate") or ""
        for n in _ID_TOK.findall(p):
            if n not in _KEYWORDS and n not in syms:
                syms.append(n)
        for n in _NUM_TOK.findall(p):
            if n not in nums:
                nums.append(n)
    return sorted(syms)[:limit], sorted(nums)[:limit]


def _validate(rec: dict, members: list, oracle_fn, collect=None):
    """The one adoption gate, shared by matched and generated transfer. A shape
    is adopted only on positive, contradiction-free target evidence:
      static          -> oracle each instance; adopt iff 0 REFUTED and >=1 CERTIFIED
      premise/monitor -> structural (predicate parses, tier matches); adopt iff
                         every instance parses; runtime stays pending.
    The general rule is never adopted blind: a target refutation scope-narrows it.
    If `collect` is a list, each (predicate, status) is appended for materialisation.
    """
    import ast as _ast
    val = dict(instances=len(members), certified=0, refuted=0, unverifiable=0,
               structural=0, method="")
    adopt = True
    if rec["tier"] == "static" and oracle_fn is not None:
        val["method"] = "oracle"
        for m in members:
            p = bare_pred(m["predicate"])
            code = m.get("unit") or (f"def f({', '.join(_pred_names(p))}):\n"
                                     f"    assert {p}\n")
            v = oracle_fn(code).get("status")
            if collect is not None:
                collect.append((m["predicate"], v))
            if v == "CERTIFIED":
                val["certified"] += 1
            elif v == "REFUTED":
                val["refuted"] += 1              # contradiction on target
            else:
                val["unverifiable"] += 1
        if val["refuted"] > 0:
            adopt = False                        # scope-narrow: not a theorem here
        elif val["certified"] == 0:
            adopt = False                        # no positive evidence on target
    else:
        val["method"] = "structural"
        for m in members:
            try:
                _ast.parse(bare_pred(m["predicate"]), mode="eval")
                val["structural"] += 1
                if collect is not None:
                    collect.append((m["predicate"], "parsed"))
            except (SyntaxError, ValueError):
                pass
        if val["structural"] < len(members):
            adopt = False
    return val, adopt


def _confirmed(from_ledger):
    """Only human-signed shape confirmations transfer; inherited/transfer records
    lack the evidence bundle and must not chain-transfer."""
    return {sid: r for sid, r in from_ledger.items()
            if r.get("decision") == "confirmed" and "evidence" in r}


def _record(sid, rec, val, adopt, note, **extra):
    return dict(
        shape_id=sid, shape=rec["shape"], construct=rec["construct"],
        tier=rec["tier"], decision="confirmed" if adopt else "scope_narrowed",
        inherited_from=dict(shape_id=sid, decision=rec["decision"],
                            by=rec.get("by"), identity=rec.get("identity"),
                            repos=rec["evidence"]["repos"]),
        validation=val, note=note, ts=_now(), shape_version=_SHAPE_V, **extra)


def transfer_validate(from_cands, to_cands, from_ledger, oracle_fn=None,
                      threshold=1.0):
    """MATCHED transfer: for every FROM-confirmed shape that already has instances
    in TO, validate those targets and adopt only on contradiction-free evidence.
    Returns per-shape records; nothing is written here (caller signs)."""
    confirmed = _confirmed(from_ledger)
    to_by_sid = {}
    for c in to_cands:
        to_by_sid.setdefault(shape_id(c), []).append(c)
    records = []
    for sid, rec in confirmed.items():
        members = to_by_sid.get(sid, [])
        if not members:
            continue
        val, adopt = _validate(rec, members, oracle_fn)
        records.append(_record(sid, rec, val, adopt,
                               "transferred: validated on target data before adoption"
                               if adopt else
                               "scope-narrowed: target validation contradicts or "
                               "lacks evidence", source="matched"))
    return records


def transfer_generate(from_cands, to_cands, from_ledger, oracle_fn=None,
                      symbols=None, literals=None, max_per_shape=256,
                      collect_instances=False):
    """GENERATIVE transfer (the cold-start path): the general rule is instantiated
    into the target's own typed environment, and the *generated* instances are then
    put through the same adoption gate. This is dual to `transfer_validate`: it does
    not wait for target instances to exist.

    Instances come from, in order:
      (a) positional substitution of the schema's variables with the variables of
          each target candidate (keeps symbols domain-cohesive);
      (b) a bounded product over an explicit/literal symbol pool when the target has
          no candidates to borrow from (pure augmentation);
      (c) a bounded sweep of the schema's constants over the target's literal pool.
    Reliability is unchanged: adoption requires >=1 CERTIFIED and 0 REFUTED (static)
    or full parse (premise/monitor). A bad instantiation is a no-op, never a false
    certificate. Nothing is written here (caller signs).
    """
    import itertools
    confirmed = _confirmed(from_ledger)
    env_syms, env_nums = environment(to_cands)
    syms = list(symbols) if symbols else env_syms
    nums = list(literals) if literals else env_nums
    to_sorted = sorted(to_cands, key=lambda c: (
        c["origin"].get("repo", ""), c["origin"].get("file", ""),
        c["origin"].get("line", 0), c.get("id", "")))
    to_meta = [(c, _ID_TOK.findall(c.get("predicate") or "")) for c in to_sorted]

    records = []
    for sid in sorted(confirmed):
        rec = confirmed[sid]
        members = [c for c in from_cands if shape_id(c) == sid]
        if not members and rec.get("sample"):
            # fall back to the schema sample persisted in the signed confirmation
            members = [{"predicate": rec["sample"], "tier": rec["tier"],
                        "origin": {"construct": rec["construct"]}}]
        if not members:
            continue
        rep = min(members, key=lambda c: (
            c["origin"].get("repo", ""), c["origin"].get("file", ""),
            c["origin"].get("line", 0)))
        tpl, meta = _template(rep["predicate"])
        nv, nn = len(meta["vars"]), len(meta["nums"])
        num_sub = {f"n{i}": meta["nums"][i] for i in range(nn)}   # constants keep
        var_sub = {f"v{i}": meta["vars"][i] for i in range(nv)}   # names keep
        gen = []

        # (a) positional: the target candidate's own symbols into the schema vars
        for tc, tn in to_meta:
            if len(tn) < nv:
                continue
            p = instantiate(tpl, {**num_sub, **{f"v{i}": tn[i] for i in range(nv)}})
            if p:
                gen.append(p)
        # (b) pure augmentation: bounded product over the symbol pool
        if nv and not gen and syms:
            for combo in itertools.islice(itertools.product(syms, repeat=nv),
                                          max_per_shape):
                p = instantiate(tpl, {**num_sub,
                                      **{f"v{i}": combo[i] for i in range(nv)}})
                if p:
                    gen.append(p)
        # (c) constant specialisation: sweep the schema's constants over the pool
        if nn and nums:
            for lit in nums[:16]:
                p = instantiate(tpl, {**var_sub,
                                      **{f"n{i}": lit for i in range(nn)}})
                if p:
                    gen.append(p)

        seen, inst = set(), []
        for p in gen:
            if p not in seen:
                seen.add(p)
                inst.append({"predicate": p, "tier": rec["tier"],
                             "origin": {"construct": rec["construct"]}})
        inst = inst[:max_per_shape]
        if not inst:
            continue
        collect = [] if collect_instances else None
        val, adopt = _validate(rec, inst, oracle_fn, collect=collect)
        val["generated"] = len(inst)
        if collect_instances:
            val["certified_instances"] = [p for p, st in collect if st == "CERTIFIED"]
            val["premise_instances"] = [p for p, st in collect if st == "parsed"]
        records.append(_record(sid, rec, val, adopt,
                               "generated on target environment and validated before "
                               "adoption" if adopt else
                               "scope-narrowed: no generated instance validated on "
                               "target", source="generated"))
    return records


def _pred_names(pred):
    import re
    names, seen = [], set()
    for n in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", pred):
        if n in seen or n == "assert":
            continue
        seen.add(n)
        names.append(n)
        if len(names) >= 12:
            break
    return names


def _now():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def apply_ledger(cands, ledger):
    """Stamp ledger decisions onto candidates (inherited).

    'confirmed' stamps proof-class provenance (drives premise injection);
    'enforced' stamps the RUNTIME-ENFORCEMENT provenance of an observation-class
    shape — bookkeeping for the enforcement plane. It never sets 'confirmed':
    evidence migrates observation -> proof only via a disposal rung through the
    verdict authority."""
    n = 0
    for c in cands:
        sid = shape_id(c)
        r = ledger.get(sid)
        if not r:
            continue
        if r["decision"] == "confirmed":
            c["provenance"]["confirmed"] = True
            c["provenance"]["confirmed_by"] = f"{r['by']}@shape:{sid}"
            n += 1
        elif r["decision"] == "enforced":
            c["provenance"]["enforced"] = True
            c["provenance"]["enforced_by"] = f"{r['by']}@shape:{sid}"
            n += 1
    return n


import os  # noqa: E402  (used by load_ledger)
