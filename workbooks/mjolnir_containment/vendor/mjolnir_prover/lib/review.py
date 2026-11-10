"""mjolnir_prover.lib.review — shape-level review workflow for the mined registry.

Turns thousands of candidate sign-offs into a handful of shape decisions:

Functions return structured results; the CLI (`mjolnir confirm ...`) wires
them and mjolnir_prover.cli.reporting renders them. Ledger:
registry/confirmations.jsonl (append-only). Gate rule: only shapes with
decision=confirmed may become enforced directives; everything else stays
fail-open (observation tier).
"""
from __future__ import annotations
import os, json
from mjolnir_prover.lib import mine, shapes
from mjolnir_prover.lib import paths as _paths
from mjolnir_prover.lib import certify

REG_DIR = _paths.registry()
LEDGER = os.path.join(REG_DIR, "confirmations.jsonl")


def load_all():
    cands = []
    for f in sorted(os.listdir(REG_DIR)):
        if f.endswith(".jsonl") and f != "confirmations.jsonl":
            with open(os.path.join(REG_DIR, f)) as fh:
                cands.extend(json.loads(l) for l in fh if l.strip())
    return cands


def _groups(registry=None):
    cands = _load_one(registry) if registry else load_all()
    return cands, shapes.collapse(cands)


def list_shapes(registry=None, limit=25):
    """Shapes ordered by review load (undecided first, then instance count)."""
    cands, groups = _groups(registry)
    ledger = shapes.load_ledger_verified(LEDGER)
    n_decided = sum(1 for g in groups.values() if g["sid"] in ledger)
    ordered = sorted(groups.values(),
                     key=lambda g: (g["sid"] in ledger, -len(g["members"])))
    rows = []
    for g in ordered[:limit]:
        r = ledger.get(g["sid"])
        rows.append({"sid": g["sid"], "status": r["decision"] if r else "PENDING",
                     "instances": len(g["members"]), "construct": g["construct"],
                     "tier": g["tier"], "shape": g["shape"]})
    return {"candidates": len(cands), "shapes": len(groups), "decided": n_decided,
            "rows": rows, "remaining": max(0, len(groups) - len(rows)), "limit": limit}


def show_shape(shape_id, registry=None):
    """Evidence bundle for one shape."""
    _cands, groups = _groups(registry)
    if shape_id not in groups:
        raise KeyError(shape_id)
    g = groups[shape_id]
    ledger = shapes.load_ledger_verified(LEDGER)
    r = ledger.get(shape_id)
    evidence = []
    for m in g["members"][:5]:
        o = m["origin"]
        evidence.append({"origin": f"{o['repo']}:{o['file']}:{o['line']}",
                         "predicate": m["predicate"]})
    return {"shape_id": shape_id, "tier": g["tier"],
            "instances": len(g["members"]), "construct": g["construct"],
            "shape": g["shape"], "oracle": dict(g["oracle"]),
            "evidence": evidence,
            "ledger": ({"decision": r["decision"], "by": r["by"],
                        "ts": r["ts"]} if r else None)}


def record_decision(shape_id, decision, by="human", note="", registry=None):
    """Append a signed shape decision to the ledger; instances inherit it."""
    _cands, groups = _groups(registry)
    if shape_id not in groups:
        raise KeyError(shape_id)
    rec = shapes.confirm(groups, shape_id, decision, by=by, note=note)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return {"decision": decision, "shape_id": shape_id, "by": by,
            "instances": len(groups[shape_id]["members"]),
            "tier": groups[shape_id]["tier"], "record": rec}


def apply_ledger(registry=None):
    """Mark every candidate whose shape is confirmed (in-memory projection)."""
    cands, _groups_map = _groups(registry)
    ledger = shapes.load_ledger_verified(LEDGER)
    n = shapes.apply_ledger(cands, ledger)
    return {"applied": n, "total": len(cands), "decisions": len(ledger)}


def _load_one(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _emit(recs):
    """Sign and append transfer records; return the summary dict."""
    if recs:
        with open(LEDGER, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(shapes.sign(r), sort_keys=True) + "\n")
    n_adopt = sum(1 for r in recs if r["decision"] == "confirmed")
    return {"records": recs, "adopted": n_adopt, "narrowed": len(recs) - n_adopt}


def transfer(from_file, to_file):
    """Adopt confirmed shapes onto a target registry, validating first (matched)."""
    from_ledger = shapes.load_ledger_verified(LEDGER)
    recs = shapes.transfer_validate(_load_one(from_file), _load_one(to_file),
                                    from_ledger, oracle_fn=certify.unit_status)
    return _emit(recs)


def transfer_generate(from_file, to_file, symbols=None, literals=None):
    """Cold-start path: instantiate confirmed shapes into the target's environment,
    validate the generated instances, and adopt only the sound ones."""
    from_ledger = shapes.load_ledger_verified(LEDGER)
    recs = shapes.transfer_generate(_load_one(from_file), _load_one(to_file),
                                    from_ledger, oracle_fn=certify.unit_status,
                                    symbols=symbols, literals=literals)
    return _emit(recs)
