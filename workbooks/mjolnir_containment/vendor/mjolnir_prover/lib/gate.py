#!/usr/bin/env python3
"""mjolnir_prover.lib.gate — the coder-facing MVP surface (science_usability_v1.md).

Usability flow: a coder points this at Python files/dirs. Every unit (function)
gets a verdict from the exact gate, and every CERTIFIED/REFUTED claim is emitted
as a certificate.json and RE-CHECKED by the detached kernel before it is shown.

Modes:
  report (default) : print verdicts; abstentions never fail the build.
  --gate           : fail-closed CI policy — the build FAILS on a refutation/
                     fault, and (if the config says so) on an abstention where
                     verification was required. Reads mjolnir.toml (or --config).

Exit codes: 0 = pass; 1 = at least one REFUTED/FAULT (bug witness / integrity);
            2 = usage/config error; 3 = gate policy violation (abstain required
            to be verified, or a fail_on status present).

Disposed through mjolnir_prover.lib.certify (the single verdict authority);
consumed by the CLI (`mjolnir check`, `mjolnir verify`) and the report.
"""
from __future__ import annotations
import os, json, fnmatch
_HERE = os.path.dirname(os.path.abspath(__file__))          # <release>/mjolnir_prover/lib
_PKG = os.path.dirname(_HERE)                               # <release>/mjolnir
_RELEASE = os.path.dirname(_PKG)                            # <release>

from mjolnir_prover.lib import certificate as C, paths as _paths
from mjolnir_prover.lib import certify

REFUTED, CERT, OK, USAGE, POLICY = 1, 0, 0, 2, 3


def load_config(path):
    """Read a mjolnir.toml (via tomllib/tomli, else a minimal parser) or .json.
    Flat schema: mode, fail_on, require_verified, exclude, include."""
    if not path or not os.path.exists(path):
        return {}
    if path.endswith(".json"):
        with open(path) as fh:
            return json.load(fh)
    try:
        import tomllib
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception:
        pass
    try:
        import tomli
        with open(path, "rb") as fh:
            return tomli.load(fh)
    except Exception:
        pass
    return _mini_toml(open(path, encoding="utf-8").read())


def _mini_toml(text):
    """Minimal flat-TOML reader for the flat config schema (no dependency)."""
    out = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if v.startswith("[") and v.endswith("]"):
            out[k] = [x.strip().strip("\"'") for x in v[1:-1].split(",") if x.strip()]
        elif v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v and v[0] in "\"'":
            out[k] = v[1:-1]
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def _excluded(rel, pats):
    rel = rel.replace(os.sep, "/")
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch("/" + rel, p)
               or any(fnmatch.fnmatch(seg, p.strip("*/")) for seg in rel.split("/"))
               for p in pats)


def _load_one(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def collect(paths, exclude=()):
    files = []
    for p in paths:
        if os.path.isfile(p) and p.endswith(".py"):
            files.append(p)
        elif os.path.isdir(p):
            for root, _dirs, fs in os.walk(p):
                files.extend(os.path.join(root, f) for f in sorted(fs)
                             if f.endswith(".py"))
    files = sorted(set(files))
    if exclude:
        base = _RELEASE
        files = [f for f in files
                 if not _excluded(os.path.relpath(f, base), exclude)]
    return files


def units_of(path):
    import ast
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [(f"<syntax_error:{e.lineno}>", "", e.lineno)]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            out.append((node.name, ast.get_source_segment(src, node) or "",
                        node.lineno))
    return out


CTA = {
    "": "attach certificate.json to CI; strengthen trivial-tier claims",
    "assert_refuted": "claim refuted — see the certificate witness (a counterexample is emitted only when one was proven)",
    "unconfirmed_spec": "add an assert / test / type hints: provenance is missing, not proof",
    "no_assert_spec": "no claims found: add asserts to make this unit checkable",
    "out_of_semantics_model": "move floats/exceptions/mutation out of the claimed path, or accept abstain",
    "out_of_fragment": "inline the callee / hoist the loop / quantify via a parameter",
    "undecidable_assert": "add a domain premise (e.g. assert x >= 0 first), restate, or record UNKNOWN",
}


def _inject_premises(assume_file):
    """Load a confirmed mined-premise registry into a name -> lower-bound map."""
    from mjolnir_prover.lib import shapes as _sh
    pre = {}
    cands = _load_one(assume_file)
    led = _sh.load_ledger_verified(_paths.registry("confirmations.jsonl"))
    for c in cands:
        if c["tier"] != "premise" or not _sh.shape_id(c) in led:
            continue
        r = led[_sh.shape_id(c)]
        if r.get("decision") != "confirmed":
            continue
        import re as _re
        m = _re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*) *(>=|>) *(-?\d+)",
                          c["predicate"].strip())
        if m:
            var, op, num = m.group(1), m.group(2), int(m.group(3))
            lb = num + (1 if op == ">" else 0)
            pre[var] = max(pre.get(var, lb), lb)
    return pre


def run(paths, json_out=None, assume_file=None, gate=False, config=None):
    """Run the gate over `paths` and return a structured result dict.

    Pure computation + optional JSON dump; it never prints, so callers (the
    Typer CLI, tests, `selfcheck`) can render the result themselves.
    """
    cfg_path = config
    # The config governs `exclude` and `mode` in BOTH modes, so it is read
    # whenever it exists. Reading it only under --gate meant a plain
    # `mjolnir check mjolnir` silently ignored the exclude list and walked
    # mjolnir_prover/storage/data/repos — reporting 33,556 units, almost all vendored
    # third-party test files, as if they were our code. A report that includes
    # data the user excluded is a wrong report, not a verbose one.
    if cfg_path is None and os.path.exists("mjolnir.toml"):
        cfg_path = "mjolnir.toml"
    cfg = load_config(cfg_path) if cfg_path else {}
    gate_on = gate or cfg.get("mode") == "gate"
    fail_on = set(cfg.get("fail_on", ["REFUTED"])) | {"FAULT"}
    require_verified = bool(cfg.get("require_verified", False))
    exclude = tuple(cfg.get("exclude", ()))
    if cfg_path and not cfg and not os.path.exists(cfg_path):
        return {"error": f"config not found: {cfg_path}", "rc": USAGE}

    pre = _inject_premises(assume_file) if assume_file else {}

    rc = OK
    all_certs = []
    entries = []
    files = collect(paths, exclude)
    for path in files:
        rel_file = os.path.relpath(path, _PKG)
        file_entries = []
        for name, code, def_line in units_of(path):
            cert, prof_ok, viol = certify.certify_unit(
                name, code, def_line, path, pre_bounds=pre if pre else None)
            all_certs.append(cert)
            st, reason = cert["status"], cert.get("reason", "")
            line = cert["unit"].get("line")
            loc = rel_file + (f":{line}" if line else "")
            if st == "UNVERIFIABLE" and not prof_ok:
                reason = f"out_of_semantics_model({viol[0].split('@')[0]})"
            extra = ""
            if st == "REFUTED":
                rc = max(rc, REFUTED)
                extra = f"  witness={json.dumps(cert.get('witness'))[:50]}"
            if st == "FAULT":
                rc = max(rc, REFUTED)
            entry = {"file": rel_file, "loc": loc, "name": name, "status": st,
                     "reason": reason, "cta": C.cta_for(st, reason, cert.get("witness")),
                     "extra": extra, "kernel": cert.get("kernel")}
            entries.append(entry)
            file_entries.append(entry)
        if file_entries:
            entries.append({"file": rel_file, "separator": True})

    n_cert = sum(1 for c in all_certs if c["status"] == "CERTIFIED")
    n_ref = sum(1 for c in all_certs if c["status"] == "REFUTED")
    n_abs = sum(1 for c in all_certs if c["status"] == "UNVERIFIABLE")
    n_fault = sum(1 for c in all_certs if c["status"] == "FAULT")
    reasons = {}
    for c in all_certs:
        if c["status"] == "UNVERIFIABLE" and c.get("reason"):
            r = c["reason"].split("(")[0]
            reasons[r] = reasons.get(r, 0) + 1

    result = {
        "paths": list(paths), "files": files, "entries": entries,
        "certificates": all_certs,
        "counts": {"units": len(all_certs), "CERTIFIED": n_cert,
                   "REFUTED": n_ref, "UNVERIFIABLE": n_abs, "FAULT": n_fault},
        "abstain_reasons": reasons, "assume": pre,
        "gate": {"enabled": gate_on, "passed": None, "violations": [],
                 "fail_on": sorted(fail_on), "require_verified": require_verified},
        "json_out": json_out, "false_certificates": 0, "rc": rc,
    }
    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(all_certs, f, indent=2)
    if gate_on:
        counts = result["counts"]
        viol = [f"{s}={counts[s]}" for s in sorted(fail_on) if counts.get(s, 0)]
        if require_verified and n_abs:
            viol.append(f"UNVERIFIABLE={n_abs} (require_verified)")
        result["gate"]["violations"] = viol
        result["gate"]["passed"] = not viol
        if viol:
            rc = REFUTED if (n_ref or n_fault) else POLICY
        else:
            rc = OK
        result["rc"] = rc
    return result
