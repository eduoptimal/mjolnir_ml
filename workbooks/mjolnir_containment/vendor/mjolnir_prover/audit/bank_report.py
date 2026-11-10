"""mjolnir_prover.audit.bank_report — fill the management Action Report template.

Reads the served template `mjolnir_prover/report/bank_exec.html`, runs the scope-aware
corpus audit + coverage ladder, and writes the populated executive page. Findings
are clickable and expand to ALL exact sites (file, line, excerpt). Manifest-sourced
findings are labelled for verification; code-derived findings are machine-grounded.

CLI: ``mjolnir bank-report --corpus <dir> [--manifest README.md] [--out FILE]``
"""
from __future__ import annotations

import html
import os
import time

from . import policy as _policy
from . import scope as _scope
from . import sql_audit

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # mjolnir_prover/
TEMPLATE = os.path.join(_ROOT, "report", "bank_exec.html")


def _sites(sites):
    if not sites:
        return "<div>no sites captured</div>"
    items = "".join(
        f"<li><code>{html.escape(str(s['file']))}:{s['line']}</code> — "
        f"{html.escape(str(s['excerpt']))}</li>" for s in sites)
    return f"<ul>{items}</ul>"


def _ev(eid: str, sections: list) -> str:
    inner = "".join(
        f"<div class='evtitle'>{html.escape(title)} "
        f"<span class='sub'>({len(st)} site{'s' if len(st) != 1 else ''})</span>"
        f"</div>{_sites(st)}" for title, st in sections)
    return (f'<tr id="{eid}" class="ev" style="display:none">'
            f'<td colspan="4"><div class="evbox">{inner}</div></td></tr>')


_SEV_CLASS = {"ok": "ok", "high": "hi", "medium-high": "md", "medium": "md",
              "low": "md"}

# Epistemic class per finding — the boundary between mjolnir's intrinsic value
# (certified/declared, produced by the authority) and frontend HEURISTICS (lint
# over code: best-practice observations, never proven). Heuristics may inform;
# they must never share a counter or a wording with certificates.
EPISTEMIC = {
    "bound": "declared",    # explicit rule→symbol bindings (assumptions)
}
_EPI_SRC = {
    "certified": "<div class='src code'>certified by the authority (single writer)</div>",
    "declared": "<div class='src'>declared contract — assumption, not a proof</div>",
    "heuristic": "<div class='src warn'>heuristic (lint over the code) — NOT certified</div>",
}


def _finding_row(f: dict) -> str:
    """One finding row: FACT (computed or manifest-quoted) + INTERPRETATION (declared
    policy R-n) + a COMPUTED severity. A zero count renders as 'no exposure detected',
    not as a hardcoded High."""
    n = f["count"]
    covered = n > 0 and bool(f.get("ev_html"))
    attrs = f' class="finding" onclick="tog(\'{f["ev_id"]}\')"' if covered else ""
    chev = " <span class='chev'>▸ click to inspect</span>" if covered else ""
    epi = f.get("epistemic", "heuristic")
    src = _EPI_SRC.get(epi, _EPI_SRC["heuristic"])
    fact = f["fact"] if n > 0 else f["fact_zero"]
    if n > 0:
        level = f["severity"]
        label = {"bound": "Bound"}.get(f["key"], level.upper())
        interp = (f"<div class='interp'><b>Interpretation — policy {f['rule']}</b>: "
                  f"{html.escape(f['interpretation'])}</div>")
        action = html.escape(f["action"])
    else:
        level, label = "ok", "None"
        interp = ("<div class='interp none'>No exposure detected — "
                  f"policy {f['rule']} applies only when the count is non-zero.</div>")
        action = "None — no exposure detected."
    row = (f"<tr{attrs}><td>{fact}{chev}{src}</td><td>{interp}</td>"
           f"<td><span class='sev {_SEV_CLASS.get(level, 'hi')}'>{label}</span>"
           f"<div class='sub'>evidence: {epi}</div></td>"
           f"<td>{action}</td></tr>")
    return row + (f["ev_html"] if covered else "")


def _findings_rows(findings: list) -> str:
    return "".join(_finding_row(f) for f in findings)


def _scope_detail(led: dict) -> str:
    if not led.get("config"):
        return ("<div class='evtitle'>Scope</div>"
                "<div>No <code>.mjolnirignore</code> — defaults "
                "(<code>**/*.sql</code>); nothing else was read.</div>")
    rows = []
    if led["ignored"]:
        rows.append("<div class='evtitle'>Ignored by policy</div><ul>"
                    + "".join(f"<li><code>{html.escape(i)}</code></li>"
                              for i in led["ignored"]) + "</ul>")
    if led.get("applied_suppressions"):
        rows.append("<div class='evtitle'>Suppressed with reason</div><ul>"
                    + "".join(
                        f"<li><code>{html.escape(s['site'])}</code> — "
                        f"{html.escape(s['reason'] or 'no reason given')}</li>"
                        for s in led["applied_suppressions"]) + "</ul>")
    if led.get("stale_suppressions"):
        rows.append("<div class='evtitle' style='color:#b45309'>Stale "
                    "suppressions (did not match any site)</div><ul>"
                    + "".join(f"<li><code>{html.escape(s['site'])}</code></li>"
                              for s in led["stale_suppressions"]) + "</ul>")
    rows.append(f"<div class='sub' style='margin-top:6px'>"
                f"{led['skipped']} file(s) out of scope · "
                f"{led['ignored_count']} ignored · "
                f"{len(led.get('applied_suppressions', []))} suppressed</div>")
    return "".join(rows)


def _bound_facts():
    """Materialised explicit bindings (rule → target symbols): the controls that
    come from banking policy NOT encoded in the code."""
    import json
    from mjolnir_prover.lib import paths as _paths
    p = _paths.out("bank_bound_claims.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _bound_counts():
    """(certified, premise) counts of explicit BINDINGS (out/bank_bound_claims.jsonl)."""
    import json
    from mjolnir_prover.lib import paths as _paths
    p = _paths.out("bank_bound_claims.jsonl")
    if not os.path.exists(p):
        return 0, 0
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return (sum(1 for r in rows if r.get("kind") == "certified"),
            sum(1 for r in rows if r.get("kind") == "premise"))


def _bound_line():
    """Explicit bindings: rules applied to specific target symbols."""
    import json
    from mjolnir_prover.lib import paths as _paths
    p = _paths.out("bank_bound_claims.jsonl")
    if not os.path.exists(p):
        return ("bound rules: none declared (write bindings.jsonl, then "
                "`mjolnir rules bind …`)")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    cert = [r for r in rows if r.get("kind") == "certified"]
    prem = [r for r in rows if r.get("kind") == "premise"]
    if not rows:
        return "bound rules: 0 materialised"
    parts = []
    if cert:
        ex = ", ".join(f"<code>{html.escape(r['predicate'])}</code> @ <code>{html.escape(r['site'])}</code>"
                       for r in cert[:2])
        parts.append(f"<strong>{len(cert)}</strong> bound control(s) proven — {ex}")
    if prem:
        ex = ", ".join(f"<code>{html.escape(r['predicate'])}</code> @ <code>{html.escape(r['site'])}</code>"
                       for r in prem[:2])
        parts.append(f"<strong>{len(prem)}</strong> bound contract(s) declared — {ex}")
    return "bound rules (explicit rule→symbol bindings): " + "; ".join(parts)


def _transferred_counts():
    """(certified, premise) counts of the materialised typed-transfer facts."""
    import json
    from mjolnir_prover.lib import paths as _paths
    p = _paths.out("bank_transferred_claims.jsonl")
    if not os.path.exists(p):
        return 0, 0
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return (sum(1 for r in rows if r.get("kind") == "certified"),
            sum(1 for r in rows if r.get("kind") == "premise"))


def _transferred_line():
    """Informative certified facts materialised from confirmed rules (the CSV-driven
    transfer). Reads the file run_transfer.py writes; vacuous identities are already
    filtered there."""
    import json
    from mjolnir_prover.lib import paths as _paths
    p = _paths.out("bank_transferred_claims.jsonl")
    if not os.path.exists(p):
        return ("transferred rules: none materialised (run shape_ui/run_transfer.py "
                "--materialize … to add confirmed rules as bank facts)")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    cert = [r for r in rows if r.get("kind") == "certified"]
    prem = [r for r in rows if r.get("kind") == "premise"]
    if not rows:
        return ("transferred rules: 0 facts materialised (no confirmed rule transfers "
                "informatively to this target)")
    parts = []
    if cert:
        ex = ", ".join(f"<code>{html.escape(r['predicate'])}</code>" for r in cert[:2])
        parts.append(f"<strong>{len(cert)}</strong> informative certified fact(s) — e.g. {ex}")
    if prem:
        ex = ", ".join(f"<code>{html.escape(r['predicate'])}</code>" for r in prem[:2])
        parts.append(f"<strong>{len(prem)}</strong> declared premise(s) over real "
                     f"columns (assumptions) — e.g. {ex}")
    return ("transferred rules (typed transfer; vacuous tautologies filtered): "
            + "; ".join(parts))


def _template_policy(html: str) -> dict:
    """Finding wording/severity is part of the REPORT ASSET (the template), not a
    corpus config: read the embedded <script id="policy"> JSON."""
    import json as _json
    import re as _re
    m = _re.search(r'<script id="policy" type="application/json">(.*?)</script>',
                   html, _re.S)
    if not m:
        return {}
    try:
        return _json.loads(m.group(1))
    except Exception:
        return {}


def build(corpus: str, out: str | None = None, include_rules: bool = True) -> str:
    sc = _scope.load_scope(corpus)
    _html = open(TEMPLATE, encoding="utf-8").read()
    rules = {k: dict(v) for k, v in _policy.DEFAULTS.items()}
    rules.update(_template_policy(_html))
    a = sql_audit.audit(corpus, sc)
    lad = sql_audit.redispose(corpus, sc)

    def lcount(ann, st):
        return lad["ladder"].get((ann, st), 0)

    cert = sum(n for (_a, st), n in lad["ladder"].items() if st == "CERTIFIED")

    # Segmentation of what was NOT proven. "declared" = a monitor/DB-enforced binding
    # with no provable spec as written (no_assert_spec); "abstained" = a proof attempt
    # left an obligation open; "refuted" = a claimed property falsified.
    _NP_LABEL = {
        "sql-column": "Schema/column contract — DB-enforced",
        "sql-cast": "Cast monitor — no exactness spec as written",
        "sql-nullcase": "Null-handling monitor",
        "sql_nullpred": "Null-predicate monitor",
    }
    declared = abstained = refuted = 0
    _np = {}
    for c in lad["claims"]:
        o = c.get("oracle") or {}
        st, rs = o.get("oracle"), o.get("reason", "")
        if st == "CERTIFIED":
            continue
        if st == "REFUTED":
            refuted += 1
            disp = "refuted"
        elif rs == "no_assert_spec":
            declared += 1
            disp = "declared (no provable spec)"
        else:
            abstained += 1
            disp = f"abstained ({rs or 'out of fragment'})"
        key = (c["annotation"], disp)
        _np[key] = _np.get(key, 0) + 1
    notproven_total = declared + abstained + refuted
    notproven_rows = "".join(
        f"<tr><td>{html.escape(_NP_LABEL.get(a, a))}</td>"
        f"<td class='note'>{html.escape(disp)}</td>"
        f"<td class='text-right'>{n:,}</td></tr>"
        for (a, disp), n in sorted(_np.items(), key=lambda kv: -kv[1]))

    def R(key):
        return rules.get(key, {})

    _bf = _bound_facts() if include_rules else []
    _bcert = [f for f in _bf if f.get("kind") == "certified"]
    _bprem = [f for f in _bf if f.get("kind") == "premise"]

    def _brow(f):
        return {"file": f.get("site", "?"), "line": "",
                "excerpt": f"{f.get('predicate', '')}  ·  rule {f.get('rule', '')}"
                           + (f"  ·  σ={f.get('binding')}" if f.get("binding") else "")}

    ev_bound = _ev("ev_bound", [("Bound controls (rule @ target)", [_brow(f) for f in _bcert]),
                                ("Declared contracts", [_brow(f) for f in _bprem])])

    def finding(key, count, fact, fact_zero, source, ev_html):
        r = R(key)
        return {"key": key, "count": count, "fact": fact, "fact_zero": fact_zero,
                "source": source, "ev_html": ev_html, "ev_id": f"ev_{key}",
                "epistemic": EPISTEMIC.get(key, "heuristic"),
                "rule": r.get("rule", "R-0"),
                "severity": _policy.severity(count, r),
                "interpretation": r.get("interpretation", ""),
                "action": r.get("action", "")}

    findings = []
    if include_rules:
        findings.append(finding("bound", len(_bcert),
                                f"<strong>{len(_bcert)} controls are proven by explicit binding</strong> "
                                f"from banking policy <em>not encoded in the code</em>"
                                + (f" — plus {len(_bprem)} declared contract(s)" if _bprem else ""),
                                "No rules bound to banking policy (no bindings declared).",
                                "binding", ev_bound))
    n_action = 0
    policy_note = "interpretations: report template (presentation, mockup)"

    vals = {
        "generated": time.strftime("%Y-%m-%d"),
        "corpus": os.path.basename(corpus.rstrip("/")),
        "repo": os.path.basename(corpus.rstrip("/")),
        "commit": (lad["claims"][0]["origin"]["commit"] if lad["claims"] else "—"),
        "authority_fp": lad["fingerprint"],
        "n_files": a["n_files"], "n_lines": f"{a['n_lines']:,}",
        "mined": lad["n"], "certified": cert,
        "notproven_total": notproven_total, "declared": declared,
        "abstained": abstained, "refuted": refuted,
        "notproven_rows": notproven_rows,
        "findings_rows": (_findings_rows(findings) if findings else
                          "<tr><td colspan='4' class='sub'>No findings — base tally "
                          "(no accepted rules applied).</td></tr>"),
        "silent_note": ("No exposures detected in this pass."
                        if n_action == 0 else
                        f"No run-time error is raised by any of these {n_action} "
                        "findings. That is precisely why they are reported as "
                        "controls, not incidents."),
        "policy_note": policy_note,
        "scope_line": sql_audit._scope_line(lad["scope"]),
        "scope_detail": _scope_detail(lad["scope"]),
        "transferred_line": (_transferred_line() if include_rules else
                             "typed transfer: disabled — base tally (no accepted rules applied)"),
        "transferred_cert": _transferred_counts()[0] if include_rules else 0,
        "transferred_prem": _transferred_counts()[1] if include_rules else 0,
        "bound_line": (_bound_line() if include_rules else
                       "bound rules: disabled — base tally (no accepted rules applied)"),
        "bound_cert": _bound_counts()[0] if include_rules else 0,
    }
    text = _html
    for k, v in vals.items():
        text = text.replace("{{" + k + "}}", str(v))
    dest = out or os.path.join(_ROOT, "storage", "data", "reports",
                               "bank_exec.filled.html")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return dest
