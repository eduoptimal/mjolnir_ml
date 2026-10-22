"""mjolnir_ml.report — one comprehensive text/table report for a decision run.

`report(dm, rows, ...)` answers, in one artifact:

    the PROBLEM      contract: features, model/policy/calibration digests, alpha
    the DECISIONS    verdict census, abstention causes, claim/certified rates
    the PAIR         precision/recall vs a pinned oracle + degenerate controls
    the HAZARDS      completeness vs an external hazard ledger (per hazard)
    the PROOF        certificates signed / replayable (when a key is given)
    the ROWS         a sample of decisions with the basis and model facts

Everything is stdlib; `report_data` returns the same content as a dict for
JSON, so a dashboard and a text export cannot disagree.
"""
from __future__ import annotations

import time

from mjolnir_ml.certificate import certify, verify_certificate
from mjolnir_ml.hazards import measure_completeness as _measure_completeness
from mjolnir_ml.model import DecisionModel, Disposition
from mjolnir_ml.pair import audit
from mjolnir_ml.vendors import wire

_HARNESS = ("missing:", "invalid:", "out_of_regime", "regime_not_fitted",
            "model_error", "invalid_prediction")


def _stage(d: Disposition) -> str:
    if d.verdict != wire.ABSTAIN_W:
        return ", ".join(d.basis)
    if d.missing and not str(d.basis[0]).startswith("out_of_regime"):
        return f"missing fact: {d.missing}"
    b = str(d.basis[0]) if d.basis else "abstained"
    for p in _HARNESS:
        if b.startswith(p):
            return b
    if b == "uncovered":
        return "uncovered (no rule fires)"
    return f"premise: {b}"


def _clip(x, n=46) -> str:
    s = str(x)
    return s if len(s) <= n else s[:n - 1] + "…"


def _table(headers, rows, widths=None) -> str:
    widths = widths or [max(len(str(h)), *(len(str(r[i])) for r in rows))
                        for i, h in enumerate(headers)]
    line = "  " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  " + "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  " + "  ".join(str(c).ljust(w)
                                    for c, w in zip(r, widths)))
    return "\n".join(out)


def _detail(d: Disposition) -> str:
    p = d.prediction or {}
    if "set" in p:
        return (f"pred={p.get('pred')} conf={_num(p.get('conf'))} "
                f"set={p.get('set')}")
    if "lo" in p:
        return (f"pred={_num(p.get('pred'))} "
                f"[{_num(p.get('lo'))}, {_num(p.get('hi'))}]")
    return f"pred={p.get('pred')}" if p else "-"


def _num(x) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


def _row_view(i, d: Disposition) -> dict:
    p = d.prediction or {}
    return {"index": i, "verdict": d.verdict, "basis": _stage(d),
            "guarantee": d.guarantee,
            "p_positive": p.get("p_positive"), "conf": p.get("conf"),
            "pred": p.get("pred"), "lo": p.get("lo"), "hi": p.get("hi"),
            "set_size": p.get("set_size"), "say": d.say}


def report_data(dm: DecisionModel, rows, *, title="Decision report",
                safe=None, decidable=None, ledger=None, key=None,
                sample=8, context=None) -> dict:
    """Collect the full report as a JSON-able dict."""
    rows = list(rows)
    disps = [dm.dispose(r) for r in rows]
    n = len(disps)
    census = {v: 0 for v in wire.WIRE}
    causes: dict = {}
    for d in disps:
        census[d.verdict] = census.get(d.verdict, 0) + 1
        if d.verdict == wire.ABSTAIN_W:
            st = _stage(d)
            causes[st] = causes.get(st, 0) + 1
    cert = [d.prediction or {} for d in disps]
    preds = {
        "n": n,
        "claims": sum(1 for d in disps if d.claims),
        "certified_share": (
            sum(1 for p in cert if p.get("certified") is True) / n
            if n and any("certified" in p for p in cert) else None),
        "mean_conf": (sum(float(p["conf"]) for p in cert if "conf" in p)
                      / max(1, sum(1 for p in cert if "conf" in p))
                      if any("conf" in p for p in cert) else None),
        "mean_set_size": (sum(float(p["set_size"]) for p in cert
                              if "set_size" in p)
                          / max(1, sum(1 for p in cert if "set_size" in p))
                          if any("set_size" in p for p in cert) else None),
        "mean_interval_width": (sum(float(p["hi"]) - float(p["lo"])
                                    for p in cert if "lo" in p and "hi" in p)
                                / max(1, sum(1 for p in cert
                                             if "lo" in p and "hi" in p))
                                if any("lo" in p for p in cert) else None),
    }

    data = {
        "schema": "mjolnir.ml.report/1",
        "title": title,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": {
            "policy": dm.policy.name, "policy_digest": dm.policy.digest,
            "model": type(dm.estimator).__name__,
            "model_digest": dm.model_digest,
            "calibration_digest": (dm.kappa.digest if dm.kappa else None),
            "alpha": (dm.kappa.alpha if dm.kappa else None),
            "adaptive": bool(dm.kappa and hasattr(dm.kappa, "observe")),
            "regime": dm.regime,
            "n_features": len(dm.features),
            "features": list(dm.features),
            "statements": [{"code": s.code, "kind": s.kind,
                            "requires": list(s.requires),
                            "predicate": s.predicate,
                            "verdict": s.verdict}
                           for s in dm.policy.statements],
        },
        "decisions": {"n": n, "verdicts": census,
                      "claim_rate": preds["claims"] / n if n else 0.0,
                      "abstain_causes": causes},
        "predictions": preds,
        "rows": [_row_view(i, d) for i, d in enumerate(disps)],
    }
    if safe is not None and decidable is not None:
        data["pair"] = audit(disps, safe, decidable)
        data["pair"]["degenerate"] = _degenerate(safe, decidable)
    if ledger is not None:
        data["hazards"] = _measure_completeness(disps, rows, ledger)
    if key is not None:
        certs = [certify(d, key=key) for d in disps]
        ok = sum(verify_certificate(c, key=key, policy=dm.policy)
                 for c in certs)
        data["certificates"] = {
            "n": n, "verified": ok,
            "replayable": sum(1 for c in certs if c["replayable"]),
            "sample": certs[0] if certs else None,
        }
    if context:
        data["context"] = dict(context)
    data["sample_size"] = min(sample, n)
    return data


def _degenerate(safe, decidable):
    from mjolnir_ml.pair import degenerate_controls
    return degenerate_controls(safe, decidable)


def report(dm: DecisionModel, rows, **kwargs) -> str:
    """Render `report_data` as a fixed-width text report."""
    sample = kwargs.pop("sample", 8)
    data = report_data(dm, rows, sample=sample, **kwargs)
    c = data["contract"]
    lines = []
    bar = "=" * 78

    def section(name):
        lines.extend(["", name, "-" * len(name)])

    lines += [bar, f" {data['title']}", bar]
    lines += [f" {data['schema']} · {data['generated_at']}"]

    section("CONTRACT (the problem)")
    lines.append(f"  policy        {c['policy']}  ({c['policy_digest']})")
    lines.append(f"  model         {c['model']}  ({c['model_digest'][:23]}…)")
    if c["alpha"] is not None:
        lines.append(f"  calibration   alpha={c['alpha']:.2f}"
                     f"{'  adaptive=ACI' if c['adaptive'] else ''}"
                     f"  ({c['calibration_digest'][:23]}…)")
    else:
        lines.append("  calibration   none (conformal facts unavailable)")
    lines.append(f"  regime        {c['regime']}"
                 f"   features={c['n_features']}")
    lines.append(f"  features      {', '.join(c['features'])}")

    section("POLICY (every statement, compiled)")
    rows_ = [[s["code"], s["kind"], ", ".join(s["requires"]),
              s["predicate"], s["verdict"] or "-"]
             for s in c["statements"]]
    lines.append(_table(["code", "kind", "requires", "predicate", "verdict"],
                        rows_))

    section("DECISIONS")
    d = data["decisions"]
    for v in wire.WIRE:
        k = d["verdicts"].get(v, 0)
        lines.append(f"  {v:<8}{k:>5}  ({100 * k / d['n']:5.1f}%)")
    lines.append(f"  claims  {d['claim_rate'] * 100:5.1f}% of rows"
                 f" · certified share "
                 f"{_num(data['predictions']['certified_share'])}")
    if d["abstain_causes"]:
        lines.append("  abstentions by cause")
        for cause, k in sorted(d["abstain_causes"].items(),
                               key=lambda kv: -kv[1]):
            lines.append(f"    {cause:<40}{k:>5}")

    if "pair" in data:
        p = data["pair"]
        q = p["pair"]
        section("OMEGA PAIR (vs the pinned oracle)")
        lines.append(f"  precision {_num(q['prec']):>6}  "
                     f"95% CI [{_num(p['prec_ci'][0])}, "
                     f"{_num(p['prec_ci'][1])}]")
        lines.append(f"  recall    {_num(q['rec']):>6}  "
                     f"95% CI [{_num(p['rec_ci'][0])}, "
                     f"{_num(p['rec_ci'][1])}]")
        lines.append(f"  claims {q['n_claim']} · certified {q['n_cert']} · "
                     f"overclaim {q['n_overclaim']} · missed {q['n_missed']}")
        deg = p["degenerate"]
        lines.append(f"  degenerate: ALWAYS prec {_num(deg['ALWAYS']['prec'])} "
                     f"rec {_num(deg['ALWAYS']['rec'])}  |  "
                     f"NEVER prec {_num(deg['NEVER']['prec'])} "
                     f"rec {_num(deg['NEVER']['rec'])}")

    if "hazards" in data:
        h = data["hazards"]
        section("HAZARDS (external ledger)")
        lines.append(f"  ledger {h['ledger_version']}  "
                     f"({h['ledger_digest'][:23]}…)")
        lines.append(f"  precision {_num(h['precision'])} · recall "
                     f"{_num(h['recall'])} · coverage {_num(h['coverage'])} · "
                     f"false certs {h['n_false_certificate']} · "
                     f"ceiling {_num(h['ceiling'])}")
        lines.append(f"  observable recall {_num(h['recall_observable'])} "
                     f"({h['n_observable_hazard']} rows) · hidden-only "
                     f"recall {_num(h['recall_hidden_only'])} "
                     f"({h['n_hidden_only']} rows)")
        if h["per_hazard"]:
            rows_ = [[code, v["rows"], v["released"],
                      "yes" if v["observable"] else "NO (outside sigma(X))"]
                     for code, v in sorted(h["per_hazard"].items())]
            lines.append("")
            lines.append(_table(["hazard", "rows", "released", "observable"],
                                rows_))

    preds = data["predictions"]
    if preds["mean_set_size"] is not None or \
            preds["mean_interval_width"] is not None:
        section("PREDICTIONS")
        lines.append(f"  mean conf {_num(preds['mean_conf'])} · "
                     f"certified share {_num(preds['certified_share'])}")
        if preds["mean_set_size"] is not None:
            lines.append(f"  mean conformal set size "
                         f"{_num(preds['mean_set_size'])}")
        if preds["mean_interval_width"] is not None:
            lines.append(f"  mean conformal interval width "
                         f"{_num(preds['mean_interval_width'])}")

    if "certificates" in data:
        cf = data["certificates"]
        section("CERTIFICATES")
        lines.append(f"  signed & verified {cf['verified']}/{cf['n']} · "
                     f"replayable from facts alone {cf['replayable']}/{cf['n']}")

    section(f"SAMPLE DECISIONS ({data['sample_size']} of {d['n']})")
    rows_ = []
    for r in data["rows"][:data["sample_size"]]:
        rows_.append([r["index"], r["verdict"], _clip(r["basis"], 34),
                      r["guarantee"], _clip(_detail_from_view(r), 52)])
    lines.append(_table(["#", "verdict", "basis", "guarantee", "model facts"],
                        rows_))
    lines.append("")
    return "\n".join(lines) + "\n"


def _detail_from_view(r) -> str:
    if r.get("set_size") is not None:
        return (f"pred={r.get('pred')} conf={_num(r.get('conf'))} "
                f"set_size={r.get('set_size')}")
    if r.get("lo") is not None:
        return (f"pred={_num(r.get('pred'))} "
                f"[{_num(r.get('lo'))}, {_num(r.get('hi'))}]")
    if r.get("p_positive") is not None:
        return (f"pred={r.get('pred')} "
                f"p_positive={_num(r.get('p_positive'))} "
                f"conf={_num(r.get('conf'))}")
    return f"pred={r.get('pred')}" if r.get("pred") is not None else "-"


__all__ = ["report", "report_data"]
