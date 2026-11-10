"""registry — inspect and re-derive the mined registry."""
from __future__ import annotations

from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def _mined_keys(key_filter: str | None = None) -> list[str]:
    from mjolnir_prover import storage
    keys = [k for k in storage.get_store().list("registry")
            if k.rsplit("/", 1)[-1].startswith("mined_") and k.endswith(".jsonl")]
    if key_filter:
        keys = [k for k in keys if key_filter in k]
    return keys


def registry_list(key_filter: str | None = None) -> list[dict]:
    from mjolnir_prover import storage
    st = storage.get_store()
    out = []
    for k in _mined_keys(key_filter):
        rows = st.read_jsonl(k)
        with_oracle = sum(1 for r in rows if (r.get("oracle") or {}).get("oracle"))
        out.append({"key": k, "entries": len(rows), "with_oracle": with_oracle})
    return out


def registry_verify(key_filter: str | None = None) -> dict:
    """Re-derive every stored-oracle entry through the single verdict authority
    and compare against the stored label."""
    from mjolnir_prover.lib import certify
    from mjolnir_prover import storage
    st = storage.get_store()
    total = ok = bad = err = with_oracle = 0
    mismatches = []
    for k in _mined_keys(key_filter):
        for r in st.read_jsonl(k):
            total += 1
            oracle = (r.get("oracle") or {}).get("oracle")
            cu = r.get("canonical_unit")
            if not (oracle and cu):
                continue
            with_oracle += 1
            try:
                # like-for-like artifact re-derivation: the stored label was
                # minted by the prover on the canonical unit, so re-derive the
                # same disposal (arbitered by the detached kernel). Reading
                # semantics live in the report, not in artifact verification.
                got = certify.unit_status(cu)["status"]
            except Exception as e:
                err += 1
                mismatches.append({"key": k, "id": r.get("id"),
                                   "why": f"error:{type(e).__name__}"})
                continue
            if got == oracle:
                ok += 1
            else:
                bad += 1
                mismatches.append({"key": k, "id": r.get("id"),
                                   "why": f"{oracle}->{got}"})
    return {"entries": total, "with_oracle": with_oracle, "reproduced": ok,
            "mismatch": bad, "error": err, "mismatches": mismatches,
            "rc": 0 if (bad == 0 and err == 0) else 1}


def register(app: typer.Typer) -> None:
    registry_app = typer.Typer(help="Inspect and re-derive the mined registry.",
                               no_args_is_help=True, add_completion=False)

    @registry_app.command("list")
    def list_(
        key: Optional[str] = typer.Option(None, "--key",
                                          help="Only keys containing this text."),
    ) -> None:
        """List mined registry keys with entry/oracle counts."""
        reporting.render_registry_list(registry_list(key))

    @registry_app.command("verify")
    def verify_(
        key: Optional[str] = typer.Option(None, "--key",
                                          help="Only keys containing this text."),
    ) -> None:
        """Re-run the gate on every stored oracle and compare verdicts."""
        res = registry_verify(key)
        reporting.render_registry_verify(res)
        raise typer.Exit(res["rc"])

    app.add_typer(registry_app, name="registry")