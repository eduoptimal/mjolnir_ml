"""harvest — mine candidate claims from a repository."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from mjolnir_prover.cli import reporting


def _detect_lang(root: str) -> str:
    """Frontend by majority source file: py | java | sql. Python is the default."""
    counts = {"py": 0, "java": 0, "sql": 0}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "node_modules")]
        for f in files:
            if f.endswith(".py"):
                counts["py"] += 1
            elif f.endswith(".java"):
                counts["java"] += 1
            elif f.endswith(".sql"):
                counts["sql"] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "py"


def run_harvest(root: str, name: str | None, out: str | None,
                lang: str | None = None) -> dict:
    from mjolnir_prover import storage

    repo_name = name or os.path.basename(os.path.abspath(root))
    lang = (lang or _detect_lang(root)).lower()
    if lang == "java":
        from mjolnir_prover.lib import frontend_java as F
        cands = F.mine_java_repo(root, repo_name=repo_name)
    elif lang == "sql":
        from mjolnir_prover.lib import frontend_sql as F
        cands = F.mine_sql_repo(root, repo_name=repo_name)
    else:
        from mjolnir_prover.lib import mine
        lang = "py"
        cands = mine.mine_repo(root, repo_name=repo_name)
    key = out or f"registry/mined_{repo_name}.jsonl"
    storage.get_store().write_jsonl(key, cands)
    confirmed = sum(1 for c in cands if c["provenance"].get("confirmed"))
    return {"name": repo_name, "candidates": len(cands), "key": key,
            "confirmed": confirmed, "lang": lang}


def register(app: typer.Typer) -> None:

    @app.command()
    def harvest(
        root: Path = typer.Argument(
            ..., exists=True, file_okay=False,
            help="Repository root to mine for candidate claims."),
        name: Optional[str] = typer.Option(
            None, "--name", help="Repository name (defaults to the dir name)."),
        out: Optional[str] = typer.Option(
            None, "--out",
            help="Storage key for candidates "
                 "(default: registry/mined_<name>.jsonl)"),
        lang: Optional[str] = typer.Option(
            None, "--lang",
            help="Frontend: py | java | sql (default: auto-detect by majority)."),
    ) -> None:
        """Mine candidate claims from a repo into the registry (proposer only)."""
        try:
            res = run_harvest(str(root), name, out, lang)
        except Exception as e:  # storage backend errors
            reporting.console.print(f"[red]mjolnir harvest:[/red] could not write "
                                    f"storage ({e})")
            raise typer.Exit(2)
        reporting.render_harvest(res)
        raise typer.Exit(0)