"""Run all ten Volume I contract labs and summarize their exit codes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABS = sorted((ROOT / "labs").glob("lab*.py"))


def main() -> int:
    failures = 0
    for lab in LABS:
        result = subprocess.run(
            [sys.executable, str(lab), "--replay"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        ok = result.returncode == 0
        if not ok:
            failures += 1
        print(f"{lab.name:32s} exit={result.returncode} {'ok' if ok else 'FAIL'}")
        if not ok:
            print(result.stderr.strip())
    print(f"run_all: {len(LABS) - failures}/{len(LABS)} labs passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
