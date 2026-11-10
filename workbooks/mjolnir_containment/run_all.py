"""Run all twelve containment labs and summarize their exit codes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABS = sorted((ROOT / "labs").glob("lab*.py"))


def main() -> int:
    failures = 0
    for lab in LABS:
        expected = 1 if lab.name == "lab08_fidelity.py" else 0
        result = subprocess.run(
            [sys.executable, str(lab), "--replay"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        ok = result.returncode == expected
        if not ok:
            failures += 1
        print(f"{lab.name:32s} exit={result.returncode} expected={expected} {'ok' if ok else 'FAIL'}")
        if not ok:
            print(result.stderr.strip())
    print(f"run_all: {len(LABS) - failures}/{len(LABS)} labs matched expected status")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
