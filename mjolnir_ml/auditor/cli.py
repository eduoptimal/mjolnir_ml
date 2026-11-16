"""Auditor CLI: verify a bundle from files and a trust store."""
from __future__ import annotations

import argparse
import json
import sys

from mjolnir_ml.auditor.bundle import load_bundle, load_trust
from mjolnir_ml.auditor.verify import verify_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mjolnir-ml-audit")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("bundle")
    verify.add_argument("--trust", required=True)
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--allow-unsigned", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = verify_bundle(
            load_bundle(args.bundle), load_trust(args.trust),
            require_signature=not args.allow_unsigned)
    except Exception as exc:
        print(json.dumps({"schema": "warrantledger.audit/1", "valid": False,
                          "errors": [f"malformed bundle: {exc}"]}))
        return 5
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if not args.json and report.errors:
        print("\n".join(report.errors), file=sys.stderr)
    return report.exit_code
