# Copyright (c) 2024-2026 Andrey Tikhonov (XTeam-Pro). All rights reserved.
#
# This file is part of Balansis.
# Balansis is dual-licensed under:
#   1. GNU Affero General Public License v3.0 (AGPLv3) for open-source use.
#   2. A Commercial License for proprietary and corporate use.
#
# See LICENSING.md in the project root for license selection details.
# For commercial licensing: andrew@xteam.pro
"""Command-line entrypoint for Balansis."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from balansis import __version__
from balansis.array import native_available, native_dot_available, native_gram_available
from balansis.core.absolute import AbsoluteValue
from balansis.core.operations import Operations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="balansis",
        description="Balansis command-line utilities for release smoke checks and basic ACT operations.",
    )
    parser.add_argument(
        "--version", action="version", version=f"balansis {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "doctor", help="Run a minimal import and operation smoke check"
    )

    add_parser = subparsers.add_parser(
        "add", help="Compensated addition of two finite numbers"
    )
    add_parser.add_argument("left", type=float)
    add_parser.add_argument("right", type=float)
    add_parser.add_argument(
        "--json", action="store_true", help="Emit a machine-readable JSON object"
    )

    return parser


def run_doctor() -> int:
    left = AbsoluteValue.from_float(1.0)
    right = AbsoluteValue.from_float(-1.0)
    result, compensation = Operations.compensated_add(left, right)
    payload = {
        "status": "ok",
        "version": __version__,
        "native": {
            "sum": native_available(),
            "dot": native_dot_available(),
            "gram": native_gram_available(),
        },
        "operation": "compensated_add",
        "result": result.to_float(),
        "compensation": compensation,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_add(left: float, right: float, emit_json: bool) -> int:
    result, compensation = Operations.compensated_add(
        AbsoluteValue.from_float(left),
        AbsoluteValue.from_float(right),
    )
    payload = {
        "operation": "compensated_add",
        "left": left,
        "right": right,
        "result": result.to_float(),
        "compensation": compensation,
    }
    if emit_json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"result={payload['result']} compensation={payload['compensation']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return run_doctor()
    if args.command == "add":
        return run_add(args.left, args.right, args.json)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
