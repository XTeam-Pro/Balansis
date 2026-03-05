#!/usr/bin/env python3
"""Validate that version is consistent across pyproject.toml and balansis/__init__.py."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print("ERROR: version not found in pyproject.toml")
        sys.exit(1)
    return match.group(1)


def get_init_version() -> str:
    text = (ROOT / "balansis" / "__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        print("ERROR: __version__ not found in balansis/__init__.py")
        sys.exit(1)
    return match.group(1)


def main() -> None:
    pyproject_ver = get_pyproject_version()
    init_ver = get_init_version()

    print(f"pyproject.toml version: {pyproject_ver}")
    print(f"__init__.py version:    {init_ver}")

    if pyproject_ver != init_ver:
        print(f"ERROR: Version mismatch! {pyproject_ver} != {init_ver}")
        sys.exit(1)

    # Validate tag if provided
    if len(sys.argv) > 1:
        tag = sys.argv[1].lstrip("v")
        if tag != pyproject_ver:
            print(f"ERROR: Tag v{tag} does not match version {pyproject_ver}")
            sys.exit(1)
        print(f"Tag v{tag} matches version.")

    print("Version validation passed.")


if __name__ == "__main__":
    main()
