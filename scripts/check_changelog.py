#!/usr/bin/env python3
"""Check that CHANGELOG.md exists and contains an entry for the current version."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        print("ERROR: CHANGELOG.md not found")
        sys.exit(1)

    text = changelog.read_text()

    # Get current version from pyproject.toml
    pyproject = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if not match:
        print("ERROR: version not found in pyproject.toml")
        sys.exit(1)
    version = match.group(1)

    # Check for version entry in changelog
    if f"[{version}]" in text or "[Unreleased]" in text:
        print(f"CHANGELOG.md contains entry for version {version} or Unreleased section.")
        print("Changelog validation passed.")
    else:
        print(f"WARNING: No entry for [{version}] or [Unreleased] in CHANGELOG.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
