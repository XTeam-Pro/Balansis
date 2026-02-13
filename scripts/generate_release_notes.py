#!/usr/bin/env python3
"""Generate release notes from CHANGELOG.md for the current version."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        print("No CHANGELOG.md found. Release notes unavailable.")
        return

    text = changelog.read_text()

    # Get version from arg or pyproject.toml
    if len(sys.argv) > 1:
        version = sys.argv[1].lstrip("v")
    else:
        pyproject = (ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
        version = match.group(1) if match else "0.0.0"

    # Extract section for this version (or Unreleased)
    # Pattern: ## [version] ... until next ## [
    for header in [f"[{version}]", "[Unreleased]"]:
        pattern = rf"## {re.escape(header)}.*?\n(.*?)(?=\n## \[|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            notes = match.group(1).strip()
            print(f"# Balansis v{version}\n")
            print(notes)
            return

    print(f"# Balansis v{version}\n")
    print("No release notes available for this version.")


if __name__ == "__main__":
    main()
