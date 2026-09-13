#!/usr/bin/env python3
"""Classify changed paths for the repository's path-aware CI workflows."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import PurePosixPath

EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ZERO_SHA = "0" * 40

DOCUMENT_FILES = {
    ".github/CODEOWNERS",
    "AGENTS.md",
    "CHANGELOG.md",
    "CLA.md",
    "COMMERCIAL_LICENSE.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSING.md",
    "NOTICE",
    "ORDER_FORM_TEMPLATE.md",
    "README.md",
    "ROADMAP.md",
    "SECURITY.md",
}
PYTHON_PREFIXES = (
    "balansis/",
    "benchmarks/",
    "examples/",
    "scripts/",
    "tests/",
    "tnsim/",
)
PYTHON_FILES = {"poetry.lock", "pyproject.toml"}
CI_PREFIXES = (".github/scripts/", ".github/workflows/")
CI_FILES = {".pre-commit-config.yaml"}


def _is_documentation(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        path in DOCUMENT_FILES
        or path.startswith("docs/")
        or pure.suffix.lower() in {".md", ".rst"}
    )


def _changed_files(base: str, head: str) -> list[str]:
    base_ref = EMPTY_TREE_SHA if base == ZERO_SHA else base
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            base_ref,
            head,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted({line for line in completed.stdout.splitlines() if line})


def classify(paths: list[str]) -> dict[str, bool | int]:
    result: dict[str, bool | int] = {
        "changed_count": len(paths),
        "docs_only": bool(paths),
        "python_required": False,
        "native_required": False,
        "formal_required": False,
        "workflow_required": False,
    }

    for path in paths:
        if _is_documentation(path):
            continue

        result["docs_only"] = False

        if path in CI_FILES or path.startswith(CI_PREFIXES):
            result["workflow_required"] = True
        elif path.startswith("balansis_native/"):
            result["native_required"] = True
        elif path.startswith("formal/"):
            result["formal_required"] = True
        elif path in PYTHON_FILES or path.startswith(PYTHON_PREFIXES):
            result["python_required"] = True
        else:
            # Unknown executable/configuration paths take the safer test path.
            result["python_required"] = True

    if not paths:
        result["python_required"] = True

    return result


def _as_output(value: bool | int) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base commit or ref")
    parser.add_argument("--head", required=True, help="Head commit or ref")
    args = parser.parse_args()

    paths = _changed_files(args.base, args.head)
    result = classify(paths)

    print(f"Changed files ({len(paths)}):")
    for path in paths:
        print(f"  {path}")
    print("Scope:")
    for key, value in result.items():
        print(f"  {key}={_as_output(value)}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            for key, value in result.items():
                output.write(f"{key}={_as_output(value)}\n")


if __name__ == "__main__":
    main()
