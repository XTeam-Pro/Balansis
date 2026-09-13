#!/usr/bin/env python3
"""Check that local links in public Markdown files resolve in the repository."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
ZERO_SHA = "0" * 40
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)")
SKIP_PARTS = {
    ".git",
    ".lake",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "htmlcov",
    "target",
}


def _changed_markdown(base: str, head: str) -> list[str]:
    base_ref = EMPTY_TREE_SHA if base == ZERO_SHA else base
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--no-renames",
            "--name-only",
            "--diff-filter=ACMRTUXB",
            base_ref,
            head,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line.endswith(".md")]


def _markdown_files(requested: list[str], *, changed_only: bool) -> list[Path]:
    if requested:
        files = [(ROOT / path).resolve() for path in requested]
    else:
        files = list(ROOT.rglob("*.md"))

    return sorted(
        path
        for path in files
        if path.is_file() and not any(part in SKIP_PARTS for part in path.parts)
    )


def _local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip("<>")
    parsed = urlsplit(target)
    if parsed.scheme or target.startswith(("#", "//")):
        return None

    decoded_path = unquote(parsed.path)
    if not decoded_path:
        return None
    if decoded_path.startswith("/"):
        return ROOT / decoded_path.lstrip("/")
    return source.parent / decoded_path


def check_file(path: Path) -> list[str]:
    failures: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        targets = [match.group(1) for match in INLINE_LINK.finditer(line)]
        reference = REFERENCE_LINK.match(line)
        if reference:
            targets.append(reference.group(1))

        for raw_target in targets:
            target = _local_target(path, raw_target)
            if target is None:
                continue

            relative_source = path.relative_to(ROOT)
            resolved = target.resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                failures.append(
                    f"{relative_source}:{line_number}: target escapes repository: "
                    f"{raw_target}"
                )
            else:
                if not resolved.exists():
                    failures.append(
                        f"{relative_source}:{line_number}: missing {raw_target}"
                    )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths", nargs="*", help="Markdown paths relative to the repository root"
    )
    parser.add_argument("--base", help="Check Markdown changed after this Git ref")
    parser.add_argument("--head", help="Check Markdown through this Git ref")
    args = parser.parse_args()

    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be supplied together")
    if args.paths and args.base:
        parser.error("paths cannot be combined with --base/--head")

    requested = _changed_markdown(args.base, args.head) if args.base else args.paths
    files = _markdown_files(requested, changed_only=bool(args.base or args.paths))
    failures = [failure for path in files for failure in check_file(path)]
    if failures:
        print("Broken local Markdown links:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    print(f"Checked local links in {len(files)} Markdown files.")


if __name__ == "__main__":
    main()
