#!/usr/bin/env python3
"""Check package source coverage and license payloads in built wheels."""

from __future__ import annotations

import argparse
import email
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGAL = ("LICENSE", "NOTICE", "LICENSING.md", "COMMERCIAL_LICENSE.md")


def source_files(package: str) -> set[str]:
    return {
        str(path.relative_to(ROOT))
        for path in (ROOT / package).rglob("*.py")
        if not any(
            part in {"tests", "examples", "build", "dist", "__pycache__"}
            for part in path.parts
        )
        and path.name != "setup.py"
    }


def validate(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_path = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_path))
        name = metadata["Name"]
        if name in {"balansis", "tnsim"}:
            missing = source_files(name) - names
            if missing:
                raise ValueError(
                    f"{wheel.name}: missing package sources: {sorted(missing)}"
                )
            stale = [
                path
                for path in source_files(name)
                if archive.read(path) != (ROOT / path).read_bytes()
            ]
            if stale:
                raise ValueError(
                    f"{wheel.name}: outdated package sources: {sorted(stale)}"
                )
        if name == "balansis" and "balansis/py.typed" not in names:
            raise ValueError("Balansis wheel must contain py.typed")
        if name == "balansis-kernels" and not any(
            entry.startswith("_balansis_kernels") and entry.endswith((".so", ".pyd"))
            for entry in names
        ):
            raise ValueError("Native wheel must contain the compiled extension")
        for filename in LEGAL:
            if not any(Path(entry).name == filename for entry in names):
                raise ValueError(f"{wheel.name}: missing {filename}")
        forbidden = {".env", ".git", ".venv", "__pycache__", "tests", "notebooks"}
        if any(forbidden.intersection(Path(entry).parts) for entry in names):
            raise ValueError(f"{wheel.name}: contains development files")
        print(f"PASS {wheel.name}: {name} {metadata['Version']}, {len(names)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="*", type=Path)
    args = parser.parse_args()
    wheels = args.wheels or [
        *ROOT.glob("dist/*.whl"),
        *ROOT.glob("native/dist/*.whl"),
        *ROOT.glob("tnsim/dist/*.whl"),
    ]
    if not wheels:
        parser.error("No wheels found; build distributions first")
    for wheel in sorted(wheels):
        validate(wheel)


if __name__ == "__main__":
    main()
