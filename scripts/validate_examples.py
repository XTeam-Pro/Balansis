#!/usr/bin/env python3
"""Execute Python examples and guide snippets in separate processes."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cases: dict[str, str] = {}
    for folder in (
        ROOT / "examples",
        ROOT / "tnsim/examples",
        ROOT / "tnsim/notebooks",
    ):
        for path in sorted(folder.glob("*.py")):
            cases[str(path.relative_to(ROOT))] = path.read_text()
        for path in sorted(folder.glob("*.ipynb")):
            notebook = json.loads(path.read_text())
            cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            if any(
                cell.get("outputs") or cell.get("execution_count") is not None
                for cell in cells
            ):
                raise ValueError(f"Notebook outputs must be empty: {path}")
            cases[str(path.relative_to(ROOT))] = "\n".join(
                "".join(cell["source"]) for cell in cells
            )
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for index, code in enumerate(
            re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
        ):
            cases[f"{path.relative_to(ROOT)}:snippet-{index + 1}"] = code
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    failures = []
    with tempfile.TemporaryDirectory(prefix="balansis-examples-") as directory:
        for name, code in cases.items():
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode:
                failures.append(name)
                print(f"FAIL {name}\n{result.stderr}")
            else:
                print(f"PASS {name}")
    print(f"Examples: {len(cases) - len(failures)} passed, {len(failures)} failed")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
