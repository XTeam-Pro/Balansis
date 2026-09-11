#!/usr/bin/env python3
"""Generate and check API signatures directly from the Python syntax tree."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    declaration = copy.deepcopy(node)
    declaration.body = [ast.Expr(value=ast.Constant(value=Ellipsis))]
    declaration.decorator_list = [
        d
        for d in declaration.decorator_list
        if isinstance(d, ast.Name)
        and d.id in {"property", "classmethod", "staticmethod"}
    ]
    return ast.unparse(declaration)


def public_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    result = []
    for node in body:
        if isinstance(node, ast.Try):
            result.extend(public_statements(node.body))
        elif isinstance(node, ast.If):
            result.extend(public_statements(node.body))
            result.extend(public_statements(node.orelse))
        else:
            result.append(node)
    return result


def render(package: Path, docs: Path) -> dict[Path, str]:
    pages = {}
    index = [
        "# API reference",
        "",
        "Signatures are generated from the source code. Each module links to its implementation.",
        "",
    ]
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if any(part.startswith("_") for part in relative.parts[1:-1]):
            continue
        if path.stem.startswith("_") and path.stem != "__init__":
            continue
        module = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
        tree = ast.parse(path.read_text())
        declarations = []
        for node in public_statements(tree.body):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("_"):
                declarations.append(signature(node))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                bases = ", ".join(ast.unparse(base) for base in node.bases)
                body = []
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                        not member.name.startswith("_") or member.name == "__init__"
                    ):
                        body.append(signature(member))
                    elif (
                        isinstance(member, ast.AnnAssign)
                        and isinstance(member.target, ast.Name)
                        and not member.target.id.startswith("_")
                    ):
                        body.append(ast.unparse(member))
                declarations.append(
                    f"class {node.name}"
                    + (f"({bases})" if bases else "")
                    + ":\n"
                    + "\n\n".join(
                        "\n".join("    " + line for line in item.splitlines())
                        for item in body or ["..."]
                    )
                )
            elif (
                path.name == "__init__.py"
                and isinstance(node, ast.ImportFrom)
                and node.module
            ):
                declarations.append(ast.unparse(node))
        page = docs / "reference" / f"{module}.md"
        content = [
            f"# {module}",
            "",
            f"[Source](../../{relative.as_posix()}) · [Reference index](index.md)",
            "",
        ]
        content += ["```python\n" + item + "\n```\n" for item in declarations]
        if not declarations:
            content.append("This module provides package configuration.\n")
        pages[page] = "\n".join(content)
        index.append(f"- [{module}]({module}.md)")
    pages[docs / "reference" / "index.md"] = "\n".join(index) + "\n"
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT / "balansis")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()
    pages = render(args.package_dir.resolve(), args.docs_dir.resolve())
    stale = []
    for path, content in pages.items():
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        elif not path.exists() or path.read_text() != content:
            stale.append(str(path.relative_to(ROOT)))
    extras = set((args.docs_dir / "reference").glob("*.md")) - set(pages)
    for path in extras:
        if args.write:
            path.unlink()
        else:
            stale.append(str(path))
    report = {
        "summary": {
            "status": "FAIL" if stale else "PASS",
            "modules_analyzed": len(pages) - 1,
        },
        "stale_pages": sorted(stale),
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(stale))


if __name__ == "__main__":
    main()
