#!/usr/bin/env python3
"""Build a leaf-first porting queue for a Python package.

Reads every module under a package root, maps imports between them, and
writes a JSON work list ordered so that each module comes after the
modules it imports. Modules that import each other (a cycle) share one
"group" number and should be ported together.

Usage:
    python build_queue.py SRC_ROOT PACKAGE [--out port/queue.json]

Example:
    python build_queue.py src holodeck --out port/queue.json

Standard library only. Re-running keeps the status, target and notes of
units already in the output file, so it is safe to refresh the queue
after the Python source changes.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


def module_name(path: Path, src_root: Path) -> str:
    parts = list(path.relative_to(src_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_relative(current: str, is_package: bool, level: int, name: str | None) -> str:
    base = current.split(".")
    if not is_package:
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    if name:
        base = base + name.split(".")
    return ".".join(base)


def longest_known(name: str, known: set[str]) -> str | None:
    parts = name.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in known:
            return candidate
        parts.pop()
    return None


def imports_of(tree: ast.AST, current: str, is_package: bool, known: set[str]) -> tuple[set[str], set[str]]:
    internal: set[str] = set()
    external: set[str] = set()

    def record(name: str) -> None:
        hit = longest_known(name, known)
        if hit is None:
            external.add(name.split(".")[0])
        elif hit != current:
            internal.add(hit)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                record(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = resolve_relative(current, is_package, node.level, node.module)
            else:
                base = node.module or ""
            for alias in node.names:
                # "from pkg import sub" may name a submodule or a symbol.
                full = f"{base}.{alias.name}" if base else alias.name
                record(full if longest_known(full, known) == full else base)
    return internal, external


def strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iterative. Returns components leaf-first."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0

    for root in sorted(graph):
        if root in index:
            continue
        work = [(root, iter(sorted(graph[root])))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph[child]))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                result.append(sorted(component))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_root", type=Path, help="directory that contains the package, e.g. src")
    parser.add_argument("package", help="top-level package name, e.g. holodeck")
    parser.add_argument("--out", type=Path, default=Path("port/queue.json"))
    args = parser.parse_args()

    package_dir = args.src_root / args.package
    if not package_dir.is_dir():
        print(f"error: {package_dir} is not a directory", file=sys.stderr)
        return 2

    files = sorted(package_dir.rglob("*.py"))
    names = {module_name(path, args.src_root): path for path in files}
    known = set(names)

    graph: dict[str, set[str]] = {}
    externals: dict[str, set[str]] = {}
    lines: dict[str, int] = {}
    skipped: list[str] = []
    for name, path in names.items():
        text = path.read_text(encoding="utf-8")
        lines[name] = text.count("\n") + 1
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            skipped.append(f"{path}: {exc.msg}")
            graph[name], externals[name] = set(), set()
            continue
        graph[name], externals[name] = imports_of(tree, name, path.name == "__init__.py", known)

    previous: dict[str, dict] = {}
    if args.out.exists():
        for unit in json.loads(args.out.read_text(encoding="utf-8")).get("units", []):
            previous[unit["module"]] = unit

    units = []
    order = 0
    for group, component in enumerate(strongly_connected(graph), start=1):
        for name in component:
            order += 1
            old = previous.get(name, {})
            units.append(
                {
                    "order": order,
                    "group": group,
                    "cycle": len(component) > 1,
                    "module": name,
                    "source": str(names[name]),
                    "lines": lines[name],
                    "imports": sorted(graph[name]),
                    "external": sorted(externals[name]),
                    "target": old.get("target", ""),
                    "status": old.get("status", "todo"),
                    "notes": old.get("notes", ""),
                }
            )

    all_external = sorted({dep for deps in externals.values() for dep in deps})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "package": args.package,
                "statuses": ["todo", "in_progress", "built", "tests_pass", "parity_pass", "reviewed", "blocked"],
                "external_packages": all_external,
                "units": units,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cycles = sum(1 for unit in units if unit["cycle"])
    print(f"{len(units)} modules, {sum(lines.values())} lines, {cycles} modules in import cycles")
    print(f"{len(all_external)} external top-level imports (includes the standard library)")
    for message in skipped:
        print(f"skipped (syntax error): {message}", file=sys.stderr)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
