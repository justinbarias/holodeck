#!/usr/bin/env python3
"""Scan ported C# for shortcuts that hide unfinished or unchecked work.

Finds port TODO markers, stubs, constructs that switch off the compiler's
checks, and product code that shells out to Python. Exits 1 when anything
is found, so it can gate a unit or a CI run.

Usage:
    python scan_port.py CSHARP_DIR [--exclude PART ...]

Example:
    python scan_port.py dotnet/src --exclude /obj/ --exclude /bin/

--exclude skips any file whose path contains the given text. Use it for
build output and for the judge's own tests, which are allowed to start
the Python implementation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CHECKS: list[tuple[str, re.Pattern[str]]] = [
    ("port TODO", re.compile(r"TODO\(port\)")),
    ("stub", re.compile(r"\bNotImplementedException\b")),
    ("dynamic type", re.compile(r"\bdynamic\b(?!\s*\()")),
    ("warning switched off", re.compile(r"#pragma\s+warning\s+disable|\[SuppressMessage")),
    ("null-forgiving operator", re.compile(r"(?<=[\w\)\]])!(?=[\.\;\,\)\[])")),
    ("calls Python", re.compile(r"Process.*[\"']python|[\"']python3?[\"']|FileName\s*=\s*[\"'](uv|python)")),
]

# The TODO marker lives in comments, so it is checked on the raw line.
RAW_LINE_CHECKS = {"port TODO"}
COMMENT = re.compile(r"//.*$")
STRING = re.compile(r'"(?:\\.|[^"\\])*"')


def scan(path: Path) -> list[tuple[int, str, str]]:
    hits = []
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        code = COMMENT.sub("", line)
        no_strings = STRING.sub('""', code)
        for label, pattern in CHECKS:
            if label in RAW_LINE_CHECKS:
                target = line
            elif label == "calls Python":
                target = code
            else:
                target = no_strings
            if pattern.search(target):
                hits.append((number, label, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--exclude", action="append", default=[], help="skip paths containing this text")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2

    total = 0
    files = 0
    for path in sorted(args.root.rglob("*.cs")):
        text = path.as_posix()
        if any(part in text for part in args.exclude):
            continue
        files += 1
        for number, label, line in scan(path):
            total += 1
            print(f"{path}:{number}: {label}: {line}")

    print(f"{files} files scanned, {total} findings")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
