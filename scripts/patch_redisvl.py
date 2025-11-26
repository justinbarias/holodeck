#!/usr/bin/env python3
"""Patch redisvl 0.4.1 for redis 6.x+ compatibility.

This script patches the redisvl package to use the new snake_case import path
for redis.commands.search.index_definition (instead of indexDefinition).

Run this after `poetry install` or `pip install redisvl==0.4.1`.
"""

import site
import sys
from pathlib import Path


def find_redisvl_path() -> Path | None:
    """Find the redisvl package installation path."""
    for site_path in site.getsitepackages():
        redisvl_path = Path(site_path) / "redisvl"
        if redisvl_path.exists():
            return redisvl_path

    # Check user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        redisvl_path = Path(user_site) / "redisvl"
        if redisvl_path.exists():
            return redisvl_path

    return None


def patch_file(file_path: Path) -> bool:
    """Patch a single file to use snake_case import.

    Args:
        file_path: Path to the file to patch

    Returns:
        True if file was patched, False if no changes needed
    """
    if not file_path.exists():
        return False

    content = file_path.read_text()
    old_import = "from redis.commands.search.indexDefinition import"
    new_import = "from redis.commands.search.index_definition import"

    if old_import not in content:
        return False

    patched_content = content.replace(old_import, new_import)
    file_path.write_text(patched_content)
    return True


def clear_pycache(redisvl_path: Path) -> None:
    """Clear __pycache__ directories to ensure patches take effect."""
    for pycache in redisvl_path.rglob("__pycache__"):
        for pyc_file in pycache.glob("*.pyc"):
            pyc_file.unlink()


def main() -> int:
    """Apply patches to redisvl for redis 6.x+ compatibility."""
    redisvl_path = find_redisvl_path()

    if not redisvl_path:
        print("Error: redisvl package not found", file=sys.stderr)
        return 1

    print(f"Found redisvl at: {redisvl_path}")

    files_to_patch = [
        redisvl_path / "index" / "index.py",
        redisvl_path / "index" / "storage.py",
    ]

    patched_count = 0
    for file_path in files_to_patch:
        if patch_file(file_path):
            print(f"Patched: {file_path}")
            patched_count += 1
        else:
            print(f"Skipped (already patched or not found): {file_path}")

    if patched_count > 0:
        clear_pycache(redisvl_path)
        print("Cleared __pycache__ directories")

    print(f"Done. Patched {patched_count} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
