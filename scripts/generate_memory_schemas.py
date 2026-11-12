"""Deterministically generate ``schemas/memory/*.json`` from the memory tool models.

python scripts/generate_memory_schemas.py            # write the files
python scripts/generate_memory_schemas.py --check    # CI: fail if stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from holodeck.lib.memory.tool_api import (
    MEMORY_TOOL_MODELS,
    MemoryToolError,
)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "memory"
_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def render_schemas() -> dict[str, str]:
    """Return the text of every schema file, keyed by file name."""
    models = {}
    for name, (args, result) in MEMORY_TOOL_MODELS.items():
        models[f"{name}.args.schema.json"] = args
        models[f"{name}.result.schema.json"] = result
    models["error.schema.json"] = MemoryToolError
    rendered = {}
    for file_name, model in models.items():
        schema = {"$schema": _DIALECT, **model.model_json_schema()}
        rendered[file_name] = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    return rendered


def stale_files(directory: Path) -> list[Path]:
    """Return the schema files in ``directory`` that are missing or differ."""
    return [
        directory / name
        for name, text in render_schemas().items()
        if not (directory / name).exists() or (directory / name).read_text() != text
    ]


def main(argv: list[str] | None = None) -> int:
    """Write the schemas, or in ``--check`` mode verify they are up to date."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        stale = stale_files(SCHEMA_DIR)
        for path in stale:
            print(f"{path} is out of date.", file=sys.stderr)
        if stale:
            print("Run: python scripts/generate_memory_schemas.py", file=sys.stderr)
            return 1
        print(f"{SCHEMA_DIR} is up to date.")
        return 0

    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in render_schemas().items():
        (SCHEMA_DIR / name).write_text(text)
        print(f"Wrote {SCHEMA_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
