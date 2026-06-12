"""Deterministically generate ``schemas/workflow.schema.json`` from the model.

The JSON Schema published for editor auto-complete/validation of
``workflow.yaml`` is derived verbatim from the Pydantic ``Workflow`` model —
never hand-edited. Run this whenever the model tree changes:

    python scripts/generate_workflow_schema.py            # write the file
    python scripts/generate_workflow_schema.py --check     # CI: fail if stale

The ``--check`` mode is also exercised by
``tests/unit/test_workflow_schema_sync.py`` so a drifted schema fails the suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from holodeck.models.workflow import Workflow

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "workflow.schema.json"
)


def render_schema() -> str:
    """Return the canonical JSON-Schema text for the Workflow model."""
    schema = Workflow.model_json_schema()
    return json.dumps(schema, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the schema, or in ``--check`` mode verify it is up to date."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed schema differs from the model.",
    )
    args = parser.parse_args(argv)

    rendered = render_schema()
    if args.check:
        current = SCHEMA_PATH.read_text() if SCHEMA_PATH.exists() else ""
        if current != rendered:
            print(
                f"{SCHEMA_PATH} is out of date. "
                "Run: python scripts/generate_workflow_schema.py",
                file=sys.stderr,
            )
            return 1
        print(f"{SCHEMA_PATH} is up to date.")
        return 0

    SCHEMA_PATH.write_text(rendered)
    print(f"Wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
