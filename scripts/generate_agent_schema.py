"""Deterministically generate ``schemas/agent.schema.json`` from the Agent model.

The JSON Schema published for editor auto-complete/validation is derived
verbatim from the Pydantic ``Agent`` model — never hand-edited. Run this
whenever the model tree changes:

    python scripts/generate_agent_schema.py            # write the file
    python scripts/generate_agent_schema.py --check     # CI: fail if stale

The ``--check`` mode is also exercised by ``tests/unit/test_agent_schema_sync.py``
so a drifted schema fails the suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from holodeck.models.agent import Agent

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "agent.schema.json"

# Authoring keys the config loader resolves *before* the Agent model validates,
# so they never appear on the model (and thus not in ``model_json_schema()``)
# yet are valid to write in an agent.yaml. They must be injected here or the
# closed (``additionalProperties: false``) schema would reject them in editors.
# Keep in sync with ``holodeck.config.loader`` resolution helpers.
_LOADER_RESOLVED_PROPERTIES: dict[str, dict[str, Any]] = {
    "test_cases_file": {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "default": None,
        "title": "Test Cases File",
        "description": (
            "Path to an external YAML file holding the test-case list, "
            "relative to the agent.yaml directory. Resolved into `test_cases` "
            "by the config loader at load time (see "
            "holodeck.config.loader._resolve_test_cases_file); mutually "
            "exclusive with an inline `test_cases`."
        ),
    },
}


def render_schema() -> str:
    """Return the canonical JSON-Schema text for the Agent model.

    Augments the model-derived schema with loader-resolved authoring keys
    (e.g. ``test_cases_file``) that are valid in an agent.yaml but stripped
    before model validation, so editors backed by the closed schema accept
    them.
    """
    schema = Agent.model_json_schema()
    schema.setdefault("properties", {}).update(_LOADER_RESOLVED_PROPERTIES)
    return json.dumps(schema, indent=2) + "\n"


def schema_paths() -> list[Path]:
    """Return every tracked copy of the schema: the published one first.

    Each ``holodeck init`` template ships a copy so that editors validate a
    new project offline. The copies must not drift from the published file.
    """
    templates = SCHEMA_PATH.parent.parent / "src" / "holodeck" / "templates"
    # List the template directories, not the existing copies: a deleted copy
    # must fail ``--check`` and be written again.
    copies = [
        path / "schemas" / "agent.schema.json"
        for path in sorted(templates.iterdir())
        if path.is_dir() and not path.name.startswith("_")
    ]
    return [SCHEMA_PATH, *copies]


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
    targets = schema_paths()
    if args.check:
        stale = [
            path
            for path in targets
            if not path.exists() or path.read_text() != rendered
        ]
        if stale:
            for path in stale:
                print(f"{path} is out of date.", file=sys.stderr)
            print("Run: python scripts/generate_agent_schema.py", file=sys.stderr)
            return 1
        print(f"{SCHEMA_PATH} is up to date.")
        return 0

    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
