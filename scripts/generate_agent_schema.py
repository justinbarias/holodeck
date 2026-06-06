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
                "Run: python scripts/generate_agent_schema.py",
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
