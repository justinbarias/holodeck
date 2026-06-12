"""The committed workflow.schema.json must match the Workflow model exactly.

Guards against hand-edits drifting from the Pydantic source of truth. If this
fails, run: ``python scripts/generate_workflow_schema.py``.
"""

import json
from pathlib import Path

import pytest

from scripts.generate_workflow_schema import render_schema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "workflow.schema.json"


@pytest.mark.unit
def test_committed_schema_matches_model() -> None:
    assert SCHEMA_PATH.read_text() == render_schema(), (
        "schemas/workflow.schema.json is stale — run "
        "`python scripts/generate_workflow_schema.py`."
    )


@pytest.mark.unit
def test_schema_is_closed_and_defines_all_node_kinds() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["additionalProperties"] is False
    defs = schema["$defs"]
    assert {"EdgeNode", "PolicyNode", "HumanNode"} <= set(defs)


@pytest.mark.unit
def test_edge_node_schema_makes_a_verdict_unrepresentable() -> None:
    # SC-008: an edge node's published schema must be closed and expose no
    # decision/hit_policy/inputs route through which AI output becomes a verdict.
    schema = json.loads(SCHEMA_PATH.read_text())
    edge = schema["$defs"]["EdgeNode"]
    assert edge["additionalProperties"] is False
    assert {"decision", "hit_policy", "inputs"}.isdisjoint(edge["properties"])
