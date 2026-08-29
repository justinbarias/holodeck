"""The committed workflow.schema.json must match the Workflow model exactly.

Guards against hand-edits drifting from the Pydantic source of truth. If this
fails, run: ``python scripts/generate_workflow_schema.py``.
"""

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from holodeck.models.workflow import _FEEL_SAFE_IDENTIFIER_PATTERN
from scripts.generate_workflow_schema import render_schema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "workflow.schema.json"


def _workflow_with(node_id: str, input_data_key: str | None = None) -> dict[str, Any]:
    """A minimal workflow document with a single edge node."""
    document: dict[str, Any] = {
        "name": "wf",
        "version": "1.0.0",
        "nodes": [
            {
                "id": node_id,
                "edge": {"agent": "agents/a.yaml"},
                "gate": {"schema": "gates/a.json"},
            }
        ],
    }
    if input_data_key is not None:
        document["input_data"] = {input_data_key: {"schema": "schemas/a.json"}}
    return document


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


# ---------------------------------------------------------------------------
# What the runtime rejects, the published schema must reject too. A model
# validator is invisible to model_json_schema(), so a constraint expressed only
# there leaves an author's editor showing green on a file the runner refuses.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("node_kind", ["EdgeNode", "PolicyNode", "HumanNode"])
def test_published_node_id_carries_the_identifier_pattern(node_kind: str) -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    assert (
        schema["$defs"][node_kind]["properties"]["id"]["pattern"]
        == _FEEL_SAFE_IDENTIFIER_PATTERN
    )


@pytest.mark.unit
def test_published_input_data_property_names_match_the_model() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["properties"]["input_data"]["propertyNames"] == {
        "pattern": _FEEL_SAFE_IDENTIFIER_PATTERN
    }


@pytest.mark.unit
def test_published_schema_rejects_a_node_id_feel_cannot_bind() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_workflow_with("prior-state"), schema)


@pytest.mark.unit
def test_published_schema_rejects_an_input_data_key_feel_cannot_bind() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_workflow_with("evidence", "prior-state"), schema)


@pytest.mark.unit
def test_published_schema_still_accepts_well_formed_names() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(_workflow_with("evidence", "prior_state"), schema)
