"""The committed agent.schema.json must match the Agent model exactly.

Guards against hand-edits drifting from the Pydantic source of truth. If this
fails, run: ``python scripts/generate_agent_schema.py``.
"""

import json
from pathlib import Path

import pytest

from holodeck.models.agent import Agent

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "agent.schema.json"


@pytest.mark.unit
def test_committed_schema_matches_model() -> None:
    expected = json.dumps(Agent.model_json_schema(), indent=2) + "\n"
    assert SCHEMA_PATH.read_text() == expected, (
        "schemas/agent.schema.json is stale — run "
        "`python scripts/generate_agent_schema.py`."
    )


@pytest.mark.unit
def test_schema_exposes_optimizer_block() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    assert "OptimizerConfig" in schema["$defs"]
    optimizer = schema["$defs"]["EvaluationConfig"]["properties"]["optimizer"]
    assert {"$ref": "#/$defs/OptimizerConfig"} in optimizer["anyOf"]
