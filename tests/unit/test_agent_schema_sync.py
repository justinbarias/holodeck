"""The committed agent.schema.json must match the Agent model exactly.

Guards against hand-edits drifting from the Pydantic source of truth. If this
fails, run: ``python scripts/generate_agent_schema.py``.
"""

import json
from pathlib import Path

import pytest

from scripts.generate_agent_schema import render_schema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "agent.schema.json"


@pytest.mark.unit
def test_committed_schema_matches_model() -> None:
    # Compare against the generator's canonical output (the single source of
    # truth), which is the model schema plus loader-resolved authoring keys.
    assert SCHEMA_PATH.read_text() == render_schema(), (
        "schemas/agent.schema.json is stale — run "
        "`python scripts/generate_agent_schema.py`."
    )


@pytest.mark.unit
def test_schema_exposes_loader_resolved_test_cases_file() -> None:
    # `test_cases_file` is resolved into `test_cases` by the loader before the
    # Agent model validates, so it is not a model field — but it is valid to
    # author and must survive in the closed schema for editor validation.
    schema = json.loads(SCHEMA_PATH.read_text())
    assert "test_cases_file" in schema["properties"]
    assert schema["additionalProperties"] is False


@pytest.mark.unit
def test_schema_exposes_optimizer_block() -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    assert "OptimizerConfig" in schema["$defs"]
    optimizer = schema["$defs"]["EvaluationConfig"]["properties"]["optimizer"]
    assert {"$ref": "#/$defs/OptimizerConfig"} in optimizer["anyOf"]
