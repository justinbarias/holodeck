"""Unit tests for holodeck.lib.backends.openai_agents_output.

The `openai-agents` package is installed (dev extra) so the lazy SDK imports in
the factory resolve; ``jsonschema`` is a hard dependency. No network calls or
credentials are required.
"""

import json
from pathlib import Path

import pytest

from holodeck.lib.backends.openai_agents_output import (
    build_output_schema,
    load_response_format_schema,
    schema_qualifies_for_strict,
)

# A loose object schema (does not set additionalProperties:false / required).
_LOOSE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
}

# A strict-qualifying schema.
_STRICT_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


@pytest.mark.unit
class TestLoadResponseFormatSchema:
    """Resolving the three response_format shapes to a schema dict."""

    def test_none_returns_none(self) -> None:
        assert load_response_format_schema(None) is None

    def test_dict_returned_as_is(self) -> None:
        assert load_response_format_schema(_LOOSE_SCHEMA) == _LOOSE_SCHEMA

    def test_str_path_loaded_relative_to_base_dir(self, tmp_path: Path) -> None:
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(_LOOSE_SCHEMA))
        loaded = load_response_format_schema("schema.json", base_dir=tmp_path)
        assert loaded == _LOOSE_SCHEMA

    def test_str_path_absolute(self, tmp_path: Path) -> None:
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(_STRICT_SCHEMA))
        loaded = load_response_format_schema(str(schema_file))
        assert loaded == _STRICT_SCHEMA

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            load_response_format_schema("missing.json", base_dir=tmp_path)

    def test_non_object_json_raises(self, tmp_path: Path) -> None:
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="JSON object"):
            load_response_format_schema("schema.json", base_dir=tmp_path)


@pytest.mark.unit
class TestSchemaQualifiesForStrict:
    """Detecting whether a schema satisfies OpenAI strict-mode constraints."""

    def test_strict_schema_qualifies(self) -> None:
        assert schema_qualifies_for_strict(_STRICT_SCHEMA) is True

    def test_loose_schema_does_not_qualify(self) -> None:
        assert schema_qualifies_for_strict(_LOOSE_SCHEMA) is False

    def test_non_object_does_not_qualify(self) -> None:
        assert schema_qualifies_for_strict({"type": "string"}) is False

    def test_missing_required_does_not_qualify(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        assert schema_qualifies_for_strict(schema) is False


@pytest.mark.unit
class TestBuildOutputSchema:
    """The JSONSchemaOutputSchema SDK subclass driven by a JSON schema dict."""

    def test_abstract_surface(self) -> None:
        out = build_output_schema(_LOOSE_SCHEMA, name="my_schema")
        assert out.is_plain_text() is False
        assert out.name() == "my_schema"
        assert out.json_schema() == _LOOSE_SCHEMA

    def test_loose_schema_is_not_strict(self) -> None:
        out = build_output_schema(_LOOSE_SCHEMA)
        assert out.is_strict_json_schema() is False

    def test_strict_schema_is_strict(self) -> None:
        out = build_output_schema(_STRICT_SCHEMA)
        assert out.is_strict_json_schema() is True

    def test_validate_json_returns_parsed_object(self) -> None:
        out = build_output_schema(_LOOSE_SCHEMA)
        assert out.validate_json('{"answer": "42"}') == {"answer": "42"}

    def test_validate_json_rejects_malformed_json(self) -> None:
        from agents.exceptions import ModelBehaviorError

        out = build_output_schema(_LOOSE_SCHEMA)
        with pytest.raises(ModelBehaviorError, match="not valid JSON"):
            out.validate_json("not json")

    def test_validate_json_rejects_schema_violation(self) -> None:
        from agents.exceptions import ModelBehaviorError

        out = build_output_schema(_STRICT_SCHEMA)
        with pytest.raises(ModelBehaviorError, match="does not match"):
            out.validate_json('{"answer": 42, "extra": true}')

    def test_subclasses_sdk_base(self) -> None:
        from agents.agent_output import AgentOutputSchemaBase

        out = build_output_schema(_LOOSE_SCHEMA)
        assert isinstance(out, AgentOutputSchemaBase)
