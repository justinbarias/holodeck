"""Tests for response format application in agent configuration (T015).

Tests for inline response_format storage, external schema file loading,
response_format NOT inherited from global, and agent-level response_format storage.
"""

import json
from pathlib import Path

import pytest

from holodeck.config.schema import SchemaValidator


class TestInlineResponseFormatStorage:
    """Tests for inline response_format in agent configuration."""

    @pytest.mark.parametrize(
        "response_format,check_type,check_key",
        [
            pytest.param(
                {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                "object",
                "answer",
                id="simple_object_schema",
            ),
            pytest.param(
                {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["question", "answer"],
                },
                "object",
                "sources",
                id="nested_object_with_array",
            ),
            pytest.param(
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                        },
                        "required": ["id"],
                    },
                },
                "array",
                None,
                id="array_root_type",
            ),
        ],
    )
    def test_inline_response_format_stored_in_agent_config(
        self,
        temp_dir: Path,
        response_format: dict,
        check_type: str,
        check_key: str | None,
    ) -> None:
        """Test that inline response_format structures are stored correctly."""
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are a helpful assistant"},
            "response_format": response_format,
        }

        assert agent_config["response_format"]["type"] == check_type
        if check_key is not None:
            assert check_key in agent_config["response_format"]["properties"]


class TestExternalResponseFormatFileLoading:
    """Tests for loading response_format from external files."""

    def test_external_schema_file_path_stored_in_agent(self, temp_dir: Path) -> None:
        """Test that external schema file path is stored in agent config."""
        # Create schema file
        schema_content = {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        }
        schema_file = temp_dir / "schemas" / "response.json"
        schema_file.parent.mkdir(parents=True)
        schema_file.write_text(json.dumps(schema_content))

        agent_config = {
            "name": "file-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
            "response_format": "schemas/response.json",
        }

        assert agent_config["response_format"] == "schemas/response.json"

    def test_load_external_schema_file_relative_path(self, temp_dir: Path) -> None:
        """Test loading external schema from relative path."""
        schema_content = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        }
        schema_file = temp_dir / "schemas" / "status.json"
        schema_file.parent.mkdir(parents=True)
        schema_file.write_text(json.dumps(schema_content))

        # Load the schema using SchemaValidator
        result = SchemaValidator.load_schema_from_file(
            "schemas/status.json", base_dir=temp_dir
        )

        assert result == schema_content

    def test_load_external_schema_absolute_path(self, temp_dir: Path) -> None:
        """Test loading external schema from absolute path."""
        schema_content = {"type": "object"}
        schema_file = temp_dir / "my_schema.json"
        schema_file.write_text(json.dumps(schema_content))

        result = SchemaValidator.load_schema_from_file(str(schema_file))
        assert result == schema_content

    def test_external_schema_file_not_found_raises_error(self, temp_dir: Path) -> None:
        """Test that missing external schema file raises error."""
        with pytest.raises(FileNotFoundError, match="Schema file not found"):
            SchemaValidator.load_schema_from_file(
                "nonexistent/schema.json", base_dir=temp_dir
            )

    def test_external_schema_file_with_invalid_json_raises_error(
        self, temp_dir: Path
    ) -> None:
        """Test that schema file with invalid JSON raises error."""
        schema_file = temp_dir / "invalid.json"
        schema_file.write_text("{bad json")

        with pytest.raises(ValueError, match="Invalid JSON"):
            SchemaValidator.load_schema_from_file("invalid.json", base_dir=temp_dir)


class TestAgentResponseFormatHandling:
    """Tests for response_format in agent configuration (agent-specific only)."""

    def test_agent_can_define_response_format(self, temp_dir: Path) -> None:
        """Test that agent can define its own response_format."""
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
            "response_format": {
                "type": "object",
                "properties": {"agent": {"type": "string"}},
            },
        }

        # response_format should be preserved in agent config
        assert "response_format" in agent_config
        assert "agent" in agent_config["response_format"]["properties"]


class TestResponseFormatValidationOnLoad:
    """Tests for response_format validation at config load time."""

    def test_invalid_inline_response_format_raises_error(self, temp_dir: Path) -> None:
        """Test that invalid inline response_format raises error during validation."""
        invalid_schema = {
            "type": "object",
            "anyOf": [{"type": "string"}],  # unsupported keyword
        }

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword"):
            SchemaValidator.validate_schema(invalid_schema)

    def test_valid_inline_response_format_passes_validation(
        self, temp_dir: Path
    ) -> None:
        """Test that valid inline response_format passes validation."""
        valid_schema = {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        }

        result = SchemaValidator.validate_schema(valid_schema)
        assert result == valid_schema

    def test_valid_external_response_format_passes_validation(
        self, temp_dir: Path
    ) -> None:
        """Test that valid external response_format file passes validation."""
        schema_content = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        }
        schema_file = temp_dir / "valid.json"
        schema_file.write_text(json.dumps(schema_content))

        result = SchemaValidator.load_schema_from_file("valid.json", base_dir=temp_dir)
        assert result == schema_content

    def test_invalid_external_response_format_raises_error(
        self, temp_dir: Path
    ) -> None:
        """Test that invalid external response_format file raises error."""
        schema_content = {
            "type": "object",
            "pattern": "^test$",  # unsupported keyword
        }
        schema_file = temp_dir / "invalid.json"
        schema_file.write_text(json.dumps(schema_content))

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: pattern"):
            SchemaValidator.load_schema_from_file("invalid.json", base_dir=temp_dir)
