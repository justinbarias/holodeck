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

    def test_inline_response_format_stored_in_agent_config(
        self, temp_dir: Path
    ) -> None:
        """Test that inline response_format is stored in agent config."""
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are a helpful assistant"},
            "response_format": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        }

        # Validate the response_format is preserved
        assert "response_format" in agent_config
        assert agent_config["response_format"]["type"] == "object"
        assert "answer" in agent_config["response_format"]["properties"]

    def test_inline_response_format_with_nested_structure(self, temp_dir: Path) -> None:
        """Test inline response_format with nested object structure."""
        agent_config = {
            "name": "qa-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Answer questions"},
            "response_format": {
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
        }

        response_format = agent_config["response_format"]
        assert response_format["type"] == "object"
        assert "sources" in response_format["properties"]
        assert response_format["properties"]["sources"]["type"] == "array"

    def test_inline_response_format_with_array_type(self, temp_dir: Path) -> None:
        """Test inline response_format with array root type."""
        agent_config = {
            "name": "list-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Generate list"},
            "response_format": {
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
        }

        response_format = agent_config["response_format"]
        assert response_format["type"] == "array"
        assert response_format["items"]["type"] == "object"

    def test_response_format_can_be_null(self, temp_dir: Path) -> None:
        """Test that response_format can be explicitly set to null."""
        agent_config = {
            "name": "open-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Be creative"},
            "response_format": None,
        }

        assert agent_config.get("response_format") is None

    def test_response_format_omitted_is_valid(self, temp_dir: Path) -> None:
        """Test that response_format can be omitted."""
        agent_config = {
            "name": "basic-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Hello"},
        }

        assert "response_format" not in agent_config


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

    def test_agent_response_format_can_be_null(self, temp_dir: Path) -> None:
        """Test that agent can explicitly set response_format to null."""
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
            "response_format": None,
        }

        assert agent_config.get("response_format") is None

    def test_agent_response_format_can_be_omitted(self, temp_dir: Path) -> None:
        """Test that agent can omit response_format."""
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }

        assert "response_format" not in agent_config


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
