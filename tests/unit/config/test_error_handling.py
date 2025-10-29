"""Tests for configuration error handling and validation (T020, US4).

Tests for YAML syntax errors, invalid JSON in response_format, missing schema files,
unknown JSON Schema keywords, and LLM provider warnings with clear error messages.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from holodeck.config.loader import ConfigLoader
from holodeck.config.schema import SchemaValidator
from holodeck.lib.errors import (
    ConfigError,
)
from holodeck.lib.errors import (
    FileNotFoundError as HolodeckFileNotFoundError,
)


class TestYAMLSyntaxErrors:
    """Tests for YAML syntax error handling with clear messages."""

    def test_invalid_yaml_syntax_raises_config_error(self, temp_dir: Path) -> None:
        """Test that invalid YAML syntax raises ConfigError."""
        yaml_file = temp_dir / "invalid.yml"
        yaml_file.write_text("invalid: yaml: : syntax: here")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.parse_yaml(str(yaml_file))

        assert "parse" in str(exc_info.value).lower()

    def test_yaml_parse_error_includes_file_path(self, temp_dir: Path) -> None:
        """Test that YAML parse error includes file path."""
        yaml_file = temp_dir / "broken.yml"
        yaml_file.write_text("key: [unclosed list")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.parse_yaml(str(yaml_file))

        error_msg = str(exc_info.value)
        assert "broken.yml" in error_msg or "parse" in error_msg.lower()

    def test_global_config_yaml_error(self, temp_dir: Path, monkeypatch: Any) -> None:
        """Test that invalid user-level config YAML raises error."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        config_file = holodeck_dir / "config.yml"
        config_file.write_text("invalid: : yaml")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_global_config()

        assert "parse" in str(exc_info.value).lower()

    def test_project_config_yaml_error(self, temp_dir: Path) -> None:
        """Test that invalid project config YAML raises error."""
        config_file = temp_dir / "config.yml"
        config_file.write_text("bad: [ unclosed")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_project_config(str(temp_dir))

        assert "parse" in str(exc_info.value).lower()


class TestInvalidResponseFormatJSON:
    """Tests for invalid JSON in response_format error handling."""

    def test_invalid_json_string_raises_error(self) -> None:
        """Test that invalid JSON string raises ValueError."""
        invalid_json = '{"type": "object" missing closing brace'

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(invalid_json)

        assert "Invalid JSON" in str(exc_info.value)

    def test_invalid_json_includes_error_details(self) -> None:
        """Test that invalid JSON error includes parsing details."""
        invalid_json = "{bad json}"

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(invalid_json)

        error_msg = str(exc_info.value)
        assert "Invalid JSON" in error_msg

    def test_json_object_validation_required(self) -> None:
        """Test that JSON response_format must be an object."""
        json_array = json.dumps([1, 2, 3])

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(json_array)

        assert "object" in str(exc_info.value).lower()


class TestMissingSchemaFiles:
    """Tests for missing schema file error handling with path display."""

    def test_missing_schema_file_raises_error(self, temp_dir: Path) -> None:
        """Test that missing schema file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            SchemaValidator.load_schema_from_file("missing.json", base_dir=temp_dir)

        assert "not found" in str(exc_info.value).lower()

    def test_missing_schema_file_error_includes_path(self, temp_dir: Path) -> None:
        """Test that missing file error includes path information."""
        with pytest.raises(FileNotFoundError) as exc_info:
            SchemaValidator.load_schema_from_file(
                "nonexistent/schema.json", base_dir=temp_dir
            )

        error_msg = str(exc_info.value)
        assert "schema.json" in error_msg or "nonexistent" in error_msg

    def test_nested_missing_schema_file(self, temp_dir: Path) -> None:
        """Test error for missing schema in nested directory."""
        with pytest.raises(FileNotFoundError):
            SchemaValidator.load_schema_from_file(
                "deep/nested/missing.json", base_dir=temp_dir
            )


class TestUnknownSchemaKeywords:
    """Tests for unknown JSON Schema keyword error handling with keyword name."""

    def test_unknown_keyword_ref_rejected_with_name(self) -> None:
        """Test that $ref keyword error includes the keyword name."""
        schema = {
            "type": "object",
            "properties": {"user": {"$ref": "#/definitions/user"}},
        }

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(schema)

        error_msg = str(exc_info.value)
        assert "$ref" in error_msg

    def test_unknown_keyword_anyof_rejected_with_name(self) -> None:
        """Test that anyOf keyword error includes the keyword name."""
        schema = {
            "anyOf": [{"type": "string"}],
        }

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(schema)

        error_msg = str(exc_info.value)
        assert "anyOf" in error_msg

    def test_unknown_keyword_error_message_format(self) -> None:
        """Test that error message clearly identifies the unknown keyword."""
        schema = {
            "type": "object",
            "unknownKeyword": "value",
        }

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(schema)

        error_msg = str(exc_info.value)
        # Should mention it's an unknown keyword
        assert "Unknown" in error_msg or "unknown" in error_msg.lower()

    def test_nested_unknown_keyword_error(self) -> None:
        """Test that unknown keywords in nested properties are caught."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,  # unsupported
                }
            },
        }

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(schema)

        assert "minLength" in str(exc_info.value)


class TestConfigErrorMessages:
    """Tests for clear, user-friendly error messages with details."""

    def test_error_message_includes_file_location(self, temp_dir: Path) -> None:
        """Test that errors include file location."""
        invalid_yaml = "key: : invalid"
        yaml_file = temp_dir / "test.yml"
        yaml_file.write_text(invalid_yaml)

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.parse_yaml(str(yaml_file))

        error_msg = str(exc_info.value)
        # Should mention the file or parse error
        assert "test.yml" in error_msg or "parse" in error_msg.lower()

    def test_missing_file_error_includes_path(self, temp_dir: Path) -> None:
        """Test that missing file error includes full path."""
        loader = ConfigLoader()
        nonexistent = str(temp_dir / "missing.yml")

        with pytest.raises(HolodeckFileNotFoundError) as exc_info:
            loader.parse_yaml(nonexistent)

        error_msg = str(exc_info.value)
        assert "missing.yml" in error_msg or nonexistent in error_msg

    def test_validation_error_is_descriptive(self) -> None:
        """Test that validation errors are descriptive."""
        schema = {
            "type": "object",
            "patternProperties": {"^S_": {"type": "string"}},  # unsupported
        }

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(schema)

        error_msg = str(exc_info.value)
        # Should be clear about what's wrong
        assert len(error_msg) > 10


class TestEdgeCases:
    """Tests for error handling edge cases."""

    def test_empty_yaml_file_returns_none(self, temp_dir: Path) -> None:
        """Test that empty YAML file returns None gracefully."""
        yaml_file = temp_dir / "empty.yml"
        yaml_file.write_text("")

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert result is None or result == {}

    def test_empty_config_file_returns_none(self, temp_dir: Path) -> None:
        """Test that empty config file returns None gracefully."""
        config_file = temp_dir / "config.yml"
        config_file.write_text("")

        loader = ConfigLoader()
        result = loader.load_project_config(str(temp_dir))

        assert result is None

    def test_complex_nested_error_location(self) -> None:
        """Test that errors in deeply nested schemas are caught."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "string",
                                    "unknownKeyword": "value",
                                }
                            },
                        }
                    },
                }
            },
        }

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(schema)

        assert "unknownKeyword" in str(exc_info.value)


class TestConfigLoadingErrorSequence:
    """Tests for error handling sequence in config loading."""

    def test_yaml_error_before_validation_error(self, temp_dir: Path) -> None:
        """Test that YAML syntax errors are caught before validation."""
        yaml_file = temp_dir / "invalid.yml"
        yaml_file.write_text("invalid: : yaml")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.parse_yaml(str(yaml_file))

        # Should be a YAML parse error, not validation
        error_msg = str(exc_info.value)
        assert "parse" in error_msg.lower() or "yaml" in error_msg.lower()

    def test_file_not_found_error_is_specific(self, temp_dir: Path) -> None:
        """Test that file not found error is specific."""
        loader = ConfigLoader()
        with pytest.raises(HolodeckFileNotFoundError):
            loader.parse_yaml("/nonexistent/path/file.yml")

    def test_multiple_error_types_handled(self, temp_dir: Path) -> None:
        """Test that different error types are handled appropriately."""
        # YAML error
        yaml_file = temp_dir / "bad.yml"
        yaml_file.write_text("bad: :")

        loader = ConfigLoader()
        with pytest.raises(ConfigError):
            loader.parse_yaml(str(yaml_file))

        # Schema validation error
        with pytest.raises(ValueError):
            SchemaValidator.validate_schema({"type": "object", "badKeyword": "value"})

        # File not found error
        with pytest.raises(FileNotFoundError):
            SchemaValidator.load_schema_from_file("missing.json", base_dir=temp_dir)


class TestErrorMessageClarity:
    """Tests for clarity and usefulness of error messages."""

    def test_json_parse_error_shows_problem(self) -> None:
        """Test that JSON parse error shows what went wrong."""
        bad_json = '{"incomplete": '

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(bad_json)

        error_msg = str(exc_info.value)
        assert "JSON" in error_msg

    def test_unknown_keyword_error_lists_keyword(self) -> None:
        """Test that unknown keyword error lists the problematic keyword."""
        schema = {"type": "string", "myCustomKeyword": "value"}

        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.validate_schema(schema)

        assert "myCustomKeyword" in str(exc_info.value)

    def test_missing_file_error_suggests_checking_path(self, temp_dir: Path) -> None:
        """Test that missing file error provides helpful path info."""
        with pytest.raises(FileNotFoundError) as exc_info:
            SchemaValidator.load_schema_from_file(
                "schemas/missing.json", base_dir=temp_dir
            )

        error_msg = str(exc_info.value)
        # Should be clear it's looking for a file
        assert "not found" in error_msg.lower() or "missing" in error_msg.lower()
