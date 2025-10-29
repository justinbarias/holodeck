"""Tests for response format schema validation (T014).

Tests for inline YAML-to-JSON schema parsing, external schema file loading,
Basic JSON Schema keyword validation, and rejection of unsupported keywords.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from holodeck.config.schema import SchemaValidator


class TestInlineSchemaValidation:
    """Tests for inline response_format schema validation."""

    def test_inline_dict_schema_validation(self) -> None:
        """Test validation of inline dict schema."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "data": {"type": "object"},
            },
            "required": ["status"],
        }

        result = SchemaValidator.validate_schema(schema)
        assert result == schema

    def test_inline_json_string_schema_validation(self) -> None:
        """Test validation of inline JSON string schema."""
        schema_dict = {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            "required": ["id", "name"],
        }
        schema_json = json.dumps(schema_dict)

        result = SchemaValidator.validate_schema(schema_json)
        assert result == schema_dict

    def test_array_schema_validation(self) -> None:
        """Test validation of array schema with items."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                },
                "required": ["id"],
            },
        }

        result = SchemaValidator.validate_schema(schema)
        assert result == schema

    def test_schema_with_enum_validation(self) -> None:
        """Test validation of schema with enum."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"],
                }
            },
        }

        result = SchemaValidator.validate_schema(schema)
        assert result == schema

    def test_schema_with_description_validation(self) -> None:
        """Test validation of schema with description."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User full name",
                }
            },
            "description": "User profile schema",
        }

        result = SchemaValidator.validate_schema(schema)
        assert result == schema

    def test_simple_string_schema(self) -> None:
        """Test validation of simple string type schema."""
        schema = {"type": "string"}

        result = SchemaValidator.validate_schema(schema)
        assert result == schema

    def test_simple_number_schema(self) -> None:
        """Test validation of simple number type schema."""
        schema = {"type": "number"}

        result = SchemaValidator.validate_schema(schema)
        assert result == schema

    def test_schema_with_minimum_maximum(self) -> None:
        """Test validation of schema with minimum/maximum."""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                }
            },
        }

        result = SchemaValidator.validate_schema(schema)
        assert result == schema

    def test_nested_properties_validation(self) -> None:
        """Test validation of deeply nested properties."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                        }
                    },
                }
            },
        }

        result = SchemaValidator.validate_schema(schema)
        assert result == schema

    def test_invalid_json_string_raises_error(self) -> None:
        """Test that invalid JSON string raises ValueError."""
        invalid_json = '{"type": "object" broken json'

        with pytest.raises(ValueError, match="Invalid JSON"):
            SchemaValidator.validate_schema(invalid_json)

    def test_json_string_not_object_raises_error(self) -> None:
        """Test that JSON string that's not an object raises ValueError."""
        not_object = json.dumps([1, 2, 3])

        with pytest.raises(ValueError, match="must be object"):
            SchemaValidator.validate_schema(not_object)


class TestUnsupportedKeywordRejection:
    """Tests for rejection of unsupported JSON Schema keywords."""

    def test_ref_keyword_rejected(self) -> None:
        """Test that $ref keyword is rejected."""
        schema = {
            "type": "object",
            "properties": {"user": {"$ref": "#/definitions/user"}},
        }

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: \\$ref"):
            SchemaValidator.validate_schema(schema)

    def test_anyof_keyword_rejected(self) -> None:
        """Test that anyOf keyword is rejected."""
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
            ]
        }

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: anyOf"):
            SchemaValidator.validate_schema(schema)

    def test_oneof_keyword_rejected(self) -> None:
        """Test that oneOf keyword is rejected."""
        schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "number"},
            ]
        }

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: oneOf"):
            SchemaValidator.validate_schema(schema)

    def test_allof_keyword_rejected(self) -> None:
        """Test that allOf keyword is rejected."""
        schema = {
            "allOf": [
                {"type": "object"},
                {"properties": {"name": {"type": "string"}}},
            ]
        }

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: allOf"):
            SchemaValidator.validate_schema(schema)

    def test_pattern_properties_keyword_rejected(self) -> None:
        """Test that patternProperties keyword is rejected."""
        schema = {
            "type": "object",
            "patternProperties": {"^[a-z]+$": {"type": "string"}},
        }

        with pytest.raises(
            ValueError, match="Unknown JSON Schema keyword: patternProperties"
        ):
            SchemaValidator.validate_schema(schema)

    def test_pattern_keyword_rejected(self) -> None:
        """Test that pattern keyword is rejected."""
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": "^[a-z]+@[a-z]+$",
                }
            },
        }

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: pattern"):
            SchemaValidator.validate_schema(schema)

    def test_min_length_keyword_rejected(self) -> None:
        """Test that minLength keyword is rejected."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                }
            },
        }

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: minLength"):
            SchemaValidator.validate_schema(schema)

    def test_max_length_keyword_rejected(self) -> None:
        """Test that maxLength keyword is rejected."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "maxLength": 100,
                }
            },
        }

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: maxLength"):
            SchemaValidator.validate_schema(schema)

    def test_unknown_custom_keyword_rejected(self) -> None:
        """Test that unknown custom keywords are rejected."""
        schema = {
            "type": "object",
            "custom_keyword": "value",
        }

        with pytest.raises(
            ValueError, match="Unknown JSON Schema keyword: custom_keyword"
        ):
            SchemaValidator.validate_schema(schema)

    def test_multiple_unsupported_keywords_shows_first(self) -> None:
        """Test that first unsupported keyword is reported."""
        schema = {
            "type": "object",
            "anyOf": [{"type": "string"}],
            "pattern": "^test$",
        }

        with pytest.raises(ValueError):
            SchemaValidator.validate_schema(schema)


class TestBasicKeywordSupport:
    """Tests for supported Basic JSON Schema keywords."""

    def test_type_keyword_supported(self) -> None:
        """Test that 'type' keyword is supported."""
        for type_val in [
            "string",
            "number",
            "integer",
            "boolean",
            "array",
            "object",
            "null",
        ]:
            schema = {"type": type_val}
            result = SchemaValidator.validate_schema(schema)
            assert result["type"] == type_val

    def test_properties_keyword_supported(self) -> None:
        """Test that 'properties' keyword is supported."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = SchemaValidator.validate_schema(schema)
        assert "properties" in result

    def test_required_keyword_supported(self) -> None:
        """Test that 'required' keyword is supported."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = SchemaValidator.validate_schema(schema)
        assert result["required"] == ["name"]

    def test_additional_properties_keyword_supported(self) -> None:
        """Test that 'additionalProperties' keyword is supported."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        result = SchemaValidator.validate_schema(schema)
        assert result["additionalProperties"] is False

    def test_items_keyword_supported(self) -> None:
        """Test that 'items' keyword is supported."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        result = SchemaValidator.validate_schema(schema)
        assert "items" in result

    def test_enum_keyword_supported(self) -> None:
        """Test that 'enum' keyword is supported."""
        schema = {
            "enum": ["active", "inactive", "pending"],
        }
        result = SchemaValidator.validate_schema(schema)
        assert result["enum"] == ["active", "inactive", "pending"]

    def test_default_keyword_supported(self) -> None:
        """Test that 'default' keyword is supported."""
        schema = {
            "type": "string",
            "default": "unknown",
        }
        result = SchemaValidator.validate_schema(schema)
        assert result["default"] == "unknown"

    def test_minimum_maximum_keywords_supported(self) -> None:
        """Test that 'minimum' and 'maximum' keywords are supported."""
        schema = {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        }
        result = SchemaValidator.validate_schema(schema)
        assert result["minimum"] == 0
        assert result["maximum"] == 100


class TestExternalSchemaFileLoading:
    """Tests for loading external schema files."""

    def test_load_schema_from_json_file(self, temp_dir: Path) -> None:
        """Test loading schema from .json file."""
        schema_content = {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        }
        schema_file = temp_dir / "schema.json"
        schema_file.write_text(json.dumps(schema_content))

        result = SchemaValidator.load_schema_from_file("schema.json", base_dir=temp_dir)
        assert result == schema_content

    def test_load_schema_with_absolute_path(self, temp_dir: Path) -> None:
        """Test loading schema with absolute file path."""
        schema_content = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        schema_file = temp_dir / "my_schema.json"
        schema_file.write_text(json.dumps(schema_content))

        result = SchemaValidator.load_schema_from_file(str(schema_file))
        assert result == schema_content

    def test_load_schema_from_nested_path(self, temp_dir: Path) -> None:
        """Test loading schema from nested directory."""
        schema_dir = temp_dir / "schemas"
        schema_dir.mkdir()

        schema_content = {"type": "object"}
        schema_file = schema_dir / "response.json"
        schema_file.write_text(json.dumps(schema_content))

        result = SchemaValidator.load_schema_from_file(
            "schemas/response.json", base_dir=temp_dir
        )
        assert result == schema_content

    def test_load_nonexistent_schema_file_raises_error(self, temp_dir: Path) -> None:
        """Test that loading nonexistent schema file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Schema file not found"):
            SchemaValidator.load_schema_from_file("nonexistent.json", base_dir=temp_dir)

    def test_load_schema_file_with_invalid_json_raises_error(
        self, temp_dir: Path
    ) -> None:
        """Test that schema file with invalid JSON raises ValueError."""
        schema_file = temp_dir / "invalid.json"
        schema_file.write_text("{ invalid json")

        with pytest.raises(ValueError, match="Invalid JSON in schema file"):
            SchemaValidator.load_schema_from_file("invalid.json", base_dir=temp_dir)

    def test_load_schema_file_with_unsupported_keyword_raises_error(
        self, temp_dir: Path
    ) -> None:
        """Test that schema file with unsupported keyword raises ValueError."""
        schema_content = {
            "type": "object",
            "anyOf": [{"type": "string"}],
        }
        schema_file = temp_dir / "unsupported.json"
        schema_file.write_text(json.dumps(schema_content))

        with pytest.raises(ValueError, match="Unknown JSON Schema keyword: anyOf"):
            SchemaValidator.load_schema_from_file("unsupported.json", base_dir=temp_dir)

    def test_load_schema_file_not_object_raises_error(self, temp_dir: Path) -> None:
        """Test that schema file that's not an object raises ValueError."""
        schema_file = temp_dir / "array.json"
        schema_file.write_text(json.dumps([1, 2, 3]))

        with pytest.raises(ValueError, match="must be JSON object"):
            SchemaValidator.load_schema_from_file("array.json", base_dir=temp_dir)

    def test_load_schema_uses_cwd_as_default_base_dir(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that current working directory is used as default base_dir."""
        schema_content = {"type": "object"}
        schema_file = temp_dir / "schema.json"
        schema_file.write_text(json.dumps(schema_content))

        # Change to temp_dir and load relative path
        monkeypatch.chdir(temp_dir)

        result = SchemaValidator.load_schema_from_file("schema.json")
        assert result == schema_content
