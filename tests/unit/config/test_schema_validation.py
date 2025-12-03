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

    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "string"},
            {"type": "number"},
        ],
        ids=["simple_string", "simple_number"],
    )
    def test_simple_type_schemas(self, schema: dict[str, Any]) -> None:
        """Test validation of simple type schemas."""
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

    @pytest.mark.parametrize(
        "keyword,schema",
        [
            (
                "$ref",
                {
                    "type": "object",
                    "properties": {"user": {"$ref": "#/definitions/user"}},
                },
            ),
            (
                "anyOf",
                {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                    ]
                },
            ),
            (
                "oneOf",
                {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"},
                    ]
                },
            ),
            (
                "allOf",
                {
                    "allOf": [
                        {"type": "object"},
                        {"properties": {"name": {"type": "string"}}},
                    ]
                },
            ),
            (
                "patternProperties",
                {
                    "type": "object",
                    "patternProperties": {"^[a-z]+$": {"type": "string"}},
                },
            ),
            (
                "pattern",
                {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "pattern": "^[a-z]+@[a-z]+$",
                        }
                    },
                },
            ),
            (
                "minLength",
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "minLength": 1,
                        }
                    },
                },
            ),
            (
                "maxLength",
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "maxLength": 100,
                        }
                    },
                },
            ),
            (
                "custom_keyword",
                {
                    "type": "object",
                    "custom_keyword": "value",
                },
            ),
        ],
        ids=[
            "$ref",
            "anyOf",
            "oneOf",
            "allOf",
            "patternProperties",
            "pattern",
            "minLength",
            "maxLength",
            "custom_keyword",
        ],
    )
    def test_unsupported_keywords_rejected(
        self, keyword: str, schema: dict[str, Any]
    ) -> None:
        """Test that unsupported JSON Schema keywords are rejected."""
        # Escape $ in regex pattern for $ref keyword
        pattern = (
            f"Unknown JSON Schema keyword: \\{keyword}"
            if keyword == "$ref"
            else f"Unknown JSON Schema keyword: {keyword}"
        )

        with pytest.raises(ValueError, match=pattern):
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

    @pytest.mark.parametrize(
        "type_val",
        ["string", "number", "integer", "boolean", "array", "object", "null"],
        ids=["string", "number", "integer", "boolean", "array", "object", "null"],
    )
    def test_type_keyword_supported(self, type_val: str) -> None:
        """Test that 'type' keyword is supported for all basic types."""
        schema = {"type": type_val}
        result = SchemaValidator.validate_schema(schema)
        assert result["type"] == type_val

    @pytest.mark.parametrize(
        "keyword,schema,expected_value",
        [
            (
                "properties",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
                None,  # Just check presence
            ),
            (
                "required",
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                ["name"],
            ),
            (
                "additionalProperties",
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": False,
                },
                False,
            ),
            (
                "items",
                {
                    "type": "array",
                    "items": {"type": "string"},
                },
                None,  # Just check presence
            ),
            (
                "enum",
                {
                    "enum": ["active", "inactive", "pending"],
                },
                ["active", "inactive", "pending"],
            ),
            (
                "default",
                {
                    "type": "string",
                    "default": "unknown",
                },
                "unknown",
            ),
        ],
        ids=[
            "properties",
            "required",
            "additionalProperties",
            "items",
            "enum",
            "default",
        ],
    )
    def test_keyword_supported(
        self, keyword: str, schema: dict[str, Any], expected_value: Any
    ) -> None:
        """Test that basic JSON Schema keywords are supported."""
        result = SchemaValidator.validate_schema(schema)
        assert keyword in result
        if expected_value is not None:
            assert result[keyword] == expected_value

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
        from holodeck.config.context import agent_base_dir

        schema_content = {"type": "object"}
        schema_file = temp_dir / "schema.json"
        schema_file.write_text(json.dumps(schema_content))

        # Change to temp_dir and load relative path
        monkeypatch.chdir(temp_dir)

        # Reset context variable to ensure cwd fallback is tested
        # (other parallel tests may have set this)
        token = agent_base_dir.set(None)
        try:
            result = SchemaValidator.load_schema_from_file("schema.json")
            assert result == schema_content
        finally:
            agent_base_dir.reset(token)

    def test_load_schema_file_os_error(self, temp_dir: Path, monkeypatch: Any) -> None:
        """Test that OSError during file read raises FileNotFoundError."""
        schema_file = temp_dir / "schema.json"
        schema_file.write_text('{"type": "object"}')

        # Mock open to raise OSError
        original_open = open

        def mock_open(*args: Any, **kwargs: Any) -> Any:
            if "schema.json" in str(args[0]):
                raise OSError("Permission denied")
            return original_open(*args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        with pytest.raises(FileNotFoundError, match="Failed to read schema file"):
            SchemaValidator.load_schema_from_file("schema.json", base_dir=temp_dir)


class TestSchemaStructureValidation:
    """Tests for _validate_schema_structure error cases."""

    def test_invalid_type_value_rejected(self) -> None:
        """Test that invalid type values are rejected."""
        schema = {"type": "invalid_type"}

        with pytest.raises(ValueError, match="Invalid type 'invalid_type'"):
            SchemaValidator.validate_schema(schema)

    def test_properties_not_object_rejected(self) -> None:
        """Test that properties must be an object."""
        schema = {
            "type": "object",
            "properties": "not_an_object",
        }

        with pytest.raises(
            ValueError, match="'properties' at schema must be an object"
        ):
            SchemaValidator.validate_schema(schema)

    def test_required_not_array_rejected(self) -> None:
        """Test that required must be an array."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": "name",  # Should be ["name"]
        }

        with pytest.raises(ValueError, match="'required' at schema must be an array"):
            SchemaValidator.validate_schema(schema)

    def test_required_items_not_strings_rejected(self) -> None:
        """Test that required items must be strings."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": [123, "name"],  # 123 is not a string
        }

        with pytest.raises(
            ValueError, match="'required' items at schema must be strings"
        ):
            SchemaValidator.validate_schema(schema)

    def test_additional_properties_invalid_type_rejected(self) -> None:
        """Test that additionalProperties must be boolean or object."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": "false",  # String instead of boolean
        }

        with pytest.raises(
            ValueError,
            match="'additionalProperties' at schema must be boolean or object",
        ):
            SchemaValidator.validate_schema(schema)

    def test_additional_properties_as_object_accepted(self) -> None:
        """Test that additionalProperties can be an object schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": {"type": "string"},
        }

        result = SchemaValidator.validate_schema(schema)
        assert result["additionalProperties"] == {"type": "string"}

    def test_items_invalid_type_rejected(self) -> None:
        """Test that items must be object or boolean."""
        schema = {
            "type": "array",
            "items": "string",  # Should be {"type": "string"}
        }

        with pytest.raises(
            ValueError, match="'items' at schema must be object or boolean"
        ):
            SchemaValidator.validate_schema(schema)

    def test_items_as_boolean_accepted(self) -> None:
        """Test that items can be a boolean."""
        schema = {
            "type": "array",
            "items": True,
        }

        result = SchemaValidator.validate_schema(schema)
        assert result["items"] is True

    def test_enum_not_array_rejected(self) -> None:
        """Test that enum must be an array."""
        schema = {
            "type": "string",
            "enum": "active",  # Should be ["active"]
        }

        with pytest.raises(ValueError, match="'enum' at schema must be an array"):
            SchemaValidator.validate_schema(schema)

    def test_minimum_not_number_rejected(self) -> None:
        """Test that minimum must be a number."""
        schema = {
            "type": "integer",
            "minimum": "0",  # String instead of number
        }

        with pytest.raises(ValueError, match="'minimum' at schema must be a number"):
            SchemaValidator.validate_schema(schema)

    def test_maximum_not_number_rejected(self) -> None:
        """Test that maximum must be a number."""
        schema = {
            "type": "integer",
            "maximum": "100",  # String instead of number
        }

        with pytest.raises(ValueError, match="'maximum' at schema must be a number"):
            SchemaValidator.validate_schema(schema)


class TestAllowedKeywordsValidation:
    """Tests for _check_allowed_keywords edge cases."""

    def test_invalid_schema_type_rejected(self) -> None:
        """Test that invalid schema types are rejected."""

        # Create a custom object that's not a valid JSON type
        class CustomObject:
            pass

        # We need to test _check_allowed_keywords directly since validate_schema
        # only accepts dict or str
        with pytest.raises(ValueError, match="must be object or boolean"):
            SchemaValidator._check_allowed_keywords(CustomObject(), "test")

    def test_boolean_schema_accepted(self) -> None:
        """Test that boolean schemas are accepted (valid in JSON Schema)."""
        # Boolean schemas are valid - True means "accept anything",
        # False means "reject all". Tests non-dict path in _check_allowed_keywords
        SchemaValidator._check_allowed_keywords(True, "test")
        SchemaValidator._check_allowed_keywords(False, "test")

    def test_none_schema_accepted(self) -> None:
        """Test that None is accepted as a valid schema value."""
        SchemaValidator._check_allowed_keywords(None, "test")

    def test_generic_exception_handling(self) -> None:
        """Test that generic exceptions during validation are wrapped."""

        # Create a schema that will cause an unexpected error
        # by having a property that raises an exception when accessed
        class BadDict(dict):  # type: ignore[type-arg]
            def items(self) -> Any:
                raise RuntimeError("Unexpected error")

        schema = {"type": "object", "properties": BadDict()}

        with pytest.raises(ValueError, match="Invalid schema schema"):
            SchemaValidator.validate_schema(schema)
