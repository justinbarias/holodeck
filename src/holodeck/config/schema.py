"""JSON Schema validation for HoloDeck response formats.

This module provides validation for response_format schemas in agent configuration.
It uses custom validation that aligns with OpenAI's structured output requirements,
supporting only Basic JSON Schema keywords for LLM compatibility:
- type
- properties
- required
- additionalProperties
- items
- enum
- description
- minimum/maximum
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supported Basic JSON Schema keywords only
ALLOWED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",  # For array type
    "enum",  # For enum type
    "default",  # Default value
    "description",  # Field descriptions
    "minimum",  # For numeric types
    "maximum",  # For numeric types
}


class SchemaValidator:
    """Validates JSON Schema definitions for response formats."""

    @staticmethod
    def validate_schema(
        schema: dict[str, Any] | str, schema_name: str = "schema"
    ) -> dict[str, Any]:
        """Validate a JSON schema against Basic JSON Schema specification.

        Args:
            schema: Schema as dict (inline) or JSON string
            schema_name: Name for error messages (e.g., "response_format")

        Returns:
            Validated schema as dictionary

        Raises:
            ValueError: If schema is invalid or uses unsupported keywords
        """
        # Convert string to dict if needed
        if isinstance(schema, str):
            try:
                parsed = json.loads(schema)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Invalid JSON in {schema_name}: must be object")
                schema_dict = parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {schema_name}: {str(e)}") from e
        else:
            schema_dict = schema

        # Validate schema structure using our custom validation
        # We don't use jsonschema's check_schema() because it validates against
        # the full JSON Schema metaschema, which is stricter than what OpenAI's
        # structured output requires (e.g., additionalProperties: false is valid
        # for OpenAI but fails Draft 4/7/2020-12 metaschema validation)
        try:
            SchemaValidator._check_allowed_keywords(schema_dict, schema_name)
            SchemaValidator._validate_schema_structure(schema_dict, schema_name)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid {schema_name} schema: {str(e)}") from e

        return schema_dict

    @staticmethod
    def load_schema_from_file(
        file_path: str, base_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """Load and validate a JSON schema from file.

        Args:
            file_path: Path to schema file (relative to base_dir or absolute)
            base_dir: Base directory for relative paths (defaults to cwd)

        Returns:
            Validated schema as dictionary

        Raises:
            FileNotFoundError: If schema file doesn't exist
            ValueError: If schema is invalid
        """
        # Resolve file path
        if base_dir is None:
            # Try to get from context variable
            from holodeck.config.context import agent_base_dir

            base_dir = agent_base_dir.get()

        base_dir = Path.cwd() if base_dir is None else Path(base_dir)

        path = Path(file_path)
        if not path.is_absolute():
            path = base_dir / file_path

        # Check file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {path}\n" f"Expected file at: {path.resolve()}"
            )

        # Read and parse JSON
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Schema file {path} must be JSON object")
                schema_dict = loaded
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file {path}: {str(e)}") from e
        except OSError as e:
            raise FileNotFoundError(
                f"Failed to read schema file {path}: {str(e)}"
            ) from e

        # Validate schema
        SchemaValidator.validate_schema(schema_dict, f"schema file {path}")

        return schema_dict

    @staticmethod
    def _check_allowed_keywords(schema: Any, path: str = "schema") -> None:
        """Recursively check that schema only uses allowed keywords.

        Args:
            schema: Schema object to validate
            path: Current path in schema (for error messages)

        Raises:
            ValueError: If schema uses unsupported keywords
        """
        if not isinstance(schema, dict):
            # Allow non-dict schemas if they're valid JSON values
            if not isinstance(schema, bool | str | int | float | list | type(None)):
                raise ValueError(f"Invalid schema at {path}: must be object or boolean")
            return

        # Check for unsupported keywords
        for key in schema:
            if key not in ALLOWED_KEYWORDS:
                allowed = ", ".join(sorted(ALLOWED_KEYWORDS))
                raise ValueError(
                    f"Unknown JSON Schema keyword: {key}\n"
                    f"Only these keywords are supported: {allowed}"
                )

        # Recursively validate nested schemas
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                SchemaValidator._check_allowed_keywords(
                    prop_schema, f"{path}.properties.{prop_name}"
                )

        if "items" in schema:
            SchemaValidator._check_allowed_keywords(schema["items"], f"{path}.items")

    @staticmethod
    def _validate_schema_structure(
        schema: dict[str, Any], path: str = "schema"
    ) -> None:
        """Validate schema structure for OpenAI compatibility.

        Args:
            schema: Schema object to validate
            path: Current path in schema (for error messages)

        Raises:
            ValueError: If schema structure is invalid
        """
        # Validate 'type' field if present
        if "type" in schema:
            valid_types = {
                "string",
                "number",
                "integer",
                "boolean",
                "array",
                "object",
                "null",
            }
            if schema["type"] not in valid_types:
                raise ValueError(
                    f"Invalid type '{schema['type']}' at {path}. "
                    f"Must be one of: {', '.join(sorted(valid_types))}"
                )

        # Validate 'properties' is a dict if present
        if "properties" in schema:
            if not isinstance(schema["properties"], dict):
                raise ValueError(f"'properties' at {path} must be an object")
            # Recursively validate nested property schemas
            for prop_name, prop_schema in schema["properties"].items():
                if isinstance(prop_schema, dict):
                    SchemaValidator._validate_schema_structure(
                        prop_schema, f"{path}.properties.{prop_name}"
                    )

        # Validate 'required' is a list of strings if present
        if "required" in schema:
            if not isinstance(schema["required"], list):
                raise ValueError(f"'required' at {path} must be an array")
            for item in schema["required"]:
                if not isinstance(item, str):
                    raise ValueError(f"'required' items at {path} must be strings")

        # Validate 'additionalProperties' is boolean or object
        if "additionalProperties" in schema:
            ap = schema["additionalProperties"]
            if not isinstance(ap, bool | dict):
                raise ValueError(
                    f"'additionalProperties' at {path} must be boolean or object"
                )

        # Validate 'items' for array types
        if "items" in schema:
            if isinstance(schema["items"], dict):
                SchemaValidator._validate_schema_structure(
                    schema["items"], f"{path}.items"
                )
            elif not isinstance(schema["items"], bool):
                raise ValueError(f"'items' at {path} must be object or boolean")

        # Validate 'enum' is a list if present
        if "enum" in schema and not isinstance(schema["enum"], list):
            raise ValueError(f"'enum' at {path} must be an array")

        # Validate numeric constraints
        for constraint in ("minimum", "maximum"):
            if constraint in schema and not isinstance(schema[constraint], int | float):
                raise ValueError(f"'{constraint}' at {path} must be a number")
