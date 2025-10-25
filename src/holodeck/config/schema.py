"""JSON Schema validation for HoloDeck response formats.

This module provides validation for response_format schemas in agent configuration.
It supports only Basic JSON Schema keywords to maintain LLM compatibility:
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

import jsonschema
from jsonschema import Draft202012Validator

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

        # Check for unsupported keywords
        SchemaValidator._check_allowed_keywords(schema_dict, schema_name)

        # Validate the schema itself is well-formed
        try:
            Draft202012Validator.check_schema(schema_dict)
        except jsonschema.SchemaError as e:
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
            if not isinstance(schema, bool | str | int | float | type(None)):
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
