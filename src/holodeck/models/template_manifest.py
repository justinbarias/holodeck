"""Pydantic models for template management and validation.

These models define the structure and metadata for HoloDeck templates,
including variable schemas, file metadata, and template manifests.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class VariableSchema(BaseModel):
    """Schema for template variables.

    Defines what values a template variable can accept,
    including type constraints, defaults, and allowed values.

    Attributes:
        type: Variable type (string, number, boolean, enum)
        description: Description of what the variable controls
        default: Default value if not provided
        required: Whether variable must be provided
        allowed_values: For enum type, list of allowed choices
    """

    type: str = Field(..., description="Variable type")
    description: str = Field(..., description="Description of the variable")
    default: Any = Field(None, description="Default value")
    required: bool = Field(True, description="Whether variable is required")
    allowed_values: list[Any] | None = Field(
        None, description="Allowed values for enum type"
    )


class FileMetadata(BaseModel):
    """Metadata for template files.

    Defines how each file in a template should be processed
    (e.g., Jinja2 rendering vs. direct copy).

    Attributes:
        path: Relative path in generated project
        template: Whether this file is a Jinja2 template
        required: Whether this file is always included
    """

    path: str = Field(..., description="Relative path in project")
    template: bool = Field(False, description="Whether file uses Jinja2")
    required: bool = Field(True, description="Whether file is required")


class TemplateManifest(BaseModel):
    """Template metadata and validation rules.

    Describes a project template including its variables,
    defaults, and file structure.

    Attributes:
        name: Template identifier (conversational, research, customer-support)
        display_name: Human-readable name for CLI output
        description: One-line description of template purpose
        category: Use case category (conversational-ai, research-analysis, etc.)
        version: Template version (semver format)
        variables: Allowed template variables with constraints
        defaults: Template-specific default values
        files: Files in template and how to process them
    """

    name: str = Field(..., description="Template identifier")
    display_name: str = Field(..., description="Human-readable template name")
    description: str = Field(..., description="Template purpose")
    category: str = Field(..., description="Use case category")
    version: str = Field(..., description="Template version (semver)")
    variables: dict[str, VariableSchema] = Field(
        default_factory=dict, description="Template variables"
    )
    defaults: dict[str, Any] = Field(
        default_factory=dict, description="Template defaults"
    )
    files: dict[str, FileMetadata] = Field(
        default_factory=dict, description="Files in template"
    )

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate version is in semver format.

        Args:
            v: The version string to validate

        Returns:
            The validated version string

        Raises:
            ValueError: If version is not valid semver
        """
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(f"Version must be semver format (MAJOR.MINOR.PATCH): {v}")
        try:
            for part in parts:
                int(part)
        except ValueError as e:
            raise ValueError(f"Version parts must be integers: {v}") from e
        return v
