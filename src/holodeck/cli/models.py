"""Pydantic models for CLI input/output and template management.

These models define the data structures used by the init command,
including user input validation, result tracking, and template metadata.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProjectInitInput(BaseModel):
    """User-provided input for project initialization.

    This model validates and stores the parameters passed to the init command,
    ensuring required fields are present and optional fields are properly typed.

    Attributes:
        project_name: Name of the project to create (alphanumeric, hyphens, underscores)
        template: Template choice (conversational, research, customer-support)
        description: Optional description of the agent
        author: Optional creator name
        output_dir: Target directory (currently CWD, but model allows future extension)
        overwrite: Whether to overwrite existing project
    """

    project_name: str = Field(..., description="Name of the project to create")
    template: str = Field(..., description="Template choice")
    description: str | None = Field(
        None, description="Optional description of the agent"
    )
    author: str | None = Field(None, description="Optional creator name")
    output_dir: str = Field(".", description="Target directory for project creation")
    overwrite: bool = Field(False, description="Whether to overwrite existing project")

    @field_validator("project_name")
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        """Validate project name format.

        Args:
            v: The project name to validate

        Returns:
            The validated project name

        Raises:
            ValueError: If project name is invalid
        """
        if not v:
            raise ValueError("Project name cannot be empty")
        if len(v) > 64:
            raise ValueError("Project name must be 64 characters or less")
        if v[0].isdigit():
            raise ValueError("Project name cannot start with a digit")
        if not all(c.isalnum() or c in "-_" for c in v):
            msg = (
                "Project name can only contain alphanumeric characters, "
                "hyphens, and underscores"
            )
            raise ValueError(msg)
        return v

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate template choice.

        Args:
            v: The template name to validate

        Returns:
            The validated template name

        Raises:
            ValueError: If template is not recognized
        """
        valid_templates = {"conversational", "research", "customer-support"}
        if v not in valid_templates:
            templates_list = ", ".join(sorted(valid_templates))
            msg = f"Unknown template: {v}. Valid templates: {templates_list}"
            raise ValueError(msg)
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        """Validate description field.

        Args:
            v: The description to validate

        Returns:
            The validated description

        Raises:
            ValueError: If description is too long
        """
        if v is not None and len(v) > 1000:
            raise ValueError("Description must be 1000 characters or less")
        return v

    @field_validator("author")
    @classmethod
    def validate_author(cls, v: str | None) -> str | None:
        """Validate author field.

        Args:
            v: The author name to validate

        Returns:
            The validated author name

        Raises:
            ValueError: If author name is too long
        """
        if v is not None and len(v) > 256:
            raise ValueError("Author name must be 256 characters or less")
        return v


class ProjectInitResult(BaseModel):
    """Outcome of project initialization.

    This model captures the result of a project initialization attempt,
    including success status, paths, file list, and any errors or warnings.

    Attributes:
        success: Whether initialization completed successfully
        project_name: Name of created project
        project_path: Absolute path to created project directory
        template_used: Which template was applied
        files_created: List of relative paths of created files
        warnings: Non-blocking issues (e.g., permission notes)
        errors: Blocking errors that prevented creation
        duration_seconds: Time taken for initialization
    """

    success: bool = Field(..., description="Whether initialization succeeded")
    project_name: str = Field(..., description="Name of created project")
    project_path: str = Field(..., description="Path to created project")
    template_used: str = Field(..., description="Template that was applied")
    files_created: list[str] = Field(default_factory=list, description="Files created")
    warnings: list[str] = Field(
        default_factory=list, description="Non-blocking warnings"
    )
    errors: list[str] = Field(default_factory=list, description="Blocking errors")
    duration_seconds: float = Field(..., description="Time taken in seconds")


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
