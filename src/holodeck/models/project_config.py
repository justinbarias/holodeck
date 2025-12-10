"""Pydantic models for project initialization configuration.

These models define the data structures used by the init command,
including user input validation and result tracking.
"""

from pydantic import BaseModel, Field, field_validator

from holodeck.lib.validation import validate_agent_name as _validate_agent_name
from holodeck.models.wizard_config import (
    VALID_EVALS,
    VALID_LLM_PROVIDERS,
    VALID_MCP_SERVERS,
    VALID_VECTOR_STORES,
    ProviderConfig,
    get_default_evals,
    get_default_mcp_servers,
)


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
        agent_name: Name of the agent to create (from wizard)
        llm_provider: Selected LLM provider (from wizard)
        provider_config: Provider-specific configuration (endpoint, deployment name)
        vector_store: Selected vector store (from wizard)
        evals: List of selected evaluation metrics (from wizard)
        mcp_servers: List of selected MCP servers (from wizard)
    """

    project_name: str = Field(..., description="Name of the project to create")
    template: str = Field(..., description="Template choice")
    description: str | None = Field(
        None, description="Optional description of the agent"
    )
    author: str | None = Field(None, description="Optional creator name")
    output_dir: str = Field(".", description="Target directory for project creation")
    overwrite: bool = Field(False, description="Whether to overwrite existing project")

    # Wizard configuration fields
    agent_name: str = Field(
        default="my-agent",
        description="Name of the agent (alphanumeric, hyphens, underscores)",
    )
    llm_provider: str = Field(
        default="ollama",
        description="Selected LLM provider",
    )
    provider_config: ProviderConfig | None = Field(
        default=None,
        description="Provider-specific configuration (endpoint, deployment name)",
    )
    vector_store: str = Field(
        default="chromadb",
        description="Selected vector store",
    )
    evals: list[str] = Field(
        default_factory=get_default_evals,
        description="List of selected evaluation metrics",
    )
    mcp_servers: list[str] = Field(
        default_factory=get_default_mcp_servers,
        description="List of selected MCP servers",
    )

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
        from holodeck.lib.template_engine import TemplateRenderer

        available_templates = set(TemplateRenderer.list_available_templates())
        if v not in available_templates:
            templates_list = ", ".join(sorted(available_templates))
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

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Validate agent name format using shared validator.

        Args:
            v: The agent name to validate

        Returns:
            The validated agent name

        Raises:
            ValueError: If agent name is invalid
        """
        return _validate_agent_name(v)

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate LLM provider choice.

        Args:
            v: The LLM provider to validate

        Returns:
            The validated LLM provider

        Raises:
            ValueError: If LLM provider is not recognized
        """
        if v not in VALID_LLM_PROVIDERS:
            valid = ", ".join(sorted(VALID_LLM_PROVIDERS))
            raise ValueError(f"Invalid LLM provider: {v}. Valid options: {valid}")
        return v

    @field_validator("vector_store")
    @classmethod
    def validate_vector_store(cls, v: str) -> str:
        """Validate vector store choice.

        Args:
            v: The vector store to validate

        Returns:
            The validated vector store

        Raises:
            ValueError: If vector store is not recognized
        """
        if v not in VALID_VECTOR_STORES:
            valid = ", ".join(sorted(VALID_VECTOR_STORES))
            raise ValueError(f"Invalid vector store: {v}. Valid options: {valid}")
        return v

    @field_validator("evals")
    @classmethod
    def validate_evals(cls, v: list[str]) -> list[str]:
        """Validate evaluation metrics choices.

        Args:
            v: The list of evaluation metrics to validate

        Returns:
            The validated list of evaluation metrics

        Raises:
            ValueError: If any eval metric is not recognized
        """
        invalid = [e for e in v if e not in VALID_EVALS]
        if invalid:
            valid = ", ".join(sorted(VALID_EVALS))
            raise ValueError(
                f"Invalid evaluation metric(s): {', '.join(invalid)}. "
                f"Valid options: {valid}"
            )
        return v

    @field_validator("mcp_servers")
    @classmethod
    def validate_mcp_servers(cls, v: list[str]) -> list[str]:
        """Validate MCP server choices.

        Args:
            v: The list of MCP servers to validate

        Returns:
            The validated list of MCP servers

        Raises:
            ValueError: If any MCP server is not recognized
        """
        invalid = [s for s in v if s not in VALID_MCP_SERVERS]
        if invalid:
            valid = ", ".join(sorted(VALID_MCP_SERVERS))
            raise ValueError(
                f"Invalid MCP server(s): {', '.join(invalid)}. Valid options: {valid}"
            )
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
