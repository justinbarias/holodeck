"""Wizard configuration models for interactive init wizard.

This module contains Pydantic models and predefined choice lists
for the interactive initialization wizard.

Note: LLM providers are derived from holodeck.models.llm.ProviderEnum.
Vector store choices are a curated subset of holodeck.models.tool.DatabaseConfig
providers optimized for the wizard experience.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from holodeck.lib.validation import AGENT_NAME_PATTERN  # noqa: F401 - re-exported
from holodeck.lib.validation import validate_agent_name as _validate_agent_name
from holodeck.models.llm import ProviderEnum


class WizardStep(str, Enum):
    """Steps in the interactive wizard flow.

    Each step represents a prompt in the wizard sequence.
    """

    AGENT_NAME = "agent_name"
    TEMPLATE = "template"
    LLM_PROVIDER = "llm_provider"
    VECTOR_STORE = "vector_store"
    EVALS = "evals"
    MCP_SERVERS = "mcp_servers"
    COMPLETE = "complete"


class WizardState(BaseModel):
    """Runtime state tracking for the wizard flow.

    This model tracks the user's progress through the wizard,
    storing intermediate selections until the wizard completes.

    Attributes:
        current_step: The current step in the wizard flow
        agent_name: Selected agent name (None until provided)
        llm_provider: Selected LLM provider (None until selected)
        vector_store: Selected vector store (None until selected)
        evals: List of selected evaluation metrics
        mcp_servers: List of selected MCP servers
        is_cancelled: Whether the user cancelled the wizard
    """

    model_config = ConfigDict(extra="forbid")

    current_step: WizardStep = Field(
        default=WizardStep.AGENT_NAME,
        description="Current step in the wizard flow",
    )
    agent_name: str | None = Field(
        default=None,
        description="Selected agent name",
    )
    llm_provider: str | None = Field(
        default=None,
        description="Selected LLM provider",
    )
    vector_store: str | None = Field(
        default=None,
        description="Selected vector store",
    )
    evals: list[str] = Field(
        default_factory=list,
        description="List of selected evaluation metrics",
    )
    mcp_servers: list[str] = Field(
        default_factory=list,
        description="List of selected MCP servers",
    )
    is_cancelled: bool = Field(
        default=False,
        description="Whether the wizard was cancelled",
    )


# Valid choices for validation
# LLM providers derived from ProviderEnum (see holodeck.models.llm)
VALID_LLM_PROVIDERS: frozenset[str] = frozenset(p.value for p in ProviderEnum)

# Vector store choices - curated subset of DatabaseConfig.provider
# (see holodeck.models.tool). Qdrant used instead of redis (not in DatabaseConfig)
VALID_VECTOR_STORES = frozenset(["chromadb", "qdrant", "in-memory"])
VALID_EVALS = frozenset(
    [
        "rag-faithfulness",
        "rag-answer_relevancy",
        "rag-context_precision",
        "rag-context_recall",
    ]
)
VALID_MCP_SERVERS = frozenset(
    [
        "brave-search",
        "memory",
        "sequentialthinking",
        "filesystem",
        "github",
        "postgres",
    ]
)


class TemplateChoice(BaseModel):
    """Template option for the wizard.

    This model represents a single project template choice displayed
    in the wizard selection prompt.

    Attributes:
        value: Template identifier (e.g., 'conversational', 'research')
        display_name: Human-readable name shown in the prompt
        description: Help text explaining the template purpose
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(..., description="Template identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Help text for the template")


def get_template_choices() -> list[TemplateChoice]:
    """Get available template choices from manifest files.

    Dynamically loads template metadata from manifest.yaml files
    in the templates directory.

    Returns:
        List of TemplateChoice objects for available templates.
    """
    from holodeck.lib.template_engine import TemplateRenderer

    templates = TemplateRenderer.get_available_templates()
    return [TemplateChoice(**t) for t in templates]


class ProviderConfig(BaseModel):
    """Provider-specific configuration collected from wizard prompts.

    This model holds provider-specific settings like endpoint URLs
    that are collected via follow-up prompts after selecting certain
    LLM providers (e.g., Azure OpenAI).

    Attributes:
        endpoint: The API endpoint URL (for Azure OpenAI)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str | None = Field(
        default=None,
        description="API endpoint URL (e.g., https://resource.openai.azure.com/)",
    )


class WizardResult(BaseModel):
    """Final validated result from the wizard.

    This immutable model contains all validated selections from the wizard,
    ready to be used for project initialization.

    Attributes:
        agent_name: Validated agent name (alphanumeric, hyphens, underscores)
        template: Selected project template
        llm_provider: Selected LLM provider
        provider_config: Provider-specific configuration (endpoint, deployment name)
        vector_store: Selected vector store
        evals: List of selected evaluation metrics (can be empty)
        mcp_servers: List of selected MCP servers (can be empty)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_name: str = Field(
        ...,
        description="Agent name (alphanumeric, hyphens, underscores)",
    )
    template: str = Field(
        default="conversational",
        description="Selected project template",
    )
    llm_provider: str = Field(
        ...,
        description="Selected LLM provider",
    )
    provider_config: ProviderConfig | None = Field(
        default=None,
        description="Provider-specific configuration (endpoint, deployment name)",
    )
    vector_store: str = Field(
        ...,
        description="Selected vector store",
    )
    evals: list[str] = Field(
        default_factory=list,
        description="List of selected evaluation metrics",
    )
    mcp_servers: list[str] = Field(
        default_factory=list,
        description="List of selected MCP servers",
    )

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

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate template is available.

        Args:
            v: The template name to validate

        Returns:
            The validated template name

        Raises:
            ValueError: If template is not recognized
        """
        from holodeck.lib.template_engine import TemplateRenderer

        templates = TemplateRenderer.get_available_templates()
        available = [t["value"] for t in templates]
        if v not in available:
            valid = ", ".join(sorted(available))
            raise ValueError(f"Invalid template: {v}. Valid options: {valid}")
        return v

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


class LLMProviderChoice(BaseModel):
    """LLM provider option for the wizard.

    This model represents a single LLM provider choice displayed
    in the wizard selection prompt.

    Attributes:
        value: Provider identifier (e.g., 'ollama', 'openai')
        display_name: Human-readable name shown in the prompt
        description: Help text explaining the provider
        is_default: Whether this is the default selection
        default_model: Default model name for this provider
        requires_api_key: Whether an API key is required
        api_key_env_var: Environment variable name for the API key
        requires_endpoint: Whether a custom endpoint is needed (Azure)
        endpoint_env_var: Environment variable name for endpoint (Azure)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(..., description="Provider identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Help text for the provider")
    is_default: bool = Field(default=False, description="Whether this is the default")
    default_model: str = Field(..., description="Default model name")
    requires_api_key: bool = Field(
        default=False, description="Whether API key is required"
    )
    api_key_env_var: str | None = Field(
        default=None, description="Environment variable for API key"
    )
    requires_endpoint: bool = Field(
        default=False, description="Whether custom endpoint is needed"
    )
    endpoint_env_var: str | None = Field(
        default=None, description="Environment variable for endpoint URL"
    )


# Predefined LLM provider choices
LLM_PROVIDER_CHOICES: list[LLMProviderChoice] = [
    LLMProviderChoice(
        value="ollama",
        display_name="Ollama (Local)",
        description="Run models locally with Ollama - no API key required",
        is_default=True,
        default_model="gpt-oss:20b",
        requires_api_key=False,
        api_key_env_var=None,
        requires_endpoint=False,
    ),
    LLMProviderChoice(
        value="openai",
        display_name="OpenAI",
        description="Use OpenAI models like GPT-4o",
        is_default=False,
        default_model="gpt-4o",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
        requires_endpoint=False,
    ),
    LLMProviderChoice(
        value="azure_openai",
        display_name="Azure OpenAI",
        description="Use Azure-hosted OpenAI models",
        is_default=False,
        default_model="gpt-4o",
        requires_api_key=True,
        api_key_env_var="AZURE_OPENAI_API_KEY",
        requires_endpoint=True,
        endpoint_env_var="AZURE_OPENAI_ENDPOINT",
    ),
    LLMProviderChoice(
        value="anthropic",
        display_name="Anthropic",
        description="Use Anthropic models like Claude",
        is_default=False,
        default_model="claude-3-5-sonnet-20241022",
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        requires_endpoint=False,
    ),
]


class VectorStoreChoice(BaseModel):
    """Vector store option for the wizard.

    This model represents a single vector store choice displayed
    in the wizard selection prompt.

    Attributes:
        value: Store identifier (e.g., 'chromadb', 'redis')
        display_name: Human-readable name shown in the prompt
        description: Help text explaining the store
        is_default: Whether this is the default selection
        default_endpoint: Default connection endpoint
        persistence: Storage type (local, remote, none)
        connection_required: Whether a connection is needed at runtime
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(..., description="Store identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Help text for the store")
    is_default: bool = Field(default=False, description="Whether this is the default")
    default_endpoint: str | None = Field(
        default=None, description="Default connection endpoint"
    )
    persistence: str = Field(..., description="Storage type: local, remote, or none")
    connection_required: bool = Field(
        default=True, description="Whether connection is needed"
    )


# Predefined vector store choices
# Aligned with DatabaseConfig.provider options in holodeck.models.tool
VECTOR_STORE_CHOICES: list[VectorStoreChoice] = [
    VectorStoreChoice(
        value="chromadb",
        display_name="ChromaDB",
        description="Local vector database with HTTP API - recommended for development",
        is_default=True,
        default_endpoint="http://localhost:8000",
        persistence="local",
        connection_required=True,
    ),
    VectorStoreChoice(
        value="qdrant",
        display_name="Qdrant",
        description="High-performance vector database with REST/gRPC API",
        is_default=False,
        default_endpoint="http://localhost:6333",
        persistence="remote",
        connection_required=True,
    ),
    VectorStoreChoice(
        value="in-memory",
        display_name="In-Memory",
        description="Ephemeral storage - data lost on restart (testing only)",
        is_default=False,
        default_endpoint=None,
        persistence="none",
        connection_required=False,
    ),
]


class EvalChoice(BaseModel):
    """Evaluation metric option for the wizard.

    This model represents a single evaluation metric choice displayed
    in the wizard multi-select prompt.

    Attributes:
        value: Metric identifier (e.g., 'rag-faithfulness')
        display_name: Human-readable name shown in the prompt
        description: Help text explaining what the metric measures
        is_default: Whether this metric is pre-selected by default
        metric_type: Type of metric ('ai' for LLM-based, 'nlp' for traditional)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(..., description="Metric identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Help text for the metric")
    is_default: bool = Field(
        default=False, description="Whether pre-selected by default"
    )
    metric_type: str = Field(..., description="Type: 'ai' or 'nlp'")


# Predefined evaluation metric choices
EVAL_CHOICES: list[EvalChoice] = [
    EvalChoice(
        value="rag-faithfulness",
        display_name="Faithfulness",
        description="Measures if the response is grounded in the retrieved context",
        is_default=True,
        metric_type="ai",
    ),
    EvalChoice(
        value="rag-answer_relevancy",
        display_name="Answer Relevancy",
        description="Measures if the response is relevant to the user's question",
        is_default=True,
        metric_type="ai",
    ),
    EvalChoice(
        value="rag-context_precision",
        display_name="Context Precision",
        description="Measures the precision of retrieved context chunks",
        is_default=False,
        metric_type="ai",
    ),
    EvalChoice(
        value="rag-context_recall",
        display_name="Context Recall",
        description="Measures the recall of relevant context chunks",
        is_default=False,
        metric_type="ai",
    ),
]


class MCPServerChoice(BaseModel):
    """MCP server option for the wizard.

    This model represents a single MCP (Model Context Protocol) server
    choice displayed in the wizard multi-select prompt.

    Attributes:
        value: Server identifier (e.g., 'brave-search')
        display_name: Human-readable name shown in the prompt
        description: Help text explaining the server's functionality
        is_default: Whether this server is pre-selected by default
        package_identifier: NPM package name for the MCP server
        command: Execution command (defaults to 'npx')
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(..., description="Server identifier")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Help text for the server")
    is_default: bool = Field(
        default=False, description="Whether pre-selected by default"
    )
    package_identifier: str = Field(..., description="NPM package name")
    command: str = Field(default="npx", description="Execution command")


# Predefined MCP server choices
MCP_SERVER_CHOICES: list[MCPServerChoice] = [
    # Default servers (pre-selected)
    MCPServerChoice(
        value="brave-search",
        display_name="Brave Search",
        description="Web search using Brave Search API",
        is_default=True,
        package_identifier="@brave/brave-search-mcp-server",
        command="npx",
    ),
    MCPServerChoice(
        value="memory",
        display_name="Memory",
        description="Persistent memory using local knowledge graph",
        is_default=True,
        package_identifier="@modelcontextprotocol/server-memory",
        command="npx",
    ),
    MCPServerChoice(
        value="sequentialthinking",
        display_name="Sequential Thinking",
        description="Dynamic problem-solving through structured thinking",
        is_default=True,
        package_identifier="@modelcontextprotocol/server-sequential-thinking",
        command="npx",
    ),
    # Optional servers
    MCPServerChoice(
        value="filesystem",
        display_name="Filesystem",
        description="Read and write files on the local filesystem",
        is_default=False,
        package_identifier="@modelcontextprotocol/server-filesystem",
        command="npx",
    ),
    MCPServerChoice(
        value="github",
        display_name="GitHub",
        description="Interact with GitHub repositories and issues",
        is_default=False,
        package_identifier="@modelcontextprotocol/server-github",
        command="npx",
    ),
    MCPServerChoice(
        value="postgres",
        display_name="PostgreSQL",
        description="Query PostgreSQL databases (read-only)",
        is_default=False,
        package_identifier="@zeddotdev/postgres-context-server",
        command="npx",
    ),
]


# Helper functions for getting defaults
def get_default_llm_provider() -> str:
    """Get the default LLM provider value.

    Returns:
        The value of the default LLM provider choice.
    """
    for choice in LLM_PROVIDER_CHOICES:
        if choice.is_default:
            return choice.value
    return "ollama"  # Fallback


def get_default_vector_store() -> str:
    """Get the default vector store value.

    Returns:
        The value of the default vector store choice.
    """
    for choice in VECTOR_STORE_CHOICES:
        if choice.is_default:
            return choice.value
    return "chromadb"  # Fallback


def get_default_evals() -> list[str]:
    """Get the list of default evaluation metric values.

    Returns:
        List of values for default evaluation metrics.
    """
    return [choice.value for choice in EVAL_CHOICES if choice.is_default]


def get_default_mcp_servers() -> list[str]:
    """Get the list of default MCP server values.

    Returns:
        List of values for default MCP servers.
    """
    return [choice.value for choice in MCP_SERVER_CHOICES if choice.is_default]
