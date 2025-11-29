# Data Model: Interactive Init Wizard

**Feature Branch**: `011-interactive-init-wizard`
**Date**: 2025-11-29

## Entity Overview

```
┌─────────────────────┐      ┌─────────────────────┐
│   WizardState       │──────│   WizardResult      │
│  (runtime tracking) │      │ (final selections)  │
└─────────────────────┘      └─────────────────────┘
         │                            │
         │                            ▼
         │                   ┌─────────────────────┐
         │                   │  ProjectInitInput   │
         │                   │ (existing, extended)│
         │                   └─────────────────────┘
         │
         ▼
┌─────────────────────┐      ┌─────────────────────┐
│ LLMProviderChoice   │      │ VectorStoreChoice   │
│  (wizard option)    │      │  (wizard option)    │
└─────────────────────┘      └─────────────────────┘

┌─────────────────────┐      ┌─────────────────────┐
│ MCPServerInfo       │◀─────│ MCPRegistryResponse │
│  (registry data)    │      │  (API response)     │
└─────────────────────┘      └─────────────────────┘
```

## Entities

### 1. WizardState

**Purpose**: Tracks user progress through the wizard and accumulates selections.

**Location**: `src/holodeck/models/wizard_config.py`

```python
from enum import Enum
from pydantic import BaseModel, Field

class WizardStep(str, Enum):
    """Current step in the wizard flow."""
    LLM_PROVIDER = "llm_provider"
    VECTOR_STORE = "vector_store"
    MCP_SERVERS = "mcp_servers"
    COMPLETE = "complete"

class WizardState(BaseModel):
    """Runtime state tracking for interactive wizard.

    Tracks current step and accumulated selections as user
    progresses through the wizard flow.
    """
    current_step: WizardStep = Field(
        default=WizardStep.LLM_PROVIDER,
        description="Current wizard step"
    )
    llm_provider: str | None = Field(
        default=None,
        description="Selected LLM provider"
    )
    vector_store: str | None = Field(
        default=None,
        description="Selected vector store"
    )
    mcp_servers: list[str] = Field(
        default_factory=list,
        description="Selected MCP server identifiers"
    )
    is_cancelled: bool = Field(
        default=False,
        description="Whether wizard was cancelled by user"
    )
```

**State Transitions**:
```
LLM_PROVIDER → (selection) → VECTOR_STORE → (selection) → MCP_SERVERS → (selection) → COMPLETE
     ↓                            ↓                            ↓
  (cancel)                     (cancel)                     (cancel)
     ↓                            ↓                            ↓
[is_cancelled=True, abort without file creation]
```

---

### 2. WizardResult

**Purpose**: Final validated selections from wizard, ready for project initialization.

**Location**: `src/holodeck/models/wizard_config.py`

```python
from pydantic import BaseModel, Field, model_validator

class WizardResult(BaseModel):
    """Final selections from interactive wizard.

    All fields are required after wizard completion.
    Validated before passing to ProjectInitializer.
    """
    llm_provider: str = Field(
        ...,
        description="Selected LLM provider: ollama, openai, azure_openai, anthropic"
    )
    vector_store: str = Field(
        ...,
        description="Selected vector store: chromadb, redis, in-memory"
    )
    mcp_servers: list[str] = Field(
        ...,
        description="Selected MCP server package identifiers"
    )

    @model_validator(mode="after")
    def validate_selections(self) -> "WizardResult":
        """Validate all selections are from allowed values."""
        valid_providers = {"ollama", "openai", "azure_openai", "anthropic"}
        valid_stores = {"chromadb", "redis", "in-memory"}

        if self.llm_provider not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")
        if self.vector_store not in valid_stores:
            raise ValueError(f"Invalid vector store: {self.vector_store}")

        return self
```

**Validation Rules**:
- `llm_provider`: Must be one of: ollama, openai, azure_openai, anthropic
- `vector_store`: Must be one of: chromadb, redis, in-memory
- `mcp_servers`: List can be empty (user deselected all)

---

### 3. LLMProviderChoice

**Purpose**: Defines a selectable LLM provider option for the wizard.

**Location**: `src/holodeck/models/wizard_config.py`

```python
from pydantic import BaseModel, Field

class LLMProviderChoice(BaseModel):
    """LLM provider option for wizard selection.

    Provides display information and configuration hints
    for each supported LLM provider.
    """
    value: str = Field(..., description="Provider identifier for config")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Brief capability description")
    is_default: bool = Field(default=False, description="Whether this is the default selection")
    requires_api_key: bool = Field(default=True, description="Whether API key is needed")
    api_key_env_var: str | None = Field(default=None, description="Environment variable for API key")
    requires_endpoint: bool = Field(default=False, description="Whether endpoint URL is needed")

# Predefined choices
LLM_PROVIDER_CHOICES = [
    LLMProviderChoice(
        value="ollama",
        display_name="Ollama (local)",
        description="Local LLM inference, no API key required",
        is_default=True,
        requires_api_key=False,
        requires_endpoint=False,  # Defaults to localhost
    ),
    LLMProviderChoice(
        value="openai",
        display_name="OpenAI",
        description="GPT-4, GPT-3.5-turbo via OpenAI API",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
    ),
    LLMProviderChoice(
        value="azure_openai",
        display_name="Azure OpenAI",
        description="OpenAI models via Azure deployment",
        requires_api_key=True,
        api_key_env_var="AZURE_OPENAI_API_KEY",
        requires_endpoint=True,
    ),
    LLMProviderChoice(
        value="anthropic",
        display_name="Anthropic Claude",
        description="Claude 3.5, Claude 3 via Anthropic API",
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
    ),
]
```

---

### 4. VectorStoreChoice

**Purpose**: Defines a selectable vector store option for the wizard.

**Location**: `src/holodeck/models/wizard_config.py`

```python
from pydantic import BaseModel, Field

class VectorStoreChoice(BaseModel):
    """Vector store option for wizard selection.

    Provides display information and configuration hints
    for each supported vector store.
    """
    value: str = Field(..., description="Store identifier for config")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Brief capability description")
    is_default: bool = Field(default=False, description="Whether this is the default selection")
    persistence: str = Field(..., description="Data persistence model")
    connection_required: bool = Field(default=False, description="Whether connection string needed")

# Predefined choices
VECTOR_STORE_CHOICES = [
    VectorStoreChoice(
        value="chromadb",
        display_name="ChromaDB (default)",
        description="Embedded vector database with local persistence",
        is_default=True,
        persistence="local file",
        connection_required=False,
    ),
    VectorStoreChoice(
        value="redis",
        display_name="Redis",
        description="Production-grade vector store with Redis Stack",
        persistence="remote server",
        connection_required=True,
    ),
    VectorStoreChoice(
        value="in-memory",
        display_name="In-Memory",
        description="Ephemeral storage for development/testing",
        persistence="none (lost on restart)",
        connection_required=False,
    ),
]
```

**Warning for In-Memory**: Per spec acceptance scenario 3.3, warn user about data loss on restart.

---

### 5. MCPServerInfo

**Purpose**: Represents an MCP server fetched from the registry.

**Location**: `src/holodeck/lib/mcp_registry.py`

```python
from pydantic import BaseModel, Field

class MCPPackage(BaseModel):
    """Package installation details for an MCP server."""
    registry_type: str = Field(..., alias="registryType", description="Package registry: npm, pypi")
    identifier: str = Field(..., description="Package name/identifier")
    transport_type: str = Field(default="stdio", description="Transport: stdio, sse, http")

class MCPServerInfo(BaseModel):
    """MCP server information from registry.

    Parsed from official MCP registry API response.
    """
    name: str = Field(..., description="Fully qualified server name")
    description: str = Field(default="", description="Server description")
    version: str = Field(default="", description="Latest version")
    packages: list[MCPPackage] = Field(default_factory=list, description="Installation packages")
    is_official: bool = Field(default=False, description="Whether maintained by MCP team")
    is_default: bool = Field(default=False, description="Whether pre-selected in wizard")

    @property
    def short_name(self) -> str:
        """Extract short name from fully qualified name."""
        # "io.github.user/server-name" → "server-name"
        return self.name.split("/")[-1] if "/" in self.name else self.name

    @property
    def primary_package(self) -> MCPPackage | None:
        """Get primary package for installation."""
        return self.packages[0] if self.packages else None

# Default servers (pre-selected)
DEFAULT_MCP_SERVERS = [
    "@modelcontextprotocol/server-filesystem",
    "@modelcontextprotocol/server-memory",
    "@modelcontextprotocol/server-sequential-thinking",
]
```

---

### 6. MCPRegistryResponse

**Purpose**: Represents the full API response from MCP registry.

**Location**: `src/holodeck/lib/mcp_registry.py`

```python
from pydantic import BaseModel, Field

class MCPRegistryMetadata(BaseModel):
    """Pagination metadata from registry response."""
    count: int = Field(default=0, description="Number of results")
    next_cursor: str | None = Field(default=None, alias="nextCursor", description="Cursor for next page")

class MCPRegistryResponse(BaseModel):
    """Full response from MCP registry API.

    Handles pagination and provides iteration support.
    """
    servers: list[MCPServerInfo] = Field(default_factory=list, description="List of servers")
    metadata: MCPRegistryMetadata = Field(default_factory=MCPRegistryMetadata)

    @property
    def has_more(self) -> bool:
        """Check if more pages available."""
        return self.metadata.next_cursor is not None
```

---

### 7. Extended ProjectInitInput

**Purpose**: Extend existing model to include wizard selections.

**Location**: `src/holodeck/models/project_config.py` (existing file, add fields)

```python
# Add to existing ProjectInitInput model

class ProjectInitInput(BaseModel):
    """Extended with wizard configuration selections."""
    # ... existing fields ...

    # New wizard-related fields
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider from wizard selection"
    )
    vector_store: str = Field(
        default="chromadb",
        description="Vector store from wizard selection"
    )
    mcp_servers: list[str] = Field(
        default_factory=lambda: [
            "@modelcontextprotocol/server-filesystem",
            "@modelcontextprotocol/server-memory",
            "@modelcontextprotocol/server-sequential-thinking",
        ],
        description="MCP servers from wizard selection"
    )
```

---

## Relationships

| From | To | Relationship | Description |
|------|-----|-------------|-------------|
| WizardState | WizardResult | Transforms to | Final state becomes result |
| WizardResult | ProjectInitInput | Merged into | Wizard selections added to init input |
| MCPRegistryResponse | MCPServerInfo | Contains many | API response contains server list |
| MCPServerInfo | MCPPackage | Contains many | Each server has installation packages |

## Validation Summary

| Entity | Validation Rule | Error Behavior |
|--------|----------------|----------------|
| WizardResult | Provider in allowed set | Raise ValueError |
| WizardResult | Vector store in allowed set | Raise ValueError |
| MCPServerInfo | Has at least one package | Log warning, skip display |
| ProjectInitInput | Template exists | Raise ValidationError |

## State Diagram

```
[Start]
    │
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = LLM_PROVIDER        │
│  llm_provider = None        │
└─────────────────────────────┘
    │ User selects provider
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = VECTOR_STORE        │
│  llm_provider = "openai"    │
└─────────────────────────────┘
    │ User selects store
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = MCP_SERVERS         │
│  vector_store = "chromadb"  │
└─────────────────────────────┘
    │ User selects servers
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = COMPLETE            │
│  mcp_servers = [...]        │
└─────────────────────────────┘
    │ Convert to WizardResult
    ▼
┌─────────────────────────────┐
│  WizardResult               │
│  (all fields required)      │
└─────────────────────────────┘
    │ Merge with ProjectInitInput
    ▼
┌─────────────────────────────┐
│  ProjectInitializer         │
│  Creates project files      │
└─────────────────────────────┘
    │
    ▼
[End: Project Created]
```
