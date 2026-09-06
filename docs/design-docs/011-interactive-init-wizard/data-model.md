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
│ EvalChoice          │      │ MCPServerChoice     │
│  (wizard option)    │      │  (wizard option)    │
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
    AGENT_NAME = "agent_name"
    LLM_PROVIDER = "llm_provider"
    VECTOR_STORE = "vector_store"
    EVALS = "evals"
    MCP_SERVERS = "mcp_servers"
    COMPLETE = "complete"

class WizardState(BaseModel):
    """Runtime state tracking for interactive wizard.

    Tracks current step and accumulated selections as user
    progresses through the wizard flow.
    """
    current_step: WizardStep = Field(
        default=WizardStep.AGENT_NAME,
        description="Current wizard step"
    )
    agent_name: str | None = Field(
        default=None,
        description="Agent name entered by user"
    )
    llm_provider: str | None = Field(
        default=None,
        description="Selected LLM provider"
    )
    vector_store: str | None = Field(
        default=None,
        description="Selected vector store"
    )
    evals: list[str] = Field(
        default_factory=list,
        description="Selected evaluation metrics"
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
AGENT_NAME → (input) → LLM_PROVIDER → (selection) → VECTOR_STORE → (selection) → EVALS → (selection) → MCP_SERVERS → (selection) → COMPLETE
     ↓                      ↓                            ↓                          ↓                        ↓
  (cancel)               (cancel)                     (cancel)                   (cancel)                 (cancel)
     ↓                      ↓                            ↓                          ↓                        ↓
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
    agent_name: str = Field(
        ...,
        description="Agent name (alphanumeric, hyphens, underscores)"
    )
    llm_provider: str = Field(
        ...,
        description="Selected LLM provider: ollama, openai, azure_openai, anthropic"
    )
    vector_store: str = Field(
        ...,
        description="Selected vector store: chromadb, redis, in-memory"
    )
    evals: list[str] = Field(
        ...,
        description="Selected evaluation metrics"
    )
    mcp_servers: list[str] = Field(
        ...,
        description="Selected MCP server identifiers"
    )

    @model_validator(mode="after")
    def validate_selections(self) -> "WizardResult":
        """Validate all selections are from allowed values."""
        import re

        valid_providers = {"ollama", "openai", "azure_openai", "anthropic"}
        valid_stores = {"chromadb", "redis", "in-memory"}
        valid_evals = {"rag-faithfulness", "rag-answer_relevancy", "rag-context_precision", "rag-context_recall"}
        valid_mcp = {"brave-search", "memory", "sequentialthinking", "filesystem", "github", "postgres"}

        # Validate agent name format
        if not re.match(r"^[a-zA-Z0-9_-]+$", self.agent_name):
            raise ValueError(f"Invalid agent name: {self.agent_name}. Use alphanumeric, hyphens, underscores only.")

        if self.llm_provider not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")
        if self.vector_store not in valid_stores:
            raise ValueError(f"Invalid vector store: {self.vector_store}")
        for eval_metric in self.evals:
            if eval_metric not in valid_evals:
                raise ValueError(f"Invalid eval metric: {eval_metric}")
        for mcp_server in self.mcp_servers:
            if mcp_server not in valid_mcp:
                raise ValueError(f"Invalid MCP server: {mcp_server}")

        return self
```

**Validation Rules**:

- `agent_name`: Alphanumeric, hyphens, underscores only
- `llm_provider`: Must be one of: ollama, openai, azure_openai, anthropic
- `vector_store`: Must be one of: chromadb, redis, in-memory
- `evals`: Must be from allowed eval metrics
- `mcp_servers`: Must be from allowed MCP servers (list can be empty)

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
    default_model: str = Field(..., description="Default model for this provider")
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
        default_model="gpt-oss:20b",
        requires_api_key=False,
        requires_endpoint=False,
    ),
    LLMProviderChoice(
        value="openai",
        display_name="OpenAI",
        description="GPT-4, GPT-3.5-turbo via OpenAI API",
        default_model="gpt-4o",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
    ),
    LLMProviderChoice(
        value="azure_openai",
        display_name="Azure OpenAI",
        description="OpenAI models via Azure deployment",
        default_model="gpt-4o",
        requires_api_key=True,
        api_key_env_var="AZURE_OPENAI_API_KEY",
        requires_endpoint=True,
    ),
    LLMProviderChoice(
        value="anthropic",
        display_name="Anthropic Claude",
        description="Claude 3.5, Claude 3 via Anthropic API",
        default_model="claude-3-5-sonnet-20241022",
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
    default_endpoint: str | None = Field(default=None, description="Default connection endpoint")
    persistence: str = Field(..., description="Data persistence model")
    connection_required: bool = Field(default=False, description="Whether connection string needed")

# Predefined choices
VECTOR_STORE_CHOICES = [
    VectorStoreChoice(
        value="chromadb",
        display_name="ChromaDB (default)",
        description="Embedded vector database with local persistence",
        is_default=True,
        default_endpoint="http://localhost:8000",
        persistence="local file",
        connection_required=False,
    ),
    VectorStoreChoice(
        value="redis",
        display_name="Redis",
        description="Production-grade vector store with Redis Stack",
        default_endpoint="redis://localhost:6379",
        persistence="remote server",
        connection_required=True,
    ),
    VectorStoreChoice(
        value="in-memory",
        display_name="In-Memory",
        description="Ephemeral storage for development/testing",
        default_endpoint=None,
        persistence="none (lost on restart)",
        connection_required=False,
    ),
]
```

**Warning for In-Memory**: Per spec acceptance scenario 3.3, warn user about data loss on restart.

---

### 5. EvalChoice

**Purpose**: Defines a selectable evaluation metric option for the wizard.

**Location**: `src/holodeck/models/wizard_config.py`

```python
from pydantic import BaseModel, Field

class EvalChoice(BaseModel):
    """Evaluation metric option for wizard selection.

    Provides display information for each supported evaluation metric.
    """
    value: str = Field(..., description="Eval metric identifier for config")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="What this metric measures")
    is_default: bool = Field(default=False, description="Whether pre-selected by default")
    metric_type: str = Field(default="ai", description="Type of metric: ai or nlp")

# Predefined choices
EVAL_CHOICES = [
    EvalChoice(
        value="rag-faithfulness",
        display_name="RAG Faithfulness",
        description="Measures if response is grounded in retrieved context",
        is_default=True,
        metric_type="ai",
    ),
    EvalChoice(
        value="rag-answer_relevancy",
        display_name="RAG Answer Relevancy",
        description="Measures if response answers the question",
        is_default=True,
        metric_type="ai",
    ),
    EvalChoice(
        value="rag-context_precision",
        display_name="RAG Context Precision",
        description="Measures precision of retrieved context",
        is_default=False,
        metric_type="ai",
    ),
    EvalChoice(
        value="rag-context_recall",
        display_name="RAG Context Recall",
        description="Measures recall of retrieved context",
        is_default=False,
        metric_type="ai",
    ),
]
```

---

### 6. MCPServerChoice

**Purpose**: Defines a selectable MCP server option for the wizard.

**Location**: `src/holodeck/models/wizard_config.py`

```python
from pydantic import BaseModel, Field

class MCPServerChoice(BaseModel):
    """MCP server option for wizard selection.

    Provides display information and package details for each MCP server.
    """
    value: str = Field(..., description="Short name for config")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Brief capability description")
    is_default: bool = Field(default=False, description="Whether pre-selected by default")
    package_identifier: str = Field(..., description="NPM package identifier")
    command: str = Field(default="npx", description="Command to run the server")

# Predefined choices
MCP_SERVER_CHOICES = [
    MCPServerChoice(
        value="brave-search",
        display_name="Brave Search",
        description="Web search capabilities",
        is_default=True,
        package_identifier="@anthropic/mcp-server-brave-search",
    ),
    MCPServerChoice(
        value="memory",
        display_name="Memory",
        description="Key-value memory storage",
        is_default=True,
        package_identifier="@modelcontextprotocol/server-memory",
    ),
    MCPServerChoice(
        value="sequentialthinking",
        display_name="Sequential Thinking",
        description="Structured reasoning chains",
        is_default=True,
        package_identifier="@modelcontextprotocol/server-sequentialthinking",
    ),
    MCPServerChoice(
        value="filesystem",
        display_name="Filesystem",
        description="File system access",
        is_default=False,
        package_identifier="@modelcontextprotocol/server-filesystem",
    ),
    MCPServerChoice(
        value="github",
        display_name="GitHub",
        description="GitHub repository access",
        is_default=False,
        package_identifier="@modelcontextprotocol/server-github",
    ),
    MCPServerChoice(
        value="postgres",
        display_name="PostgreSQL",
        description="PostgreSQL database access",
        is_default=False,
        package_identifier="@modelcontextprotocol/server-postgres",
    ),
]
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
    agent_name: str = Field(
        default="my-agent",
        description="Agent name from wizard input"
    )
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider from wizard selection"
    )
    vector_store: str = Field(
        default="chromadb",
        description="Vector store from wizard selection"
    )
    evals: list[str] = Field(
        default_factory=lambda: [
            "rag-faithfulness",
            "rag-answer_relevancy",
        ],
        description="Evaluation metrics from wizard selection"
    )
    mcp_servers: list[str] = Field(
        default_factory=lambda: [
            "brave-search",
            "memory",
            "sequentialthinking",
        ],
        description="MCP servers from wizard selection"
    )
```

---

## Relationships

| From         | To               | Relationship  | Description                           |
| ------------ | ---------------- | ------------- | ------------------------------------- |
| WizardState  | WizardResult     | Transforms to | Final state becomes result            |
| WizardResult | ProjectInitInput | Merged into   | Wizard selections added to init input |

## Validation Summary

| Entity           | Validation Rule                         | Error Behavior        |
| ---------------- | --------------------------------------- | --------------------- |
| WizardResult     | Agent name format (alphanumeric, -, \_) | Raise ValueError      |
| WizardResult     | Provider in allowed set                 | Raise ValueError      |
| WizardResult     | Vector store in allowed set             | Raise ValueError      |
| WizardResult     | Evals in allowed set                    | Raise ValueError      |
| WizardResult     | MCP servers in allowed set              | Raise ValueError      |
| ProjectInitInput | Template exists                         | Raise ValidationError |

## State Diagram

```
[Start]
    │
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = AGENT_NAME          │
│  agent_name = None          │
└─────────────────────────────┘
    │ User enters name
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = LLM_PROVIDER        │
│  agent_name = "my-agent"    │
└─────────────────────────────┘
    │ User selects provider
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = VECTOR_STORE        │
│  llm_provider = "ollama"    │
└─────────────────────────────┘
    │ User selects store
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = EVALS               │
│  vector_store = "chromadb"  │
└─────────────────────────────┘
    │ User selects evals
    ▼
┌─────────────────────────────┐
│  WizardState                │
│  step = MCP_SERVERS         │
│  evals = [...]              │
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
