# Data Model: Choose Your Backend

**Feature**: 023-choose-your-backend | **Date**: 2026-03-15

## Entities

### 1. ProviderEnum (Modified)

**Source**: `src/holodeck/models/llm.py`
**Change**: Add two new enum values

| Value | String | Backend Class |
|-------|--------|---------------|
| OPENAI | `"openai"` | SKBackend (existing) |
| AZURE_OPENAI | `"azure_openai"` | SKBackend (existing) |
| ANTHROPIC | `"anthropic"` | ClaudeBackend (existing) |
| OLLAMA | `"ollama"` | SKBackend (existing) |
| **GOOGLE_ADK** | `"google_adk"` | **ADKBackend (new)** |
| **AGENT_FRAMEWORK** | `"agent_framework"` | **AFBackend (new)** |

### 2. GoogleADKConfig (New)

**File**: `src/holodeck/models/google_adk_config.py`
**Parent**: `pydantic.BaseModel`
**Pattern**: Follows `ClaudeConfig` (optional backend-specific section in Agent model)
**Validation**: `ConfigDict(extra="forbid")`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `streaming_mode` | `StreamingMode` (enum: `none`, `sse`) | `none` | How responses are delivered |
| `code_execution` | `bool` | `False` | Enable ADK's built-in code execution capability |
| `max_iterations` | `int \| None` | `None` | Max agent loop turns (`ge=1`) |
| `output_key` | `str` | `"output"` | Key in ADK session state for final response |

**Nested Types**:
- `StreamingMode(str, Enum)`: `none = "none"`, `sse = "sse"`

### 3. AgentFrameworkConfig (New)

**File**: `src/holodeck/models/af_config.py`
**Parent**: `pydantic.BaseModel`
**Pattern**: Follows `ClaudeConfig`
**Validation**: `ConfigDict(extra="forbid")`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sub_provider` | `AFSubProvider \| None` | `None` | Explicit client type override (auto-detected from model name if omitted) |
| `compaction_strategy` | `CompactionStrategy` | `none` | Message history compaction for long conversations |
| `max_tool_rounds` | `int \| None` | `None` | Max tool invocation rounds per turn (`ge=1`) |
| `use_native_embeddings` | `bool` | `False` | Use AF's native embedding clients instead of `embedding_provider` |

**Nested Types**:
- `AFSubProvider(str, Enum)`: `openai = "openai"`, `azure_openai = "azure_openai"`, `anthropic = "anthropic"`, `ollama = "ollama"`
- `CompactionStrategy(str, Enum)`: `none = "none"`, `sliding_window = "sliding_window"`, `summarization = "summarization"`

### 4. Agent Model (Modified)

**File**: `src/holodeck/models/agent.py`
**Change**: Add two optional fields following the `claude` field pattern

| New Field | Type | Default | Condition |
|-----------|------|---------|-----------|
| `google_adk` | `GoogleADKConfig \| None` | `None` | Validated when `model.provider == google_adk`; silently ignored otherwise |
| `agent_framework` | `AgentFrameworkConfig \| None` | `None` | Validated when `model.provider == agent_framework`; silently ignored otherwise |

### 5. EmbeddingService Protocol (New)

**File**: `src/holodeck/lib/embedding_protocol.py`
**Type**: `typing.Protocol` (`@runtime_checkable`)
**Purpose**: Decouple `tool_initializer.py` from SK embedding classes

| Method/Property | Signature | Description |
|-----------------|-----------|-------------|
| `embed_batch` | `async (texts: list[str]) -> list[list[float]]` | Generate embeddings for a batch of texts |
| `dimensions` | `@property -> int` | Embedding vector dimensionality |
| `model_id` | `@property -> str` | Identifier of the embedding model |

**Adapters**:

| Adapter | Wraps | File |
|---------|-------|------|
| `SKEmbeddingAdapter` | SK `TextEmbedding` classes (`OpenAITextEmbedding`, `AzureTextEmbedding`, `OllamaTextEmbedding`) | `embedding_protocol.py` |
| `AFEmbeddingAdapter` | AF `OpenAIEmbeddingClient` / `AzureOpenAIEmbeddingClient` | `af_embedding_adapter.py` |

### 6. ADKBackend + ADKSession (New)

**File**: `src/holodeck/lib/backends/adk_backend.py`
**Implements**: `AgentBackend`, `AgentSession` protocols

**ADKBackend State**:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_agent` | `google.adk.agents.LlmAgent` | Configured ADK agent instance |
| `_runner` | `google.adk.runners.Runner` | ADK runner for invocations |
| `_session_service` | `InMemorySessionService` | Session state management |
| `_tools` | `list[Any]` | Adapted tool instances |
| `_config` | `GoogleADKConfig \| None` | Backend-specific config |

**ADKSession State**:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_runner` | `Runner` | Shared runner reference |
| `_user_id` | `str` | Unique user ID for session |
| `_session_id` | `str` | Unique session ID |

### 7. AFBackend + AFSession (New)

**File**: `src/holodeck/lib/backends/af_backend.py`
**Implements**: `AgentBackend`, `AgentSession` protocols

**AFBackend State**:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_agent` | `agent_framework.Agent` | Configured AF agent instance |
| `_client` | `BaseChatClient` | Provider-specific LLM client |
| `_tools` | `list[Any]` | Adapted tool instances |
| `_config` | `AgentFrameworkConfig \| None` | Backend-specific config |

**AFSession State**:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_agent` | `Agent` | Shared agent reference |
| `_session` | `AgentSession` | AF session with history |

## Relationships

```
Agent (model)
├── model.provider: ProviderEnum ──→ BackendSelector routing
├── claude: ClaudeConfig? ─────────→ ClaudeBackend (existing)
├── google_adk: GoogleADKConfig? ──→ ADKBackend (new)
├── agent_framework: AFConfig? ────→ AFBackend (new)
├── embedding_provider: ... ───────→ EmbeddingService (protocol)
└── tools: list[ToolUnion] ────────→ Per-backend tool adapters

EmbeddingService (protocol)
├── SKEmbeddingAdapter ────→ SK TextEmbedding (default)
└── AFEmbeddingAdapter ────→ AF EmbeddingClient (optional)

BackendSelector
├── openai/azure_openai/ollama ──→ SKBackend
├── anthropic ───────────────────→ ClaudeBackend
├── google_adk ──────────────────→ ADKBackend (lazy import)
└── agent_framework ─────────────→ AFBackend (lazy import)
```

## Validation Rules

1. **Backend-specific config sections**: Only validated when matching provider is selected. Present but non-matching sections are silently ignored (not rejected).
2. **`embedding_provider` requirement**: When `provider` is `google_adk` or `anthropic` and vectorstore/hierarchical_document tools are configured, `embedding_provider` MUST be set.
3. **AF sub-provider auto-detection**: When `sub_provider` is not set in `AgentFrameworkConfig`, model name prefix matching determines the client class. Unrecognizable model names raise `ValidationError`.
4. **Lazy import guard**: ADK and AF backend modules are only imported when their provider is selected. Missing packages raise `BackendInitError` with installation instructions.
