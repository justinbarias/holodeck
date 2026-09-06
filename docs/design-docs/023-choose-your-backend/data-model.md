# Data Model: Choose Your Backend

**Feature**: 023-choose-your-backend | **Date**: 2026-03-15

## Entities

### 1. ProviderEnum (Modified)

**Source**: `src/holodeck/models/llm.py`
**Change**: Add one new enum value; remove runtime-as-provider values

| Value | String | Default Backend |
|-------|--------|----------------|
| OPENAI | `"openai"` | AFBackend (default) |
| AZURE_OPENAI | `"azure_openai"` | AFBackend (default) |
| ANTHROPIC | `"anthropic"` | ClaudeBackend (default) |
| OLLAMA | `"ollama"` | ClaudeBackend (default) |
| **GOOGLE** | `"google"` | **ADKBackend (default)** |

### 1b. BackendEnum (New)

**Source**: `src/holodeck/models/llm.py`
**Purpose**: Identifies the agent runtime, decoupled from LLM provider

| Value | String | Backend Class |
|-------|--------|---------------|
| SEMANTIC_KERNEL | `"semantic_kernel"` | SKBackend |
| CLAUDE | `"claude"` | ClaudeBackend |
| GOOGLE_ADK | `"google_adk"` | ADKBackend |
| AGENT_FRAMEWORK | `"agent_framework"` | AFBackend |

### Default Routing (when `backend` omitted)

| Provider | Default Backend | Rationale |
|----------|----------------|-----------|
| `openai` | `agent_framework` | AF is the recommended OpenAI runtime (SK planned for deprecation) |
| `azure_openai` | `agent_framework` | AF has native Azure OpenAI client support |
| `anthropic` | `claude` | Claude Agent SDK is the native Anthropic runtime |
| `ollama` | `claude` | Claude SDK supports Ollama models natively |
| `google` | `google_adk` | ADK is the native Google/Gemini runtime |

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
| `compaction_strategy` | `CompactionStrategy` | `none` | Message history compaction for long conversations |
| `max_tool_rounds` | `int \| None` | `None` | Max tool invocation rounds per turn (`ge=1`) |

**Nested Types**:
- `CompactionStrategy(str, Enum)`: `none = "none"`, `sliding_window = "sliding_window"`, `summarization = "summarization"`

### 4. Agent Model (Modified)

**File**: `src/holodeck/models/agent.py`
**Change**: Add three optional fields — `backend` override plus two backend-specific config sections

| New Field | Type | Default | Condition |
|-----------|------|---------|-----------|
| `backend` | `BackendEnum \| None` | `None` | Agent runtime override. When `None`, auto-detected from `model.provider` using default routing table. |
| `google_adk` | `GoogleADKConfig \| None` | `None` | Validated when resolved backend is `google_adk`; silently ignored otherwise |
| `agent_framework` | `AgentFrameworkConfig \| None` | `None` | Validated when resolved backend is `agent_framework`; silently ignored otherwise |

### 5. EmbeddingService Protocol + LiteLLM Adapter (New)

**File**: `src/holodeck/lib/embedding_protocol.py`
**Type**: `typing.Protocol` (`@runtime_checkable`)
**Purpose**: Replace SK embedding classes with a unified LiteLLM-based embedding service for all backends

| Method/Property | Signature | Description |
|-----------------|-----------|-------------|
| `embed_batch` | `async (texts: list[str]) -> list[list[float]]` | Generate embeddings for a batch of texts |
| `dimensions` | `@property -> int` | Embedding vector dimensionality |
| `model_id` | `@property -> str` | Identifier of the embedding model |

**Single Adapter — `LiteLLMEmbeddingAdapter`** (in same file):

| Field | Type | Description |
|-------|------|-------------|
| `_model_id` | `str` | LiteLLM model string (e.g., `"text-embedding-3-small"`, `"azure/my-deployment"`, `"ollama/nomic-embed-text"`) |
| `_api_key` | `str \| None` | API key for the embedding provider |
| `_api_base` | `str \| None` | Custom endpoint URL (maps from `embedding_provider.endpoint`) |
| `_dimensions` | `int` | Embedding vector dimensionality |

**Provider mapping from `embedding_provider` YAML to LiteLLM model string**:

| YAML `provider` | YAML `name` | LiteLLM `model` |
|-----------------|-------------|-----------------|
| `openai` | `text-embedding-3-small` | `"text-embedding-3-small"` |
| `azure_openai` | `my-deployment` | `"azure/my-deployment"` |
| `ollama` | `nomic-embed-text` | `"ollama/nomic-embed-text"` |

**What is removed**:
- SK `OpenAITextEmbedding`, `AzureTextEmbedding`, `OllamaTextEmbedding` imports from `tool_initializer.py`
- No `SKEmbeddingAdapter` or `AFEmbeddingAdapter` needed

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

> **Import Aliasing**: The AF SDK's `AgentSession` type collides with HoloDeck's `AgentSession` protocol (from `base.py`). In `af_backend.py`, import the AF type with an alias: `from agent_framework import AgentSession as AFNativeSession` to avoid ambiguity.

### 8. SkillTool (New — replaces PromptTool)

**File**: `src/holodeck/models/tool.py`
**Replaces**: `PromptTool` (type: prompt) — removed from `ToolUnion`
**Spec**: [Agent Skills specification](https://agentskills.io/specification)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Skill identifier (1-64 chars, lowercase alphanumeric + hyphens, matches Agent Skills spec naming) |
| `description` | `str \| None` | `None` | What the skill does and when to use it (max 1024 chars per spec). Required for inline skills; optional for file-based skills (falls back to SKILL.md frontmatter `description`). |
| `type` | `Literal["skill"]` | `"skill"` | Tool type discriminator |
| `instructions` | `str \| None` | `None` | Inline skill instructions (mutually exclusive with `path`) |
| `path` | `str \| None` | `None` | Path to skill directory containing SKILL.md (mutually exclusive with `instructions`) |
| `allowed_tools` | `list[str] \| None` | `None` | Names of parent agent tools this skill can access. `None` = no tool access. YAML-only — not merged from SKILL.md frontmatter. |

**Validation Rules**:
- Exactly one of `instructions` or `path` must be provided
- `description` is required when `instructions` is set (inline skill). When `path` is set (file-based skill), `description` is optional and falls back to the SKILL.md frontmatter `description` field. A `model_post_init` validator enforces that at least one source of description exists.
- `name` uses a DIFFERENT pattern from other tool types: `^[a-z0-9]+(-[a-z0-9]+)*$` (Agent Skills spec: lowercase alphanumeric + hyphens, no leading/trailing/consecutive hyphens, 1-64 chars). Other tool types use `^[0-9A-Za-z_]+$`.
- When `path` is provided, the directory must contain a valid `SKILL.md` with required frontmatter (`name`, `description`). Each backend handles SKILL.md parsing natively via its own skill runtime.
- When `allowed_tools` is provided, referenced tool names are validated against the parent agent's `tools` list at config time. `allowed_tools` is specified ONLY in agent.yaml — the SKILL.md `allowed-tools` frontmatter field is for the backend's native tool permissions and is NOT merged.

**Inline form** (simple skills):
```yaml
tools:
  - name: sentiment-analyzer
    type: skill
    description: "Analyze sentiment and extract key emotions from text"
    instructions: "Analyze the given text for sentiment. Return overall sentiment, confidence, and key emotions."
    allowed_tools: [knowledge_base]
```

**File-based form** (complex skills with scripts/references/assets):
```yaml
tools:
  - name: research-assistant
    type: skill
    path: ./skills/research-assistant/
    # SKILL.md in that directory provides name, description, instructions
    # Each backend handles SKILL.md parsing natively via its skill runtime
    allowed_tools: [knowledge_base, web_search]  # YAML-only, not from SKILL.md
```

**Backend Adaptation**:
Each backend uses its native skill/sub-agent runtime. SK is excluded (planned for deprecation).

| Backend | Native Runtime |
|---------|---------------|
| ADK | ADK's native agent composition — skill as a sub-agent within the ADK agent graph |
| AF | AF's native agent delegation — skill via AF's built-in sub-agent orchestration |
| Claude | Claude Agent SDK's native sub-agent system |

## Relationships

```
Agent (model)
├── backend: BackendEnum? ────────→ BackendSelector (explicit routing)
├── model.provider: ProviderEnum ─→ BackendSelector (auto-detect fallback)
├── claude: ClaudeConfig? ────────→ ClaudeBackend (existing)
├── google_adk: GoogleADKConfig? ─→ ADKBackend (new)
├── agent_framework: AFConfig? ───→ AFBackend (new)
├── embedding_provider: ... ──────→ EmbeddingService (protocol)
└── tools: list[ToolUnion] ───────→ Per-backend tool adapters

EmbeddingService (protocol)
└── LiteLLMEmbeddingAdapter ──→ litellm.aembedding() (all providers)

BackendSelector
├── Explicit backend field ────────→ Use directly
└── Auto-detect from provider:
    ├── openai/azure_openai ───────→ AFBackend
    ├── anthropic ─────────────────→ ClaudeBackend
    ├── ollama ────────────────────→ ClaudeBackend
    └── google ────────────────────→ ADKBackend (lazy import)
```

## Validation Rules

1. **Backend-specific config sections**: Only validated when matching backend is resolved. Present but non-matching sections are silently ignored (not rejected).
2. **`embedding_provider` requirement**: When the resolved `backend` is `google_adk`, `agent_framework`, or `claude` and vectorstore/hierarchical_document tools are configured, `embedding_provider` MUST be set. All embedding is handled via `LiteLLMEmbeddingAdapter`. **Implementation note**: This validator does NOT currently exist in `agent.py` — it must be created as a new `@model_validator(mode='after')`, not extended from an existing validator. This fixes the pre-existing gap for the `anthropic` provider and covers both new backends.
3. **Backend/provider compatibility**: When `backend` is explicitly set, the system validates that `model.provider` is compatible with the chosen backend. Incompatible combinations raise `ValidationError` with clear error messages.
4. **Default routing resolution**: When `backend` is `None`, it is resolved from `model.provider` using the default routing table before backend initialization.
5. **Lazy import guard**: ADK and AF backend modules are only imported when their backend is selected. Missing packages raise `BackendInitError` with installation instructions.
