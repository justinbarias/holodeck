# Feature Specification: Choose Your Backend

**Feature Branch**: `023-choose-your-backend`
**Created**: 2026-03-15
**Status**: Draft
**Input**: User description: "Add multi-backend support for Google ADK and Microsoft Agent Framework alongside existing Semantic Kernel and Claude Agent SDK backends"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure a Google ADK Agent via YAML (Priority: P1)

A platform user wants to use Google's Gemini models through the Google ADK (Agent Development Kit) backend. They create an `agent.yaml` file specifying `provider: google` and a Gemini model name, along with their system instructions and tools. When they run `holodeck test` or `holodeck chat`, HoloDeck auto-detects the ADK backend from the `google` provider, initializes the agent, and executes the request using the Google ADK runtime.

**Why this priority**: Google ADK is a major agent framework with native Gemini support and broad multi-model capabilities via LiteLLM. Adding it as a first-class backend expands HoloDeck's reach to Google Cloud-centric teams and developers who prefer ADK's agent orchestration features.

**Independent Test**: Can be fully tested by creating an agent.yaml with `provider: google`, running `holodeck test`, and verifying the agent responds correctly using a Gemini model (backend auto-detected to ADK). Delivers immediate value to users who want Gemini-powered agents.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml with `model.provider: google` and `model.name: gemini-2.5-flash`, **When** the user runs `holodeck test`, **Then** HoloDeck auto-detects the ADK backend, invokes the agent, and returns a response with populated text and token usage.
2. **Given** an agent.yaml with `provider: google` and vectorstore tools defined, **When** the user runs `holodeck test`, **Then** tools are initialized and available to the ADK agent during execution.
3. **Given** an agent.yaml with `provider: google` and MCP tools configured, **When** the agent processes a query requiring tool use, **Then** the MCP tools are invoked and results are returned with tool call and tool result records.
4. **Given** an agent.yaml with `provider: google`, **When** the user runs `holodeck chat`, **Then** a multi-turn chat session is established and conversation context is maintained across turns.

---

### User Story 2 - Configure a Microsoft Agent Framework Agent via YAML (Priority: P1)

A platform user wants to use the Microsoft Agent Framework backend for agent execution. They configure `provider: openai` (or another supported provider) in their agent.yaml, specifying a model name and any Agent Framework-specific settings. HoloDeck auto-detects the AF backend from the `openai` provider, creates the appropriate provider client, and executes agent interactions through the Agent Framework runtime.

**Why this priority**: Microsoft Agent Framework is a multi-provider agent runtime with native support for OpenAI, Azure OpenAI, Anthropic, Ollama, and AWS Bedrock. It provides an alternative execution engine with middleware, compaction, and workflow capabilities that appeal to enterprise teams.

**Independent Test**: Can be tested by creating an agent.yaml with `provider: openai` and `model.name: gpt-4o`, running `holodeck test`, and verifying the agent responds correctly through the AF runtime (backend auto-detected to Agent Framework).

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml with `model.provider: openai` and `model.name: gpt-4o`, **When** the user runs `holodeck test`, **Then** HoloDeck auto-detects the AF backend, creates an OpenAI client, and returns a valid response.
2. **Given** an agent.yaml with `provider: azure_openai` and Azure OpenAI endpoint configured, **When** the user runs `holodeck test`, **Then** an Azure OpenAI client is used via the AF backend and the agent responds correctly.
3. **Given** an agent.yaml with `provider: openai` and MCP stdio tools, **When** the agent processes a query requiring tool use, **Then** MCP tools are invoked via AF's native MCP integration and results are captured.
4. **Given** an agent.yaml with `provider: openai`, **When** the user runs `holodeck chat`, **Then** multi-turn conversation state is maintained across turns using AF sessions.

---

### User Story 3 - Seamless Tool Portability Across Backends (Priority: P2)

A platform user has an existing agent configuration with vectorstore, function, and MCP tools. They want to switch from one backend to another (e.g., from Semantic Kernel to Google ADK) by changing only the `backend` field. All tools defined in the YAML continue to work without modification because HoloDeck's tool adapters translate tool definitions to each backend's native format.

**Why this priority**: Tool portability is the key differentiator of HoloDeck's multi-backend architecture. Without it, users are locked into a single backend, defeating the purpose of the abstraction layer.

**Independent Test**: Can be tested by taking an existing agent.yaml with tools, changing the `backend` field to `google_adk` or `agent_framework`, and verifying all tools still function correctly.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with vectorstore tools and `provider: openai`, **When** the user adds `backend: google_adk` and runs `holodeck test`, **Then** vectorstore tools are adapted to the ADK format and produce identical search results.
2. **Given** an agent.yaml with MCP stdio tools and `provider: anthropic`, **When** the user changes to `backend: agent_framework` and runs `holodeck test`, **Then** MCP tools are adapted to AF's native MCP format and function correctly.
3. **Given** an agent.yaml with the same `provider: openai` and `model.name: gpt-4o`, **When** the user switches `backend` from `agent_framework` to `semantic_kernel`, **Then** tool behavior remains identical across both backends.
4. **Given** an agent.yaml with function tools, **When** switching between any two supported backends, **Then** function tools are loaded and callable with the same behavior.

---

### User Story 4 - Backend-Specific Configuration (Priority: P2)

A platform user wants to tune backend-specific features that don't apply universally. For Google ADK, they want to configure streaming mode and code execution. For Agent Framework, they want to set message compaction strategy and client type. These settings are specified in optional, backend-specific YAML sections (`google_adk:` or `agent_framework:`) that are ignored when using other backends.

**Why this priority**: Each backend has unique capabilities that users should be able to leverage without being constrained by a lowest-common-denominator interface.

**Independent Test**: Can be tested by adding backend-specific configuration sections to agent.yaml and verifying they are applied when the matching backend is selected, and ignored otherwise.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `backend: google_adk` and a `google_adk:` section specifying `streaming_mode: sse`, **When** the user runs `holodeck chat`, **Then** streaming responses are delivered progressively.
2. **Given** an agent.yaml with `backend: agent_framework` and an `agent_framework:` section specifying `compaction_strategy: sliding_window`, **When** the agent processes a long conversation, **Then** message history is compacted according to the strategy.
3. **Given** an agent.yaml with `provider: openai` and a `google_adk:` section present, **When** the user runs `holodeck test`, **Then** the `google_adk:` section is ignored without errors and the auto-detected backend is used.

---

### User Story 5 - Cross-Provider Backend Selection (Priority: P2)

A platform user wants to run an agent on a specific backend runtime while using a model from a different provider. For example, they want to use Google ADK's agent orchestration features with an OpenAI model, or run the Agent Framework runtime with an Anthropic model. They set the `backend` field to their desired runtime and `model.provider` to the LLM provider, and HoloDeck routes accordingly.

**Why this priority**: Cross-provider backend selection unlocks the full power of decoupling runtime from provider. Users are no longer constrained to the default backend for their chosen model provider.

**Independent Test**: Can be tested by configuring `backend: google_adk` with `provider: openai`, running `holodeck test`, and verifying the ADK runtime executes with an OpenAI model.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `backend: google_adk` and `model.provider: openai`, `model.name: gpt-4o`, **When** the user runs `holodeck test`, **Then** HoloDeck selects the ADK backend and the ADK runtime uses the OpenAI model via LiteLLM.
2. **Given** an agent.yaml with `backend: agent_framework` and `model.provider: anthropic`, `model.name: claude-sonnet-4-20250514`, **When** the user runs `holodeck test`, **Then** HoloDeck selects the AF backend and creates an Anthropic client.
3. **Given** an agent.yaml with `backend: claude` and `model.provider: openai`, `model.name: gpt-4o`, **When** the user runs `holodeck test`, **Then** the system raises a clear error explaining that the Claude backend only supports the `anthropic` provider.

---

### User Story 6 - Unified Embedding Provider via LiteLLM (Priority: P1)

A platform user has vectorstore or hierarchical_document tools configured and wants embeddings to work consistently regardless of which backend they use. Today, embeddings hard-depend on Semantic Kernel's `TextEmbedding` classes, which limits embedding support to OpenAI, Azure OpenAI, and Ollama — and couples all backends to SK. With LiteLLM as the unified embedding provider, the existing `embedding_provider` YAML config maps to `litellm.aembedding()` internally, supporting 20+ providers (OpenAI, Azure, Ollama, Gemini, Cohere, Vertex, Bedrock, HuggingFace, etc.) without backend-specific code.

**Why this priority**: This is a prerequisite for the new backends (neither ADK nor AF has native embedding support) and accelerates SK deprecation by removing the last hard dependency on SK classes for non-SK backends. It also expands the set of available embedding providers from 3 to 20+.

**Independent Test**: Can be tested by configuring any backend with `embedding_provider` and a vectorstore tool, running `holodeck test`, and verifying embeddings are generated correctly via LiteLLM.

**Acceptance Scenarios**:

1. **Given** an existing agent.yaml with `provider: openai` and vectorstore tools using `embedding_provider: { provider: openai, name: text-embedding-3-small }`, **When** the user runs `holodeck test`, **Then** embeddings are generated via LiteLLM (replacing SK's `OpenAITextEmbedding`) and vectorstore search produces identical results.
2. **Given** an agent.yaml with `provider: google` and vectorstore tools with `embedding_provider` configured, **When** the user runs `holodeck test`, **Then** embeddings are generated via LiteLLM without any SK dependency (backend auto-detected to ADK).
3. **Given** an agent.yaml with `provider: openai` and vectorstore tools with `embedding_provider` configured, **When** the user runs `holodeck test`, **Then** embeddings are generated via LiteLLM and vectorstore tools function correctly (backend auto-detected to AF).
4. **Given** an agent.yaml with `embedding_provider: { provider: ollama, name: nomic-embed-text, endpoint: http://localhost:11434 }`, **When** the user runs `holodeck test`, **Then** the provider/name/endpoint fields are mapped to LiteLLM parameters (`model="ollama/nomic-embed-text"`, `api_base="http://localhost:11434"`) and embeddings are generated correctly.
5. **Given** an agent.yaml with `provider: anthropic` or `google` and vectorstore tools but NO `embedding_provider` section, **When** the user loads the config, **Then** a clear validation error is raised explaining that `embedding_provider` is required for this backend.

---

### User Story 7 - Agent Skills Replace Prompt Tools (Priority: P2)

A platform user wants to define reusable, scoped sub-agent capabilities within their agent configuration. Previously, HoloDeck had an unimplemented `PromptTool` (type: prompt) concept based on Jinja2 templates. This is replaced by `SkillTool` (type: skill) following the [Agent Skills specification](https://agentskills.io/specification). Skills are sub-agent invocations that inherit the parent's backend and can access a subset of the parent's tools via `allowed_tools`. Two forms are supported: inline (instructions in agent.yaml) for simple skills, and file-based (path to a skill directory with SKILL.md) for complex skills with scripts, references, and assets.

**Why this priority**: Skills provide a clean, standards-based abstraction for sub-agent delegation that replaces the never-implemented prompt tool. They compose naturally with the multi-backend architecture — each backend uses its native skill/sub-agent runtime. This is P2 because it depends on the new backends being functional first.

**Independent Test**: Can be tested by adding a `type: skill` tool to an agent.yaml, running `holodeck test` or `holodeck chat`, and verifying the skill is invoked as a sub-agent with the correct instructions and tool access.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with an inline skill (`type: skill`, `instructions`, `description`), **When** the agent invokes the skill during execution, **Then** a sub-agent is created with the skill's instructions using the parent's backend and model, and the skill's response is returned to the parent agent.
2. **Given** an agent.yaml with a file-based skill (`type: skill`, `path: ./skills/my-skill/`), **When** the config is loaded, **Then** SKILL.md frontmatter is validated (`name`, `description` required) and the skill directory is passed to the backend's native skill runtime.
3. **Given** an inline skill with `allowed_tools: [knowledge_base]`, **When** the skill is invoked, **Then** the skill can only access the `knowledge_base` tool from the parent agent's tool set — other parent tools are not available to the skill.
4. **Given** a skill with `allowed_tools: [nonexistent_tool]`, **When** the config is loaded, **Then** Pydantic validation rejects the configuration with a clear error listing the invalid tool name.
5. **Given** an agent.yaml with `backend: semantic_kernel` and a skill tool configured, **When** the config is loaded, **Then** the system raises an error explaining that skill tools are not supported on the SK backend (planned for deprecation).
6. **Given** a file-based skill with `path` but no `description` in the YAML, **When** the config is loaded, **Then** `description` falls back to the SKILL.md frontmatter `description` field.

---

### Edge Cases

- What happens when a user specifies `backend: google_adk` but the `google-adk` package is not installed? The system should raise a clear error with installation instructions.
- What happens when a user specifies `backend: agent_framework` but the `agent-framework-core` package is not installed? The system should raise a clear error with installation instructions.
- What happens when a user specifies an incompatible backend/provider combination (e.g., `backend: claude` with `provider: openai`)? The system should raise a clear error listing compatible providers for the requested backend.
- What happens when a user specifies an invalid `backend` value? Pydantic validation should reject the configuration with an error listing valid backend values.
- What happens when backend-specific config sections have invalid fields? Pydantic validation should reject the configuration with clear error messages.
- What happens when a tool type is not supported by the target backend? The system should raise an error at initialization time explaining which tool types are unsupported for that backend.
- What happens when the ADK or AF backend encounters a transient error during agent invocation? The system should apply retry logic consistent with existing backends.
- What happens when a skill references `allowed_tools` names that don't exist in the parent agent's tools? Pydantic validation should reject the configuration with a clear error listing the invalid tool names.
- What happens when a file-based skill's `path` directory doesn't contain a valid SKILL.md? The system should raise a `ConfigError` explaining the missing/invalid file and linking to the Agent Skills spec.
- What happens when a skill's SKILL.md has invalid frontmatter (missing `name` or `description`)? The system should raise a validation error with the specific missing fields.
- What happens when `embedding_provider` specifies an unsupported provider name? LiteLLM should raise a clear error identifying the unrecognized model string.
- What happens when `embedding_provider` credentials are invalid or missing? LiteLLM should surface the authentication error with the provider name and expected env var (e.g., `OPENAI_API_KEY`).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support `backend: google_adk` to route agent execution to the Google ADK backend.
- **FR-002**: System MUST support `backend: agent_framework` to route agent execution to the Microsoft Agent Framework backend.
- **FR-003**: Both new backends MUST implement the existing `AgentBackend` protocol (`initialize()`, `invoke_once()`, `create_session()`, `teardown()`).
- **FR-004**: Both new backends MUST implement the existing `AgentSession` protocol (`send()`, `send_streaming()`, `close()`).
- **FR-005**: Both new backends MUST return provider-agnostic result objects with all fields populated (response text, tool call records, tool result records, token usage, turn count, error state).
- **FR-006**: System MUST provide tool adapters that translate HoloDeck tool definitions (vectorstore, function, MCP, skill, hierarchical_document) to each new backend's native tool format.
- **FR-007**: System MUST support MCP tools (stdio, SSE, HTTP transports) on both new backends using each backend's native MCP integration. WebSocket transport SHOULD be supported where the backend provides native WebSocket MCP support.
- **FR-008**: System MUST support optional backend-specific configuration via `google_adk:` and `agent_framework:` YAML sections in the agent configuration.
- **FR-009**: Backend-specific YAML sections MUST be validated and rejected with clear errors if invalid.
- **FR-010**: Backend-specific YAML sections MUST be silently ignored when a different backend is selected.
- **FR-011**: System MUST raise a clear, actionable error message when a backend's required package is not installed, including the installation command.
- **FR-012**: System MUST use lazy imports for new backend modules so that uninstalled backend packages do not cause import-time failures for other backends.
- **FR-013**: The backend selector MUST route by the `backend` field when explicitly set, falling back to auto-detection from `model.provider` when `backend` is omitted.
- **FR-014**: Both new backends MUST support multi-turn conversation sessions with state preserved across turns.
- **FR-015**: Both new backends MUST support streaming responses via the streaming method.
- **FR-016**: Both new backends MUST return results compatible with all existing evaluation metric types (standard NLP, G-Eval, RAG).
- **FR-018**: The Agent Framework backend MUST read `model.provider` to determine the appropriate provider client.
- **FR-019**: System MUST abstract embedding service creation behind an `EmbeddingService` protocol backed by a single `LiteLLMEmbeddingAdapter` implementation using `litellm.aembedding()`. This replaces all Semantic Kernel embedding classes (`OpenAITextEmbedding`, `AzureTextEmbedding`, `OllamaTextEmbedding`) and provides unified embedding support for all backends. LiteLLM MUST be added as a core dependency.
- **FR-020**: When the resolved backend is `google_adk`, `agent_framework`, or `claude` and vectorstore or hierarchical_document tools are configured, the system MUST require `embedding_provider` configuration, since none of these backends have native embedding support. The existing `embedding_provider` YAML schema is preserved; fields are mapped to LiteLLM parameters internally.
- **FR-021**: _Removed_ — AF native embedding support (`OpenAIEmbeddingClient`) is unverified and unnecessary with LiteLLM as the unified provider.
- **FR-022**: System MUST replace the unimplemented `PromptTool` (type: prompt) with `SkillTool` (type: skill) following the [Agent Skills specification](https://agentskills.io/specification).
- **FR-023**: `SkillTool` MUST support two forms: inline (instructions defined in agent.yaml) and file-based (path to a skill directory containing SKILL.md with frontmatter and markdown body).
- **FR-024**: `SkillTool` MUST support an `allowed_tools` field that references parent agent tools by name, scoping the skill's tool access to a subset of the parent's configured tools.
- **FR-025**: Skills MUST run on the same backend as the parent agent. Skills inherit the parent's provider and backend runtime.
- **FR-026**: The ADK, AF, and Claude backend tool adapters MUST translate `SkillTool` into a sub-agent invocation using each backend's native skill/sub-agent runtime. SK is excluded from SkillTool scope (planned for deprecation).
- **FR-027**: System MUST support an optional top-level `backend` field in agent configuration with values: `semantic_kernel`, `claude`, `google_adk`, `agent_framework`.
- **FR-028**: When `backend` is omitted, the system MUST auto-detect the backend from `model.provider` using the default routing table.
- **FR-029**: System MUST support `google` as a valid `model.provider` value for Google AI / Vertex AI Gemini models.
- **FR-030**: System MUST validate backend/provider compatibility and raise clear errors for incompatible combinations.

### Key Entities

- **Provider Enumeration**: Extended set of supported LLM providers, adding `google` to the existing set (openai, azure_openai, anthropic, ollama).
- **Backend Enumeration**: New enumeration of supported agent runtime backends: `semantic_kernel`, `claude`, `google_adk`, `agent_framework`.
- **Google ADK Configuration**: Backend-specific settings for ADK features such as streaming mode, code execution capability, and max agent loop turns.
- **Agent Framework Configuration**: Backend-specific settings for AF features such as message compaction strategy and max tool invocation rounds. The AF backend reads `model.provider` directly to select the appropriate client class.
- **ADK Backend / ADK Session**: Backend and session implementations wrapping the Google ADK Runner and Agent classes.
- **AF Backend / AF Session**: Backend and session implementations wrapping the Microsoft Agent Framework Agent and session classes.
- **ADK Tool Adapters**: Translators from HoloDeck tool definitions to ADK-native tool objects (plain callables, MCP toolsets).
- **AF Tool Adapters**: Translators from HoloDeck tool definitions to AF-native tool objects (function tools, MCP tools).
- **Embedding Service Protocol + LiteLLM Adapter**: Provider-agnostic interface for embedding generation, backed by a single `LiteLLMEmbeddingAdapter` using `litellm.aembedding()`. Replaces all SK embedding classes. Supports 20+ embedding providers (OpenAI, Azure, Ollama, Gemini, Cohere, Vertex, Bedrock, HuggingFace, etc.).
- **Skill Tool**: Replaces `PromptTool`. A sub-agent invocation tool following the [Agent Skills specification](https://agentskills.io/specification). Supports inline definition (instructions + allowed_tools in agent.yaml) or file-based definition (path to a skill directory with SKILL.md). Skills inherit the parent agent's backend and can access a subset of the parent's tools.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can switch between any of the four supported backends (SK, Claude, ADK, AF) by changing only the `backend` field (or `model.provider` and `model.name` for auto-detected backends), with all configured tools continuing to function, except skill tools which are not supported on the SK backend (planned for deprecation).
- **SC-002**: All existing tests continue to pass without modification after adding the new backends.
- **SC-003**: Each new backend passes a functional test suite covering single-turn invocation, multi-turn sessions, streaming, tool calling, and error handling.
- **SC-004**: Agent configurations with new backends validate successfully, and invalid configurations produce actionable error messages within 1 second.
- **SC-005**: Users who have not installed the new backend packages experience no import errors or degraded functionality when using existing backends.
- **SC-006**: New backend invocations return complete result objects that are fully compatible with all three evaluation metric types (standard NLP, G-Eval, RAG).
- **SC-007**: Each new backend supports all 5 HoloDeck tool types (vectorstore, function, MCP, skill, hierarchical_document) with working tool adapters.

## Clarifications

### Session 2026-03-15

- Q: Should new backends use the SK-dependent embedding path for vectorstore tools, or should embedding service creation be abstracted? → A: Abstract embeddings into a protocol backed by LiteLLM (`litellm.aembedding()`). SK embedding classes are removed entirely. Neither ADK nor AF has confirmed native embedding support, so LiteLLM serves as the single unified embedding provider for all backends.
- Q: Should all 5 tool types be required for new backends, or is partial coverage acceptable? → A: All 5 tool types (vectorstore, function, MCP, skill, hierarchical_document) are required on day one for both new backends. FR-006 and SC-007 are now aligned.
- Q: Should the unimplemented `PromptTool` (type: prompt) be carried forward to new backends? → A: No. Replace `PromptTool` with `SkillTool` (type: skill) following the [Agent Skills specification](https://agentskills.io/specification). Skills are sub-agent invocations that inherit the parent's backend and can access a subset of the parent's tools via `allowed_tools`. Two forms: inline (instructions in agent.yaml) and file-based (path to a skill directory containing SKILL.md). The `PromptTool` model is removed from `ToolUnion`.
- Q: What naming convention should the new backend enum values follow? → A: Use product-specific names: `google_adk` and `agent_framework`. These match package names and are unambiguous if the vendors release additional frameworks.
- Q: Should new providers follow the Anthropic pattern (require `embedding_provider` for vectorstore tools)? → A: Yes. Neither ADK nor AF has confirmed native embedding support. All non-SK backends (Anthropic, ADK, AF) require `embedding_provider` when vectorstore/hierarchical_document tools are configured. Embeddings are handled uniformly via LiteLLM.
- Q: Where should the `backend` field be placed in agent.yaml? → A: Top-level, alongside `model`, `instructions`, etc. Not nested inside `model`.
- Q: Should backend/provider combinations be validated? → A: Yes. Each backend supports specific providers. Invalid combinations raise clear errors at config validation time.
- Q: Why does `openai` default to `agent_framework` instead of `semantic_kernel`? → A: SK is planned for deprecation. AF is the recommended runtime for OpenAI models going forward. This is a breaking change with a documented migration path.
- Q: What stability posture for pre-release dependencies? → A: Pin to specific tested RC versions in optional deps with a documented upgrade path when GA arrives. Both backends are labeled as preview/experimental in documentation until their upstream packages reach stable GA releases.

## Assumptions

- Google ADK Python package (`google-adk`) is available on PyPI and provides a usable API for agent creation, tool registration, and session management. Optional dependency is pinned to a specific tested version.
- Microsoft Agent Framework package (`agent-framework-core`) is available on PyPI at a tested RC version (currently v1.0.0rc4). Optional dependency is pinned to this specific version. Both new backends are labeled as preview/experimental in documentation until their upstream packages reach stable GA releases. A documented upgrade path is provided for when GA versions are released.
- Both new backends are added as optional dependencies; they are not required for existing functionality.
- The existing shared tool initialization path is refactored to use an `EmbeddingService` protocol backed by `LiteLLMEmbeddingAdapter`. All SK embedding class imports (`OpenAITextEmbedding`, `AzureTextEmbedding`, `OllamaTextEmbedding`) are removed from `tool_initializer.py`. LiteLLM is added as a core dependency. The existing `embedding_provider` YAML schema is preserved — fields are mapped to LiteLLM parameters internally.
- Token usage reporting formats differ across backends but can be normalized to the existing token usage model.
- In-memory session management (ADK's InMemorySessionService, AF's InMemoryHistoryProvider) is sufficient; persistent session storage is out of scope.
- The Agent Framework backend reads `model.provider` directly to select the appropriate client class, eliminating the need for sub-provider auto-detection.
- The Semantic Kernel backend is planned for deprecation. SK receives no new feature work (no SkillTool support). Existing SK functionality is maintained as-is during this feature. SK deprecation is a separate future effort.
