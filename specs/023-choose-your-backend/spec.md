# Feature Specification: Choose Your Backend

**Feature Branch**: `023-choose-your-backend`
**Created**: 2026-03-15
**Status**: Draft
**Input**: User description: "Add multi-backend support for Google ADK and Microsoft Agent Framework alongside existing Semantic Kernel and Claude Agent SDK backends"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure a Google ADK Agent via YAML (Priority: P1)

A platform user wants to use Google's Gemini models through the Google ADK (Agent Development Kit) backend. They create an `agent.yaml` file specifying `provider: google_adk` and a Gemini model name, along with their system instructions and tools. When they run `holodeck test` or `holodeck chat`, HoloDeck automatically routes to the ADK backend, initializes the agent, and executes the request using the Google ADK runtime.

**Why this priority**: Google ADK is a major agent framework with native Gemini support and broad multi-model capabilities via LiteLLM. Adding it as a first-class backend expands HoloDeck's reach to Google Cloud-centric teams and developers who prefer ADK's agent orchestration features.

**Independent Test**: Can be fully tested by creating an agent.yaml with `provider: google_adk`, running `holodeck test`, and verifying the agent responds correctly using a Gemini model. Delivers immediate value to users who want Gemini-powered agents.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml with `model.provider: google_adk` and `model.name: gemini-2.5-flash`, **When** the user runs `holodeck test`, **Then** HoloDeck selects the ADK backend, invokes the agent, and returns a response with populated text and token usage.
2. **Given** an agent.yaml with `provider: google_adk` and vectorstore tools defined, **When** the user runs `holodeck test`, **Then** tools are initialized and available to the ADK agent during execution.
3. **Given** an agent.yaml with `provider: google_adk` and MCP tools configured, **When** the agent processes a query requiring tool use, **Then** the MCP tools are invoked and results are returned with tool call and tool result records.
4. **Given** an agent.yaml with `provider: google_adk`, **When** the user runs `holodeck chat`, **Then** a multi-turn chat session is established and conversation context is maintained across turns.

---

### User Story 2 - Configure a Microsoft Agent Framework Agent via YAML (Priority: P1)

A platform user wants to use the Microsoft Agent Framework backend for agent execution. They configure `provider: agent_framework` in their agent.yaml, specifying a model name and any Agent Framework-specific settings. HoloDeck auto-selects the AF backend, creates the appropriate provider client (OpenAI, Azure OpenAI, Anthropic, Ollama), and executes agent interactions through the Agent Framework runtime.

**Why this priority**: Microsoft Agent Framework is a multi-provider agent runtime with native support for OpenAI, Azure OpenAI, Anthropic, Ollama, and AWS Bedrock. It provides an alternative execution engine with middleware, compaction, and workflow capabilities that appeal to enterprise teams.

**Independent Test**: Can be tested by creating an agent.yaml with `provider: agent_framework` and `model.name: gpt-4o`, running `holodeck test`, and verifying the agent responds correctly through the AF runtime.

**Acceptance Scenarios**:

1. **Given** a valid agent.yaml with `model.provider: agent_framework` and `model.name: gpt-4o`, **When** the user runs `holodeck test`, **Then** HoloDeck selects the AF backend, creates an OpenAI client, and returns a valid response.
2. **Given** an agent.yaml with `provider: agent_framework` and Azure OpenAI endpoint configured, **When** the user runs `holodeck test`, **Then** an Azure OpenAI client is used and the agent responds correctly.
3. **Given** an agent.yaml with `provider: agent_framework` and MCP stdio tools, **When** the agent processes a query requiring tool use, **Then** MCP tools are invoked via AF's native MCP integration and results are captured.
4. **Given** an agent.yaml with `provider: agent_framework`, **When** the user runs `holodeck chat`, **Then** multi-turn conversation state is maintained across turns using AF sessions.

---

### User Story 3 - Seamless Tool Portability Across Backends (Priority: P2)

A platform user has an existing agent configuration with vectorstore, function, and MCP tools. They want to switch from one backend to another (e.g., from Semantic Kernel to Google ADK) by changing only the `model.provider` and `model.name` fields. All tools defined in the YAML continue to work without modification because HoloDeck's tool adapters translate tool definitions to each backend's native format.

**Why this priority**: Tool portability is the key differentiator of HoloDeck's multi-backend architecture. Without it, users are locked into a single backend, defeating the purpose of the abstraction layer.

**Independent Test**: Can be tested by taking an existing agent.yaml with tools, changing the provider from `openai` to `google_adk` or `agent_framework`, and verifying all tools still function correctly.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with vectorstore tools and `provider: openai`, **When** the user changes to `provider: google_adk` and runs `holodeck test`, **Then** vectorstore tools are adapted to the ADK format and produce identical search results.
2. **Given** an agent.yaml with MCP stdio tools and `provider: anthropic`, **When** the user changes to `provider: agent_framework` and runs `holodeck test`, **Then** MCP tools are adapted to AF's native MCP format and function correctly.
3. **Given** an agent.yaml with function tools, **When** switching between any two supported backends, **Then** function tools are loaded and callable with the same behavior.

---

### User Story 4 - Backend-Specific Configuration (Priority: P2)

A platform user wants to tune backend-specific features that don't apply universally. For Google ADK, they want to configure streaming mode and code execution. For Agent Framework, they want to set message compaction strategy and client type. These settings are specified in optional, backend-specific YAML sections (`google_adk:` or `agent_framework:`) that are ignored when using other backends.

**Why this priority**: Each backend has unique capabilities that users should be able to leverage without being constrained by a lowest-common-denominator interface.

**Independent Test**: Can be tested by adding backend-specific configuration sections to agent.yaml and verifying they are applied when the matching backend is selected, and ignored otherwise.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `provider: google_adk` and a `google_adk:` section specifying `streaming_mode: sse`, **When** the user runs `holodeck chat`, **Then** streaming responses are delivered progressively.
2. **Given** an agent.yaml with `provider: agent_framework` and an `agent_framework:` section specifying `compaction_strategy: sliding_window`, **When** the agent processes a long conversation, **Then** message history is compacted according to the strategy.
3. **Given** an agent.yaml with `provider: openai` and a `google_adk:` section present, **When** the user runs `holodeck test`, **Then** the `google_adk:` section is ignored without errors and the SK backend is used.

---

### User Story 5 - Google ADK Multi-Model Support (Priority: P3)

A platform user wants to use non-Google models through the ADK backend by leveraging ADK's built-in LiteLLM integration. They configure `provider: google_adk` with a LiteLLM-prefixed model name (e.g., `openai/gpt-4o`) and HoloDeck routes through ADK's LiteLLM model registry, allowing them to use any LiteLLM-supported provider while benefiting from ADK's agent orchestration.

**Why this priority**: This expands the ADK backend's utility beyond just Gemini models, making it a viable alternative execution engine for any model provider.

**Independent Test**: Can be tested by configuring `provider: google_adk` with `model.name: openai/gpt-4o` and verifying the ADK backend resolves and invokes the OpenAI model via LiteLLM.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `provider: google_adk` and `model.name: openai/gpt-4o`, **When** the user runs `holodeck test`, **Then** ADK routes through LiteLLM to OpenAI and returns a valid response.
2. **Given** an agent.yaml with `provider: google_adk` and `model.name: anthropic/claude-sonnet-4-20250514`, **When** the user runs `holodeck test`, **Then** ADK routes through LiteLLM to Anthropic and returns a valid response.

---

### Edge Cases

- What happens when a user specifies `provider: google_adk` but the `google-adk` package is not installed? The system should raise a clear error with installation instructions.
- What happens when a user specifies `provider: agent_framework` with a model name that doesn't match any known sub-provider pattern? The system should raise a descriptive error listing supported model prefixes or prompt the user to specify `sub_provider` explicitly.
- What happens when backend-specific config sections have invalid fields? Pydantic validation should reject the configuration with clear error messages.
- What happens when a tool type is not supported by the target backend? The system should raise an error at initialization time explaining which tool types are unsupported for that backend.
- What happens when the ADK or AF backend encounters a transient error during agent invocation? The system should apply retry logic consistent with existing backends.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support `provider: google_adk` in `model.provider` to route agent execution to the Google ADK backend.
- **FR-002**: System MUST support `provider: agent_framework` in `model.provider` to route agent execution to the Microsoft Agent Framework backend.
- **FR-003**: Both new backends MUST implement the existing `AgentBackend` protocol (`initialize()`, `invoke_once()`, `create_session()`, `teardown()`).
- **FR-004**: Both new backends MUST implement the existing `AgentSession` protocol (`send()`, `send_streaming()`, `close()`).
- **FR-005**: Both new backends MUST return provider-agnostic result objects with all fields populated (response text, tool call records, tool result records, token usage, turn count, error state).
- **FR-006**: System MUST provide tool adapters that translate HoloDeck tool definitions (vectorstore, function, MCP, prompt, hierarchical_document) to each new backend's native tool format.
- **FR-007**: System MUST support MCP tools (stdio, SSE, HTTP, WebSocket transports) on both new backends using each backend's native MCP integration.
- **FR-008**: System MUST support optional backend-specific configuration via `google_adk:` and `agent_framework:` YAML sections in the agent configuration.
- **FR-009**: Backend-specific YAML sections MUST be validated and rejected with clear errors if invalid.
- **FR-010**: Backend-specific YAML sections MUST be silently ignored when a different backend is selected.
- **FR-011**: System MUST raise a clear, actionable error message when a backend's required package is not installed, including the installation command.
- **FR-012**: System MUST use lazy imports for new backend modules so that uninstalled backend packages do not cause import-time failures for other backends.
- **FR-013**: The backend selector MUST route `provider: google_adk` to the ADK backend and `provider: agent_framework` to the AF backend.
- **FR-014**: Both new backends MUST support multi-turn conversation sessions with state preserved across turns.
- **FR-015**: Both new backends MUST support streaming responses via the streaming method.
- **FR-016**: Both new backends MUST return results compatible with all existing evaluation metric types (standard NLP, G-Eval, RAG).
- **FR-017**: The Google ADK backend MUST support multi-model routing via ADK's LiteLLM integration when model names contain provider prefixes (e.g., `openai/gpt-4o`).
- **FR-018**: The Agent Framework backend MUST auto-detect the appropriate provider client based on model name and configuration, with an option to specify explicitly.
- **FR-019**: System MUST abstract embedding service creation behind a protocol so that tool initialization (vectorstore, hierarchical_document) does not hard-depend on any single backend's embedding classes. The current Semantic Kernel embedding implementation becomes the default adapter behind this protocol.
- **FR-020**: When `provider: google_adk` is used with vectorstore or hierarchical_document tools, the system MUST require `embedding_provider` configuration (same pattern as `provider: anthropic`), since Google ADK has no native embedding support.
- **FR-021**: When `provider: agent_framework` is used with vectorstore or hierarchical_document tools, the system SHOULD support Agent Framework's native embedding clients (`OpenAIEmbeddingClient`, `AzureOpenAIEmbeddingClient`) as an adapter behind the embedding protocol, with fallback to `embedding_provider` configuration.

### Key Entities

- **Provider Enumeration**: Extended set of supported backend providers, adding `google_adk` and `agent_framework` values to the existing set (openai, azure_openai, anthropic, ollama).
- **Google ADK Configuration**: Backend-specific settings for ADK features such as streaming mode, code execution capability, and max agent loop turns.
- **Agent Framework Configuration**: Backend-specific settings for AF features such as client type (chat vs. responses), message compaction strategy, and max tool invocation rounds.
- **ADK Backend / ADK Session**: Backend and session implementations wrapping the Google ADK Runner and Agent classes.
- **AF Backend / AF Session**: Backend and session implementations wrapping the Microsoft Agent Framework Agent and session classes.
- **ADK Tool Adapters**: Translators from HoloDeck tool definitions to ADK-native tool objects (plain callables, MCP toolsets).
- **AF Tool Adapters**: Translators from HoloDeck tool definitions to AF-native tool objects (function tools, MCP tools).
- **Embedding Service Protocol**: Provider-agnostic interface for embedding generation, decoupling tool initialization from any specific backend's embedding classes. SK embedding classes serve as the default implementation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can switch between any of the four supported backends (SK, Claude, ADK, AF) by changing only the `model.provider` and `model.name` fields, with all configured tools continuing to function.
- **SC-002**: All existing tests continue to pass without modification after adding the new backends.
- **SC-003**: Each new backend passes a functional test suite covering single-turn invocation, multi-turn sessions, streaming, tool calling, and error handling.
- **SC-004**: Agent configurations with new backends validate successfully, and invalid configurations produce actionable error messages within 1 second.
- **SC-005**: Users who have not installed the new backend packages experience no import errors or degraded functionality when using existing backends.
- **SC-006**: New backend invocations return complete result objects that are fully compatible with all three evaluation metric types (standard NLP, G-Eval, RAG).
- **SC-007**: Each new backend supports all 5 HoloDeck tool types (vectorstore, function, MCP, prompt, hierarchical_document) with working tool adapters.

## Clarifications

### Session 2026-03-15

- Q: Should new backends use the SK-dependent embedding path for vectorstore tools, or should embedding service creation be abstracted? → A: Abstract embeddings into a protocol; SK becomes one implementation, with optional ADK/AF-native embedding adapters.
- Q: Should all 5 tool types be required for new backends, or is partial coverage acceptable? → A: All 5 tool types (vectorstore, function, MCP, prompt, hierarchical_document) are required on day one for both new backends. FR-006 and SC-007 are now aligned.
- Q: What naming convention should the new provider enum values follow? → A: Use product-specific names: `google_adk` and `agent_framework`. These match package names and are unambiguous if the vendors release additional frameworks.
- Q: Should new providers follow the Anthropic pattern (require `embedding_provider` for vectorstore tools)? → A: Google ADK has no native embedding support, so it follows the Anthropic pattern (requires `embedding_provider`). Microsoft Agent Framework has native embedding clients (`OpenAIEmbeddingClient`, `AzureOpenAIEmbeddingClient` via `SupportsGetEmbeddings` protocol), so it can optionally use its own embedding adapter behind the new embedding protocol, or fall back to `embedding_provider`.
- Q: What stability posture for pre-release dependencies? → A: Pin to specific tested RC versions in optional deps with a documented upgrade path when GA arrives. Both backends are labeled as preview/experimental in documentation until their upstream packages reach stable GA releases.

## Assumptions

- Google ADK Python package (`google-adk`) is available on PyPI and provides a usable API for agent creation, tool registration, and session management. Optional dependency is pinned to a specific tested version.
- Microsoft Agent Framework package (`agent-framework-core`) is available on PyPI at a tested RC version (currently v1.0.0rc4). Optional dependency is pinned to this specific version. Both new backends are labeled as preview/experimental in documentation until their upstream packages reach stable GA releases. A documented upgrade path is provided for when GA versions are released.
- Both new backends are added as optional dependencies; they are not required for existing functionality.
- The existing shared tool initialization path is refactored to use an embedding service protocol rather than hard-coding Semantic Kernel embedding classes. SK embedding classes become the default implementation, with optional ADK/AF-native adapters possible in the future. The tool initialization entry point remains shared across all backends.
- Token usage reporting formats differ across backends but can be normalized to the existing token usage model.
- In-memory session management (ADK's InMemorySessionService, AF's InMemoryHistoryProvider) is sufficient; persistent session storage is out of scope.
- The Agent Framework's sub-provider auto-detection covers the most common model name patterns; exotic configurations may require explicit sub-provider specification in the backend config section.
