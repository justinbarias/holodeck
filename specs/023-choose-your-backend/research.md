# Research: Choose Your Backend

**Feature**: 023-choose-your-backend
**Date**: 2026-03-15
**Status**: Complete

## R1: Google ADK Python SDK API Surface

**Decision**: Use `google.adk.agents.Agent` (alias for `LlmAgent`) with `Runner` for invocation and `InMemorySessionService` for session management.

**Rationale**: ADK's `Runner.run_async()` returns an `AsyncGenerator[Event]` which maps cleanly to HoloDeck's `ExecutionResult`. The `InMemorySessionService` handles multi-turn state without external dependencies. Tools can be plain callables (auto-wrapped as `FunctionTool`) or `McpToolset` instances.

**Key API Mapping**:
- Agent creation: `LlmAgent(name, model, instruction, tools, generate_content_config)`
- Invocation: `runner.run_async(user_id, session_id, new_message)` → `AsyncGenerator[Event]`
- Response extraction: `event.is_final_response()`, `event.content.parts`, `event.get_function_calls()`, `event.usage_metadata`
- Streaming: `RunConfig(streaming_mode=StreamingMode.SSE)` → partial events
- MCP tools: `McpToolset(connection_params=StdioConnectionParams(command, args))`
- Multi-model: For `provider: google`, model name passes directly (e.g., `"gemini-2.5-flash"`). For cross-provider (`backend: google_adk` + other provider), HoloDeck constructs the LiteLLM prefix internally from `model.provider` (e.g., `provider: openai` + `name: gpt-4o` → `"openai/gpt-4o"`).

**Alternatives Considered**:
- Direct Gemini API (google-genai): Too low-level, no tool loop, no MCP support
- LangChain Google integration: Adds unnecessary dependency layer

## R2: Microsoft Agent Framework API Surface

**Decision**: Use `BaseChatClient.as_agent()` pattern with provider-specific client classes (`OpenAIChatClient`, `AzureOpenAIChatClient`, `AnthropicClient`, `OllamaChatClient`).

**Rationale**: AF's `agent.run()` returns `AgentResponse` (non-streaming) or `ResponseStream[AgentResponseUpdate, AgentResponse]` (streaming), mapping directly to `ExecutionResult`. The `@tool` decorator and `FunctionTool` class handle function tools. MCP tools are first-class via `MCPStdioTool`, `MCPStreamableHTTPTool`, `MCPWebsocketTool`.

**Key API Mapping**:
- Client creation: Selected based on `model.provider` (e.g., `provider: openai` → `OpenAIChatClient(api_key, model)`)
- Agent creation: `client.as_agent(name, instructions, tools, default_options, context_providers)`
- Invocation: `await agent.run(message)` → `AgentResponse`
- Response extraction: `result.text`, `result.usage_details`, `result.messages` (for tool calls)
- Streaming: `agent.run(message, stream=True)` → `ResponseStream`
- Sessions: `agent.create_session()`, pass to `agent.run(message, session=session)`
- MCP: `MCPStdioTool(name, command, args)`, requires `async with agent:` context manager

**Alternatives Considered**:
- Semantic Kernel (already used): SK is the existing backend; AF provides a different runtime with middleware/compaction features
- AutoGen: Not protocol-driven, harder to adapt

## R3: Embedding Service Abstraction

**Decision**: Create an `EmbeddingService` protocol with `embed_batch(texts) -> list[list[float]]` method. Single implementation: `LiteLLMEmbeddingAdapter` wrapping `litellm.aembedding()`. Replaces all SK `TextEmbedding` classes entirely.

**Rationale**: The current `tool_initializer.py` hard-depends on SK's `OpenAITextEmbedding`, `AzureTextEmbedding`, and `OllamaTextEmbedding`. Research confirmed:
- ADK has **no native embedding support** (confirmed via PyPI and ADK docs)
- AF (`agent-framework-core`) has **no confirmed embedding support** in the core package (`OpenAIEmbeddingClient` is unverified)
- SK is planned for deprecation

LiteLLM (`litellm.aembedding()`) provides:
1. 20+ embedding providers (OpenAI, Azure, Gemini, Cohere, Vertex, Bedrock, HuggingFace, Ollama, etc.)
2. Unified OpenAI-compatible async API
3. Already a transitive dependency of `google-adk`
4. MIT licensed, actively maintained (v1.82.2)

**Protocol Design**:
```python
@runtime_checkable
class EmbeddingService(Protocol):
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...
    @property
    def model_id(self) -> str: ...
```

**Single Adapter — `LiteLLMEmbeddingAdapter`**:
- Maps existing `embedding_provider` YAML config to LiteLLM params:
  - `provider: openai` + `name: text-embedding-3-small` → `model="text-embedding-3-small"`
  - `provider: azure_openai` + `name: my-deployment` → `model="azure/my-deployment"`, `api_base=endpoint`
  - `provider: ollama` + `name: nomic-embed-text` → `model="ollama/nomic-embed-text"`, `api_base=endpoint`
- `api_key`, `endpoint` → passed through as `api_key`, `api_base`
- `api_version` → passed through for Azure

**What is removed**:
- All SK `TextEmbedding` imports from `tool_initializer.py`
- No `SKEmbeddingAdapter` (SK embedding classes no longer used)
- No `AFEmbeddingAdapter` / `af_embedding_adapter.py` (AF has no confirmed embedding API)
- `use_native_embeddings` field removed from `AgentFrameworkConfig`

**Alternatives Considered**:
- Keep SK embedding classes as default adapter: Rejected (SK is being deprecated; adds unnecessary dependency)
- Per-backend embedding adapters (SK, AF, ADK): Rejected (neither ADK nor AF has confirmed embedding support; multiple adapters for no benefit)
- No protocol, call LiteLLM inline: Rejected (protocol enables testability via mocking)

## R4: Tool Adapter Patterns Per Backend

**Decision**: Create per-backend tool adapter modules (`adk_tool_adapters.py`, `af_tool_adapters.py`) following the existing `tool_adapters.py` (Claude) pattern.

**Tool Type Mapping**:

| HoloDeck Tool | ADK Adaptation | AF Adaptation |
|---|---|---|
| VectorstoreTool | Plain async callable wrapping `instance.search()` | `FunctionTool(func=search_fn)` |
| HierarchicalDocumentTool | Plain async callable wrapping `instance.search()` | `FunctionTool(func=search_fn)` |
| FunctionTool | Load Python function, pass directly (ADK auto-wraps) | `FunctionTool(func=loaded_fn)` or `@tool` |
| MCPTool (stdio) | `McpToolset(StdioConnectionParams(command, args))` | `MCPStdioTool(name, command, args)` |
| MCPTool (sse/http) | `McpToolset(SseConnectionParams(url))` / `McpToolset(StreamableHTTPConnectionParams(url))` | `MCPStreamableHTTPTool(name, url)` |
| MCPTool (websocket) | `McpToolset(SseConnectionParams(url))` (ADK may not have native WS) | `MCPWebsocketTool(name, url)` |
| SkillTool | Native skill runtime: ADK agent composition with skill instructions + filtered parent tools | Native skill runtime: AF agent delegation with skill instructions + filtered parent tools |

**Alternatives Considered**:
- Single universal adapter module: Rejected (each backend has different tool APIs; separate modules are cleaner)
- Wrapping all tools as MCP servers: Over-engineering; function tools don't need MCP overhead

## R5: Provider Client Selection (Agent Framework)

**Decision**: AF reads `model.provider` directly to select the appropriate client class. No auto-detection or model name prefix matching needed.

**Provider-to-Client Mapping**:

| `model.provider` | AF Client Class |
|---|---|
| `openai` | `OpenAIChatClient` |
| `azure_openai` | `AzureOpenAIChatClient` |
| `anthropic` | `AnthropicClient` |
| `ollama` | `OllamaChatClient` |

When `model.provider` is set, the AF backend directly instantiates the corresponding client. No `sub_provider` field is needed.

**Alternatives Considered**:
- Model name prefix matching with `sub_provider` override: Rejected — `model.provider` already contains the information; prefix matching adds fragile heuristics
- Always require explicit client specification: Rejected — `model.provider` is always present and unambiguous

## R6: Dependency Versioning Strategy

**Decision**: Pin to specific tested versions in optional dependency groups. Label backends as "preview" in documentation.

**Pinned Versions**:
- `google-adk`: Pin to latest stable or RC at time of implementation (e.g., `google-adk>=1.2.0,<2.0.0`)
- `agent-framework-core`: Pin to `==1.0.0rc4` (exact RC pin until GA)
- Provider-specific AF packages: `agent-framework-anthropic`, `agent-framework-ollama` as optional sub-extras

**pyproject.toml additions**:
```toml
[project.optional-dependencies]
google-adk = ["google-adk>=1.2.0,<2.0.0"]
agent-framework = [
    "agent-framework-core==1.0.0rc4",
]
agent-framework-anthropic = [
    "agent-framework-core==1.0.0rc4",
    "agent-framework-anthropic==1.0.0rc4",
]
agent-framework-ollama = [
    "agent-framework-core==1.0.0rc4",
    "agent-framework-ollama==1.0.0rc4",
]
```

**Note**: Version ranges above are placeholders. Exact package names and available versions must be validated as a Phase 0 prerequisite task before implementation begins. See plan.md for the validation task.

**Alternatives Considered**:
- Wait for GA: Blocks feature indefinitely; RC APIs are stable enough for preview
- Unpinned ranges: Risk of silent breakage from RC API changes

## R7: SkillTool Design (Replacing PromptTool)

**Decision**: Replace the unimplemented `PromptTool` (type: prompt) with `SkillTool` (type: skill) following the [Agent Skills specification](https://agentskills.io/specification). Skills use each backend's native skill/sub-agent runtime.

**Rationale**: `PromptTool` was defined in the Pydantic schema but never implemented — no backend initializes, adapts, or executes prompt tools. The concept (Jinja2 template → LLM call) conflated template rendering with agentic invocation. The Agent Skills spec provides a cleaner model: skills are sub-agent invocations with their own instructions and scoped tool access, running on the parent's backend via its native runtime.

**Key Design Decisions**:
- **Hybrid form**: Inline (instructions in agent.yaml) for simple skills; file-based (path to skill directory with SKILL.md) for complex skills with scripts/references/assets
- **Tool access**: Skills reference parent tools by name via `allowed_tools` in agent.yaml. `None` = no tool access. SKILL.md's `allowed-tools` frontmatter is for the backend's native tool permissions and is NOT merged.
- **Backend inheritance**: Skills run on the same backend as the parent agent. No cross-backend skill invocation. Each backend uses its native skill/sub-agent runtime.
- **SKILL.md parsing**: Each backend (ADK, AF, Claude) handles SKILL.md content natively via its own skill runtime. HoloDeck only needs to parse the YAML frontmatter (via existing PyYAML dependency) for config-time validation (`name`, `description` required fields). No dedicated `skill_loader.py` needed.
- **No model override**: Skills inherit the parent agent's model entirely. No model field on SkillTool.
- **Name format**: Follows Agent Skills spec — lowercase alphanumeric + hyphens, 1-64 chars. This is a different pattern from other tool types which use `^[0-9A-Za-z_]+$`.
- **SK excluded**: SK backend is planned for deprecation and receives no SkillTool support.

**Alternatives Considered**:
- Keep PromptTool and implement it: Rejected — Jinja2 template rendering is too limited; skills are a strictly better abstraction
- Skills as a separate top-level config (not in `tools`): Rejected — skills are invoked as tools by the agent, so they belong in the tools list
- Allow skills to define their own tools (not just reference parent tools): Rejected — creates tool lifecycle complexity; skills should compose from the parent's tool set

## R8: Backend/Provider Decoupling

**Decision**: Introduce a separate `BackendEnum` (`semantic_kernel`, `claude`, `google_adk`, `agent_framework`) alongside the existing `ProviderEnum`. Add an optional top-level `backend` field to agent configuration. When omitted, the backend is auto-detected from `model.provider`.

**Rationale**: The original design overloaded `model.provider` with two independent concerns:
1. **Provider** — which LLM API serves the model (authentication, API format)
2. **Backend** — which agent runtime executes the agent (tool handling, sessions, orchestration)

This prevented valid combinations like "run ADK with an OpenAI model" or "run Agent Framework with an Anthropic model."

**Default Routing Table**:

| Provider | Default Backend | Rationale |
|---|---|---|
| `openai` | `agent_framework` | AF is the recommended OpenAI runtime; SK planned for deprecation |
| `azure_openai` | `agent_framework` | AF has native Azure OpenAI client support |
| `anthropic` | `claude` | Claude Agent SDK is the native Anthropic runtime |
| `ollama` | `claude` | Claude SDK supports Ollama models natively |
| `google` | `google_adk` | ADK is the native Google/Gemini runtime |

**Breaking Change**: `openai`, `azure_openai`, and `ollama` previously defaulted to `semantic_kernel`. Users who depend on SK behavior must add `backend: semantic_kernel` explicitly. Migration path: add `backend: semantic_kernel` to existing agent.yaml files that use these providers and rely on SK-specific behavior.

**Backend/Provider Compatibility Matrix**:

| Backend | Compatible Providers |
|---|---|
| `semantic_kernel` | `openai`, `azure_openai`, `ollama` |
| `claude` | `anthropic` |
| `google_adk` | `google`, `openai`, `azure_openai`, `anthropic`, `ollama` (via LiteLLM) |
| `agent_framework` | `openai`, `azure_openai`, `anthropic`, `ollama` |

**Alternatives Rejected**:
- Nested `model.backend` field: Rejected — `backend` is an agent-level concern (runtime), not a model-level concern (API)
- Required `backend` field: Rejected — auto-detection provides good defaults for the common case; explicit field is for advanced cross-provider scenarios
- Provider-as-backend (status quo): Rejected — prevents cross-provider backend selection and creates artificial provider values that don't represent LLM APIs
