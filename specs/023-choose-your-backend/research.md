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
- Multi-model: Model strings pass directly (e.g., `"gemini-2.5-flash"`, `"openai/gpt-4o"` via LiteLLM)

**Alternatives Considered**:
- Direct Gemini API (google-genai): Too low-level, no tool loop, no MCP support
- LangChain Google integration: Adds unnecessary dependency layer

## R2: Microsoft Agent Framework API Surface

**Decision**: Use `BaseChatClient.as_agent()` pattern with provider-specific client classes (`OpenAIChatClient`, `AzureOpenAIChatClient`, `AnthropicClient`, `OllamaChatClient`).

**Rationale**: AF's `agent.run()` returns `AgentResponse` (non-streaming) or `ResponseStream[AgentResponseUpdate, AgentResponse]` (streaming), mapping directly to `ExecutionResult`. The `@tool` decorator and `FunctionTool` class handle function tools. MCP tools are first-class via `MCPStdioTool`, `MCPStreamableHTTPTool`, `MCPWebsocketTool`.

**Key API Mapping**:
- Client creation: Provider-specific (e.g., `OpenAIChatClient(api_key, model)`)
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

**Decision**: Create an `EmbeddingService` protocol with `embed_batch(texts) -> list[list[float]]` method. Wrap existing SK `TextEmbedding` classes as `SKEmbeddingAdapter`. Optionally provide `AFEmbeddingAdapter` wrapping AF's `OpenAIEmbeddingClient`.

**Rationale**: The current `tool_initializer.py` creates SK-specific embedding instances in `create_embedding_service()`. Extracting a protocol:
1. Removes hard SK dependency for non-SK backends
2. Enables AF-native embeddings via `AFEmbeddingAdapter`
3. Keeps SK as the default (no breaking change for existing users)
4. Future-proofs for additional embedding providers

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

**Alternatives Considered**:
- Keep SK embedding classes as-is for all backends: Rejected (creates unnecessary SK dependency for ADK/AF users)
- Use a third-party embedding library (e.g., sentence-transformers): Adds dependency, SK/AF already have embedding clients

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
| PromptTool | Render Jinja2 template → plain callable returning rendered text | Render Jinja2 template → `FunctionTool` wrapping render |

**Alternatives Considered**:
- Single universal adapter module: Rejected (each backend has different tool APIs; separate modules are cleaner)
- Wrapping all tools as MCP servers: Over-engineering; function tools don't need MCP overhead

## R5: Provider Client Auto-Detection (Agent Framework)

**Decision**: Use model name prefix matching with an explicit `sub_provider` override in `AgentFrameworkConfig`.

**Auto-Detection Rules**:
```
gpt-*, o1*, o3*, o4*        → OpenAIChatClient
claude-*                     → AnthropicClient
gemini-*                     → OllamaChatClient (via LiteLLM) or error
llama-*, mistral-*, phi-*    → OllamaChatClient
*                            → Fall back to sub_provider or error
```

**When `endpoint` contains "azure"** → `AzureOpenAIChatClient` (overrides model name detection)

**When `sub_provider` is explicitly set** → Use that client directly (overrides auto-detection)

**Alternatives Considered**:
- Always require `sub_provider`: Too verbose for common cases (gpt-4o → obviously OpenAI)
- Use AF's own model registry: AF doesn't have one; client selection is manual

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

**Alternatives Considered**:
- Wait for GA: Blocks feature indefinitely; RC APIs are stable enough for preview
- Unpinned ranges: Risk of silent breakage from RC API changes
