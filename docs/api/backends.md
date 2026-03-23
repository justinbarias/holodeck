# Backend Abstraction Layer

The backend abstraction layer provides a provider-agnostic interface for agent
execution. Downstream consumers (test runner, chat session, serve endpoint)
depend only on the protocols defined in `holodeck.lib.backends.base` -- no
provider-specific types leak through.

## Routing

`BackendSelector` inspects `model.provider` and instantiates the correct
backend automatically:

| Provider                        | Backend        |
|---------------------------------|----------------|
| `openai`, `azure_openai`, `ollama` | `SKBackend`    |
| `anthropic`                     | `ClaudeBackend` |

---

## `holodeck.lib.backends.base` -- Core Protocols & Data Classes

Defines the provider-agnostic contracts that every backend must satisfy and the
unified result types returned to callers.

### ExecutionResult

::: holodeck.lib.backends.base.ExecutionResult
    options:
      docstring_style: google
      show_source: true

### ToolEvent

::: holodeck.lib.backends.base.ToolEvent
    options:
      docstring_style: google
      show_source: true

### AgentSession

::: holodeck.lib.backends.base.AgentSession
    options:
      docstring_style: google
      show_source: true

### AgentBackend

::: holodeck.lib.backends.base.AgentBackend
    options:
      docstring_style: google
      show_source: true

### ContextGenerator

::: holodeck.lib.backends.base.ContextGenerator
    options:
      docstring_style: google
      show_source: true

### Exceptions

::: holodeck.lib.backends.base.BackendError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.backends.base.BackendInitError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.backends.base.BackendSessionError
    options:
      docstring_style: google
      show_source: true

::: holodeck.lib.backends.base.BackendTimeoutError
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.lib.backends.selector` -- Backend Routing

Routes an `Agent` configuration to the correct backend based on
`model.provider`.

### BackendSelector

::: holodeck.lib.backends.selector.BackendSelector
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.lib.backends.sk_backend` -- Semantic Kernel Backend

Wraps the existing `AgentFactory` / `AgentThreadRun` infrastructure behind the
provider-agnostic backend interfaces. Handles OpenAI, Azure OpenAI, and Ollama
providers.

### SKBackend

::: holodeck.lib.backends.sk_backend.SKBackend
    options:
      docstring_style: google
      show_source: true

### SKSession

::: holodeck.lib.backends.sk_backend.SKSession
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.lib.backends.claude_backend` -- Claude Agent SDK Backend

Implements the backend for `provider: anthropic`. Single-turn invocations use
the top-level `query()` SDK function; multi-turn chat sessions use
`ClaudeSDKClient`.

### ClaudeBackend

::: holodeck.lib.backends.claude_backend.ClaudeBackend
    options:
      docstring_style: google
      show_source: true

### ClaudeSession

::: holodeck.lib.backends.claude_backend.ClaudeSession
    options:
      docstring_style: google
      show_source: true

### build_options

::: holodeck.lib.backends.claude_backend.build_options
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.lib.backends.tool_adapters` -- Claude SDK Tool Adapters

Wraps HoloDeck vectorstore and hierarchical-document tools as `@tool`-decorated
functions, bundles them into an in-process MCP server, and provides a factory
for `ClaudeBackend` to call during initialization.

### VectorStoreToolAdapter

::: holodeck.lib.backends.tool_adapters.VectorStoreToolAdapter
    options:
      docstring_style: google
      show_source: true

### HierarchicalDocToolAdapter

::: holodeck.lib.backends.tool_adapters.HierarchicalDocToolAdapter
    options:
      docstring_style: google
      show_source: true

### create_tool_adapters

::: holodeck.lib.backends.tool_adapters.create_tool_adapters
    options:
      docstring_style: google
      show_source: true

### build_holodeck_sdk_server

::: holodeck.lib.backends.tool_adapters.build_holodeck_sdk_server
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.lib.backends.mcp_bridge` -- MCP Configuration Bridge

Translates HoloDeck `MCPTool` configurations into Claude Agent SDK
`McpStdioServerConfig` format for subprocess-based MCP servers. Only `stdio`
transport tools are supported.

### build_claude_mcp_configs

::: holodeck.lib.backends.mcp_bridge.build_claude_mcp_configs
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.lib.backends.otel_bridge` -- Observability Bridge

Translates HoloDeck `ObservabilityConfig` into environment variable dicts that
configure OpenTelemetry for the Claude subprocess.

### translate_observability

::: holodeck.lib.backends.otel_bridge.translate_observability
    options:
      docstring_style: google
      show_source: true

---

## `holodeck.lib.backends.validators` -- Startup Validators

Pre-flight checks called by `ClaudeBackend.initialize()` before spawning the
Claude subprocess. These surface configuration errors at startup rather than at
runtime.

### validate_nodejs

::: holodeck.lib.backends.validators.validate_nodejs
    options:
      docstring_style: google
      show_source: true

### validate_credentials

::: holodeck.lib.backends.validators.validate_credentials
    options:
      docstring_style: google
      show_source: true

### validate_embedding_provider

::: holodeck.lib.backends.validators.validate_embedding_provider
    options:
      docstring_style: google
      show_source: true

### validate_tool_filtering

::: holodeck.lib.backends.validators.validate_tool_filtering
    options:
      docstring_style: google
      show_source: true

### validate_working_directory

::: holodeck.lib.backends.validators.validate_working_directory
    options:
      docstring_style: google
      show_source: true

### validate_response_format

::: holodeck.lib.backends.validators.validate_response_format
    options:
      docstring_style: google
      show_source: true
