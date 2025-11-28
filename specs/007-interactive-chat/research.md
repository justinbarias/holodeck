# Research: Interactive Agent Testing

**Feature**: `007-interactive-chat`
**Date**: 2025-11-22
**Status**: Complete

## Overview

This document consolidates research findings for implementing the interactive chat command using Semantic Kernel for agent execution and OpenTelemetry for observability.

---

## 1. Agent Execution Runtime

### Decision: Reuse `AgentFactory` with Semantic Kernel `ChatCompletionAgent`

Use the existing `holodeck.lib.test_runner.agent_factory.AgentFactory` as the runtime bridge, swapping the chat implementation to Semantic Kernel's `ChatCompletionAgent` (with `ChatHistoryAgentThread`) instead of raw `ChatCompletion` calls. This keeps chat aligned with the test runner and centralizes provider wiring, retries, and history management.

### Rationale

- **Single bridge**: Reuses the same factory used by the test executor, reducing duplicate Semantic Kernel setup and keeping provider parity.
- **ChatCompletionAgent**: Higher-level agent abstraction with built-in threading/history support, better fit for interactive chat than direct `ChatCompletionService` calls.
- **Streaming/tool extraction**: AgentFactory already extracts tool calls and manages retries/timeouts; extending it for chat keeps behavior consistent.
- **Provider abstraction**: AgentFactory already supports OpenAI, Azure OpenAI, and Anthropic via Semantic Kernel services.

### Implementation Notes

**Core Components** (reused):
- `AgentFactory`: Creates kernel + `ChatCompletionAgent`, loads instructions, manages retries/timeouts, extracts tool calls.
- `ChatHistoryAgentThread`: Maintains conversation context for the agent.
- `ChatHistory`: Conversation history container passed into the agent thread.

**Key Pattern (aligned to AgentFactory)**:
```python
from holodeck.lib.test_runner.agent_factory import AgentFactory
from semantic_kernel.contents import ChatHistory

# Initialize factory with AgentConfig (already used by test executor)
factory = AgentFactory(agent_config)

# Maintain history via ChatHistoryAgentThread within AgentFactory
history = ChatHistory()
history.add_user_message(user_input)

# Invoke using the agent_factory (internally uses ChatCompletionAgent)
result = await factory.invoke(user_input)

# result.chat_history contains updated history; result.tool_calls holds tool metadata
```

### Alternatives Considered

- **LangChain**: More feature-rich but heavier dependency, different abstraction model
- **Direct LLM SDK calls**: Would require reimplementing conversation management and tool orchestration
- **Custom agent framework**: Unnecessary complexity when Semantic Kernel already provides needed functionality

---

## 1.1 MCP Tool Integration

### Decision: Use Semantic Kernel MCP Plugins Directly

Integrate MCP tools using Semantic Kernel's native MCP connector (`semantic_kernel.connectors.mcp`). The SK plugins handle complete server lifecycle management automatically.

### Rationale

- **Lifecycle handled**: SK's `MCPPluginBase` manages spawn/connect/close via async context managers
- **Resource cleanup**: Uses `AsyncExitStack` for deterministic cleanup on exit
- **Transport abstraction**: Supports stdio, SSE, WebSocket, HTTP without custom code
- **Tool discovery**: Automatic tool/prompt loading from MCP servers via `list_tools()`/`list_prompts()`
- **Notification handling**: Responds to `tools/list_changed` and `prompts/list_changed` events
- **No custom wrapper needed**: HoloDeck doesn't need to reimplement lifecycle management

### Implementation Notes

**Key SK Plugin Classes**:
- `MCPStdioPlugin`: Local servers via subprocess (command + args)
- `MCPSsePlugin`: Remote servers via Server-Sent Events
- `MCPWebsocketPlugin`: Bidirectional real-time communication
- `MCPStreamableHttpPlugin`: HTTP with streaming response support

**Usage Pattern**:
```python
from semantic_kernel.connectors.mcp import MCPStdioPlugin

# Context manager handles full lifecycle automatically
async with MCPStdioPlugin(
    name="filesystem",
    description="File system access",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
) as plugin:
    # Plugin is connected, tools are loaded
    kernel.add_plugin(plugin)
    result = await kernel.invoke_prompt("Read file.txt")

# Exiting context: server process terminated, resources cleaned up
```

**Lifecycle Methods** (handled by SK):
```python
# Internal to MCPPluginBase:
async def __aenter__(self) -> Self:
    await self.connect()  # Spawns process, initializes session, loads tools
    return self

async def __aexit__(self, exc_type, exc_value, traceback) -> None:
    await self.close()  # Sets stop event, awaits task completion, cleans up
```

**Integration with AgentFactory**:
The `AgentFactory` can load MCP tool configurations from `agent.yaml` and instantiate the appropriate SK plugin based on the `transport` field. No additional lifecycle management code is needed in HoloDeck.

### Alternatives Considered

- **Custom MCP lifecycle management**: Unnecessary duplication of SK's existing implementation
- **Direct mcp-sdk usage**: Would bypass SK's conveniences and require more code
- **Wrapper abstraction**: Adds complexity without value since SK plugins are already well-designed

---

## 2. OpenTelemetry Observability

### Decision: Semantic Kernel Native OpenTelemetry Integration

Use Semantic Kernel's built-in OpenTelemetry instrumentation following GenAI Semantic Conventions.

### Rationale

- **Native support**: Semantic Kernel 1.37+ includes OpenTelemetry instrumentation
- **GenAI conventions**: Follows official OpenTelemetry semantic conventions for generative AI
- **Automatic instrumentation**: LLM calls, tool executions, and prompts automatically traced
- **Standard attributes**: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`, `gen_ai.prompt`, `gen_ai.completion`
- **Cost tracking**: Token usage metrics automatically captured

### Implementation Notes

**Setup**:
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.semantic_kernel import SemanticKernelInstrumentor

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer_provider = trace.get_tracer_provider()

# Add exporters (console for dev, Jaeger/OTLP for production)
tracer_provider.add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Instrument Semantic Kernel
SemanticKernelInstrumentor().instrument()
```

**Key Traces**:
- **Session span**: Covers entire chat session lifecycle
- **Turn spans**: Each user message → agent response cycle
- **LLM call spans**: Individual model invocations with token counts
- **Tool execution spans**: Function/plugin calls with parameters and results

**Metrics Tracked**:
- `gen_ai.client.token.usage` (prompt tokens, completion tokens, total tokens)
- `gen_ai.client.operation.duration` (LLM call latency)
- Session duration, message count, tool execution count

**Privacy Considerations**:
- Conversation content logging: **Disabled by default** (PII concerns)
- Enable via `--debug` flag for troubleshooting
- Tool parameters: **Sanitized** (redact sensitive values like API keys)

### Alternatives Considered

- **Manual instrumentation**: More control but high maintenance burden
- **LangSmith/LangFuse**: Vendor-specific, less portable
- **No observability**: Violates Constitution Principle IV

---

## 3. Terminal Interface

### Decision: Click + Python's input() for MVP

Use Click for command structure and built-in `input()` for interactive prompts. Defer rich terminal features to future iterations.

### Rationale

- **Simplicity**: Minimal dependencies, works cross-platform
- **Click integration**: Already using Click for CLI framework
- **MVP-appropriate**: Meets basic requirements without over-engineering
- **Future extensibility**: Can upgrade to `prompt_toolkit` or `rich` if needed

### Implementation Notes

**Basic REPL Pattern**:
```python
import click

@click.command()
@click.argument('agent_config_path', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Show detailed tool execution')
def chat(agent_config_path: str, verbose: bool) -> None:
    """Start interactive chat session with an agent."""
    # Load agent config
    agent = load_agent(agent_config_path)

    click.echo(f"Starting chat with {agent.name}...")
    click.echo("Type 'exit' or 'quit' to end session.\n")

    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                click.echo("Goodbye!")
                break

            if not user_input:
                click.echo("Please enter a message.")
                continue

            # Process message
            response = await agent.chat(user_input)
            click.echo(f"Agent: {response}")

        except (KeyboardInterrupt, EOFError):
            click.echo("\nGoodbye!")
            break
```

**Tool Execution Display** (verbose mode):
```
You: What's the weather in SF?

[Tool Call] get_weather(location="San Francisco, CA")
[Tool Result] 72°F, Sunny (executed in 0.3s)

Agent: The current weather in San Francisco is 72°F and sunny.
```

### Alternatives Considered

- **prompt_toolkit**: Rich features (syntax highlighting, auto-complete) but adds complexity
- **rich**: Beautiful terminal output but overkill for MVP
- **curses**: Cross-platform issues, high complexity

---

## 4. Input Validation & Sanitization

### Decision: Custom Validation Pipeline with Extensible Architecture

Implement validation as a pipeline of filters with clear extension points for future safety features.

### Rationale

- **Extensibility**: Easy to add prompt injection detection, content filtering later
- **Separation of concerns**: Each validator handles specific concern
- **Testability**: Each validator can be unit tested independently
- **Performance**: Fast-path for common cases (empty input, size limits)

### Implementation Notes

**Architecture**:
```python
from abc import ABC, abstractmethod
from typing import Protocol

class MessageValidator(Protocol):
    """Protocol for message validators."""
    def validate(self, message: str) -> tuple[bool, str | None]:
        """Validate message. Returns (is_valid, error_message)."""
        ...

class ValidationPipeline:
    """Extensible validation pipeline."""

    def __init__(self):
        self.validators: list[MessageValidator] = [
            EmptyMessageValidator(),
            SizeLimitValidator(max_chars=10000),
            ControlCharacterValidator(),
            UTF8Validator(),
        ]

    def add_validator(self, validator: MessageValidator) -> None:
        """Add custom validator to pipeline."""
        self.validators.append(validator)

    def validate(self, message: str) -> tuple[bool, str | None]:
        """Run message through all validators."""
        for validator in self.validators:
            is_valid, error = validator.validate(message)
            if not is_valid:
                return False, error
        return True, None
```

**Built-in Validators**:
- `EmptyMessageValidator`: Reject empty/whitespace-only messages
- `SizeLimitValidator`: Enforce ~10K character limit
- `ControlCharacterValidator`: Strip/reject dangerous control characters
- `UTF8Validator`: Ensure valid UTF-8 encoding

**Output Sanitization**:
```python
import html
import re

def sanitize_tool_output(output: str) -> str:
    """Escape tool outputs to prevent terminal injection."""
    # Strip ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', output)

    # Escape HTML entities (for terminal display safety)
    cleaned = html.escape(cleaned)

    # Truncate extremely long outputs
    if len(cleaned) > 5000:
        cleaned = cleaned[:5000] + "\n... (output truncated)"

    return cleaned
```

**Future Extension Points**:
- `PromptInjectionValidator`: Detect malicious prompts
- `ContentFilterValidator`: Block inappropriate content
- `RateLimitValidator`: Prevent abuse

### Alternatives Considered

- **Library-based validation** (e.g., `pydantic`, `cerberus`): Too heavy for simple text validation
- **LLM-based safety filters**: Expensive, adds latency, better as optional add-on
- **No validation**: Security risk, violates spec requirements

---

## 5. Conversation Context Management

### Decision: Semantic Kernel ChatHistory with Size Monitoring

Use Semantic Kernel's `ChatHistory` class with proactive warnings when approaching context limits.

### Rationale

- **Built-in support**: ChatHistory manages message ordering, role tracking
- **Provider-agnostic**: Works with all LLM providers
- **Context window awareness**: Can track token counts
- **Extensible**: Easy to add summarization or history pruning later

### Implementation Notes

**Context Monitoring**:
```python
from semantic_kernel.contents import ChatHistory

class ManagedChatHistory:
    """ChatHistory wrapper with context limit monitoring."""

    def __init__(self, max_messages: int = 50):
        self.history = ChatHistory()
        self.max_messages = max_messages

    def add_message(self, role: str, content: str) -> None:
        """Add message and check limits."""
        self.history.add_message(role, content)

        # Warn at 80% capacity
        if len(self.history.messages) >= int(self.max_messages * 0.8):
            click.secho(
                f"\n⚠️  Conversation approaching limit "
                f"({len(self.history.messages)}/{self.max_messages} messages). "
                f"Consider starting a new session.\n",
                fg='yellow'
            )

    def clear(self) -> None:
        """Clear history (keep system message)."""
        system_msg = self.history.messages[0]
        self.history.clear()
        self.history.add_message(system_msg.role, system_msg.content)
```

**In-session commands** (future enhancement):
- `/clear`: Clear history but keep session
- `/save <session-id>`: Save session for later resume (P3 feature)
- `/history`: Display conversation summary

### Alternatives Considered

- **Manual list management**: Reinventing the wheel, error-prone
- **Database storage**: Overkill for MVP (in-memory sufficient)
- **Automatic summarization**: Complex, adds latency, defer to future iteration

---

## Summary

**Technology Stack**:
- **Agent Runtime**: Semantic Kernel 1.37+ (ChatHistory, chat completions, plugins)
- **MCP Tools**: Semantic Kernel MCP plugins (stdio, SSE, WebSocket, HTTP transports)
- **Observability**: OpenTelemetry with Semantic Kernel native instrumentation
- **CLI Framework**: Click (existing) + built-in input() for interactive prompts
- **Validation**: Custom pipeline architecture (extensible for future safety filters)
- **Context Management**: Semantic Kernel ChatHistory with size monitoring

**Key Architectural Decisions**:
1. Leverage Semantic Kernel's native capabilities (chat, tools, observability)
2. Use SK MCP plugins directly for tool lifecycle management (no custom wrappers)
3. Keep terminal interface simple for MVP (upgrade to rich UI later if needed)
4. Design validation pipeline for extensibility (prompt injection, content filters)
5. In-memory conversation storage (file persistence deferred to P3)
6. Stream tool execution events inline with conversation flow

**No unresolved NEEDS CLARIFICATION items remain.** All technical decisions documented with rationale.
