# Research: Interactive Agent Testing

**Feature**: `007-interactive-chat`
**Date**: 2025-11-22
**Status**: Complete

## Overview

This document consolidates research findings for implementing the interactive chat command using Semantic Kernel for agent execution and OpenTelemetry for observability.

---

## 1. Agent Execution Runtime

### Decision: Semantic Kernel Chat Completion API

Use Semantic Kernel's built-in chat completion services with conversation history management.

### Rationale

- **Native chat support**: Semantic Kernel provides `ChatHistory` class for multi-turn conversations
- **Tool integration**: Plugins/functions integrate seamlessly with chat completions
- **Provider abstraction**: Supports OpenAI, Anthropic, Azure OpenAI through unified interface
- **Streaming support**: Built-in streaming for real-time responses
- **Already integrated**: Project dependency (semantic-kernel>=1.37.1)

### Implementation Notes

**Core Components**:
- `Kernel`: Main orchestration object for agent execution
- `ChatHistory`: Maintains conversation context across turns
- `ChatCompletionService`: LLM provider interface (Anthropic, OpenAI, etc.)
- `KernelFunction`: Tool/plugin execution wrapper
- `FunctionChoiceBehavior`: Controls automatic tool execution

**Key Pattern**:
```python
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion

# Initialize kernel with agent config
kernel = Kernel()
chat_service = AnthropicChatCompletion(
    model_id=agent_config.model.name,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
kernel.add_service(chat_service)

# Add tools/plugins
for tool in agent_config.tools:
    kernel.add_plugin(load_tool(tool))

# Maintain conversation history
history = ChatHistory()
history.add_system_message(agent_config.instructions)

# Interactive loop
while True:
    user_input = get_user_input()
    history.add_user_message(user_input)

    # Execute with streaming
    response = await kernel.invoke_stream(
        chat_service,
        chat_history=history,
        settings=chat_settings
    )

    # Stream response and tool calls
    async for chunk in response:
        display_chunk(chunk)

    history.add_assistant_message(response.content)
```

### Alternatives Considered

- **LangChain**: More feature-rich but heavier dependency, different abstraction model
- **Direct LLM SDK calls**: Would require reimplementing conversation management and tool orchestration
- **Custom agent framework**: Unnecessary complexity when Semantic Kernel already provides needed functionality

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
- **Observability**: OpenTelemetry with Semantic Kernel native instrumentation
- **CLI Framework**: Click (existing) + built-in input() for interactive prompts
- **Validation**: Custom pipeline architecture (extensible for future safety filters)
- **Context Management**: Semantic Kernel ChatHistory with size monitoring

**Key Architectural Decisions**:
1. Leverage Semantic Kernel's native capabilities (chat, tools, observability)
2. Keep terminal interface simple for MVP (upgrade to rich UI later if needed)
3. Design validation pipeline for extensibility (prompt injection, content filters)
4. In-memory conversation storage (file persistence deferred to P3)
5. Stream tool execution events inline with conversation flow

**No unresolved NEEDS CLARIFICATION items remain.** All technical decisions documented with rationale.
