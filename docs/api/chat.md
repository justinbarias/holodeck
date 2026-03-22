# Chat API Reference

The chat subsystem provides interactive, multi-turn conversation capabilities
for HoloDeck agents. It coordinates message validation, agent execution via
the provider-agnostic backend layer, streaming responses, tool execution
progress tracking, and session lifecycle management.

## Module: `holodeck.chat.executor`

Orchestrates agent execution for chat sessions using the backend abstraction
layer (`AgentBackend` / `AgentSession`). Supports both synchronous turn-based
and streaming response modes, with lazy backend initialization and optional
task-bound session wrapping for HTTP server contexts.

### AgentResponse

::: holodeck.chat.executor.AgentResponse
    options:
      docstring_style: google
      show_source: true

### AgentExecutor

::: holodeck.chat.executor.AgentExecutor
    options:
      docstring_style: google
      show_source: true

## Module: `holodeck.chat.message`

Validates user messages before they reach the agent, enforcing content
standards such as empty-message detection, size limits, control-character
filtering, and UTF-8 validation.

### MessageValidator

::: holodeck.chat.message.MessageValidator
    options:
      docstring_style: google
      show_source: true

## Module: `holodeck.chat.session`

Manages chat session lifecycle and state, coordinating between message
validation, agent execution, token tracking, and session statistics.

### ChatSessionManager

::: holodeck.chat.session.ChatSessionManager
    options:
      docstring_style: google
      show_source: true

## Module: `holodeck.chat.streaming`

Streams tool execution events to callers in real time, allowing UIs to
display progress as tools start, run, and complete (or fail).

### ToolExecutionStream

::: holodeck.chat.streaming.ToolExecutionStream
    options:
      docstring_style: google
      show_source: true

### ToolEvent and ToolEventType

`ToolEvent` and `ToolEventType` are re-exported from `holodeck.models.tool_event`.
See the [Models API Reference](models.md#tool-execution-and-events) for full documentation.

## Module: `holodeck.chat.progress`

Tracks and displays chat session progress with animated spinners and
adaptive status output (inline for default mode, rich panel for verbose mode).

### ChatProgressIndicator

::: holodeck.chat.progress.ChatProgressIndicator
    options:
      docstring_style: google
      show_source: true
