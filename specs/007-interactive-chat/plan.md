# Implementation Plan: Interactive Agent Testing

**Branch**: `007-interactive-chat` | **Date**: 2025-11-22 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/007-interactive-chat/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement an interactive chat command (`holodeck chat agent.yaml`) that enables developers to test and debug agents in real-time through a terminal-based interface. The command will load agent configurations, establish a REPL-style chat session, maintain conversation context across multiple turns, stream tool execution events, and handle errors gracefully with comprehensive input validation and output sanitization.

## Technical Context

**Language/Version**: Python 3.10+ (as defined in pyproject.toml)
**Primary Dependencies**:

- Click 8.0+ (CLI framework)
- Semantic Kernel 1.37+ (agent execution runtime)
- Pydantic 2.0+ (configuration validation)
- PyYAML 6.0+ (configuration loading)
- Anthropic 0.72+ (LLM provider integration)
- Azure AI Evaluation 1.13+ (evaluation metrics)

**Storage**: In-memory only for MVP (conversation history cleared on session exit)
**Testing**: pytest (unit tests with @pytest.mark.unit, integration tests with @pytest.mark.integration)
**Target Platform**: Cross-platform CLI (macOS, Linux, Windows with terminal support)
**Project Type**: Single project (CLI-based agent development tool)
**Performance Goals**:

- Session startup: <1 second
- Response processing: <5 seconds (excluding LLM latency)
- Multi-turn conversations: 10+ exchanges without degradation
- Session stability: 30+ minutes without memory leaks

**Constraints**:

- Terminal-based interaction only (no web UI)
- Single-agent sessions (no multi-agent orchestration)
- Input size limit: ~10K characters per message
- Context window management required for long conversations

**Scale/Scope**:

- MVP supports individual developers testing agents locally
- No concurrent session support required
- Conversation history: 50 message warning threshold

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

### Principle I: No-Code-First Agent Definition

**Status**: ✅ **PASS** - This feature enables testing of agents defined via YAML. The chat command loads existing agent configurations without requiring code changes.

**Evaluation**: Interactive chat is a testing tool that operates on declarative YAML configurations. Users do not write Python code to use the chat feature; they provide YAML paths.

### Principle II: MCP for API Integrations

**Status**: ✅ **PASS** - No new API integrations introduced. Chat command executes existing agent tools (which may include MCP servers).

**Evaluation**: This feature does not create new API tool types. It leverages the existing tool execution framework defined in agent YAML files.

### Principle III: Test-First with Multimodal Support

**Status**: ✅ **PASS** - Chat feature supports agents with multimodal capabilities defined in their configurations.

**Evaluation**: The chat interface will support sending messages to agents that have multimodal tools configured. File inputs can be tested via the agent's tool execution (e.g., vector search with document inputs). Unit and integration tests will validate chat session behavior.

### Principle IV: OpenTelemetry-Native Observability

**Status**: ⚠️ **NEEDS CLARIFICATION** - Observability integration approach needs definition.

**Evaluation**: Chat sessions should emit traces/metrics for LLM calls, tool executions, and session events. Research needed to determine:

- How to instrument interactive sessions with OpenTelemetry
- Whether to log conversation history for debugging
- Cost tracking for interactive LLM usage

### Principle V: Evaluation Flexibility with Model Overrides

**Status**: ✅ **PASS** - Chat command respects agent configuration's LLM settings.

**Evaluation**: Agents loaded in chat mode use their configured LLM provider and model settings. No evaluation metrics are executed during interactive chat (evaluation is separate via `holodeck test`).

### Architecture Constraints

**Status**: ✅ **PASS** - Chat feature integrates with Agent Engine without tight coupling.

**Evaluation**:

- Uses Agent Engine for LLM execution and tool calls
- Does not interfere with Evaluation Framework or Deployment Engine
- CLI command structure maintains separation of concerns

### Code Quality & Testing Discipline

**Status**: ✅ **PASS** - Standard testing and quality requirements apply.

**Evaluation**:

- Python 3.10+ target enforced via pyproject.toml
- pytest unit and integration tests required
- Type hints, docstrings, and Google Python Style Guide compliance
- Pre-commit hooks (Black, Ruff, MyPy, Bandit) will validate code

## Project Structure

### Documentation (this feature)

```
specs/007-interactive-chat/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
src/holodeck/
├── cli/
│   ├── commands/
│   │   ├── chat.py              # NEW: Chat command implementation
│   │   ├── init.py              # Existing
│   │   ├── test.py              # Existing
│   │   └── config.py            # Existing
│   ├── main.py                  # UPDATE: Register chat command
│   └── exceptions.py            # UPDATE: Add chat-specific exceptions
│
├── agent/                       # NEW: Agent execution runtime
│   ├── __init__.py
│   ├── executor.py              # Agent execution orchestration
│   ├── session.py               # Chat session state management
│   ├── message.py               # Message handling and validation
│   └── streaming.py             # Tool execution event streaming
│
├── models/
│   ├── agent.py                 # Existing (AgentConfig)
│   ├── llm.py                   # Existing (LLMConfig)
│   └── chat.py                  # NEW: Chat session models
│
├── lib/
│   ├── errors.py                # UPDATE: Add chat-specific errors
│   ├── validation.py            # NEW: Input/output sanitization
│   └── observability.py         # NEW: OpenTelemetry instrumentation
│
└── config/
    └── loader.py                # Existing (YAML config loading)

tests/
├── unit/
│   ├── cli/
│   │   └── commands/
│   │       └── test_chat.py     # NEW: Chat command unit tests
│   ├── agent/
│   │   ├── test_executor.py     # NEW: Agent executor tests
│   │   ├── test_session.py      # NEW: Session management tests
│   │   ├── test_message.py      # NEW: Message validation tests
│   │   └── test_streaming.py    # NEW: Event streaming tests
│   └── lib/
│       ├── test_validation.py   # NEW: Input/output sanitization tests
│       └── test_observability.py # NEW: Observability tests
│
└── integration/
    └── test_chat_integration.py # NEW: End-to-end chat tests
```

**Structure Decision**: Single project structure (Option 1). HoloDeck is a CLI-based tool with a unified codebase. The new `agent/` module provides the runtime for executing agents during interactive chat sessions and will be reused for test execution. The `cli/commands/chat.py` module implements the chat command interface, while `agent/` handles the business logic of agent execution and conversation management.

## Complexity Tracking

_Fill ONLY if Constitution Check has violations that must be justified_

**Status**: No violations requiring justification. One area (OpenTelemetry observability) marked as "NEEDS CLARIFICATION" will be resolved through Phase 0 research.
