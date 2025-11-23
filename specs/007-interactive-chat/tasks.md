# Implementation Tasks: Interactive Agent Testing (holodeck chat)

**Feature**: Interactive Agent Testing (`holodeck chat`)
**Focus**: User Stories 1-2 + Error Handling (P1/P2 scope), P3 session persistence optional
**Branch**: `007-interactive-chat`
**Date Generated**: 2025-11-22
**Planning**: `/speckit.plan` from `/specs/007-interactive-chat/plan.md`
**Approach**: TDD for models/runtime/CLI with Semantic Kernel mocks; observability behind flag

---

## Executive Summary

This task list delivers the terminal-based `holodeck chat` command with full
multi-turn conversation support, validation, streaming tool events, and optional
OpenTelemetry instrumentation. Tasks are grouped by dependency, with tests
written first for models, runtime, and CLI layers. P3 session persistence is
scoped as optional follow-up.

- **Total Tasks**: 37 (22 implementation + 12 testing + 3 documentation/quality)
- **Primary Scope (P1/P2)**: Basic chat loop, multi-turn context, validation,
  streaming tool execution, CLI UX, observability flag
- **Deferred (P3)**: Session save/resume plumbing (stubbed for future)
- **Independent Test Criteria**: CLI invocation succeeds, multi-turn context
  preserved, validation blocks bad input, tool events stream, observability
  toggle works

---

## Phase 0: Pre-flight and Foundations

_Ensure specs, contracts, and dependencies are aligned before coding._

### Phase 0 Goals

- Confirm specification alignment (spec, research, contracts)
- Verify dependency expectations (Semantic Kernel, OpenTelemetry)
- Prepare fixtures for CLI and runtime tests

### Phase 0 Independent Test Criteria

- Contract files referenced in tests without missing paths
- Test fixtures load valid agent YAML for chat scenarios

### Tasks

- [x] T201 Read/align spec, plan, research, and contracts to extract acceptance
      criteria for CLI, runtime, validation, and observability
- [x] T202 Verify `pyproject.toml` has required deps (semantic-kernel,
      opentelemetry) and note gaps for implementation tasks
- [x] T203 Add/reuse agent YAML fixtures for chat scenarios (happy path,
      missing tool, invalid config) under `tests/fixtures/agents/`

---

## Phase 1: Module Scaffolding and Error Surface

_Create runtime module layout and error types used across tasks._

### Phase 1 Goals

- Agent runtime package scaffolded (`src/holodeck/agent/`)
- Error hierarchy supports chat-specific failures

### Phase 1 Independent Test Criteria

- Importing `holodeck.chat` modules succeeds
- Error classes surfaced through CLI exceptions mapping

### Tasks

- [x] T204 [P] Scaffold `src/holodeck/chat/{executor.py,session.py,message.py,streaming.py,__init__.py}`
      with placeholders referenced by tests
- [x] T205 [P] Extend `holodeck.lib.errors` and `holodeck.cli.exceptions` with
      chat error classes and exit code mapping per contract

---

## Phase 2: Models and Validation (Tests First)

_Define chat models and validation pipeline with Pydantic + custom sanitizers._

### Phase 2 Goals

- ChatSession, Message, ToolExecution, TokenUsage, ChatConfig models validated
- Validation pipeline rejects bad input and sanitizes outputs

### Phase 2 Independent Test Criteria

- Pydantic validators enforce limits (10K chars, tool call rules)
- Sanitization strips ANSI/control sequences and truncates long outputs

### Phase 2 - Tests First

- [x] T206 [P] Write model tests for ChatSession/Message/ChatConfig in
      `tests/unit/models/test_chat.py`
- [x] T207 [P] Write model tests for ToolExecution/TokenUsage in
      `tests/unit/models/test_tool_execution.py`
- [x] T208 [P] Write validation pipeline tests (empty, size limit, control
      chars, UTF-8, sanitization) in `tests/unit/lib/test_validation.py`

### Phase 2 - Implementation

- [x] T209 [P] Implement `src/holodeck/models/chat.py` per data-model.md
- [x] T210 [P] Implement `src/holodeck/models/tool_execution.py` and
      `src/holodeck/models/token_usage.py` per data-model.md
- [x] T211 Implement validation pipeline and sanitizers in
      `src/holodeck/lib/validation.py`; export helpers in `lib/__init__.py`

---

## Phase 3: Agent Runtime (Tests First)

_Build executor, session manager, streaming, and message validation plumbing._

### Phase 3 Goals

- AgentExecutor reuses AgentFactory (Semantic Kernel ChatCompletionAgent) with tool execution streaming
- ChatSessionManager tracks history, warnings, lifecycle
- Message validation pipeline enforced before execution

### Phase 3 Independent Test Criteria

- Executor returns AgentResponse with tool metadata and token usage
- Session warns at 80% of max_messages and handles exit commands cleanly
- Message validator rejects empty/oversized/invalid UTF-8 messages

### Phase 3 - Tests First

- [x] T213 Write executor unit tests with mocked AgentFactory/Semantic Kernel in
      `tests/unit/agent/test_executor.py`
- [x] T214 Write session manager tests (start/process/warn/terminate) in
      `tests/unit/agent/test_session.py`
- [x] T215 Write message validator tests (pipeline integration, error messages)
      in `tests/unit/agent/test_message.py`
- [x] T216 Write streaming tests (ToolEvent lifecycle, verbose vs standard) in
      `tests/unit/agent/test_streaming.py`

### Phase 3 - Implementation

- [x] T217 Implement `AgentExecutor` in `src/holodeck/chat/executor.py`
      reusing `holodeck.lib.test_runner.agent_factory.AgentFactory`
- [x] T218 Implement `ChatSessionManager` in `src/holodeck/chat/session.py`
- [x] T219 Implement validation orchestrator in `src/holodeck/chat/message.py`
      using `lib.validation` pipeline
- [x] T220 Implement `ToolExecutionStream` in `src/holodeck/chat/streaming.py`
- [x] T221 Export runtime APIs in `src/holodeck/chat/__init__.py`

---

## Phase 4: CLI Command (Tests First)

_Expose chat runtime via Click command with correct UX and exit codes._

### Phase 4 Goals

- `holodeck chat` command wired to runtime with options from contract
- Clear error handling and exit codes (0,1,2,130)
- Graceful exit on exit/quit/interrupt

### Phase 4 Independent Test Criteria

- CLI accepts agent path, --verbose, --observability, --max-messages
- Error output matches contract for config/agent errors and validation failures

### Phase 4 - Tests First

- [x] T222 Write CLI tests for happy path, exit commands, option parsing in
      `tests/unit/cli/commands/test_chat.py`
- [x] T223 Write CLI error-handling tests (invalid path, bad config, runtime
      failure, interrupt) in `tests/unit/cli/commands/test_chat_errors.py`

### Phase 4 - Implementation

- [x] T224 Implement `src/holodeck/cli/commands/chat.py` (Click command +
      REPL loop + streaming display)
- [x] T225 Register chat command in `src/holodeck/cli/main.py`
- [x] T226 Map agent errors to exit codes/messages in
      `src/holodeck/cli/exceptions.py`
- [x] T227 Update CLI help/docs strings to reference chat usage and flags

---

## Phase 5: Observability Toggle

_Enable optional OpenTelemetry instrumentation for chat sessions._

### Phase 5 Goals

- Observability flag (`--observability`) initializes OTEL instrumentation
- Token usage and tool executions traced per research.md guidance

### Phase 5 Independent Test Criteria

- Observability disabled by default; enabling sets up tracer provider
- Traces include session span and per-turn spans (mock exporter in tests)

### Tasks

- [ ] T228 Write observability tests (flag handling, exporter wiring, safe
      defaults) in `tests/unit/lib/test_observability.py`
- [ ] T229 Implement `src/holodeck/lib/observability.py` and integrate in
      executor/session startup paths

---

## Phase 6: Integration Tests

_Validate end-to-end chat flows with multi-turn context and tool streaming._

### Phase 6 Goals

- CLI end-to-end covers basic chat, multi-turn context, validation errors
- Tool execution stream visible in standard and verbose modes

### Phase 6 Independent Test Criteria

- `pytest tests/integration/test_chat_integration.py -v` passes with mocked
  Semantic Kernel service and fixture agent configs

### Tasks

- [ ] T230 Write integration test: happy path multi-turn with tool events and
      context retention
- [ ] T231 Write integration test: validation failure, config error, tool
      failure recovery paths
- [ ] T232 Write integration test: observability flag ensures tracer invoked
      without leaking content (mock exporter)

---

## Phase 7: Session Persistence (P3 Optional)

_Prepare stubs for future save/resume without blocking MVP._

### Phase 7 Goals

- Optional save/resume hooks do not impact MVP runtime

### Tasks

- [ ] T233 Add stubbed persistence interfaces (no-op) and guarded CLI flags for
      resume/save, covered by unit tests skipping by default (`@pytest.mark.skip`)

---

## Phase 8: Documentation and DX

_Document usage, quickstart, and contracts._

### Phase 8 Goals

- CLI help updated, docs provide quickstart and troubleshooting

### Phase 8 Independent Test Criteria

- Documentation references correct command syntax and flags

### Tasks

- [x] T234 Add `specs/007-interactive-chat/quickstart.md` outlining CLI usage,
      verbose mode, observability flag, and sample transcript
- [x] T235 Update `docs/` or README snippet for new chat command (link to spec
      and quickstart)

---

## Phase 9: Quality Gates

_Run full quality suite and ensure release readiness._

### Phase 9 Goals

- Formatting, linting, typing, security checks clean
- Coverage threshold (>=80%) met across new modules

### Tasks

- [ ] T236 Run `make format` and `make format-check`; fix style issues
- [ ] T237 Run `make lint`, `make type-check`, `make security`; resolve
      findings and type errors
- [ ] T238 Run `make test-coverage` (or equivalent) to confirm coverage >=80%
      including new runtime and CLI code

---

## Dependency Notes

```
Phase 0 → Phase 1 → Phase 2 Tests → Phase 2 Impl → Phase 3 Tests → Phase 3 Impl
       → Phase 4 Tests → Phase 4 Impl → Phase 5 → Phase 6 → Phase 8 → Phase 9
                                   ↘ (optional) Phase 7 (persistence stubs)
```

**Critical Path**: T201 → T204 → T206-T208 → T209-T211 → T213-T216 → T217-T221
→ T222-T223 → T224-T227 → T228-T229 → T230-T232 → T236-T238

---

## TDD Workflow Reference

- Write unit tests first for models, runtime, CLI (`tests/unit/...`), ensure
  failures
- Implement minimal code to satisfy tests, then refactor with type/lint checks
- Add integration tests after unit tests pass; iterate until green suite

---

## Success Criteria for MVP (P1/P2)

1. `holodeck chat <agent.yaml>` starts in <1 second and shows prompt
2. Multi-turn conversations preserve context and warn at 80% message threshold
3. Tool execution events stream inline; verbose mode shows parameters/results
4. Validation rejects empty/oversized/invalid inputs with friendly errors
5. Observability flag enables OTEL tracing without leaking content by default
6. All tests pass with coverage ≥80% and quality gates clean

---
