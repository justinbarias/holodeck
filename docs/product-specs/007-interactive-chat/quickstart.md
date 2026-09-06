# Quickstart: Interactive Agent Testing

**Feature**: `007-interactive-chat`
**Date**: 2025-11-22
**Audience**: Developers implementing the chat feature

## Overview

This quickstart provides a step-by-step guide for implementing the interactive chat command. Follow these steps in order to build the feature incrementally with continuous validation.

---

## Prerequisites

- Python 3.10+ environment activated
- Dependencies installed: `poetry install`
- Familiarity with Semantic Kernel concepts (Kernel, ChatHistory, plugins)
- Understanding of Click CLI framework

---

## Implementation Sequence

### Phase 1: Core Data Models (Day 1)

**Goal**: Implement Pydantic models for chat sessions, messages, and tool execution.

**Files to Create**:

1. `src/holodeck/models/chat.py`
2. `src/holodeck/models/tool_execution.py`
3. `src/holodeck/models/token_usage.py`

**Steps**:

1. Implement `MessageRole`, `SessionState`, `ToolStatus`, `ToolEventType` enums
2. Implement `TokenUsage` model with validation
3. Implement `ToolExecution` model with sanitization
4. Implement `Message` model with content validation
5. Implement `ChatSession` model with state management
6. Implement `ChatConfig` model with path validation

**Validation**:

```bash
# Run unit tests
pytest tests/unit/models/test_chat.py -v
pytest tests/unit/models/test_tool_execution.py -v

# Type checking
mypy src/holodeck/models/chat.py
```

**Success Criteria**:

- All models pass Pydantic validation
- Field validators enforce constraints (size limits, enum values)
- Type hints pass MyPy strict checks

---

### Phase 2: Input Validation & Sanitization (Day 1-2)

**Goal**: Implement extensible validation pipeline and output sanitization.

**Files to Create**:

1. `src/holodeck/lib/validation.py`

**Steps**:

1. Define `MessageValidator` protocol
2. Implement `ValidationPipeline` class
3. Implement built-in validators:
   - `EmptyMessageValidator`
   - `SizeLimitValidator` (~10K chars)
   - `ControlCharacterValidator`
   - `UTF8Validator`
4. Implement `sanitize_tool_output()` function (ANSI escape removal, HTML escaping)

**Validation**:

```bash
pytest tests/unit/lib/test_validation.py -v

# Test edge cases
pytest tests/unit/lib/test_validation.py::test_empty_message -v
pytest tests/unit/lib/test_validation.py::test_oversized_message -v
pytest tests/unit/lib/test_validation.py::test_terminal_injection -v
```

**Success Criteria**:

- Validators correctly reject invalid inputs
- Sanitization removes terminal escape sequences
- Pipeline is extensible (easy to add new validators)

---

### Phase 3: Agent Executor (Day 2-3)

**Goal**: Implement Semantic Kernel integration for agent execution.

**Files to Create**:

1. `src/holodeck/agent/__init__.py`
2. `src/holodeck/agent/executor.py`

**Steps**:

1. Implement `AgentExecutor.__init__()`:
   - Initialize Semantic Kernel `Kernel`
   - Configure LLM service from `AgentConfig`
   - Load tools/plugins
2. Implement `AgentExecutor.execute_turn()`:
   - Add user message to ChatHistory
   - Invoke kernel with streaming
   - Capture tool executions
   - Track token usage
   - Return `AgentResponse`
3. Implement `AgentExecutor.get_history()`
4. Implement `AgentExecutor.clear_history()`
5. Implement `AgentExecutor.shutdown()`

**Validation**:

```bash
# Unit tests with mocked Semantic Kernel
pytest tests/unit/agent/test_executor.py -v

# Integration test with real LLM (requires API key)
pytest tests/integration/test_agent_executor.py -v --slow
```

**Success Criteria**:

- Agent successfully initializes with valid config
- Multi-turn conversations maintain context
- Tool executions are captured and returned
- Token usage is tracked correctly

---

### Phase 4: Session Manager (Day 3-4)

**Goal**: Implement chat session lifecycle and context management.

**Files to Create**:

1. `src/holodeck/agent/session.py`

**Steps**:

1. Implement `ChatSessionManager.__init__()`
2. Implement `ChatSessionManager.start()`:
   - Create `ChatSession` instance
   - Initialize `AgentExecutor`
   - Set state to ACTIVE
3. Implement `ChatSessionManager.process_message()`:
   - Validate input via `ValidationPipeline`
   - Pass to `AgentExecutor.execute_turn()`
   - Increment message count
   - Return response
4. Implement `ChatSessionManager.should_warn_context_limit()`:
   - Check if message_count >= max_messages \* 0.8
5. Implement `ChatSessionManager.terminate()`:
   - Set state to TERMINATED
   - Shutdown executor

**Validation**:

```bash
pytest tests/unit/agent/test_session.py -v

# Test context limit warnings
pytest tests/unit/agent/test_session.py::test_context_warning -v
```

**Success Criteria**:

- Session lifecycle managed correctly
- Context limit warnings triggered at 80%
- Invalid inputs rejected with clear errors

---

### Phase 5: CLI Command (Day 4-5)

**Goal**: Implement interactive chat command interface.

**Files to Create/Update**:

1. `src/holodeck/cli/commands/chat.py` (NEW)
2. `src/holodeck/cli/main.py` (UPDATE: register chat command)
3. `src/holodeck/cli/exceptions.py` (UPDATE: add chat exceptions)

**Steps**:

1. Implement `chat()` Click command:
   - Define arguments and options
   - Load agent configuration
   - Create `ChatConfig` from CLI options
   - Initialize `ChatSessionManager`
2. Implement interactive REPL loop:
   - Display welcome message
   - Accept user input via `input()`
   - Handle exit commands ('exit', 'quit')
   - Handle Ctrl+C gracefully
3. Implement response display:
   - Standard mode: `You:` / `Agent:` format
   - Verbose mode: detailed tool execution
   - Context warnings when limit approached
4. Implement error handling:
   - Configuration errors (exit code 1)
   - Initialization errors (exit code 2)
   - Runtime errors (display, continue session)
5. Register command in `cli/main.py`

**Validation**:

```bash
# Unit test CLI command
pytest tests/unit/cli/commands/test_chat.py -v

# Manual testing
holodeck chat examples/conversational/agent.yaml

# Test verbose mode
holodeck chat examples/conversational/agent.yaml --verbose
```

**Success Criteria**:

- Command starts session in <1 second
- User can send messages and receive responses
- Exit commands work correctly
- Errors display helpful messages

---

### Phase 6: Tool Execution Streaming (Day 5-6)

**Goal**: Implement real-time tool execution event streaming.

**Files to Create**:

1. `src/holodeck/agent/streaming.py`

**Steps**:

1. Implement `ToolExecutionStream` class
2. Implement `stream_execution()` async generator:
   - Yield `ToolEvent(STARTED)` when tool begins
   - Yield `ToolEvent(PROGRESS)` during execution (if supported)
   - Yield `ToolEvent(COMPLETED)` or `ToolEvent(FAILED)` when done
3. Integrate with `AgentExecutor.execute_turn()`
4. Update CLI to display events in real-time

**Validation**:

```bash
pytest tests/unit/agent/test_streaming.py -v

# Integration test with tool execution
pytest tests/integration/test_tool_streaming.py -v --slow
```

**Success Criteria**:

- Tool events stream in real-time (<100ms latency)
- Events display correctly in terminal
- Failed tools show clear error messages

---

### Phase 7: OpenTelemetry Integration (Day 6-7)

**Goal**: Implement observability instrumentation.

**Files to Create**:

1. `src/holodeck/lib/observability.py`

**Steps**:

1. Implement OpenTelemetry setup:
   - Initialize `TracerProvider`
   - Configure exporters (console, OTLP)
   - Instrument Semantic Kernel
2. Add session-level tracing:
   - Create span for entire chat session
   - Add session_id, agent_name attributes
3. Add turn-level tracing:
   - Create span for each message turn
   - Track LLM calls, tool executions
4. Add metrics:
   - Token usage counts
   - Session duration
   - Tool execution counts
5. Integrate with CLI `--observability` flag

**Validation**:

```bash
pytest tests/unit/lib/test_observability.py -v

# Manual test with observability enabled
holodeck chat examples/conversational/agent.yaml --observability

# Verify traces in console output
```

**Success Criteria**:

- Traces exported to configured backend
- GenAI semantic attributes present
- Token usage metrics accurate
- No performance degradation

---

### Phase 8: Error Handling & Edge Cases (Day 7)

**Goal**: Comprehensive error handling and edge case coverage.

**Steps**:

1. Add error classes to `src/holodeck/lib/errors.py`
2. Implement error handling in all modules
3. Add edge case tests:
   - Extremely long messages
   - Rapid message submission
   - LLM provider downtime
   - Tool timeouts
   - Invalid agent configurations
   - Unicode edge cases

**Validation**:

```bash
# Run all tests including edge cases
pytest tests/ -v --cov=src/holodeck/agent --cov=src/holodeck/cli/commands/chat.py

# Coverage report
pytest --cov-report=html
open htmlcov/index.html
```

**Success Criteria**:

- All error scenarios handled gracefully
- No uncaught exceptions in normal usage
- Clear error messages for users
- 80%+ code coverage

---

### Phase 9: Integration Testing (Day 8)

**Goal**: End-to-end integration tests with real agents.

**Files to Create**:

1. `tests/integration/test_chat_integration.py`

**Steps**:

1. Create test agent configurations
2. Implement integration test scenarios:
   - Basic chat session
   - Multi-turn conversations
   - Tool execution
   - Context limit warnings
   - Error recovery
3. Test with multiple LLM providers (OpenAI, Anthropic)

**Validation**:

```bash
# Run integration tests
pytest tests/integration/test_chat_integration.py -v --slow

# Test with different providers
ANTHROPIC_API_KEY=... pytest tests/integration/ -v --slow
```

**Success Criteria**:

- All user stories from spec validated
- Tests pass with real LLM providers
- Performance targets met (SC-001 through SC-007)

---

### Phase 10: Documentation & Polish (Day 9)

**Goal**: User documentation and final polish.

**Steps**:

1. Add docstrings to all public functions/classes
2. Update README with chat command examples
3. Create example agent configurations
4. Run code quality checks:
   ```bash
   make format
   make lint
   make type-check
   make security
   ```
5. Run full CI pipeline:
   ```bash
   make ci
   ```

**Validation**:

```bash
# All checks must pass
make ci

# Manual smoke test
holodeck chat examples/conversational/agent.yaml
```

**Success Criteria**:

- All code quality checks pass
- Documentation complete and accurate
- Examples work out of the box

---

## Testing Strategy

### Unit Tests (Fast, Isolated)

- Mock Semantic Kernel components
- Test each class independently
- Focus on validation logic, error handling
- Run with every commit

### Integration Tests (Slow, LLM-dependent)

- Use real LLM providers (test API keys)
- Test end-to-end workflows
- Run before PR merge

### Manual Testing Checklist

- [ ] Start chat with valid agent config
- [ ] Send multiple messages, verify context preserved
- [ ] Trigger tool execution, verify display
- [ ] Test exit commands ('exit', 'quit', Ctrl+C)
- [ ] Test invalid inputs (empty, oversized)
- [ ] Test with invalid agent config (expect clear error)
- [ ] Test verbose mode (`--verbose`)
- [ ] Test observability mode (`--observability`)
- [ ] Test context limit warning (send 40+ messages)
- [ ] Test session stability (30-minute conversation)

---

## Debugging Tips

**Enable verbose logging**:

```bash
export HOLODECK_LOG_LEVEL=DEBUG
holodeck chat agent.yaml --verbose
```

**Enable OpenTelemetry console export**:

```bash
holodeck chat agent.yaml --observability
```

**Run with debugger**:

```python
# In chat.py, add breakpoint
import pdb; pdb.set_trace()
```

**Profile performance**:

```bash
python -m cProfile -o chat.prof src/holodeck/cli/main.py chat agent.yaml
python -m pstats chat.prof
```

---

## Common Issues & Solutions

| Issue              | Solution                                              |
| ------------------ | ----------------------------------------------------- |
| LLM API key errors | Check `.env` file has correct keys                    |
| Import errors      | Ensure all `__init__.py` files created                |
| Type errors        | Run `mypy src/holodeck/agent/` to identify issues     |
| Test failures      | Check test fixtures in `tests/fixtures/`              |
| Slow tests         | Use `@pytest.mark.slow` and skip with `-m "not slow"` |

---

## Success Metrics

At the end of implementation, verify:

- ✅ All functional requirements (FR-001 through FR-016) implemented
- ✅ All success criteria (SC-001 through SC-007) met
- ✅ All acceptance scenarios from spec pass
- ✅ Code coverage ≥ 80%
- ✅ All CI checks pass (`make ci`)
- ✅ Manual testing checklist complete
- ✅ Constitution Check passes (re-evaluate post-implementation)

---

## Next Steps After Completion

1. Create PR with implementation
2. Request code review from team
3. Address review feedback
4. Merge to main branch
5. Tag release (v0.1.0 milestone)
6. Update user documentation
7. Plan P2/P3 features (session persistence, rich UI)
