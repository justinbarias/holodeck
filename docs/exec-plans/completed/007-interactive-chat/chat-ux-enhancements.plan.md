# Chat UX Enhancements Implementation Plan

## Overview
Full integration with spinner, token tracking, and adaptive status display (minimal by default, rich in verbose mode). Spinner always shows regardless of quiet mode.

## Requirements

### User Preferences
1. **Integration Approach**: Full Integration (comprehensive implementation)
2. **Status Panel Design**: Adaptive - inline minimal by default, rich panel in verbose mode
3. **Token Tracking**: Implement in this iteration
4. **Quiet Mode Behavior**: Always show spinner (even when `--quiet` is enabled)

## Implementation Tasks

### 1. Create ChatProgressIndicator Class
**New file**: `src/holodeck/chat/progress.py`

**Purpose**: Track and display chat session progress with spinner and status information.

**Based on**: `src/holodeck/lib/test_runner/progress.py` pattern

**Responsibilities**:
- Track message count, total tokens, session time, last response time
- Generate animated spinner text
- Provide adaptive status display (minimal/rich)
- TTY detection for terminal compatibility

**Methods**:
- `__init__(max_messages: int, quiet: bool, verbose: bool)`
- `get_spinner_line() -> str` - Returns animated spinner text
- `update(response: AgentResponse)` - Update stats after each turn
- `get_status_inline() -> str` - Minimal status for default mode
- `get_status_panel() -> str` - Rich panel for verbose mode
- `is_tty: bool` - Property for TTY detection via `sys.stdout.isatty()`

**State Tracking**:
```python
- max_messages: int
- current_messages: int = 0
- total_tokens: int = 0
- prompt_tokens: int = 0
- completion_tokens: int = 0
- session_start: datetime
- last_response_time: float | None = None
- quiet: bool
- verbose: bool
- _spinner_index: int = 0
- _spinner_chars: list[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
```

### 2. Add Token Tracking to AgentFactory
**Modify**: `src/holodeck/lib/test_runner/agent_factory.py`

**Changes**:
1. Add `token_usage: TokenUsage | None` field to `AgentExecutionResult` dataclass (line ~230)
2. Create `_extract_token_usage(response: Any) -> TokenUsage | None` method
3. Extract token usage in `_invoke_agent_impl()` (line ~334-370)
4. Handle None gracefully (some LLM providers don't expose usage)

**Token Extraction Logic**:
```python
def _extract_token_usage(self, response: Any) -> TokenUsage | None:
    """Extract token usage from agent response.

    Checks for usage attribute on response object and maps to TokenUsage model.
    Returns None if provider doesn't expose usage data.
    """
    try:
        if hasattr(response, 'usage'):
            usage = response.usage
            return TokenUsage(
                prompt_tokens=getattr(usage, 'prompt_tokens', 0),
                completion_tokens=getattr(usage, 'completion_tokens', 0),
                total_tokens=getattr(usage, 'total_tokens', 0),
            )
    except Exception as e:
        logger.warning(f"Failed to extract token usage: {e}")
    return None
```

**Integration Point**:
- Call in `_invoke_agent_impl()` after agent invocation
- Return in `AgentExecutionResult`

### 3. Pass Token Usage Through AgentExecutor
**Modify**: `src/holodeck/chat/executor.py`

**Changes**:
1. Update `execute_turn()` to extract token usage from factory result (line 104)
2. Change `tokens_used=None` to `tokens_used=self._extract_token_usage(result)`
3. Add execution callbacks to `__init__` signature:
   - `on_execution_start: Callable[[str], None] | None = None`
   - `on_execution_complete: Callable[[AgentResponse], None] | None = None`
4. Call callbacks in `execute_turn()`:
   - Before: `if self.on_execution_start: self.on_execution_start(message)`
   - After: `if self.on_execution_complete: self.on_execution_complete(response)`

**Modified Method Signature**:
```python
def __init__(
    self,
    agent_config: Agent,
    enable_observability: bool = False,
    timeout: float | None = 60.0,
    max_retries: int = 3,
    on_execution_start: Callable[[str], None] | None = None,
    on_execution_complete: Callable[[AgentResponse], None] | None = None,
) -> None:
```

### 4. Enhance ChatSessionManager
**Modify**: `src/holodeck/chat/session.py`

**Changes**:
1. Add cumulative token tracking to session state
2. Accumulate tokens in `process_message()` after each turn
3. Add method to retrieve session statistics

**New Fields**:
```python
self.total_tokens = TokenUsage(
    prompt_tokens=0,
    completion_tokens=0,
    total_tokens=0,
)
```

**New Method**:
```python
def get_session_stats(self) -> dict[str, Any]:
    """Get current session statistics.

    Returns:
        Dict with message_count, total_tokens, session_duration
    """
    return {
        'message_count': self.message_count,
        'total_tokens': self.total_tokens,
        'session_duration': (datetime.now() - self.session_start_time).total_seconds(),
    }
```

### 5. Integrate Spinner in Chat Command
**Modify**: `src/holodeck/cli/commands/chat.py`

**Changes**:
1. Create `ChatSpinnerThread` class (similar to test runner's `SpinnerThread`)
2. Initialize `ChatProgressIndicator` before REPL loop
3. Modify REPL loop to use spinner around agent execution
4. Display status based on verbose flag
5. Always show spinner regardless of `--quiet` flag

**ChatSpinnerThread Implementation** (lines ~25-60):
```python
class ChatSpinnerThread(threading.Thread):
    """Background thread for displaying animated spinner during agent execution."""

    def __init__(self, progress: ChatProgressIndicator) -> None:
        super().__init__(daemon=True)
        self.progress = progress
        self._stop_event = threading.Event()
        self._running = False

    def run(self) -> None:
        self._running = True
        while not self._stop_event.is_set():
            line = self.progress.get_spinner_line()
            if line:
                sys.stdout.write(f"\r{line}")
                sys.stdout.flush()
            time.sleep(0.1)  # 10 FPS update rate
        self._running = False

    def stop(self) -> None:
        self._stop_event.set()
        if self._running:
            # Clear spinner line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
```

**REPL Loop Integration**:
```python
# Initialize progress indicator
progress = ChatProgressIndicator(
    max_messages=max_messages,
    quiet=quiet,
    verbose=verbose,
)

# REPL loop
while True:
    user_input = click.prompt("You", default="").strip()

    if user_input.lower() in ("exit", "quit"):
        break

    # Start spinner (always show, regardless of quiet mode)
    spinner = None
    if sys.stdout.isatty():
        spinner = ChatSpinnerThread(progress)
        spinner.start()

    try:
        response = await session_manager.process_message(user_input)
    finally:
        # Stop spinner
        if spinner:
            spinner.stop()
            spinner.join()

    # Update progress
    if response:
        progress.update(response)

        # Display response with status
        if verbose:
            click.echo(progress.get_status_panel())
            click.echo(f"Agent: {response.content}\n")
        else:
            # Inline status
            status = progress.get_status_inline()
            click.echo(f"Agent: {response.content} {status}\n")
```

### 6. Update Models
**Modify**: `src/holodeck/lib/test_runner/agent_factory.py`

**Change**: Add `token_usage: TokenUsage | None` to `AgentExecutionResult` dataclass

```python
@dataclass
class AgentExecutionResult:
    """Result from agent execution."""

    tool_calls: list[dict[str, Any]]
    chat_history: ChatHistory
    token_usage: TokenUsage | None = None  # NEW FIELD
```

## Status Display Behavior

### Default Mode (neither quiet nor verbose)
```
You: What is the weather?
⠋ Thinking...
Agent: It's sunny today! [3/50 | 1.2s]
```

**Format**: `[messages_current/messages_max | execution_time]`

### Verbose Mode (`--verbose`)
```
╭─── Chat Status ─────────────────────────╮
│ Session Time: 00:05:23                   │
│ Messages: 3 / 50 (6%)                    │
│ Total Tokens: 1,234                      │
│   ├─ Prompt: 890                         │
│   └─ Completion: 344                     │
│ Last Response: 2.3s                      │
│ Tools Used: weather_api                  │
╰─────────────────────────────────────────╯
You: What is the weather?
⠋ Thinking...
Agent: It's sunny today!
```

### Quiet Mode (`--quiet`)
- Spinner still shows (per requirement: always show spinner)
- No status display (inline or panel)
- Only response content shown
```
You: What is the weather?
⠋ Thinking...
Agent: It's sunny today!
```

### Non-TTY Mode (CI/CD, piped output)
- No spinner animation
- Plain text status messages
- No ANSI colors or line overwriting
```
You: What is the weather?
[Processing...]
Agent: It's sunny today! [3/50 | 1.2s]
```

## Files to Modify

1. **NEW**: `src/holodeck/chat/progress.py` (~200 lines)
   - ChatProgressIndicator class
   - Spinner animation logic
   - Status display formatting

2. **MODIFY**: `src/holodeck/lib/test_runner/agent_factory.py` (+50 lines)
   - Add token_usage to AgentExecutionResult
   - Implement _extract_token_usage()
   - Extract tokens in _invoke_agent_impl()

3. **MODIFY**: `src/holodeck/chat/executor.py` (+30 lines)
   - Add execution callbacks to __init__
   - Extract token usage in execute_turn()
   - Call callbacks before/after execution

4. **MODIFY**: `src/holodeck/chat/session.py` (+20 lines)
   - Add total_tokens accumulation
   - Implement get_session_stats()

5. **MODIFY**: `src/holodeck/cli/commands/chat.py` (+80 lines)
   - Add ChatSpinnerThread class
   - Initialize ChatProgressIndicator
   - Integrate spinner in REPL loop
   - Display adaptive status

## Testing Strategy

### Unit Tests
- Test `ChatProgressIndicator` methods
- Test token extraction with mock responses
- Test callback invocation in executor

### Integration Tests
- Test spinner animation in TTY environment
- Test non-TTY fallback (plain text)
- Verify token accumulation across multiple turns
- Test status display in default/verbose/quiet modes

### Manual Testing
- Run in interactive terminal (macOS/Linux)
- Test with `--quiet`, `--verbose`, and default flags
- Verify spinner appears regardless of quiet mode
- Test token tracking with OpenAI/Azure providers
- Test graceful handling when token_usage is None
- Test in piped output (non-TTY): `echo "test" | holodeck chat agent.yaml`

## Edge Cases to Handle

1. **No Token Usage**: Some providers don't expose usage
   - Display "N/A" or omit token count
   - Don't crash if token_usage is None

2. **Terminal Width**: Status panel should fit in narrow terminals
   - Minimum width: 60 characters
   - Truncate long tool names

3. **Rapid Responses**: Very fast agent responses (<100ms)
   - Spinner may flash briefly
   - Acceptable UX tradeoff

4. **Thread Cleanup**: Ensure spinner thread stops on:
   - Normal exit (user types "exit")
   - Ctrl+C interrupt
   - Error during agent execution

5. **Non-TTY Environments**:
   - No spinner animation
   - Fall back to plain text status
   - No ANSI color codes

## Dependencies

**Existing**:
- `threading` (stdlib)
- `sys` (stdlib)
- `time` (stdlib)
- `click` (already used)

**Models**:
- `TokenUsage` from `src/holodeck/models/token_usage.py`
- `AgentResponse` from `src/holodeck/chat/executor.py`

## Estimated Effort

**Implementation**: ~2-3 hours
- ChatProgressIndicator class: 1 hour
- Token tracking: 30 minutes
- Executor/session updates: 30 minutes
- Chat command integration: 1 hour

**Testing**: ~1 hour
- Unit tests: 30 minutes
- Integration/manual testing: 30 minutes

**Total**: ~3-4 hours

## Success Criteria

- ✅ Spinner displays during agent execution in TTY mode
- ✅ Spinner always shows (ignores `--quiet` flag)
- ✅ Token usage tracked and accumulated across conversation
- ✅ Default mode shows inline status: `[messages | time]`
- ✅ Verbose mode shows rich status panel with tokens breakdown
- ✅ Graceful fallback when token_usage is None
- ✅ Works in non-TTY environments (CI/CD)
- ✅ Passes all unit and integration tests
- ✅ No thread leaks or cleanup issues
- ✅ Code quality checks pass (format, lint, type-check)

## Future Enhancements (Out of Scope)

- Streaming responses with progressive token display
- Real-time token cost calculation
- Session export with token usage statistics
- Context window usage percentage
- Tool execution timing in verbose mode
- Customizable status panel layout
