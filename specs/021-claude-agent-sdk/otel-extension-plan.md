# Plan: OTel Tracing for Claude Backend via SDK Hooks

## Context

The Claude Agent SDK spawns a subprocess that generates its own OTel traces with independent trace IDs. These traces appear as separate root spans in the OTel dashboard — not as children of HoloDeck's `holodeck.cli.test` span. The fix: use the SDK's hooks API (`PreToolUse`, `PostToolUse`, `PostToolUseFailure`) to create OTel spans from HoloDeck's own `TracerProvider`, so all Claude operations nest under the existing application spans.

**Key finding (verified via `scripts/test_hooks_query.py`)**: Hooks only work with
`ClaudeSDKClient` (streaming input mode), NOT with `query()` (single message mode).
Although `InternalClient.process_query()` wires `options.hooks` through at `_internal/client.py:107`,
the hook callbacks fail with `Stream closed` — the bidirectional `sendRequest` channel that hooks
depend on is torn down before callbacks can execute. The official docs confirm this limitation:
single message mode does not support hook integration.

**Consequence**: `invoke_once()` must switch from `query()` to `ClaudeSDKClient` to support
OTel hook-based tracing. Each stateless test invocation wraps a short-lived `ClaudeSDKClient`
session (connect → send → receive → disconnect). This is a net-new change to `claude_backend.py`.

## Span Hierarchy (when observability enabled)

```
holodeck.cli.test                       (existing — test.py)
  └── holodeck.claude.invoke            (NEW — wraps ClaudeSDKClient session / session.send)
        ├── holodeck.claude.tool        (NEW — from PreToolUse/PostToolUse hooks)
        ├── holodeck.claude.tool        (NEW — per tool call)
        └── holodeck.claude.tool        (NEW — per tool call)
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/holodeck/lib/backends/claude_otel_hooks.py` | **CREATE** — Hook callbacks + span tracking |
| `src/holodeck/lib/backends/claude_backend.py` | **MODIFY** — Wire hooks into `build_options()`, add wrapper spans |
| `tests/unit/lib/backends/test_claude_otel_hooks.py` | **CREATE** — Unit tests for hooks module |
| `tests/unit/lib/backends/test_claude_backend.py` | **MODIFY** — Tests for hooks integration |

## Implementation Steps

### Step 1: Create `src/holodeck/lib/backends/claude_otel_hooks.py`

New module with `ClaudeOtelHookFactory` class:

- `_ActiveToolSpan` dataclass: holds `span`, `context_token`, `start_time`
- `ClaudeOtelHookFactory.__init__(observability_config)`: stores config, creates empty `_active_spans: dict[str, _ActiveToolSpan]`
- `build_hooks() -> dict[str, list[HookMatcher]]`: returns hook config for `PreToolUse`, `PostToolUse`, `PostToolUseFailure`
- `_on_pre_tool_use()`: creates span `holodeck.claude.tool` with `tool.name`, `tool.use_id` attributes; stores in `_active_spans[tool_use_id]`; optionally captures `tool.input` if `capture_content=True`
- `_on_post_tool_use()`: ends span with `StatusCode.OK`, sets `tool.duration_s`, `tool.status=success`; optionally captures `tool.response`
- `_on_post_tool_use_failure()`: ends span with `StatusCode.ERROR`, records error message
- `cleanup_orphaned_spans()`: ends any uncompleted spans during teardown

**Error handling**: Every hook callback wrapped in `try/except Exception` — logs at DEBUG, always returns `{"continue_": True}`. Hooks must never block the SDK.

**Privacy**: `tool.input` and `tool.response` only captured when `observability_config.traces.capture_content is True` (matches existing pattern in evaluators).

### Step 2: Update `build_options()` in `claude_backend.py`

Add optional `hooks: dict[str, list[Any]] | None = None` parameter. Pass through to `ClaudeAgentOptions(**opts_kwargs)` when not None.

### Step 3: Update `ClaudeBackend.initialize()` in `claude_backend.py`

After step 7 (OTel env vars), before step 8 (build options):
- Check `_is_observability_enabled()` (new helper: checks `agent.observability.enabled` and `agent.observability.traces.enabled`)
- If enabled: create `ClaudeOtelHookFactory`, call `build_hooks()`, pass to `build_options()`
- Store factory as `self._otel_hook_factory` for cleanup

### Step 4: Switch `invoke_once()` from `query()` to `ClaudeSDKClient`

**Why**: `query()` does not support hooks (see Key Finding above). `invoke_once()` must use
`ClaudeSDKClient` to get OTel hook callbacks.

**Change in `claude_backend.py`**: Replace the current `_invoke_query()` implementation that
iterates over `query(prompt=message, options=self._options)` with a short-lived
`ClaudeSDKClient` session:

```python
async def _invoke_with_client(self, message: str) -> ExecutionResult:
    async with ClaudeSDKClient(self._options) as client:
        await client.query(message)
        async for msg in client.receive_response():
            # Same message processing as current _invoke_query()
            ...
```

Each test invocation opens and closes its own session. This is equivalent to the current
`query()` behavior (fresh session per call) but enables the hooks bidirectional channel.

**Wrapper span**: Wrap entire method body in `holodeck.claude.invoke` span using existing
`nullcontext()` pattern:
- Attributes: `holodeck.input_length`, `holodeck.num_turns`, `holodeck.token_usage.prompt`, `holodeck.token_usage.completion`, `holodeck.tool_calls_count`

### Step 5: Add wrapper span in `ClaudeSession.send()`

Same pattern. Pass `observability_enabled: bool` through `ClaudeSession.__init__()` (set by `ClaudeBackend.create_session()`).

### Step 6: Update `ClaudeBackend.teardown()`

Call `self._otel_hook_factory.cleanup_orphaned_spans()` if factory exists.

### Step 7: Write tests

**`test_claude_otel_hooks.py`** (9 tests):
- `build_hooks()` returns correct structure (3 hook events)
- `_on_pre_tool_use` creates span with correct attributes
- `_on_post_tool_use` ends span with OK status and duration
- `_on_post_tool_use_failure` ends span with ERROR status
- Exception in hook is swallowed (returns `{"continue_": True}`)
- Unknown `tool_use_id` in PostToolUse is a no-op
- `cleanup_orphaned_spans` ends all active spans
- Content capture respects `capture_content` toggle
- `tool_use_id=None` handled gracefully

**`test_claude_backend.py`** updates (4-5 tests):
- `build_options()` with hooks passes them through
- `initialize()` with observability creates hook factory
- `initialize()` without observability skips hooks
- `teardown()` calls cleanup

### Step 8: Code quality

`make format && make lint-fix && make type-check && pytest tests/unit/lib/backends/ -n auto -v`

## Span Attributes

| Attribute | Span | Type | Description |
|-----------|------|------|-------------|
| `holodeck.input_length` | `invoke` | int | User input char count |
| `holodeck.num_turns` | `invoke` | int | SDK agentic turns |
| `holodeck.token_usage.prompt` | `invoke` | int | Prompt tokens |
| `holodeck.token_usage.completion` | `invoke` | int | Completion tokens |
| `holodeck.tool_calls_count` | `invoke` | int | Number of tool calls |
| `tool.name` | `tool` | str | Tool name from SDK |
| `tool.use_id` | `tool` | str | Tool use ID |
| `tool.status` | `tool` | str | "success" or "error" |
| `tool.duration_s` | `tool` | float | Wall-clock seconds |
| `tool.input` | `tool` | str | Tool input (opt-in) |
| `tool.response` | `tool` | str | Tool response (opt-in) |
| `tool.error` | `tool` | str | Error message (failures) |

## What Changes (beyond OTel)

- `invoke_once()` switches from `query()` to `ClaudeSDKClient` — same observable behavior
  (fresh session per call), but enables hooks. This is required regardless of OTel; it also
  fixes a latent issue where `query()` cannot support future hook-dependent features.

## What Does NOT Change

- `otel_bridge.py` (subprocess env vars) — complementary, not replaced
- `ClaudeSession` (chat path) — already uses `ClaudeSDKClient`, hooks work as-is
- `ObservabilityConfig` model
- CLI commands (test.py, chat.py)
- SK backend path (agent_factory.py)
- No new external dependencies
