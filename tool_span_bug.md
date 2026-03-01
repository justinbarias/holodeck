# Bug: `execute_tool` spans missing for `ClaudeSDKClient` (session) path

## Summary

The `otel-instrumentation-claude-agent-sdk` instrumentor fails to create `execute_tool` spans when using the `ClaudeSDKClient` (multi-turn session) code path. The standalone `query()` path works correctly.

This affects any consumer that uses `ClaudeSDKClient` for multi-turn conversations, including HoloDeck's `holodeck chat` command.

## Root Cause

**ContextVar timing mismatch.** The `_read_messages` background task is spawned during `connect()`, before the `InvocationContext` is set in the ContextVar. Hook callbacks that fire inside `_read_messages` read a stale `None` from the ContextVar and silently no-op.

## Detailed Analysis

### Standalone `query()` path (works)

```
_instrumented_query()
  1. set_invocation_context(ctx)        # ContextVar SET here
  2. wrapped(*args, **kwargs)
       -> InternalClient.process_query()
            -> Query(hooks=...)
            -> query.start()
                 -> _tg.start_soon(_read_messages)   # Task INHERITS ctx
            -> query.initialize()
            -> async for data in query.receive_messages():
                 hook_callback -> _on_pre_tool_use()
                                  -> get_invocation_context()  # Returns ctx ✅
                                  -> create_execute_tool_span()
```

The `InvocationContext` is set **before** the background task is spawned, so the task inherits the ContextVar value.

### `ClaudeSDKClient` path (broken)

```
_wrap_client_init()
  -> hooks injected into options ✅

client.connect()                               # Step 1
  -> Query(hooks=...)
  -> query.start()
       -> _tg.start_soon(_read_messages)       # Task spawned, ContextVar is None
  -> query.initialize()                        # Hooks registered with subprocess

_wrap_client_query(message)                    # Step 2
  -> create invoke_agent span
  -> set_invocation_context(ctx)               # ContextVar SET here (too late!)
  -> wrapped(message)                          # Sends message to subprocess

client.receive_response()                      # Step 3
  -> query.receive_messages()
       -> _read_messages processes hook_callback
            -> _on_pre_tool_use()
                 -> get_invocation_context()    # Reads from _read_messages task
                                               # context — still None ❌
                 -> returns {} (silent no-op)
```

The `_read_messages` task is spawned during `connect()` (Step 1). The `InvocationContext` is set during `client.query()` (Step 2). In Python's asyncio, `ContextVar` values are **copied at task creation time**. The background task has a snapshot from Step 1 — which is `None`.

### Why it's silent

The hook callbacks guard with an early return:

```python
# _hooks.py
async def _on_pre_tool_use(input_data, tool_use_id=None, context=None, **kwargs):
    ctx = get_invocation_context()
    if ctx is None or tool_use_id is None:
        return {}  # Silent no-op — no span created, no error logged
```

## Affected Components

| Component | Version | Path |
|-----------|---------|------|
| `otel-instrumentation-claude-agent-sdk` | `_instrumentor.py` | `_wrap_client_query` sets context too late |
| `otel-instrumentation-claude-agent-sdk` | `_hooks.py` | `_on_pre_tool_use` / `_on_post_tool_use` get `None` context |
| `claude-agent-sdk` | `client.py` | `connect()` spawns `_read_messages` task before hooks can set context |
| `claude-agent-sdk` | `_internal/query.py:170` | `_tg.start_soon(self._read_messages)` — task creation point |

## Reproduction

```python
import asyncio
import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from opentelemetry.instrumentation.claude_agent_sdk import ClaudeAgentSdkInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
tp = TracerProvider()
tp.add_span_processor(SimpleSpanProcessor(exporter))

instrumentor = ClaudeAgentSdkInstrumentor()
instrumentor.instrument(tracer_provider=tp, agent_name="test", capture_content=True)

async def test():
    client = claude_agent_sdk.ClaudeSDKClient(
        options=ClaudeAgentOptions(
            model="claude-sonnet-4-6",
            system_prompt="You are helpful. Use web search.",
            permission_mode="bypassPermissions",
            max_turns=3,
            web_search=True,
        )
    )
    await client.connect()
    await client.query("Search the web for today's date.")
    async for msg in client.receive_response():
        pass
    await client.close()

    tp.force_flush()
    spans = exporter.get_finished_spans()
    print(f"Spans: {len(spans)}")
    for s in spans:
        print(f"  {s.name}")
    # Expected: invoke_agent + execute_tool WebSearch
    # Actual:   invoke_agent only (execute_tool missing)

asyncio.run(test())
instrumentor.uninstrument()
```

## Potential Fixes (in instrumentor)

### Option A: Set context before `connect()`

Move `InvocationContext` creation from `_wrap_client_query` to `_wrap_client_init`, so it's set before `connect()` spawns the background task.

**Downside:** `__init__` happens once, but `query()` can be called multiple times per session. Each turn needs its own `invoke_agent` span.

### Option B: Store context on the client instance (bypass ContextVar)

Instead of using ContextVar for the `ClaudeSDKClient` path, store the `InvocationContext` directly on the client instance. Modify hook callbacks to check the client instance first, then fall back to ContextVar.

```python
# In _wrap_client_init:
instance._otel_invocation_ctx = None  # placeholder

# In _wrap_client_query:
ctx = InvocationContext(invocation_span=span, ...)
instance._otel_invocation_ctx = ctx  # store on instance

# In hook callbacks (closures that capture instance):
ctx = instance._otel_invocation_ctx or get_invocation_context()
```

**Downside:** Requires hooks to be closures that capture the client instance.

### Option C: Re-set ContextVar inside `_read_messages` task

Wrap `_read_messages` or inject a hook-dispatch middleware that re-sets the ContextVar from the client instance before calling each hook callback.

### Option D: Workaround in HoloDeck (outside instrumentor)

In `ClaudeSession._ensure_client()`, manually set the `InvocationContext` before calling `client.connect()`, ensuring the background task inherits it. This only works for the first turn.

## Recommended Action

File an issue against `otel-instrumentation-claude-agent-sdk` requesting **Option B** — store the `InvocationContext` on the client instance and have hook callbacks read from there for the `ClaudeSDKClient` path. This is the cleanest fix because:

1. It doesn't change the ContextVar semantics for the standalone `query()` path
2. It works across multiple `client.query()` turns (each turn updates the instance attribute)
3. The instrumentor already stores other OTel state on the instance (`_otel_tracer`, `_otel_meter`, etc.)

## Verified Working Path

The standalone `query()` path (used by `invoke_once()` / `holodeck test`) correctly produces both `invoke_agent` and `execute_tool` spans. This was confirmed via integration tests with `InMemorySpanExporter` and OTLP export to Aspire dashboard.
