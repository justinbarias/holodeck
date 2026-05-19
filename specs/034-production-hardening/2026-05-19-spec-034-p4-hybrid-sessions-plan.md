# Spec 034 P4 — Hybrid Sessions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ClaudeSession`'s persistent `ClaudeSDKClient` with per-turn top-level `query()` calls using `ClaudeAgentOptions.resume=<session_id>`, so idle sessions cost ~0 resident memory and the `SessionStore` cap correctly applies to concurrent active turns instead of open threads.

**Architecture:** Keep `ClaudeSession`'s public protocol stable (`send`, `send_streaming`, `prepare`, `close` — all callers downstream of `_TaskBoundSession` and the AG-UI / REST bridges are unchanged). Swap the internals from "one persistent connected `ClaudeSDKClient` per session" to "one fresh `query()` invocation per turn, replaying conversation state from the on-disk JSONL transcript via `resume=session_id`". Capture `session_id` from `ResultMessage` on turn 1 and feed it back into `options.resume` on turn 2+. Use streaming-mode prompts (`AsyncIterable[dict]`) to keep stdin open for SDK MCP tool callbacks — string-mode `query()` calls `end_input()` and deadlocks. On `close()`, delete the on-disk transcript at `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`.

**Tech Stack:**
- `claude-agent-sdk==0.1.44` — `query()`, `ClaudeAgentOptions.resume`
- pytest with `pytest-asyncio`, `unittest.mock.{AsyncMock, MagicMock, patch}`
- Pydantic v2 (config models)
- Python 3.10+ (`dataclasses.replace`, `asyncio.Lock`)

**Branch:** `feature/034-p4-hybrid-sessions` off `feature/034-p1-stability-permissions` (or off `main` if P1 has merged by the time work starts).

**Pre-flight (run from repo root):**
```bash
source .venv/bin/activate
git checkout feature/034-p1-stability-permissions
git pull origin feature/034-p1-stability-permissions
git checkout -b feature/034-p4-hybrid-sessions
```

---

## Task 1: Add streaming-prompt envelope helper

**Why:** Top-level `query()` accepts two prompt shapes — `str` (one-shot, calls `end_input()` after writing) or `AsyncIterable[dict]` (streaming, keeps stdin open via a background `stream_input` task). P4 turns need streaming-mode because the agent will invoke SDK MCP tools, and the SDK's reverse control channel (tool callbacks) needs stdin to stay open. Spike v1 surfaced this — string-mode crashes with `ProcessTransport is not ready for writing` mid-turn.

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py` (add helper near the existing `_wrap_prompt` at line 81)
- Test: `tests/unit/lib/backends/test_claude_backend.py` (add `TestStreamingUserEnvelope` class)

- [ ] **Step 1: Ensure test-module imports cover P4 tasks**

`asyncio` and `pathlib.Path` are referenced by tests added in later tasks (Tasks 2, 5, 7) but are not currently imported in `tests/unit/lib/backends/test_claude_backend.py`. Add them once now so every subsequent task can rely on them.

Edit the import block at the top of `tests/unit/lib/backends/test_claude_backend.py`. After the `import json` line (around line 13), insert:

```python
import asyncio
from pathlib import Path
```

(`patch` is already imported from `unittest.mock` — no change needed there.)

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/lib/backends/test_claude_backend.py`:

```python
# ---------------------------------------------------------------------------
# Streaming-mode envelope (spec 034 P4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingUserEnvelope:
    """The envelope wraps a string into the AsyncIterable[dict] shape that
    ``query()`` needs in streaming mode. String-mode prompts cause query()
    to call end_input() after writing, deadlocking SDK MCP tool callbacks.
    """

    @pytest.mark.asyncio
    async def test_yields_single_user_message(self) -> None:
        from holodeck.lib.backends.claude_backend import (
            _streaming_user_envelope,
        )

        msgs = [m async for m in _streaming_user_envelope("hello")]

        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["type"] == "user"
        assert msg["session_id"] == ""
        assert msg["message"] == {"role": "user", "content": "hello"}
        assert msg["parent_tool_use_id"] is None
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestStreamingUserEnvelope -v
```

Expected: `ImportError: cannot import name '_streaming_user_envelope' from 'holodeck.lib.backends.claude_backend'`.

- [ ] **Step 4: Add the helper**

Edit `src/holodeck/lib/backends/claude_backend.py`. After the existing `_wrap_prompt` function (ends around line 99), add:

```python
async def _streaming_user_envelope(
    message: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Wrap a user message in the streaming-mode envelope ``query()`` expects.

    ``query()`` accepts either a ``str`` prompt or an ``AsyncIterable[dict]``.
    The ``str`` form writes the message and immediately closes stdin via
    ``end_input()``, which deadlocks any subsequent SDK-MCP tool callback
    (control-channel writes fail with ``ProcessTransport is not ready for
    writing``). Streaming mode keeps stdin open via the SDK's background
    ``stream_input`` task, so tool callbacks can write responses back.

    See spec 034 P4 spike v1 (2026-05-19) for the surfacing.
    """
    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": message},
        "parent_tool_use_id": None,
    }
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestStreamingUserEnvelope -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/holodeck/lib/backends/claude_backend.py tests/unit/lib/backends/test_claude_backend.py
git commit -m "feat(spec-034 P4): add streaming-mode prompt envelope helper"
```

---

## Task 2: Add `_sdk_session_id` + `_send_lock` fields to `ClaudeSession`

**Why:** Under P4, conversation state lives in `session_id` (a string captured from `ResultMessage`) rather than in a long-lived subprocess. The lock serializes concurrent `send()` calls on the same session — two simultaneous turns with the same `resume=` would race writing to the same JSONL transcript.

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py:687-696` (the `ClaudeSession.__init__`)
- Test: `tests/unit/lib/backends/test_claude_backend.py` (extend existing `TestClaudeSessionEnsureClient` neighborhood)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/lib/backends/test_claude_backend.py`:

```python
@pytest.mark.unit
class TestClaudeSessionP4Fields:
    """P4 session model: session_id captures CLI-assigned id on turn 1,
    feeds it into ``options.resume`` on turn 2+. Lock serialises concurrent
    sends to prevent transcript-write races.
    """

    def test_init_sets_p4_fields(self) -> None:
        session = ClaudeSession(options=MagicMock())
        assert session._sdk_session_id is None
        assert isinstance(session._send_lock, asyncio.Lock)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionP4Fields -v
```

Expected: `AttributeError: 'ClaudeSession' object has no attribute '_sdk_session_id'`.

- [ ] **Step 3: Add the fields**

Edit `src/holodeck/lib/backends/claude_backend.py`. Modify `ClaudeSession.__init__` (around line 687):

```python
def __init__(self, options: ClaudeAgentOptions) -> None:
    """Initialize session with base options.

    Args:
        options: Base options (immutable reference for the session lifetime).
    """
    self._base_options = options
    self._client: ClaudeSDKClient | None = None
    self._turn_count: int = 0
    self._tool_event_queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
    # spec 034 P4 — hybrid-session state.
    # CLI-assigned conversation id; captured from ResultMessage on turn 1
    # and fed into ``options.resume`` on turn 2+ so each fresh subprocess
    # rehydrates the JSONL transcript at
    # ``~/.claude/projects/<encoded-cwd>/<sdk_session_id>.jsonl``.
    self._sdk_session_id: str | None = None
    # Serialises concurrent send() / send_streaming() on the same session.
    # Two concurrent turns with the same resume= would race the transcript.
    self._send_lock: asyncio.Lock = asyncio.Lock()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionP4Fields -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/lib/backends/claude_backend.py tests/unit/lib/backends/test_claude_backend.py
git commit -m "feat(spec-034 P4): add _sdk_session_id and _send_lock to ClaudeSession"
```

---

## Task 3: Re-implement `ClaudeSession.send()` on top of `query(resume=)`

**Why:** This is the core P4 swap. Replace the `_ensure_client() + client.query() + client.receive_response()` flow with `query(prompt=streaming_envelope, options=replace(opts, resume=sdk_session_id))` + async-iter drain. Spike v2 (2026-05-19) confirmed this preserves conversation context across turns including tool-use/tool-result blocks. The ExecutionResult fields downstream (response, tool_calls, tool_results, token_usage, structured_output, num_turns) are populated from the same SDK message types.

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py:759-864` (the `send()` method body)
- Test: `tests/unit/lib/backends/test_claude_backend.py` — rewrite the existing `TestClaudeSessionSend` class to mock `query` instead of `ClaudeSDKClient`

- [ ] **Step 1: Write the failing test (multi-turn resume= propagation)**

Add a new test class to `tests/unit/lib/backends/test_claude_backend.py` (delete the existing `TestClaudeSessionSend` class and replace it with the below — the multi-turn test is the load-bearing one):

```python
@pytest.mark.unit
class TestClaudeSessionSend:
    """ClaudeSession.send() under spec 034 P4 (hybrid sessions).

    send() calls top-level claude_agent_sdk.query() with a streaming-mode
    prompt envelope and ``options.resume = self._sdk_session_id``. session_id
    is captured from ResultMessage on turn 1.
    """

    @pytest.mark.asyncio
    async def test_send_returns_execution_result(self) -> None:
        assistant = _make_assistant_message(
            [_make_text_block("Hello "), _make_text_block("world!")]
        )
        result_msg = _make_result_message(session_id="sdk-sess-001")

        async def fake_query(prompt, options):
            for m in (assistant, result_msg):
                yield m

        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        with patch(
            "holodeck.lib.backends.claude_backend.query", side_effect=fake_query
        ):
            result = await session.send("Hi")

        assert isinstance(result, ExecutionResult)
        assert result.response == "Hello world!"
        assert result.token_usage.prompt_tokens == 10
        assert session._sdk_session_id == "sdk-sess-001"

    @pytest.mark.asyncio
    async def test_send_enriches_tool_results_with_name(self) -> None:
        """Regression: tool_results must carry tool name (call_id → name lookup)."""
        tool_use = _make_tool_use_block(
            tool_id="call_abc",
            name="mcp__holodeck_tools__legislation_search_search",
            inp={"query": "sec3"},
        )
        assistant = _make_assistant_message([_make_text_block("ok"), tool_use])

        tool_result_block = _make_tool_result_block("call_abc", "chunk text", False)
        user_msg = MagicMock()
        user_msg.content = [tool_result_block]
        user_msg.__class__.__name__ = "UserMessage"

        async def fake_query(prompt, options):
            for m in (assistant, user_msg, _make_result_message()):
                yield m

        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        with patch(
            "holodeck.lib.backends.claude_backend.query", side_effect=fake_query
        ):
            result = await session.send("search")

        assert result.tool_calls[0]["name"] == (
            "mcp__holodeck_tools__legislation_search_search"
        )
        assert result.tool_results[0]["name"] == (
            "mcp__holodeck_tools__legislation_search_search"
        )
        assert result.tool_results[0]["call_id"] == "call_abc"

    @pytest.mark.asyncio
    async def test_send_propagates_session_id_into_resume_on_turn_2(
        self,
    ) -> None:
        """Turn 1 captures session_id from ResultMessage; turn 2's query() call
        must pass that id back as ``options.resume`` so the SDK rehydrates the
        on-disk transcript.
        """
        captured_options: list[Any] = []

        async def fake_query(prompt, options):
            captured_options.append(options)
            yield _make_assistant_message()
            yield _make_result_message(session_id="sdk-sess-XYZ")

        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        with patch(
            "holodeck.lib.backends.claude_backend.query", side_effect=fake_query
        ):
            await session.send("Turn 1")
            await session.send("Turn 2")

        assert len(captured_options) == 2
        # Turn 1: no prior session_id, resume must be None.
        assert getattr(captured_options[0], "resume", None) is None
        # Turn 2: must carry the captured session_id from turn 1.
        assert captured_options[1].resume == "sdk-sess-XYZ"
        # Session is the same across turns.
        assert session._sdk_session_id == "sdk-sess-XYZ"
        assert session._turn_count == 2

    @pytest.mark.asyncio
    async def test_send_uses_streaming_mode_prompt(self) -> None:
        """The prompt passed to query() must be an AsyncIterable, not a str —
        string-mode closes stdin and deadlocks SDK MCP tool callbacks (spec
        034 P4 spike v1).
        """
        captured_prompts: list[Any] = []

        async def fake_query(prompt, options):
            captured_prompts.append(prompt)
            # Drain the prompt iterable so the test sees what was sent.
            async for _ in prompt:
                pass
            yield _make_result_message()

        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        with patch(
            "holodeck.lib.backends.claude_backend.query", side_effect=fake_query
        ):
            await session.send("Hi")

        assert len(captured_prompts) == 1
        # Must be an async iterable, not a str.
        assert not isinstance(captured_prompts[0], str)
        assert hasattr(captured_prompts[0], "__aiter__")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionSend -v
```

Expected: 4 failures. The existing implementation calls `client.query()` not the top-level `query()`, and never sets `_sdk_session_id`.

- [ ] **Step 3: Re-implement `send()`**

Edit `src/holodeck/lib/backends/claude_backend.py`. Replace the entire body of `ClaudeSession.send()` (currently lines 759-864) with:

```python
async def send(self, message: str) -> ExecutionResult:
    """Send a message and collect the full response.

    Args:
        message: User message text.

    Returns:
        ``ExecutionResult`` with the agent's response.

    Raises:
        BackendSessionError: On subprocess or SDK error.
    """
    async with self._send_lock:
        turn_no = self._turn_count + 1
        logger.debug(
            "[trace] ClaudeSession.send turn=%d: entering, resume=%s",
            turn_no,
            self._sdk_session_id,
        )
        try:
            options = self._options_with_hooks()
            if self._sdk_session_id is not None:
                import dataclasses

                options = dataclasses.replace(
                    options, resume=self._sdk_session_id
                )

            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            token_usage = TokenUsage.zero()
            num_turns = 1
            structured_output: Any = None

            msg_count = 0
            async for msg in query(
                prompt=_streaming_user_envelope(message), options=options
            ):
                msg_count += 1
                logger.debug(
                    "[trace] ClaudeSession.send turn=%d: msg #%d type=%s",
                    turn_no,
                    msg_count,
                    msg.__class__.__name__,
                )
                _maybe_emit_subagent_message(msg, self._tool_event_queue)
                text_parts, tool_calls, tool_results = _process_message(
                    msg, text_parts, tool_calls, tool_results
                )
                if msg.__class__.__name__ == "ResultMessage":
                    rm = cast(Any, msg)
                    usage = rm.usage or {}
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion = usage.get("output_tokens", 0)
                    token_usage = TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion,
                        total_tokens=prompt_tokens + completion,
                    )
                    num_turns = rm.num_turns
                    structured_output = rm.structured_output
                    if self._sdk_session_id is None:
                        captured = getattr(rm, "session_id", None)
                        if isinstance(captured, str) and captured:
                            self._sdk_session_id = captured

            logger.debug(
                "[trace] ClaudeSession.send turn=%d: exited, "
                "msg_count=%d, num_turns=%d, sdk_session_id=%s",
                turn_no,
                msg_count,
                num_turns,
                self._sdk_session_id,
            )
            self._turn_count += 1

            _enrich_tool_results(tool_calls, tool_results)

            response_text = "".join(text_parts)
            if structured_output is not None:
                response_text = json.dumps(structured_output)

            return ExecutionResult(
                response=response_text,
                tool_calls=tool_calls,
                tool_results=tool_results,
                token_usage=token_usage,
                structured_output=structured_output,
                num_turns=num_turns,
            )
        except (ProcessError, CLIConnectionError) as exc:
            raise BackendSessionError(
                f"subprocess terminated unexpectedly: {exc}"
            ) from exc
```

Then add `query` to the top-level SDK import (the module already imports `ClaudeAgentOptions, ClaudeSDKClient` around line 19-25). Modify that import block:

```python
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    query,
)
```

- [ ] **Step 4: Run the rewritten send() tests**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionSend -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the full claude_backend test module to surface regressions**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py -n auto -v
```

Expected: many passes, plus some failures in `TestClaudeSessionStreaming`, `TestClaudeSessionEnsureClient`, `TestClaudeSessionClose`, `TestClaudeBackendLazyInit::test_session_prepare_invokes_connect`. Those are intentional — Tasks 4-6 will fix them. Capture the failing test names; if any test outside that list fails, stop and investigate before continuing.

- [ ] **Step 6: Commit**

```bash
git add src/holodeck/lib/backends/claude_backend.py tests/unit/lib/backends/test_claude_backend.py
git commit -m "feat(spec-034 P4): ClaudeSession.send() uses query(resume=)"
```

---

## Task 4: Re-implement `ClaudeSession.send_streaming()` on top of `query(resume=)`

**Why:** Same architectural swap as Task 3 but for the AG-UI hot path. `send_streaming()` is what the AG-UI bridge calls per turn. Yields text chunks progressively as `AssistantMessage` frames arrive; captures `session_id` from `ResultMessage` on turn 1.

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py:866-893` (the `send_streaming()` method body)
- Test: `tests/unit/lib/backends/test_claude_backend.py` — rewrite the existing `TestClaudeSessionStreaming` class

- [ ] **Step 1: Write the failing tests**

Replace the existing `TestClaudeSessionStreaming` class with:

```python
@pytest.mark.unit
class TestClaudeSessionStreaming:
    """ClaudeSession.send_streaming() under spec 034 P4."""

    @pytest.mark.asyncio
    async def test_send_streaming_yields_chunks(self) -> None:
        chunk1 = _make_assistant_message([_make_text_block("Hello ")])
        chunk2 = _make_assistant_message([_make_text_block("world!")])
        result_msg = _make_result_message(session_id="sdk-stream-001")

        async def fake_query(prompt, options):
            for m in (chunk1, chunk2, result_msg):
                yield m

        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        with patch(
            "holodeck.lib.backends.claude_backend.query", side_effect=fake_query
        ):
            chunks: list[str] = []
            async for chunk in session.send_streaming("Hi"):
                chunks.append(chunk)

        assert chunks == ["Hello ", "world!"]
        assert session._sdk_session_id == "sdk-stream-001"

    @pytest.mark.asyncio
    async def test_send_streaming_propagates_session_id_on_turn_2(self) -> None:
        captured_options: list[Any] = []

        async def fake_query(prompt, options):
            captured_options.append(options)
            yield _make_assistant_message([_make_text_block("ok")])
            yield _make_result_message(session_id="sdk-stream-XYZ")

        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        with patch(
            "holodeck.lib.backends.claude_backend.query", side_effect=fake_query
        ):
            async for _ in session.send_streaming("Turn 1"):
                pass
            async for _ in session.send_streaming("Turn 2"):
                pass

        assert getattr(captured_options[0], "resume", None) is None
        assert captured_options[1].resume == "sdk-stream-XYZ"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionStreaming -v
```

Expected: 2 failures (current impl still calls `client.query`).

- [ ] **Step 3: Re-implement `send_streaming()`**

Edit `src/holodeck/lib/backends/claude_backend.py`. Replace the entire body of `send_streaming()` (currently lines 866-893):

```python
async def send_streaming(
    self, message: str
) -> AsyncGenerator[str, None]:
    """Send a message and yield text chunks progressively.

    Args:
        message: User message text.

    Yields:
        Text chunks as they arrive from the SDK.

    Raises:
        BackendSessionError: On subprocess or SDK error.
    """
    async with self._send_lock:
        options = self._options_with_hooks()
        if self._sdk_session_id is not None:
            import dataclasses

            options = dataclasses.replace(
                options, resume=self._sdk_session_id
            )

        try:
            async for msg in query(
                prompt=_streaming_user_envelope(message), options=options
            ):
                _maybe_emit_subagent_message(msg, self._tool_event_queue)
                if msg.__class__.__name__ == "AssistantMessage":
                    for block in cast(Any, msg).content:
                        if block.__class__.__name__ == "TextBlock" and block.text:
                            yield block.text
                elif msg.__class__.__name__ == "ResultMessage":
                    self._turn_count += 1
                    if self._sdk_session_id is None:
                        captured = getattr(msg, "session_id", None)
                        if isinstance(captured, str) and captured:
                            self._sdk_session_id = captured
        except (ProcessError, CLIConnectionError) as exc:
            raise BackendSessionError(
                f"subprocess terminated unexpectedly: {exc}"
            ) from exc
```

- [ ] **Step 4: Run the streaming tests**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionStreaming -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/lib/backends/claude_backend.py tests/unit/lib/backends/test_claude_backend.py
git commit -m "feat(spec-034 P4): ClaudeSession.send_streaming() uses query(resume=)"
```

---

## Task 5: `close()` deletes the on-disk JSONL transcript

**Why:** Without cleanup, every ever-opened session leaves a JSONL file at `~/.claude/projects/<encoded-cwd>/<session_id>.jsonl` forever. For a long-lived deployment that's effectively a memory leak on disk. On `close()`, delete the transcript whose name matches `self._sdk_session_id`. The path-encoding rule used by the Claude CLI is to replace `/` with `-` in the absolute cwd.

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py:895-917` (`release_transport`, `close`)
- Test: `tests/unit/lib/backends/test_claude_backend.py` — rewrite `TestClaudeSessionClose`

- [ ] **Step 1: Write the failing test**

Replace the existing `TestClaudeSessionClose` class with:

```python
@pytest.mark.unit
class TestClaudeSessionClose:
    """ClaudeSession.close() under spec 034 P4 — transcript cleanup."""

    @pytest.mark.asyncio
    async def test_close_deletes_transcript_file(self, tmp_path) -> None:
        """close() deletes ``~/.claude/projects/<encoded-cwd>/<id>.jsonl``."""
        # Stand up a fake projects dir; point the helper at it.
        cwd_encoded = str(tmp_path / "fake-cwd").replace("/", "-")
        transcript = tmp_path / "projects" / cwd_encoded / "sess-001.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text('{"type":"user"}\n')

        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        session._sdk_session_id = "sess-001"

        with patch(
            "holodeck.lib.backends.claude_backend._transcript_path",
            return_value=transcript,
        ):
            await session.close()

        assert not transcript.exists()
        assert session._sdk_session_id is None

    @pytest.mark.asyncio
    async def test_close_is_safe_when_transcript_missing(self) -> None:
        """No-op when the transcript file doesn't exist (race / never-spawned)."""
        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        session._sdk_session_id = "never-spawned"
        with patch(
            "holodeck.lib.backends.claude_backend._transcript_path",
            return_value=Path("/nonexistent/path.jsonl"),
        ):
            await session.close()  # must not raise

    @pytest.mark.asyncio
    async def test_close_no_session_id_is_noop(self) -> None:
        """Sessions that never sent a turn have no transcript to delete."""
        session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
        await session.close()  # must not raise
        assert session._sdk_session_id is None

    def test_implements_agent_session_protocol(self) -> None:
        """ClaudeSession still satisfies the AgentSession protocol."""
        assert isinstance(
            ClaudeSession(options=MagicMock()), AgentSession
        )
```

Also add `from pathlib import Path` and `from unittest.mock import patch` to the test imports if not already present.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionClose -v
```

Expected: 3 failures (helper `_transcript_path` doesn't exist yet; current `close()` calls `client.disconnect()` and won't unset `_sdk_session_id`). The protocol-conformance test should already pass.

- [ ] **Step 3: Add `_transcript_path` helper + rewrite `close()`**

Edit `src/holodeck/lib/backends/claude_backend.py`. Add this helper near the other module-level helpers (e.g., just before `class ClaudeSession`):

```python
def _transcript_path(session_id: str, cwd: Path | None = None) -> Path:
    """Return the on-disk JSONL transcript path for a session_id.

    The Claude CLI writes per-session transcripts to
    ``~/.claude/projects/<encoded-cwd>/<session_id>.jsonl`` where
    ``encoded-cwd`` is the absolute cwd with ``/`` replaced by ``-``.
    """
    base = cwd if cwd is not None else Path.cwd()
    encoded = str(base.resolve()).replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded / f"{session_id}.jsonl"
```

Add `from pathlib import Path` to the module-level imports if not already there.

Replace the body of `release_transport()` and `close()` (currently lines 895-917) with:

```python
async def release_transport(self) -> None:
    """No-op under spec 034 P4.

    Retained for backwards compatibility with the chat executor's
    ``_TaskBoundSession``. Under the hybrid-session model each turn's
    subprocess is created and torn down inside ``query()``; there is no
    persistent transport to release between turns.
    """
    return None

async def close(self) -> None:
    """Delete the on-disk JSONL transcript and clear session state.

    Under spec 034 P4 the session has no persistent subprocess to
    disconnect. Conversation state lives on disk at
    ``~/.claude/projects/<encoded-cwd>/<sdk_session_id>.jsonl``. Closing
    the session permanently discards that transcript so the next
    open of the same threadId starts fresh.
    """
    if self._sdk_session_id is not None:
        path = _transcript_path(self._sdk_session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "Failed to delete transcript %s: %s", path, exc
            )
        self._sdk_session_id = None
```

- [ ] **Step 4: Run the close tests**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionClose -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/lib/backends/claude_backend.py tests/unit/lib/backends/test_claude_backend.py
git commit -m "feat(spec-034 P4): ClaudeSession.close() deletes JSONL transcript"
```

---

## Task 6: Strip the dead persistent-client machinery

**Why:** Under P4 nothing reads `self._client`, nothing calls `_ensure_client()`, nothing benefits from `_patch_hooks_for_context_propagation`, and `_DEFAULT_SESSION_ID` is no longer relevant. Leaving them in place would mislead future readers ("does the session use a persistent client or per-turn query?"). Delete the dead code.

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py` — delete `_DEFAULT_SESSION_ID` (around line 69-79), `_patch_hooks_for_context_propagation` (around line 485-540), `_ensure_client` method (around line 739-757), `_client` field, `claude_agent_sdk` module-level import (used only inside `_ensure_client`)
- Modify: `ClaudeSession.prepare()` body becomes a no-op (kept as a shim — `_TaskBoundSession` calls it).
- Test: delete `TestClaudeSessionEnsureClient` class entirely. Modify `TestClaudeBackendLazyInit::test_session_prepare_invokes_connect` to assert `prepare()` is a no-op (no connect to invoke).

- [ ] **Step 1: Delete the dead code in source**

Edit `src/holodeck/lib/backends/claude_backend.py`:

1. Remove the `_DEFAULT_SESSION_ID` constant block (around lines 69-79).
2. Remove `_patch_hooks_for_context_propagation` function entirely.
3. Remove `_ensure_client()` method from `ClaudeSession`.
4. Remove `self._client: ClaudeSDKClient | None = None` from `__init__`.
5. Replace `ClaudeSession.prepare()` body with:

```python
async def prepare(self) -> None:
    """No-op under spec 034 P4.

    Retained for backwards compatibility with the chat executor's
    ``_TaskBoundSession``. Under the hybrid-session model the SDK's
    anyio task group is created inside each ``query()`` call frame,
    so there is no task-binding to do up front.
    """
    return None
```

6. Remove `import claude_agent_sdk` at the top of the file (it was only used inside `_ensure_client`). Leave `from claude_agent_sdk import ...` (used by Task 3).
7. Remove `ClaudeSDKClient` from the `from claude_agent_sdk import (...)` block — it is no longer referenced.

- [ ] **Step 2: Delete the obsolete test classes**

In `tests/unit/lib/backends/test_claude_backend.py`:

1. Delete the entire `TestClaudeSessionEnsureClient` class (around line 2101).
2. Modify `TestClaudeBackendLazyInit::test_session_prepare_invokes_connect`. Find it (around line 1448) and replace its body with:

```python
@pytest.mark.asyncio
async def test_session_prepare_is_noop(self) -> None:
    """spec 034 P4: ClaudeSession.prepare() no longer connects a client —
    each query() call manages its own transport. prepare() must succeed
    without raising so existing _TaskBoundSession callers stay working.
    """
    session = ClaudeSession(options=MagicMock())
    await session.prepare()  # must not raise
    # No client to assert against; just confirm field is gone.
    assert not hasattr(session, "_client")
```

- [ ] **Step 3: Run the full claude_backend test module**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py -n auto -v
```

Expected: all green. If anything fails outside the test classes you modified, stop and inspect — it is likely a test that was reaching into a deleted attribute.

- [ ] **Step 4: Run the full unit suite to catch fan-out regressions**

```bash
pytest tests/unit -n auto -q
```

Expected: all green. Watch for failures in `tests/unit/serve/` (the AG-UI bridge tests use real `ClaudeSession` instances via the bridge — they may patch removed attributes).

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/lib/backends/claude_backend.py tests/unit/lib/backends/test_claude_backend.py
git commit -m "refactor(spec-034 P4): drop dead persistent-client machinery"
```

---

## Task 7: Add unit test for the lock — concurrent sends serialise

**Why:** Two concurrent `send()` calls on the same session with the same `resume=` would each spawn a fresh subprocess reading and appending the same JSONL transcript. The `_send_lock` added in Task 2 serialises them. Add an explicit test so the next person who tries to "optimise" by removing the lock breaks a test instead of production.

**Files:**
- Test: `tests/unit/lib/backends/test_claude_backend.py` — extend `TestClaudeSessionP4Fields` from Task 2

- [ ] **Step 1: Write the failing test**

Append to `TestClaudeSessionP4Fields`:

```python
@pytest.mark.asyncio
async def test_concurrent_sends_serialise(self) -> None:
    """Two concurrent send() calls must not invoke query() in parallel.
    The lock prevents transcript-write races under the resume= model.
    """
    in_flight = 0
    max_concurrent = 0

    async def fake_query(prompt, options):
        nonlocal in_flight, max_concurrent
        in_flight += 1
        max_concurrent = max(max_concurrent, in_flight)
        # Yield control so the second send() has a chance to enter.
        await asyncio.sleep(0)
        yield _make_assistant_message()
        yield _make_result_message(session_id="sess-conc")
        in_flight -= 1

    session = ClaudeSession(options=MagicMock(spec=ClaudeAgentOptions))
    with patch(
        "holodeck.lib.backends.claude_backend.query", side_effect=fake_query
    ):
        await asyncio.gather(session.send("a"), session.send("b"))

    assert max_concurrent == 1, (
        "expected lock to serialise concurrent sends, "
        f"got max_concurrent={max_concurrent}"
    )
```

- [ ] **Step 2: Run it**

```bash
pytest tests/unit/lib/backends/test_claude_backend.py::TestClaudeSessionP4Fields::test_concurrent_sends_serialise -v
```

Expected: PASS (the lock was added in Task 2 and used in Tasks 3-4). If it fails, the lock is not being acquired — go inspect `send()` / `send_streaming()`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/lib/backends/test_claude_backend.py
git commit -m "test(spec-034 P4): cover send-lock serialisation under concurrency"
```

---

## Task 8: Quality gates

**Why:** Before deploy validation, lock in the static-analysis story so the deploy iteration loop doesn't surface formatting / typing surprises.

- [ ] **Step 1: Format + lint**

```bash
make format && make lint
```

Expected: both pass. If lint fails, fix the report.

- [ ] **Step 2: Type-check**

```bash
rm -rf .mypy_cache && make type-check
```

Expected: no new errors compared to baseline (pre-existing errors in `keyword_search.py`, `vector_store.py`, `hierarchical_document_tool.py`, `serve/server.py`, etc. are unrelated to P4 and may stay). If the diff between this run and `git stash && make type-check` shows new errors in `claude_backend.py`, fix them.

- [ ] **Step 3: Security scan**

```bash
make security
```

Expected: no high-severity findings introduced by P4.

- [ ] **Step 4: Commit any auto-fixes from format/lint**

```bash
git status
# If anything changed under src/ or tests/:
git add -p  # interactive review
git commit -m "chore(spec-034 P4): formatting + lint cleanup"
```

If nothing changed, skip this step.

---

## Task 9: End-to-end deploy validation

**Why:** The unit tests cover the session-model swap, but the real binary risks (AG-UI stream-shape compat, OTel context propagation, subprocess spawn jitter under concurrent load) only surface in a live deployment. Run the deploy validation loop from `CLAUDE.md` against `sample/financial-assistant/claude`.

**This task must be approved by the human operator before running** (it builds a docker image, pushes to GHCR, rolls a live ACA revision). Do not run automatically.

- [ ] **Step 1: Get approval**

Ask: "Ready to run the end-to-end deploy validation loop for P4 on `sample/financial-assistant/claude`? It will rebuild the base image with the working-tree wheel, push, and roll a new ACA revision."

Block until the operator confirms.

- [ ] **Step 2: Run the validation loop**

Follow the exact sequence in `CLAUDE.md` under "End-to-End Deploy Validation Loop":

```bash
# 1. Local wheel
rm -rf dist && uv build --wheel

# 2. Local base image
docker buildx build --platform linux/amd64 --no-cache \
    -f docker/Dockerfile.local \
    -t ghcr.io/justinbarias/holodeck-base:latest --load .
docker run --rm --entrypoint python ghcr.io/justinbarias/holodeck-base:latest \
    -c "import holodeck; print(holodeck.__version__)"

# 3. Flip pull=True → pull=False in src/holodeck/deploy/builder.py (will revert
#    in step 7).

# 4. Agent image
cd sample/financial-assistant/claude
holodeck deploy build
docker push ghcr.io/justinbarias/holodeck-financial-assistant:<tag>

# 5. ACA deploy
holodeck deploy run

# 6. Health + smoke
URL=https://financial-assistant.nicemoss-50caf9f5.eastus.azurecontainerapps.io
until curl -sf -o /dev/null --max-time 5 "$URL/health"; do sleep 3; done
curl -sS -X POST "$URL/awp" -H 'content-type: application/json' \
    -d '{"threadId":"p4-smoke","runId":"r1","state":{},"messages":[{"id":"m1","role":"user","content":"What were ALXN'\''s 2007 rental payments?"}],"tools":[],"context":[],"forwardedProps":{}}' \
    --max-time 180

# 7. Revert builder.py pull=False → pull=True.
```

- [ ] **Step 3: Verify AG-UI stream shape**

The smoke curl must return a 200 with a coherent answer referencing ALXN/2007. If the stream looks malformed (events out of order, missing TEXT_MESSAGE_CHUNK, missing RUN_FINISHED), capture the raw response and stop — the SDK-message-to-AG-UI translation layer has a P4-specific bug that needs fixing.

- [ ] **Step 4: Multi-turn AG-UI verification**

Re-use the same `threadId` (`p4-smoke`) and send a follow-up:

```bash
curl -sS -X POST "$URL/awp" -H 'content-type: application/json' \
    -d '{"threadId":"p4-smoke","runId":"r2","state":{},"messages":[{"id":"m1","role":"user","content":"What were ALXN'\''s 2007 rental payments?"},{"id":"m2","role":"assistant","content":"<answer-from-r1>"},{"id":"m3","role":"user","content":"And the prior year?"}],"tools":[],"context":[],"forwardedProps":{}}' \
    --max-time 180
```

The response must reference ALXN/2007 (the anchor from turn 1) — if it says "which company?" or asks for clarification, `resume=` is not actually being propagated into the bridge. Spike v2 proved the SDK side works; this would mean the AG-UI bridge is dropping the session id between turns.

- [ ] **Step 5: Concurrent-burst validation (regression of P1 OOM scenario)**

In one shell:

```bash
for tid in p4-burst-{1..5}; do
  curl -sS -X POST "$URL/awp" -H 'content-type: application/json' \
      -d "{\"threadId\":\"$tid\",\"runId\":\"r1\",\"state\":{},\"messages\":[{\"id\":\"m1\",\"role\":\"user\",\"content\":\"What were ALXN's 2007 rental payments?\"}],\"tools\":[],\"context\":[],\"forwardedProps\":{}}" \
      --max-time 180 \
      -o /tmp/p4-burst-$tid.out -w "%{http_code} t=%{time_total}s\n" &
done
wait
```

Expect: all five return 200 within ~60s. Under P1 with a 2 GiB replica + cap=4, the 5th burst returned 429. Under P4 with idle sessions costing ~0 memory, the cap should not bind here. If anything returns 429 or 5xx, capture container memory at the time of the burst and add to the spec.

- [ ] **Step 6: Capture observations**

Record in `specs/034-production-hardening/2026-05-18-production-hardening-for-claude-agents.md` under the Phase 4 section:
- Per-turn AG-UI latency vs P1 baseline
- Whether OTel traces propagated (eyeball one trace in Aspire / Azure Monitor)
- Whether the 5-concurrent burst succeeded
- Replica memory at concurrent-burst peak

- [ ] **Step 7: Commit the spec update**

```bash
git add specs/034-production-hardening/2026-05-18-production-hardening-for-claude-agents.md
git commit -m "docs(spec-034): P4 end-to-end deploy validation results"
```

- [ ] **Step 8: Open PR**

```bash
git push -u origin feature/034-p4-hybrid-sessions
gh pr create --title "feat(spec-034): P4 hybrid sessions (subprocess pooling via query+resume)" --body "$(cat <<'EOF'
## Summary
- Swap `ClaudeSession`'s persistent `ClaudeSDKClient` for per-turn `query(resume=session_id)` calls.
- Idle sessions now cost ~0 resident memory; the `SessionStore` cap correctly applies to concurrent active turns.
- Streaming-mode prompts (AsyncIterable envelope) are mandatory — string mode closes stdin and deadlocks SDK MCP tool callbacks (spike v1 surfaced this).
- `close()` deletes the on-disk JSONL transcript; per-session `asyncio.Lock` serialises concurrent sends to prevent transcript-write races.
- Spike data and design recorded in `specs/034-production-hardening/2026-05-18-production-hardening-for-claude-agents.md` Phase 4.

## Test plan
- [x] Unit: `TestClaudeSessionSend`, `TestClaudeSessionStreaming`, `TestClaudeSessionClose`, `TestClaudeSessionP4Fields`, `TestStreamingUserEnvelope`.
- [x] Lint + type + security gates green.
- [x] End-to-end deploy validation on `sample/financial-assistant/claude` (single-turn smoke, multi-turn resume, 5-concurrent burst).
- [x] AG-UI stream shape preserved across the bridge.

## Closes
Phase 4 of spec 034 (hybrid sessions / subprocess pooling).
EOF
)"
```

---

## Known follow-ups (do NOT include in this PR)

These were surfaced during spike v2 and remain open. Add as TODOs against spec 034 P4 but do not implement in this PR (scope discipline):

1. **Transcript size budget.** Long-running sessions accumulate every tool-use/tool-result block in the JSONL; 50 turns × 5 retrievals × 8 KiB ≈ 2 MiB transcript that gets re-read every turn. Add a `claude.max_transcript_bytes` knob + rolling-summary fallback before exposing P4 to chat UIs with multi-hour sessions.
2. **Idle-session TTL.** `ServerSession` objects in `SessionStore` still pin a Python object per threadId forever; the on-disk transcript also lives forever between turns. Add a TTL (e.g. 1 hour idle) that calls `ClaudeSession.close()` and removes the entry from `SessionStore`.
3. **OTel context propagation.** The `_patch_hooks_for_context_propagation` workaround targeted a persistent `ClaudeSDKClient` that no longer exists under P4. Verify the OTel instrumentor still wires invocation context through top-level `query()` — if not, file a follow-up to either patch `claude_agent_sdk.query` or set the ContextVar before the call in `send()` / `send_streaming()`.
4. **Drop `_TaskBoundSession`.** Under P4 it's a no-op wrapper (the persistent-transport-task-binding problem it solved no longer exists). Removing it simplifies `chat/executor.py` and `serve/server.py`. Separate PR.
5. **Repurpose `SessionStore` cap from "open sessions" to "concurrent active turns".** Today it counts threadIds; under P4 the OOM constraint is concurrent active turns, not idle threads. Until this lands, the cap is still binding but for the wrong reason (it'll reject a 6th threadId even if all 5 prior threads are idle).
