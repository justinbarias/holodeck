"""Additional coverage tests for chat/executor.py.

Covers untested code paths: callbacks, _TaskBoundSession actor,
error handling branches, edge cases in _convert_tool_calls,
shutdown/clear_history edge cases, and streaming history.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.chat.executor import (
    AgentExecutor,
    AgentResponse,
    _require_agui_session,
    _TaskBoundSession,
)
from holodeck.lib.backends.base import (
    AgentSession,
    BackendInitError,
    BackendSessionError,
    ExecutionResult,
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    """Test on_execution_start and on_execution_complete callbacks."""

    @pytest.mark.asyncio
    async def test_on_execution_start_called(
        self, make_agent, make_mock_backend
    ) -> None:
        """on_execution_start callback receives the user message."""
        mock_backend, _ = make_mock_backend("Reply")
        start_cb = MagicMock()

        executor = AgentExecutor(
            make_agent(), backend=mock_backend, on_execution_start=start_cb
        )
        await executor.execute_turn("Hello")

        start_cb.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_on_execution_complete_called(
        self, make_agent, make_mock_backend
    ) -> None:
        """on_execution_complete callback receives the AgentResponse."""
        mock_backend, _ = make_mock_backend("Reply")
        complete_cb = MagicMock()

        executor = AgentExecutor(
            make_agent(), backend=mock_backend, on_execution_complete=complete_cb
        )
        await executor.execute_turn("Hello")

        complete_cb.assert_called_once()
        response_arg = complete_cb.call_args[0][0]
        assert isinstance(response_arg, AgentResponse)
        assert response_arg.content == "Reply"

    @pytest.mark.asyncio
    async def test_both_callbacks_called_in_order(
        self, make_agent, make_mock_backend
    ) -> None:
        """Both callbacks fire in correct order: start before complete."""
        mock_backend, _ = make_mock_backend("Reply")
        call_order: list[str] = []
        start_cb = MagicMock(side_effect=lambda msg: call_order.append("start"))
        complete_cb = MagicMock(side_effect=lambda resp: call_order.append("complete"))

        executor = AgentExecutor(
            make_agent(),
            backend=mock_backend,
            on_execution_start=start_cb,
            on_execution_complete=complete_cb,
        )
        await executor.execute_turn("Hello")

        assert call_order == ["start", "complete"]


# ---------------------------------------------------------------------------
# Error handling in execute_turn
# ---------------------------------------------------------------------------


class TestExecuteTurnErrors:
    """Test error handling branches in execute_turn."""

    @pytest.mark.asyncio
    async def test_backend_init_error_wrapped_as_runtime(
        self, make_agent, make_mock_backend
    ) -> None:
        """BackendInitError is wrapped as RuntimeError."""
        mock_backend, _ = make_mock_backend()
        mock_backend.create_session.side_effect = BackendInitError("Init failed")

        executor = AgentExecutor(make_agent(), backend=mock_backend)

        with pytest.raises(RuntimeError, match="Init failed"):
            await executor.execute_turn("Hello")

    @pytest.mark.asyncio
    async def test_generic_exception_wrapped_as_runtime(
        self, make_agent, make_mock_backend
    ) -> None:
        """Generic exceptions are wrapped as RuntimeError."""
        mock_backend, mock_session = make_mock_backend()
        mock_session.send.side_effect = ValueError("Unexpected error")

        executor = AgentExecutor(make_agent(), backend=mock_backend)

        with pytest.raises(RuntimeError, match="Unexpected error"):
            await executor.execute_turn("Hello")

    @pytest.mark.asyncio
    async def test_runtime_error_passthrough(
        self, make_agent, make_mock_backend
    ) -> None:
        """RuntimeError is re-raised directly, not double-wrapped."""
        mock_backend, mock_session = make_mock_backend()
        mock_session.send.side_effect = RuntimeError("Direct runtime error")

        executor = AgentExecutor(make_agent(), backend=mock_backend)

        with pytest.raises(RuntimeError, match="Direct runtime error"):
            await executor.execute_turn("Hello")


class TestExecuteTurnAgui:
    """Test AgentExecutor.execute_turn_agui()."""

    @pytest.mark.asyncio
    async def test_execute_turn_agui_streams_backend_events(self, make_agent) -> None:
        """execute_turn_agui() delegates to an AG-UI-capable session."""
        from ag_ui.core import EventType, RunAgentInput, RunStartedEvent, UserMessage

        class AguiSession:
            async def prepare(self) -> None:
                return None

            async def send(self, message: str) -> ExecutionResult:
                return ExecutionResult(response="unused")

            async def send_streaming(self, message: str):
                yield "unused"

            async def send_agui(
                self,
                input_data: RunAgentInput,
                message_override: str | None = None,
            ):
                assert message_override == "override"
                yield RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    input=input_data,
                )

            async def close(self) -> None:
                return None

        mock_backend = AsyncMock()
        mock_backend.create_session = AsyncMock(return_value=AguiSession())
        input_data = RunAgentInput(
            thread_id="thread-1",
            run_id="run-1",
            messages=[UserMessage(id="msg-1", role="user", content="hi")],
            tools=[],
            context=[],
            state=None,
            forwarded_props=None,
        )

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        events = [
            event async for event in executor.execute_turn_agui(input_data, "override")
        ]

        assert [event.type for event in events] == [EventType.RUN_STARTED]

    @pytest.mark.asyncio
    async def test_execute_turn_agui_wraps_backend_init_error(self, make_agent) -> None:
        """BackendInitError is wrapped with AG-UI-specific context."""
        from ag_ui.core import RunAgentInput, UserMessage

        mock_backend = AsyncMock()
        mock_backend.create_session.side_effect = BackendInitError("init failed")
        input_data = RunAgentInput(
            thread_id="thread-1",
            run_id="run-1",
            messages=[UserMessage(id="msg-1", role="user", content="hi")],
            tools=[],
            context=[],
            state=None,
            forwarded_props=None,
        )

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        with pytest.raises(RuntimeError, match="Agent AG-UI execution failed"):
            async for _ in executor.execute_turn_agui(input_data):
                pass

    @pytest.mark.asyncio
    async def test_backend_session_error_propagates(
        self, make_agent, make_mock_backend
    ) -> None:
        """BackendSessionError propagates without wrapping."""
        mock_backend, mock_session = make_mock_backend()
        mock_session.send.side_effect = BackendSessionError("Session broke")

        executor = AgentExecutor(make_agent(), backend=mock_backend)

        with pytest.raises(BackendSessionError, match="Session broke"):
            await executor.execute_turn("Hello")


# ---------------------------------------------------------------------------
# Error handling in execute_turn_streaming
# ---------------------------------------------------------------------------


class TestStreamingErrors:
    """Test error handling branches in execute_turn_streaming."""

    @pytest.mark.asyncio
    async def test_streaming_backend_session_error(
        self, make_agent, make_mock_backend
    ) -> None:
        """BackendSessionError propagates from streaming."""
        mock_backend, mock_session = make_mock_backend()

        async def _bad_stream(message: str):
            raise BackendSessionError("Stream broke")
            yield  # make it a generator  # pragma: no cover

        mock_session.send_streaming = _bad_stream

        executor = AgentExecutor(make_agent(), backend=mock_backend)

        with pytest.raises(BackendSessionError, match="Stream broke"):
            async for _ in executor.execute_turn_streaming("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_streaming_backend_init_error(self, make_agent) -> None:
        """BackendInitError during streaming is wrapped as RuntimeError."""
        mock_backend = AsyncMock()
        mock_backend.create_session.side_effect = BackendInitError("No init")

        executor = AgentExecutor(make_agent(), backend=mock_backend)

        with pytest.raises(RuntimeError, match="No init"):
            async for _ in executor.execute_turn_streaming("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_streaming_tracks_history(
        self, make_agent, make_mock_backend
    ) -> None:
        """Streaming appends user/assistant entries to history."""
        mock_backend, _ = make_mock_backend()

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        chunks: list[str] = []
        async for chunk in executor.execute_turn_streaming("Hi"):
            chunks.append(chunk)

        history = executor.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1] == {"role": "assistant", "content": "Hello world"}


# ---------------------------------------------------------------------------
# _convert_tool_calls edge cases
# ---------------------------------------------------------------------------


class TestConvertToolCalls:
    """Test _convert_tool_calls edge cases."""

    @pytest.mark.asyncio
    async def test_tool_call_missing_name_defaults_to_unknown(
        self, make_agent, make_mock_backend
    ) -> None:
        """Tool call without 'name' key defaults to 'unknown'."""
        mock_backend, _ = make_mock_backend(
            response_text="done",
            tool_calls=[{"arguments": {"q": "test"}}],
        )

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        response = await executor.execute_turn("Go")

        assert len(response.tool_executions) == 1
        assert response.tool_executions[0].tool_name == "unknown"

    @pytest.mark.asyncio
    async def test_tool_call_missing_arguments_defaults_to_empty(
        self, make_agent, make_mock_backend
    ) -> None:
        """Tool call without 'arguments' key defaults to empty dict."""
        mock_backend, _ = make_mock_backend(
            response_text="done",
            tool_calls=[{"name": "search"}],
        )

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        response = await executor.execute_turn("Go")

        assert len(response.tool_executions) == 1
        assert response.tool_executions[0].parameters == {}

    def test_convert_tool_calls_exception_returns_empty(self, make_agent) -> None:
        """Exception during conversion returns empty list."""
        executor = AgentExecutor(make_agent())
        # Pass something that will cause .get() to fail
        result = executor._convert_tool_calls([None])  # type: ignore[list-item]
        assert result == []


# ---------------------------------------------------------------------------
# Shutdown and clear_history edge cases
# ---------------------------------------------------------------------------


class TestShutdownEdgeCases:
    """Test shutdown and clear_history in various states."""

    @pytest.mark.asyncio
    async def test_shutdown_when_idle(self, make_agent) -> None:
        """Shutdown with no session or backend is a no-op."""
        executor = AgentExecutor(make_agent())
        await executor.shutdown()
        assert executor._session is None
        assert executor._backend is None

    @pytest.mark.asyncio
    async def test_shutdown_exception_swallowed(
        self, make_agent, make_mock_backend
    ) -> None:
        """Shutdown logs but doesn't raise on error."""
        mock_backend, mock_session = make_mock_backend()
        mock_session.close.side_effect = RuntimeError("Close failed")

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        await executor.execute_turn("Hello")

        # Should not raise
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_clear_history_when_no_session(self, make_agent) -> None:
        """clear_history without a session is safe."""
        executor = AgentExecutor(make_agent())
        await executor.clear_history()
        assert executor.get_history() == []


# ---------------------------------------------------------------------------
# get_history returns a copy
# ---------------------------------------------------------------------------


class TestGetHistoryCopy:
    """Test that get_history returns a copy, not a reference."""

    @pytest.mark.asyncio
    async def test_get_history_returns_copy(
        self, make_agent, make_mock_backend
    ) -> None:
        """Mutating returned history does not affect internal state."""
        mock_backend, _ = make_mock_backend("Reply")

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        await executor.execute_turn("Hello")

        history = executor.get_history()
        history.clear()

        # Internal history should still have entries
        assert len(executor.get_history()) == 2


# ---------------------------------------------------------------------------
# _TaskBoundSession actor
# ---------------------------------------------------------------------------


class TestTaskBoundSession:
    """Test the _TaskBoundSession actor wrapper."""

    @pytest.mark.asyncio
    async def test_send_agui_delegates_to_inner_session(self) -> None:
        """send_agui() yields events from an AG-UI-capable inner session."""
        from ag_ui.core import EventType, RunAgentInput, RunStartedEvent, UserMessage

        class AguiCapableSession:
            async def prepare(self) -> None:
                return None

            async def send(self, message: str) -> ExecutionResult:
                return ExecutionResult(response="unused")

            async def send_streaming(self, message: str):
                yield "unused"

            async def send_agui(
                self,
                input_data: RunAgentInput,
                message_override: str | None = None,
            ):
                yield RunStartedEvent(
                    type=EventType.RUN_STARTED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    input=input_data,
                )

            async def close(self) -> None:
                return None

        input_data = RunAgentInput(
            thread_id="thread-1",
            run_id="run-1",
            messages=[UserMessage(id="msg-1", role="user", content="hi")],
            tools=[],
            context=[],
            state=None,
            forwarded_props=None,
        )

        actor = _TaskBoundSession(AguiCapableSession())
        events = [event async for event in actor.send_agui(input_data)]

        assert [event.type for event in events] == [EventType.RUN_STARTED]

    def test_require_agui_session_rejects_missing_capability(self) -> None:
        """A non-AG-UI session fails with a clear AttributeError."""
        with pytest.raises(AttributeError, match="does not support AG-UI"):
            _require_agui_session(None)

    @pytest.mark.asyncio
    async def test_send_delegates_to_inner_session(self) -> None:
        """send() delegates to the underlying session via the actor."""
        inner = AsyncMock(spec=AgentSession)
        inner.send.return_value = ExecutionResult(response="Hi")

        actor = _TaskBoundSession(inner)
        await actor.start()

        result = await actor.send("Hello")

        assert result.response == "Hi"
        inner.send.assert_awaited_once_with("Hello")
        await actor.close()

    @pytest.mark.asyncio
    async def test_streaming_delegates_to_inner_session(self) -> None:
        """send_streaming() yields chunks from the underlying session."""
        inner = AsyncMock(spec=AgentSession)

        async def _stream(msg: str):
            for chunk in ["A", "B", "C"]:
                yield chunk

        inner.send_streaming = _stream

        actor = _TaskBoundSession(inner)
        await actor.start()

        chunks: list[str] = []
        async for chunk in actor.send_streaming("Go"):
            chunks.append(chunk)

        assert chunks == ["A", "B", "C"]
        await actor.close()

    @pytest.mark.asyncio
    async def test_send_propagates_exception(self) -> None:
        """Exceptions from inner session.send() propagate to caller."""
        inner = AsyncMock(spec=AgentSession)
        inner.send.side_effect = BackendSessionError("Boom")

        actor = _TaskBoundSession(inner)
        await actor.start()

        with pytest.raises(BackendSessionError, match="Boom"):
            await actor.send("Hello")

        await actor.close()

    @pytest.mark.asyncio
    async def test_streaming_propagates_exception(self) -> None:
        """Exceptions from inner send_streaming() propagate to caller."""
        inner = AsyncMock(spec=AgentSession)

        async def _bad_stream(msg: str):
            yield "partial"
            raise BackendSessionError("Stream failed")

        inner.send_streaming = _bad_stream

        actor = _TaskBoundSession(inner)
        await actor.start()

        with pytest.raises(BackendSessionError, match="Stream failed"):
            async for _ in actor.send_streaming("Go"):
                pass

        await actor.close()

    @pytest.mark.asyncio
    async def test_close_shuts_down_actor_and_inner(self) -> None:
        """close() stops the actor loop and closes the inner session."""
        inner = AsyncMock(spec=AgentSession)
        inner.send.return_value = ExecutionResult(response="Hi")

        actor = _TaskBoundSession(inner)
        await actor.start()

        await actor.send("Hello")
        await actor.close()

        inner.close.assert_awaited_once()
        assert actor._task is not None
        assert actor._task.done()

    @pytest.mark.asyncio
    async def test_cancelled_future_at_pickup_is_skipped(self) -> None:
        """Items cancelled before the actor picks them up don't burn a turn."""

        inner = AsyncMock(spec=AgentSession)
        inner.send = AsyncMock(return_value=ExecutionResult(response="ok"))

        actor = _TaskBoundSession(inner)
        # Don't start the loop yet — queue an already-cancelled item first.
        future: asyncio.Future[ExecutionResult] = (
            asyncio.get_running_loop().create_future()
        )
        future.cancel()
        await actor._queue.put(("ghost", future, None))

        await actor.start()

        # A real send should still go through.
        result = await actor.send("real")
        assert result.response == "ok"
        # Inner.send must have been called exactly once — the cancelled item
        # was skipped, not dispatched.
        assert inner.send.await_count == 1

        await actor.close()

    @pytest.mark.asyncio
    async def test_start_prepares_inner_session_in_actor_task(self) -> None:
        """Actor calls session.prepare() at startup so connect() binds here.

        Regression: the SDK's anyio task group is bound to whichever task
        called ``connect()``. If that's an HTTP request task, the request
        ends after turn 1 and the SDK's ``_read_messages`` background task
        gets cancelled, leaving turn 2's ``receive_response()`` hanging on
        a memory stream nobody fills. The actor must own the connect call,
        which it performs by invoking ``session.prepare()`` in its own task.
        """
        inner = AsyncMock(spec=AgentSession)
        inner.send.return_value = ExecutionResult(response="ok")

        actor = _TaskBoundSession(inner)
        await actor.start()

        inner.prepare.assert_awaited_once()
        await actor.close()

    @pytest.mark.asyncio
    async def test_start_propagates_prepare_failure(self) -> None:
        """If prepare() fails during startup, start() raises the error."""
        inner = AsyncMock(spec=AgentSession)
        inner.prepare = AsyncMock(side_effect=RuntimeError("connect boom"))

        actor = _TaskBoundSession(inner)
        with pytest.raises(RuntimeError, match="connect boom"):
            await actor.start()

    @pytest.mark.asyncio
    async def test_prepare_on_actor_is_noop(self) -> None:
        """_TaskBoundSession.prepare() is a no-op; actor handles inner.prepare()
        inside ``start()`` to bind the connect to the actor task.
        """
        inner = AsyncMock(spec=AgentSession)

        actor = _TaskBoundSession(inner)
        assert await actor.prepare() is None

    @pytest.mark.asyncio
    async def test_close_idempotent_when_task_done(self) -> None:
        """close() handles already-finished task gracefully."""
        inner = AsyncMock(spec=AgentSession)

        actor = _TaskBoundSession(inner)
        await actor.start()
        await actor.close()
        # Second close should be safe
        await actor.close()

        # inner.close called at least once
        assert inner.close.await_count >= 1


# ---------------------------------------------------------------------------
# release_transport_after_turn path
# ---------------------------------------------------------------------------


class TestReleaseTransportAfterTurn:
    """Test _TaskBoundSession wrapping via release_transport_after_turn."""

    @pytest.mark.asyncio
    async def test_task_bound_session_used_when_flag_set(
        self, make_agent, make_mock_backend
    ) -> None:
        """Session is wrapped in _TaskBoundSession when flag is True."""
        mock_backend, _ = make_mock_backend("Reply")

        executor = AgentExecutor(
            make_agent(),
            backend=mock_backend,
            release_transport_after_turn=True,
        )
        await executor.execute_turn("Hello")

        assert isinstance(executor._session, _TaskBoundSession)
        await executor.shutdown()

    @pytest.mark.asyncio
    async def test_raw_session_used_when_flag_not_set(
        self, make_agent, make_mock_backend
    ) -> None:
        """Session is NOT wrapped when flag is False (default)."""
        mock_backend, _ = make_mock_backend("Reply")

        executor = AgentExecutor(make_agent(), backend=mock_backend)
        await executor.execute_turn("Hello")

        assert not isinstance(executor._session, _TaskBoundSession)
        await executor.shutdown()


class TestTaskBoundSessionOwnership:
    """The actor builds, prepares, closes, and tears down inside its own task.

    Regression for serve turn-2 failures on the OpenAI Agents backend: the
    backend used to be initialised (stdio MCP servers connected) in the HTTP
    request task of turn 1, whose anyio streams died with that task, so the
    next turn on the same thread failed with ``ClosedResourceError``.
    """

    @staticmethod
    def _session(record: dict[str, asyncio.Task[None] | None]) -> AgentSession:
        class Inner:
            async def prepare(self) -> None:
                record["prepare"] = asyncio.current_task()

            async def send(self, message: str) -> ExecutionResult:
                record["send"] = asyncio.current_task()
                return ExecutionResult(response="ok")

            async def send_streaming(self, message: str):
                yield "ok"

            async def close(self) -> None:
                record["close"] = asyncio.current_task()

        return Inner()  # type: ignore[return-value]

    @pytest.mark.asyncio
    async def test_factory_prepare_close_and_teardown_share_actor_task(
        self,
    ) -> None:
        record: dict[str, asyncio.Task[None] | None] = {}

        async def factory() -> AgentSession:
            record["factory"] = asyncio.current_task()
            return self._session(record)

        async def teardown() -> None:
            record["teardown"] = asyncio.current_task()

        actor = _TaskBoundSession(session_factory=factory, teardown=teardown)
        await actor.start()
        await actor.send("hi")
        await actor.close()

        tasks = {record[k] for k in ("factory", "prepare", "send", "close", "teardown")}
        assert len(tasks) == 1, "every lifecycle step must run in the actor task"
        assert tasks != {asyncio.current_task()}

    @pytest.mark.asyncio
    async def test_factory_failure_surfaces_from_start(self) -> None:
        async def factory() -> AgentSession:
            raise BackendInitError("mcp connect failed")

        actor = _TaskBoundSession(session_factory=factory)
        with pytest.raises(BackendInitError, match="mcp connect failed"):
            await actor.start()

    def test_requires_exactly_one_source(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _TaskBoundSession()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        record: dict[str, asyncio.Task[None] | None] = {}
        calls: list[str] = []

        async def teardown() -> None:
            calls.append("teardown")

        actor = _TaskBoundSession(session=self._session(record), teardown=teardown)
        await actor.start()
        await actor.close()
        await actor.close()
        assert calls == ["teardown"]

    @pytest.mark.asyncio
    async def test_executor_selects_backend_inside_actor_task(
        self, make_agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With release_transport_after_turn, BackendSelector runs in the actor."""
        from holodeck.chat import executor as executor_module

        record: dict[str, asyncio.Task[None] | None] = {}
        backend = MagicMock()
        backend.create_session = AsyncMock(
            side_effect=lambda **_: self._session(record)
        )
        backend.teardown = AsyncMock(
            side_effect=lambda: record.__setitem__("teardown", asyncio.current_task())
        )

        async def select(*args, **kwargs):
            record["select"] = asyncio.current_task()
            return backend

        monkeypatch.setattr(executor_module.BackendSelector, "select", select)

        executor = AgentExecutor(make_agent(), release_transport_after_turn=True)
        response = await executor.execute_turn("Hello")
        assert response.content == "ok"
        await executor.shutdown()

        assert executor._backend is None
        tasks = {record[k] for k in ("select", "prepare", "send", "close", "teardown")}
        assert len(tasks) == 1
        assert tasks != {asyncio.current_task()}


class TestExecuteTurnErrorResults:
    """Backends report failures as ``is_error`` results; chat must surface them."""

    @pytest.mark.asyncio
    async def test_error_result_without_content_raises(
        self, make_agent, make_mock_backend
    ) -> None:
        mock_backend, mock_session = make_mock_backend()
        mock_session.send.return_value = ExecutionResult(
            response="", is_error=True, error_reason="ClosedResourceError: "
        )
        executor = AgentExecutor(make_agent(), backend=mock_backend)
        with pytest.raises(BackendSessionError, match="ClosedResourceError"):
            await executor.execute_turn("Hello")

    @pytest.mark.asyncio
    async def test_error_result_with_partial_content_is_returned(
        self, make_agent, make_mock_backend
    ) -> None:
        mock_backend, mock_session = make_mock_backend()
        mock_session.send.return_value = ExecutionResult(
            response="partial", is_error=True, error_reason="budget exceeded"
        )
        executor = AgentExecutor(make_agent(), backend=mock_backend)
        response = await executor.execute_turn("Hello")
        assert response.content == "partial"


@pytest.mark.unit
class TestTaskBoundSessionFailureAndCancellation:
    """Stack-review findings: cleanup after a failed start, abandoned close."""

    @staticmethod
    def _inner(
        record: dict[str, object],
        *,
        prepare_error: Exception | None = None,
        block: asyncio.Event | None = None,
    ) -> AgentSession:
        class Inner:
            async def prepare(self) -> None:
                record["prepare_task"] = asyncio.current_task()
                if prepare_error is not None:
                    raise prepare_error

            async def send(self, message: str) -> ExecutionResult:
                if block is not None:
                    await block.wait()
                return ExecutionResult(response="ok")

            async def send_streaming(self, message: str):
                yield "ok"

            async def close(self) -> None:
                record["close_task"] = asyncio.current_task()
                record.setdefault("closes", 0)
                record["closes"] = int(record["closes"]) + 1  # type: ignore[arg-type]

        return Inner()  # type: ignore[return-value]

    @pytest.mark.asyncio
    async def test_prepare_failure_closes_and_tears_down_in_actor(self) -> None:
        record: dict[str, object] = {}
        teardowns: list[asyncio.Task[None] | None] = []

        async def factory() -> AgentSession:
            return self._inner(record, prepare_error=BackendInitError("mcp down"))

        async def teardown() -> None:
            teardowns.append(asyncio.current_task())

        actor = _TaskBoundSession(session_factory=factory, teardown=teardown)
        with pytest.raises(BackendInitError, match="mcp down"):
            await actor.start()

        assert record["closes"] == 1
        assert teardowns and teardowns[0] is record["prepare_task"]
        assert record["close_task"] is record["prepare_task"]
        await actor.close()  # idempotent after a failed start
        assert record["closes"] == 1 and len(teardowns) == 1

    @pytest.mark.asyncio
    async def test_abandoned_close_cancels_actor_and_fails_inflight_turn(
        self,
    ) -> None:
        record: dict[str, object] = {}
        teardowns: list[str] = []
        block = asyncio.Event()

        async def teardown() -> None:
            teardowns.append("teardown")

        actor = _TaskBoundSession(
            session=self._inner(record, block=block), teardown=teardown
        )
        await actor.start()
        inflight = asyncio.create_task(actor.send("stuck"))
        await asyncio.sleep(0)  # let the actor dequeue and block in send()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(actor.close(), timeout=0.05)
        with pytest.raises(BackendSessionError, match="closed while a turn"):
            await inflight
        assert actor._task is not None
        with pytest.raises(asyncio.CancelledError):
            await actor._task

        assert record["closes"] == 1
        assert teardowns == ["teardown"]

    @pytest.mark.asyncio
    async def test_close_from_inside_actor_does_not_deadlock(self) -> None:
        record: dict[str, object] = {}
        teardowns: list[str] = []
        holder: dict[str, _TaskBoundSession] = {}

        class Inner:
            async def prepare(self) -> None:
                return None

            async def send(self, message: str) -> ExecutionResult:
                await holder["actor"].close()  # a tool/hook ending the session
                return ExecutionResult(response="bye")

            async def send_streaming(self, message: str):
                yield "bye"

            async def close(self) -> None:
                record["closes"] = int(record.get("closes", 0)) + 1  # type: ignore[arg-type]

        async def teardown() -> None:
            teardowns.append("teardown")

        actor = _TaskBoundSession(session=Inner(), teardown=teardown)  # type: ignore[arg-type]
        holder["actor"] = actor
        await actor.start()
        result = await asyncio.wait_for(actor.send("end"), timeout=1)
        assert result.response == "bye"
        await asyncio.wait_for(actor.close(), timeout=1)
        assert record["closes"] == 1
        assert teardowns == ["teardown"]

    @pytest.mark.asyncio
    async def test_concurrent_first_turns_share_one_backend(
        self, make_agent, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from holodeck.chat import executor as executor_module

        selects: list[int] = []
        backend = MagicMock()

        class Inner:
            async def prepare(self) -> None:
                return None

            async def send(self, message: str) -> ExecutionResult:
                await asyncio.sleep(0)
                return ExecutionResult(response=message)

            async def send_streaming(self, message: str):
                yield message

            async def close(self) -> None:
                return None

        backend.create_session = AsyncMock(side_effect=lambda **_: Inner())
        backend.teardown = AsyncMock()

        async def select(*args, **kwargs):
            selects.append(1)
            await asyncio.sleep(0)
            return backend

        monkeypatch.setattr(executor_module.BackendSelector, "select", select)
        executor = AgentExecutor(make_agent(), release_transport_after_turn=True)
        a, b = await asyncio.gather(
            executor.execute_turn("a"), executor.execute_turn("b")
        )
        assert {a.content, b.content} == {"a", "b"}
        assert len(selects) == 1
        await executor.shutdown()
        backend.teardown.assert_awaited_once()
