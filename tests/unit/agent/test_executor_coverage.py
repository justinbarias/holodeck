"""Additional coverage tests for chat/executor.py.

Covers untested code paths: callbacks, _TaskBoundSession actor,
error handling branches, edge cases in _convert_tool_calls,
shutdown/clear_history edge cases, and streaming history.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.chat.executor import AgentExecutor, AgentResponse, _TaskBoundSession
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
