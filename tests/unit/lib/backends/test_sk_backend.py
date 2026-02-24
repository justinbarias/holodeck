"""Unit tests for holodeck.lib.backends.sk_backend (Phase 4A TDD).

All tests in this module will fail with ImportError until Phase 4B provides
the sk_backend module. This is intentional TDD behaviour.

Classes:
    TestExtractResponse (T024): 4 tests for the _extract_response() helper.
    TestSKBackend       (T025): 8 tests for SKBackend implementing AgentBackend.
    TestSKSession       (T026): 5 tests for SKSession implementing AgentSession.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from semantic_kernel.contents import ChatHistory

from holodeck.lib.backends.base import AgentBackend, AgentSession, ExecutionResult

# ImportError expected until Phase 4B creates this module.
from holodeck.lib.backends.sk_backend import SKBackend, SKSession, _extract_response
from holodeck.lib.test_runner.agent_factory import AgentExecutionResult
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.token_usage import TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> Agent:
    """Create a minimal Agent instance for tests."""
    return Agent(
        name="test-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="Be helpful."),
    )


def _make_exec_result(
    response_text: str = "Agent response",
    tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    token_usage: TokenUsage | None = None,
) -> AgentExecutionResult:
    """Create an AgentExecutionResult with one assistant message in its ChatHistory."""
    history = ChatHistory()
    history.add_assistant_message(response_text)
    return AgentExecutionResult(
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        chat_history=history,
        token_usage=token_usage,
    )


# ---------------------------------------------------------------------------
# T024 — _extract_response helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractResponse:
    """Tests for the _extract_response(history) private helper."""

    def test_multi_message_extraction(self) -> None:
        """Last assistant message content is returned from a mixed-role history."""
        history = ChatHistory()
        history.add_user_message("Hello")
        history.add_assistant_message("Hi there!")

        result = _extract_response(history)

        assert result == "Hi there!"

    def test_empty_history_returns_empty_string(self) -> None:
        """Empty ChatHistory yields an empty string without raising."""
        history = ChatHistory()

        result = _extract_response(history)

        assert result == ""

    def test_no_assistant_messages_returns_empty_string(self) -> None:
        """History containing only user messages returns empty string."""
        history = ChatHistory()
        history.add_user_message("Hello")

        result = _extract_response(history)

        assert result == ""

    def test_none_content_treated_as_empty(self) -> None:
        """Assistant message with None content does not raise; returns empty string."""
        history = ChatHistory()
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = None
        history.messages.append(msg)

        result = _extract_response(history)

        assert result == ""


# ---------------------------------------------------------------------------
# T025 — SKBackend
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSKBackend:
    """Tests for SKBackend implementing the AgentBackend Protocol."""

    # ------------------------------------------------------------------
    # Internal helper to wire up the AgentFactory mock consistently.
    # ------------------------------------------------------------------

    def _setup_factory_mock(
        self,
        mock_factory_cls: Any,
        thread_run: Any = None,
    ) -> MagicMock:
        """Return a configured AgentFactory instance mock.

        Attaches async stubs for every method SKBackend calls so that any
        test which forgets to set up a thread_run still gets a sensible mock.
        """
        mock_factory = MagicMock()
        mock_factory._ensure_tools_initialized = AsyncMock()
        mock_factory.create_thread_run = AsyncMock(return_value=thread_run)
        mock_factory.shutdown = AsyncMock()
        mock_factory_cls.return_value = mock_factory
        return mock_factory

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @patch("holodeck.lib.backends.sk_backend.AgentFactory")
    def test_isinstance_agentbackend(self, mock_factory_cls: Any) -> None:
        """SKBackend satisfies the AgentBackend runtime-checkable Protocol."""
        self._setup_factory_mock(mock_factory_cls)
        backend = SKBackend(agent_config=_make_agent())

        assert isinstance(backend, AgentBackend)

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.sk_backend.AgentFactory")
    async def test_initialize_calls_ensure_tools_initialized(
        self, mock_factory_cls: Any
    ) -> None:
        """initialize() delegates to factory._ensure_tools_initialized."""
        mock_factory = self._setup_factory_mock(mock_factory_cls)
        backend = SKBackend(agent_config=_make_agent())

        await backend.initialize()

        mock_factory._ensure_tools_initialized.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.sk_backend.AgentFactory")
    async def test_invoke_once_returns_execution_result(
        self, mock_factory_cls: Any
    ) -> None:
        """invoke_once returns an ExecutionResult instance."""
        mock_thread_run = MagicMock()
        mock_thread_run.invoke = AsyncMock(return_value=_make_exec_result())
        self._setup_factory_mock(mock_factory_cls, thread_run=mock_thread_run)

        backend = SKBackend(agent_config=_make_agent())
        await backend.initialize()
        result = await backend.invoke_once("Hello")

        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.sk_backend.AgentFactory")
    async def test_invoke_once_maps_tool_calls(self, mock_factory_cls: Any) -> None:
        """invoke_once maps AgentExecutionResult.tool_calls onto ExecutionResult."""
        expected = [{"name": "search", "arguments": {"query": "test"}}]
        mock_thread_run = MagicMock()
        mock_thread_run.invoke = AsyncMock(
            return_value=_make_exec_result(tool_calls=expected)
        )
        self._setup_factory_mock(mock_factory_cls, thread_run=mock_thread_run)

        backend = SKBackend(agent_config=_make_agent())
        await backend.initialize()
        result = await backend.invoke_once("Hello")

        assert result.tool_calls == expected

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.sk_backend.AgentFactory")
    async def test_invoke_once_maps_tool_results(self, mock_factory_cls: Any) -> None:
        """invoke_once maps AgentExecutionResult.tool_results onto ExecutionResult."""
        expected = [{"name": "search", "result": "42 results found"}]
        mock_thread_run = MagicMock()
        mock_thread_run.invoke = AsyncMock(
            return_value=_make_exec_result(tool_results=expected)
        )
        self._setup_factory_mock(mock_factory_cls, thread_run=mock_thread_run)

        backend = SKBackend(agent_config=_make_agent())
        await backend.initialize()
        result = await backend.invoke_once("Hello")

        assert result.tool_results == expected

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.sk_backend.AgentFactory")
    async def test_invoke_once_maps_token_usage(self, mock_factory_cls: Any) -> None:
        """invoke_once maps AgentExecutionResult.token_usage onto ExecutionResult."""
        expected = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_thread_run = MagicMock()
        mock_thread_run.invoke = AsyncMock(
            return_value=_make_exec_result(token_usage=expected)
        )
        self._setup_factory_mock(mock_factory_cls, thread_run=mock_thread_run)

        backend = SKBackend(agent_config=_make_agent())
        await backend.initialize()
        result = await backend.invoke_once("Hello")

        assert result.token_usage == expected

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.sk_backend.AgentFactory")
    async def test_invoke_once_response_populated_via_extract_response(
        self, mock_factory_cls: Any
    ) -> None:
        """response is extracted from ChatHistory via _extract_response, never empty."""
        mock_thread_run = MagicMock()
        mock_thread_run.invoke = AsyncMock(
            return_value=_make_exec_result("Non-empty agent reply")
        )
        self._setup_factory_mock(mock_factory_cls, thread_run=mock_thread_run)

        backend = SKBackend(agent_config=_make_agent())
        await backend.initialize()
        result = await backend.invoke_once("Hello")

        assert result.response != ""
        assert result.response == "Non-empty agent reply"

    @pytest.mark.asyncio
    @patch("holodeck.lib.backends.sk_backend.AgentFactory")
    async def test_teardown_calls_shutdown(self, mock_factory_cls: Any) -> None:
        """teardown() delegates to factory.shutdown()."""
        mock_factory = self._setup_factory_mock(mock_factory_cls)
        backend = SKBackend(agent_config=_make_agent())
        await backend.initialize()

        await backend.teardown()

        mock_factory.shutdown.assert_awaited_once()


# ---------------------------------------------------------------------------
# T026 — SKSession
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSKSession:
    """Tests for SKSession implementing the AgentSession Protocol."""

    def _make_thread_run(self, response_text: str = "Session response") -> MagicMock:
        """Create a mock AgentThreadRun that returns the given text."""
        mock_thread_run = MagicMock()
        mock_thread_run.invoke = AsyncMock(
            return_value=_make_exec_result(response_text)
        )
        return mock_thread_run

    def test_isinstance_agentsession(self) -> None:
        """SKSession satisfies the AgentSession runtime-checkable Protocol."""
        session = SKSession(thread_run=self._make_thread_run())

        assert isinstance(session, AgentSession)

    @pytest.mark.asyncio
    async def test_send_returns_execution_result(self) -> None:
        """send() returns an ExecutionResult instance."""
        session = SKSession(thread_run=self._make_thread_run())

        result = await session.send("Hello")

        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_multi_turn_uses_same_thread_run(self) -> None:
        """Two consecutive send() calls route through the same thread_run."""
        mock_thread_run = self._make_thread_run()
        session = SKSession(thread_run=mock_thread_run)

        await session.send("First message")
        await session.send("Second message")

        assert mock_thread_run.invoke.call_count == 2

    @pytest.mark.asyncio
    async def test_send_streaming_yields_string_chunks(self) -> None:
        """send_streaming() yields at least one string chunk."""
        session = SKSession(thread_run=self._make_thread_run("Stream response"))
        chunks: list[str] = []

        async for chunk in session.send_streaming("Hello"):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    @pytest.mark.asyncio
    async def test_close_is_awaitable_without_error(self) -> None:
        """close() can be awaited without raising an exception."""
        session = SKSession(thread_run=self._make_thread_run())

        await session.close()  # Must not raise
