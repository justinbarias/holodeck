"""Unit tests for CLI chat command error handling.

Tests cover:
- Configuration loading errors (invalid path, bad YAML)
- Agent initialization errors (tool/LLM failures)
- Keyboard interrupt (Ctrl+C)
- Input validation errors (empty, oversized messages)
- Runtime execution errors
"""

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner, Result

from holodeck.lib.errors import (
    AgentInitializationError,
    ChatValidationError,
    ConfigError,
    ExecutionError,
)
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig


@pytest.fixture
def chat_tmp_path():
    """Create and clean up a temporary YAML file for chat tests."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


def _create_agent() -> Agent:
    """Create a minimal Agent instance for testing."""
    return Agent(
        name="test_agent",
        description="Test agent",
        model={"provider": "openai", "name": "gpt-4"},
        instructions={"inline": "Test instructions"},
    )


def _run_async_helper(coro):
    """Helper to execute async chat sessions in tests.

    This function properly runs async coroutines by creating a new event loop,
    which allows Click CLI input handling to work correctly with mocked asyncio.run.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _streaming_session(*replies: Exception | list[str]) -> AsyncMock:
    """Build a session mock whose streaming turns follow ``replies`` in order.

    Each reply is either a list of chunks to stream or an exception the
    stream raises before yielding anything.
    """
    pending = list(replies)
    session = AsyncMock()
    session.start = AsyncMock()
    session.terminate = AsyncMock()
    session._executor = None
    session.should_warn_context_limit = MagicMock(return_value=False)

    async def stream(_message: str) -> AsyncIterator[str]:
        reply = pending.pop(0)
        if isinstance(reply, Exception):
            raise reply
        for chunk in reply:
            yield chunk

    session.process_message_streaming = MagicMock(side_effect=stream)
    return session


class TestConfigurationErrors:
    """Tests for configuration loading errors."""

    @pytest.mark.parametrize(
        "error_msg, check_error_output",
        [
            pytest.param(
                "File not found",
                True,
                id="missing_file",
            ),
            pytest.param(
                "Invalid YAML syntax",
                False,
                id="invalid_yaml",
            ),
            pytest.param(
                "Missing required field: model",
                False,
                id="missing_required_fields",
            ),
            pytest.param(
                "Failed to load agent configuration",
                True,
                id="error_message_displayed",
            ),
        ],
    )
    def test_exit_code_one_on_config_error(
        self, chat_tmp_path, error_msg, check_error_output
    ):
        """Exit code 1 when config error: {error_msg}."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch("holodeck.cli.commands.chat.ChatSessionManager"),
        ):
            mock_load.side_effect = ConfigError("agent", error_msg)

            result = runner.invoke(chat, [chat_tmp_path])

            assert result.exit_code == 1
            if check_error_output:
                assert "Error" in result.output or "error" in result.output.lower()


class TestAgentInitializationErrors:
    """Tests for agent initialization errors."""

    @pytest.mark.parametrize(
        "error_msg, check_error_output",
        [
            pytest.param(
                "Could not connect to LLM",
                False,
                id="agent_init_failure",
            ),
            pytest.param(
                "Invalid tool configuration",
                False,
                id="invalid_tools",
            ),
            pytest.param(
                "Could not connect to Anthropic API",
                False,
                id="llm_connection_failure",
            ),
            pytest.param(
                "Failed to initialize agent",
                True,
                id="error_message_displayed",
            ),
        ],
    )
    def test_exit_code_two_on_agent_init_error(
        self, chat_tmp_path, error_msg, check_error_output
    ):
        """Exit code 2 when agent init error: {error_msg}."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )

            mock_session.side_effect = AgentInitializationError("test_agent", error_msg)

            result = runner.invoke(chat, [chat_tmp_path])

            assert result.exit_code == 2
            if check_error_output:
                assert "Error" in result.output


class TestValidationErrors:
    """Tests for input validation errors (session continues)."""

    def _invoke(
        self, chat_tmp_path: str, session: AsyncMock, user_input: str
    ) -> Result:
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()
        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch(
                "holodeck.cli.commands.chat.ChatSessionManager", return_value=session
            ),
            patch(
                "holodeck.cli.commands.chat.asyncio.run",
                side_effect=_run_async_helper,
            ),
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )
            return runner.invoke(chat, [chat_tmp_path], input=user_input)

    def test_validation_error_does_not_exit_session(self, chat_tmp_path: str) -> None:
        """A validation error is displayed and the next turn still runs."""
        session = _streaming_session(
            ChatValidationError("Message validation failed"), ["recovered reply"]
        )

        result = self._invoke(chat_tmp_path, session, "hello\nagain\nexit\n")

        assert result.exit_code == 0, result.output
        assert "Error: Message validation failed" in result.output
        assert "recovered reply" in result.output

    def test_empty_message_is_skipped(self, chat_tmp_path: str) -> None:
        """An empty line never reaches the session and does not end the chat."""
        session = _streaming_session()

        result = self._invoke(chat_tmp_path, session, "\n\nexit\n")

        assert result.exit_code == 0, result.output
        assert "Goodbye!" in result.output
        session.process_message_streaming.assert_not_called()


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    def _invoke_with_ctrl_c(self, chat_tmp_path: str) -> tuple[Result, AsyncMock]:
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()
        session = _streaming_session()

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch(
                "holodeck.cli.commands.chat.ChatSessionManager", return_value=session
            ),
            patch(
                "holodeck.cli.commands.chat.asyncio.run",
                side_effect=_run_async_helper,
            ),
            patch(
                "holodeck.cli.commands.chat.click.prompt",
                side_effect=KeyboardInterrupt,
            ),
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )
            result = runner.invoke(chat, [chat_tmp_path])
        return result, session

    def test_keyboard_interrupt_exits_gracefully(self, chat_tmp_path: str) -> None:
        """Ctrl+C at the prompt terminates the session and exits with 130."""
        result, session = self._invoke_with_ctrl_c(chat_tmp_path)

        assert result.exit_code == 130
        session.terminate.assert_awaited_once()

    def test_keyboard_interrupt_displays_goodbye(self, chat_tmp_path: str) -> None:
        """Ctrl+C at the prompt prints the goodbye message."""
        result, _ = self._invoke_with_ctrl_c(chat_tmp_path)

        assert "Goodbye!" in result.output


class TestRuntimeErrors:
    """Tests for runtime execution errors."""

    @pytest.mark.parametrize(
        "error_msg, first_message",
        [
            pytest.param(
                "Agent execution failed",
                "hello",
                id="execution_error",
            ),
            pytest.param(
                "Tool execution timeout",
                "test",
                id="tool_execution_error",
            ),
        ],
    )
    def test_execution_error_handled(
        self, chat_tmp_path: str, error_msg: str, first_message: str
    ) -> None:
        """A failed turn shows the error and the session keeps accepting input."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()
        session = _streaming_session(ExecutionError(error_msg), ["recovered reply"])

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch(
                "holodeck.cli.commands.chat.ChatSessionManager", return_value=session
            ),
            patch(
                "holodeck.cli.commands.chat.asyncio.run",
                side_effect=_run_async_helper,
            ),
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )

            result = runner.invoke(
                chat, [chat_tmp_path], input=f"{first_message}\nagain\nexit\n"
            )

        assert result.exit_code == 0, result.output
        assert f"Error: {error_msg}" in result.output
        # The session survived the error: the next turn streamed a reply.
        assert "recovered reply" in result.output
        assert "Goodbye!" in result.output
        session.terminate.assert_awaited_once()

    def test_streaming_reply_is_displayed(self, chat_tmp_path: str) -> None:
        """A successful turn streams every chunk to the terminal."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()
        session = _streaming_session(["Hello", ", ", "world"])

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch(
                "holodeck.cli.commands.chat.ChatSessionManager", return_value=session
            ),
            patch(
                "holodeck.cli.commands.chat.asyncio.run",
                side_effect=_run_async_helper,
            ),
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )

            result = runner.invoke(chat, [chat_tmp_path], input="hi\nexit\n")

        assert result.exit_code == 0, result.output
        assert "Hello, world" in result.output
        assert "Error" not in result.output
        session.process_message_streaming.assert_called_once_with("hi")

    def test_unexpected_exception_displays_error(self, chat_tmp_path):
        """Unexpected exceptions display error message."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with patch("holodeck.config.loader.load_agent_with_config") as mock_load:
            # Raise unexpected exception
            mock_load.side_effect = RuntimeError("Unexpected error")

            result = runner.invoke(chat, [chat_tmp_path])

            # Should display error
            assert result.exit_code != 0
            assert "Error" in result.output or "error" in result.output.lower()
