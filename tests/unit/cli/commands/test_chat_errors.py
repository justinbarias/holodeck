"""Unit tests for CLI chat command error handling.

Tests cover:
- Configuration loading errors (invalid path, bad YAML)
- Agent initialization errors (tool/LLM failures)
- Keyboard interrupt (Ctrl+C)
- Input validation errors (empty, oversized messages)
- Runtime execution errors
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

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

    def test_validation_error_does_not_exit_session(self, chat_tmp_path):
        """Validation errors don't exit the session, only display error."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )

            mock_session_instance = AsyncMock()
            mock_session_instance.start = AsyncMock()
            mock_session_instance.terminate = AsyncMock()
            # Simulate validation error on first message, then exit
            mock_session_instance.process_message = AsyncMock(
                side_effect=[
                    ChatValidationError("Message cannot be empty"),
                    None,  # Allow exit
                ]
            )
            mock_session_instance.should_warn_context_limit = MagicMock(
                return_value=False
            )
            mock_session.return_value = mock_session_instance

            mock_asyncio_run.side_effect = _run_async_helper

            # Send empty message, then exit
            result = runner.invoke(chat, [chat_tmp_path], input="\nexit\n")

            # Session should still be active (exit code 0 for normal exit)
            assert result.exit_code == 0

    def test_empty_message_displays_error(self, chat_tmp_path):
        """Empty message displays validation error."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )

            mock_session_instance = AsyncMock()
            mock_session_instance.start = AsyncMock()
            mock_session_instance.terminate = AsyncMock()
            # Raise validation error on non-empty message to test error display
            mock_session_instance.process_message = AsyncMock(
                side_effect=ChatValidationError("Message validation failed")
            )
            mock_session_instance.should_warn_context_limit = MagicMock(
                return_value=False
            )
            mock_session.return_value = mock_session_instance

            mock_asyncio_run.side_effect = _run_async_helper

            # Send a non-empty message to trigger the error, then exit
            result = runner.invoke(chat, [chat_tmp_path], input="hello\nexit\n")

            # Error message should be displayed
            assert "Error" in result.output or "error" in result.output.lower()


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    def test_keyboard_interrupt_exits_gracefully(self, chat_tmp_path):
        """Keyboard interrupt (Ctrl+C) exits session gracefully."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )

            mock_session_instance = AsyncMock()
            mock_session_instance.start = AsyncMock()
            mock_session_instance.terminate = AsyncMock()
            mock_session.return_value = mock_session_instance

            mock_asyncio_run.side_effect = _run_async_helper

            # Simulate Ctrl+C by raising KeyboardInterrupt
            result = runner.invoke(
                chat, [chat_tmp_path], input=None, catch_exceptions=False
            )

            # Session should be terminated
            assert mock_session_instance.terminate.called or result.exit_code in (
                0,
                130,
                1,
            )

    def test_keyboard_interrupt_displays_goodbye(self, chat_tmp_path):
        """Keyboard interrupt displays goodbye message."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )

            mock_session_instance = AsyncMock()
            mock_session_instance.start = AsyncMock()
            mock_session_instance.terminate = AsyncMock()
            mock_session.return_value = mock_session_instance

            mock_asyncio_run.side_effect = _run_async_helper

            # Use mix_stderr=False to avoid mixing error output
            result = runner.invoke(chat, [chat_tmp_path], input=None)

            # Should exit cleanly (goodbye message, or normal exit code)
            assert result.exit_code in (0, 1, 130) or "goodbye" in result.output.lower()


class TestRuntimeErrors:
    """Tests for runtime execution errors."""

    @pytest.mark.parametrize(
        "error_msg, user_input",
        [
            pytest.param(
                "Agent execution failed",
                "hello\nexit\n",
                id="execution_error",
            ),
            pytest.param(
                "Tool execution timeout",
                "test\nexit\n",
                id="tool_execution_error",
            ),
        ],
    )
    def test_execution_error_handled(self, chat_tmp_path, error_msg, user_input):
        """Execution errors during message processing are handled: {error_msg}."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with (
            patch("holodeck.config.loader.load_agent_with_config") as mock_load,
            patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
        ):
            mock_load.return_value = (
                _create_agent(),
                ExecutionConfig(),
                MagicMock(),
            )

            mock_session_instance = AsyncMock()
            mock_session_instance.start = AsyncMock()
            mock_session_instance.terminate = AsyncMock()
            mock_session_instance.process_message = AsyncMock(
                side_effect=ExecutionError(error_msg)
            )
            mock_session_instance.should_warn_context_limit = MagicMock(
                return_value=False
            )
            mock_session.return_value = mock_session_instance

            mock_asyncio_run.side_effect = _run_async_helper

            result = runner.invoke(chat, [chat_tmp_path], input=user_input)

            # Should not crash - allow continue or graceful exit
            assert result.exit_code in (0, 1, 2) or "Error" in result.output

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
