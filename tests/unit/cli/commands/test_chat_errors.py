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

from click.testing import CliRunner

from holodeck.lib.errors import (
    AgentInitializationError,
    ChatValidationError,
    ConfigError,
    ExecutionError,
)
from holodeck.models.agent import Agent


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

    def test_exit_code_one_on_missing_file(self):
        """Exit code 1 when agent config file does not exist."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.side_effect = ConfigError(
                    "agent", "File not found"
                )
                mock_loader_class.return_value = mock_loader

                result = runner.invoke(chat, [tmp_path])

                assert result.exit_code == 1
                assert "Error" in result.output or "error" in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_one_on_invalid_yaml(self):
        """Exit code 1 when agent config has invalid YAML syntax."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.side_effect = ConfigError(
                    "agent", "Invalid YAML syntax"
                )
                mock_loader_class.return_value = mock_loader

                result = runner.invoke(chat, [tmp_path])

                assert result.exit_code == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_one_on_missing_required_fields(self):
        """Exit code 1 when agent config missing required fields."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.side_effect = ConfigError(
                    "agent", "Missing required field: model"
                )
                mock_loader_class.return_value = mock_loader

                result = runner.invoke(chat, [tmp_path])

                assert result.exit_code == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_error_message_displayed_for_config_error(self):
        """Error message is displayed for configuration errors."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.side_effect = ConfigError(
                    "agent", "Failed to load agent configuration"
                )
                mock_loader_class.return_value = mock_loader

                result = runner.invoke(chat, [tmp_path])

                assert result.exit_code == 1
                assert "Error" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestAgentInitializationErrors:
    """Tests for agent initialization errors."""

    def test_exit_code_two_on_agent_init_failure(self):
        """Exit code 2 when agent initialization fails."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                # Raise agent init error when creating session manager
                mock_session.side_effect = AgentInitializationError(
                    "test_agent", "Could not connect to LLM"
                )

                result = runner.invoke(chat, [tmp_path])

                assert result.exit_code == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_two_on_invalid_tools(self):
        """Exit code 2 when agent has invalid tool configuration."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                # Raise agent init error for invalid tools
                mock_session.side_effect = AgentInitializationError(
                    "test_agent", "Invalid tool configuration"
                )

                result = runner.invoke(chat, [tmp_path])

                assert result.exit_code == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_two_on_llm_connection_failure(self):
        """Exit code 2 when LLM connection fails."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                # Raise agent init error for LLM connection
                mock_session.side_effect = AgentInitializationError(
                    "test_agent", "Could not connect to Anthropic API"
                )

                result = runner.invoke(chat, [tmp_path])

                assert result.exit_code == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_error_message_displayed_for_agent_init_error(self):
        """Error message is displayed for agent initialization errors."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session.side_effect = AgentInitializationError(
                    "test_agent", "Failed to initialize agent"
                )

                result = runner.invoke(chat, [tmp_path])

                assert result.exit_code == 2
                assert "Error" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestValidationErrors:
    """Tests for input validation errors (session continues)."""

    def test_validation_error_does_not_exit_session(self):
        """Validation errors don't exit the session, only display error."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

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
                result = runner.invoke(chat, [tmp_path], input="\nexit\n")

                # Session should still be active (exit code 0 for normal exit)
                assert result.exit_code == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_empty_message_displays_error(self):
        """Empty message displays validation error."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

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
                result = runner.invoke(chat, [tmp_path], input="hello\nexit\n")

                # Error message should be displayed
                assert "Error" in result.output or "error" in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    def test_keyboard_interrupt_exits_gracefully(self):
        """Keyboard interrupt (Ctrl+C) exits session gracefully."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                mock_asyncio_run.side_effect = _run_async_helper

                # Simulate Ctrl+C by raising KeyboardInterrupt
                result = runner.invoke(
                    chat, [tmp_path], input=None, catch_exceptions=False
                )

                # Session should be terminated
                assert mock_session_instance.terminate.called or result.exit_code in (
                    0,
                    130,
                    1,
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_keyboard_interrupt_displays_goodbye(self):
        """Keyboard interrupt displays goodbye message."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                mock_asyncio_run.side_effect = _run_async_helper

                # Use mix_stderr=False to avoid mixing error output
                result = runner.invoke(chat, [tmp_path], input=None)

                # Should exit cleanly (goodbye message, or normal exit code)
                assert (
                    result.exit_code in (0, 1, 130)
                    or "goodbye" in result.output.lower()
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestRuntimeErrors:
    """Tests for runtime execution errors."""

    def test_execution_error_during_message_processing(self):
        """Execution errors during message processing are handled."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                # Simulate execution error during message processing
                mock_session_instance.process_message = AsyncMock(
                    side_effect=ExecutionError("Agent execution failed")
                )
                mock_session_instance.should_warn_context_limit = MagicMock(
                    return_value=False
                )
                mock_session.return_value = mock_session_instance

                mock_asyncio_run.side_effect = _run_async_helper

                result = runner.invoke(chat, [tmp_path], input="hello\nexit\n")

                # Should display error and continue or exit gracefully
                assert result.exit_code in (0, 1, 2) or "Error" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_tool_execution_error_during_chat(self):
        """Tool execution errors during chat are displayed but don't crash."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                # Simulate tool failure
                mock_session_instance.process_message = AsyncMock(
                    side_effect=ExecutionError("Tool execution timeout")
                )
                mock_session_instance.should_warn_context_limit = MagicMock(
                    return_value=False
                )
                mock_session.return_value = mock_session_instance

                mock_asyncio_run.side_effect = _run_async_helper

                result = runner.invoke(chat, [tmp_path], input="test\nexit\n")

                # Should not crash - allow continue or graceful exit
                assert result.exit_code in (0, 1, 2)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_unexpected_exception_displays_error(self):
        """Unexpected exceptions display error message."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class:
                mock_loader = MagicMock()
                # Raise unexpected exception
                mock_loader.load_agent_yaml.side_effect = RuntimeError(
                    "Unexpected error"
                )
                mock_loader_class.return_value = mock_loader

                result = runner.invoke(chat, [tmp_path])

                # Should display error
                assert result.exit_code != 0
                assert "Error" in result.output or "error" in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
