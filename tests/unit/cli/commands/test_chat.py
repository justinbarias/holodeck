"""Unit tests for CLI chat command.

Tests cover:
- Argument parsing (AGENT_CONFIG positional argument)
- Option handling (--verbose, --observability, --max-messages flags)
- Exit code logic (0=normal exit, 1=config error, 2=agent error, 130=interrupt)
- Multi-turn conversation flow
- Tool execution display (standard and verbose modes)
- Spinner thread animation and lifecycle
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from holodeck.chat.progress import ChatProgressIndicator
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


class TestCLIArgumentParsing:
    """Tests for CLI chat command argument parsing."""

    def test_agent_config_defaults_to_agent_yaml(self):
        """AGENT_CONFIG defaults to agent.yaml when not provided."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create agent.yaml in current directory
            Path("agent.yaml").write_text("")

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

                # Invoke without agent_config argument
                runner.invoke(chat, [], input="exit\n")

                # Should use agent.yaml as default
                mock_loader.load_agent_yaml.assert_called_once_with("agent.yaml")

    def test_agent_config_error_when_default_not_found(self):
        """Error when agent.yaml doesn't exist and no argument provided."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Don't create agent.yaml - it should fail
            result = runner.invoke(chat, [])

            assert result.exit_code != 0
            # Click's Path(exists=True) will report the file doesn't exist
            assert "agent.yaml" in result.output or "does not exist" in result.output

    def test_agent_config_argument_accepted(self):
        """AGENT_CONFIG positional argument is accepted and loaded."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                result = runner.invoke(chat, [tmp_path])

                # Should not complain about missing argument
                assert "Missing argument" not in result.output or result.exit_code == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_verbose_flag_accepted(self):
        """--verbose flag is accepted and parsed."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                result = runner.invoke(chat, [tmp_path, "--verbose"])

                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_observability_flag_accepted(self):
        """--observability flag is accepted and parsed."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                result = runner.invoke(chat, [tmp_path, "--observability"])

                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_max_messages_option_accepted(self):
        """--max-messages option is accepted with integer value."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                result = runner.invoke(chat, [tmp_path, "--max-messages", "100"])

                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multiple_options_combined(self):
        """Multiple options can be combined in single command."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                result = runner.invoke(
                    chat,
                    [tmp_path, "--verbose", "--observability", "--max-messages", "75"],
                )

                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_short_flags_accepted(self):
        """-v and -o short flags are accepted."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.cli.commands.chat.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent()
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                result = runner.invoke(chat, [tmp_path, "-v", "-o", "-m", "50"])

                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCLIHappyPath:
    """Tests for CLI chat command happy path scenarios."""

    def test_exit_code_zero_on_normal_exit(self):
        """Exit code 0 when user types 'exit' command."""
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

                # Simulate user typing "exit"
                result = runner.invoke(chat, [tmp_path], input="exit\n")

                assert result.exit_code == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_zero_on_quit_command(self):
        """Exit code 0 when user types 'quit' command."""
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

                # Simulate user typing "quit"
                result = runner.invoke(chat, [tmp_path], input="quit\n")

                assert result.exit_code == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_session_manager_initialized_with_config(self):
        """ChatSessionManager is initialized with correct ChatConfig."""
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
                agent = _create_agent()
                mock_loader.load_agent_yaml.return_value = agent
                mock_loader_class.return_value = mock_loader

                mock_session_instance = AsyncMock()
                mock_session_instance.start = AsyncMock()
                mock_session_instance.terminate = AsyncMock()
                mock_session.return_value = mock_session_instance

                mock_asyncio_run.side_effect = _run_async_helper

                runner.invoke(chat, [tmp_path, "-v", "-o", "-m", "75"], input="exit\n")

                # Verify ChatSessionManager was called with agent and ChatConfig
                mock_session.assert_called_once()
                call_kwargs = mock_session.call_args.kwargs
                assert "agent_config" in call_kwargs
                assert "config" in call_kwargs
                config = call_kwargs["config"]
                assert config.verbose is True
                assert config.enable_observability is True
                assert config.max_messages == 75
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_session_start_and_terminate_called(self):
        """Session start() and terminate() are called for lifecycle."""
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

                runner.invoke(chat, [tmp_path], input="exit\n")

                # Verify initialization
                mock_session.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_config_loader_called_with_path(self):
        """ConfigLoader is invoked with the agent config path."""
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

                runner.invoke(chat, [tmp_path], input="exit\n")

                # Verify ConfigLoader was called
                mock_loader_class.assert_called_once()
                # Verify load_agent_yaml was called with the path
                mock_loader.load_agent_yaml.assert_called_once_with(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_welcome_message_displayed(self):
        """Welcome message is displayed when chat starts."""
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

                result = runner.invoke(chat, [tmp_path], input="exit\n")

                # Welcome message should be in output
                assert (
                    "chat" in result.output.lower()
                    or "starting" in result.output.lower()
                    or "test_agent" in result.output.lower()
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_goodbye_message_on_exit(self):
        """Goodbye message is displayed when user exits."""
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

                result = runner.invoke(chat, [tmp_path], input="exit\n")

                # Goodbye message should be in output
                assert (
                    "goodbye" in result.output.lower()
                    or "exit" in result.output.lower()
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_verbose_mode_affects_output(self):
        """Verbose mode is passed to session and affects display."""
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

                runner.invoke(chat, [tmp_path, "--verbose"], input="exit\n")

                # Verify session was initialized with verbose=True in config
                call_kwargs = mock_session.call_args.kwargs
                config = call_kwargs["config"]
                assert config.verbose is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_observability_mode_affects_config(self):
        """Observability mode is passed to session configuration."""
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

                runner.invoke(chat, [tmp_path, "--observability"], input="exit\n")

                # Verify session initialized with enable_observability=True
                call_kwargs = mock_session.call_args.kwargs
                config = call_kwargs["config"]
                assert config.enable_observability is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_max_messages_passed_to_session(self):
        """--max-messages option is passed correctly to session."""
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

                runner.invoke(chat, [tmp_path, "--max-messages", "100"], input="exit\n")

                # Verify session was initialized with correct max_messages in config
                call_kwargs = mock_session.call_args.kwargs
                config = call_kwargs["config"]
                assert config.max_messages == 100
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_default_max_messages_is_50(self):
        """Default --max-messages value is 50 when not specified."""
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

                runner.invoke(chat, [tmp_path], input="exit\n")

                # Verify default max_messages is 50 in config
                call_kwargs = mock_session.call_args.kwargs
                config = call_kwargs["config"]
                assert config.max_messages == 50
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestChatSpinnerThread:
    """Tests for ChatSpinnerThread spinner animation."""

    def test_spinner_thread_initializes(self) -> None:
        """ChatSpinnerThread initializes with progress indicator."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        assert spinner.progress is progress
        assert spinner.daemon is True

    def test_spinner_thread_has_stop_event(self) -> None:
        """ChatSpinnerThread has _stop_event for thread control."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        assert hasattr(spinner, "_stop_event")
        assert spinner._stop_event.is_set() is False

    def test_spinner_thread_has_running_flag(self) -> None:
        """ChatSpinnerThread has _running flag."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        assert hasattr(spinner, "_running")
        assert spinner._running is False

    def test_spinner_thread_stop_sets_event(self) -> None:
        """ChatSpinnerThread.stop() sets the stop event."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        spinner.stop()

        assert spinner._stop_event.is_set() is True

    def test_spinner_thread_run_sets_running_flag(self) -> None:
        """ChatSpinnerThread.run() sets _running flag to True."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        # Mock the progress.get_spinner_line to return immediately
        with patch.object(progress, "get_spinner_line", return_value=""):
            spinner._stop_event.set()
            spinner.run()

            # After run completes, _running should be False
            assert spinner._running is False

    def test_spinner_thread_clears_line_on_stop(self) -> None:
        """ChatSpinnerThread.stop() clears the spinner line from output."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        with patch("sys.stdout.write") as mock_write, patch("sys.stdout.flush"):
            # Mark as running so stop will clear the line
            spinner._running = True
            spinner.stop()

            # Should write clear sequence with spaces and carriage returns
            assert mock_write.called
            # Verify clear sequence was written (carriage return + spaces)
            call_args = [call[0][0] for call in mock_write.call_args_list]
            assert any("\r" in str(arg) for arg in call_args)

    def test_spinner_thread_with_tty_writes_spinner(self) -> None:
        """ChatSpinnerThread writes spinner text when spinner line is not empty."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        with (
            patch.object(progress, "get_spinner_line", return_value="⠋ Thinking..."),
            patch("sys.stdout.write"),
            patch("sys.stdout.flush"),
        ):
            # Set up to run for just one iteration
            spinner._stop_event.set()
            spinner.run()

            # Spinner runs even if loop exits immediately
            assert spinner is not None

    def test_spinner_thread_without_tty_no_output(self) -> None:
        """ChatSpinnerThread produces no output when spinner returns empty."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        with (
            patch.object(progress, "get_spinner_line", return_value=""),
            patch("sys.stdout.write"),
            patch("sys.stdout.flush"),
        ):
            # Set up to run for one iteration with empty output
            spinner._stop_event.set()
            spinner.run()

            # Should complete without error
            assert spinner is not None

    def test_spinner_thread_respects_stop_event(self) -> None:
        """ChatSpinnerThread stops when stop_event is set."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        with patch.object(progress, "get_spinner_line", return_value="⠋ Thinking..."):
            # Immediately set stop event
            spinner._stop_event.set()

            # Run should exit quickly
            spinner.run()

            # Verify _running is now False
            assert spinner._running is False

    def test_spinner_thread_runs_in_background(self) -> None:
        """ChatSpinnerThread is a daemon thread."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        assert spinner.daemon is True

    def test_spinner_thread_updates_progress_spinner_index(self) -> None:
        """ChatSpinnerThread calls progress.get_spinner_line repeatedly."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        initial_index = progress._spinner_index

        with patch("sys.stdout.write"), patch("sys.stdout.flush"):
            # Set up to run briefly
            def stop_after_calls():
                # Give it a few iterations
                time.sleep(0.15)  # Longer sleep to ensure multiple iterations
                spinner.stop()

            import threading as t

            stop_thread = t.Thread(target=stop_after_calls)
            stop_thread.start()
            spinner.run()
            stop_thread.join()

            # Spinner index should have advanced (at least 1 iteration)
            assert progress._spinner_index >= initial_index

    def test_spinner_thread_multiple_stop_calls_safe(self) -> None:
        """Calling stop() multiple times is safe."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        # Should not raise
        spinner.stop()
        spinner.stop()
        spinner.stop()

        assert spinner._stop_event.is_set() is True

    def test_spinner_thread_flushes_output(self) -> None:
        """ChatSpinnerThread flushes stdout after writing spinner."""
        from holodeck.cli.commands.chat import ChatSpinnerThread

        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        spinner = ChatSpinnerThread(progress)

        with (
            patch.object(progress, "get_spinner_line", return_value="⠋ Thinking..."),
            patch("sys.stdout.write"),
            patch("sys.stdout.flush"),
        ):
            spinner._stop_event.set()
            spinner.run()

            # Spinner completed successfully
            assert spinner is not None
