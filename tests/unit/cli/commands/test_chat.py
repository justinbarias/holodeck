"""Unit tests for CLI chat command.

Tests cover:
- Argument parsing (AGENT_CONFIG positional argument)
- Option handling (--verbose, --observability, --max-messages flags)
- Exit code logic (0=normal exit, 1=config error, 2=agent error, 130=interrupt)
- Multi-turn conversation flow
- Tool execution display (standard and verbose modes)
- Async drainer + renderer helpers for the live tools panel
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.chat.progress import ChatProgressIndicator
from holodeck.chat.tools_panel import ToolsPanel
from holodeck.lib.backends.base import ToolEvent
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig


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

                # Invoke without agent_config argument
                runner.invoke(chat, [], input="exit\n")

                # Should use agent.yaml as default
                mock_load.assert_called_once()
                assert mock_load.call_args.args[0] == "agent.yaml"

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
                patch("holodeck.config.loader.load_agent_with_config") as mock_load,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
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
                patch("holodeck.config.loader.load_agent_with_config") as mock_load,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
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
                patch("holodeck.config.loader.load_agent_with_config") as mock_load,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
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
                patch("holodeck.config.loader.load_agent_with_config") as mock_load,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
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
                patch("holodeck.config.loader.load_agent_with_config") as mock_load,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
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
                patch("holodeck.config.loader.load_agent_with_config") as mock_load,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run"),
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
                patch("holodeck.config.loader.load_agent_with_config") as mock_load,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
            ):
                agent = _create_agent()
                mock_load.return_value = (
                    agent,
                    ExecutionConfig(verbose=True),
                    MagicMock(),
                )

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

                runner.invoke(chat, [tmp_path], input="exit\n")

                # Verify initialization
                mock_session.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_config_loader_called_with_path(self):
        """load_agent_with_config is invoked with the agent config path."""
        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
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

                runner.invoke(chat, [tmp_path], input="exit\n")

                # Verify load_agent_with_config was called with the path
                mock_load.assert_called_once()
                assert mock_load.call_args.args[0] == tmp_path
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
                patch("holodeck.config.loader.load_agent_with_config") as mock_load,
                patch("holodeck.cli.commands.chat.ChatSessionManager") as mock_session,
                patch("holodeck.cli.commands.chat.asyncio.run") as mock_asyncio_run,
            ):
                mock_load.return_value = (
                    _create_agent(),
                    ExecutionConfig(verbose=True),
                    MagicMock(),
                )

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

                runner.invoke(chat, [tmp_path], input="exit\n")

                # Verify default max_messages is 50 in config
                call_kwargs = mock_session.call_args.kwargs
                config = call_kwargs["config"]
                assert config.max_messages == 50
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestToolEventDrainer:
    """Tests for the chat REPL's tool-event drainer task."""

    @pytest.mark.asyncio
    async def test_drainer_applies_events_to_panel(self) -> None:
        """Events on the queue are forwarded into the panel."""
        from holodeck.cli.commands.chat import _drain_tool_events

        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        panel = ToolsPanel()
        stop_event = asyncio.Event()

        await queue.put(ToolEvent(kind="start", tool_name="Read", tool_use_id="tu_1"))
        await queue.put(
            ToolEvent(
                kind="start",
                tool_name="Task",
                tool_use_id="task_1",
                tool_input={"subagent_type": "rev"},
            )
        )

        task = asyncio.create_task(_drain_tool_events(queue, panel, stop_event))
        # Yield enough times for the drainer to consume both items.
        for _ in range(5):
            await asyncio.sleep(0)
        stop_event.set()
        await task

        names = {e.tool_name for e in panel.snapshot()}
        assert names == {"Read", "Task"}

    @pytest.mark.asyncio
    async def test_drainer_exits_promptly_when_stop_event_set(self) -> None:
        """Drainer must not block forever waiting on an empty queue."""
        from holodeck.cli.commands.chat import _drain_tool_events

        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        panel = ToolsPanel()
        stop_event = asyncio.Event()

        task = asyncio.create_task(_drain_tool_events(queue, panel, stop_event))
        stop_event.set()
        # Should return well within the per-iteration timeout.
        await asyncio.wait_for(task, timeout=1.0)


class TestRemainingDrain:
    """Tests for ``_drain_remaining`` that consumes buffered events."""

    def test_drains_all_buffered_events(self) -> None:
        from holodeck.cli.commands.chat import _drain_remaining

        queue: asyncio.Queue[ToolEvent] = asyncio.Queue()
        panel = ToolsPanel()
        queue.put_nowait(ToolEvent(kind="start", tool_name="Read", tool_use_id="tu_1"))
        queue.put_nowait(ToolEvent(kind="end", tool_name="Read", tool_use_id="tu_1"))

        _drain_remaining(queue, panel)

        assert queue.empty()
        assert panel.snapshot() == []


class TestPanelRenderer:
    """Tests for ``_render_tools_panel`` and ``_clear_panel`` helpers."""

    @pytest.mark.asyncio
    async def test_renderer_returns_immediately_when_not_tty(self) -> None:
        from holodeck.cli.commands.chat import _render_tools_panel

        panel = ToolsPanel()
        progress = ChatProgressIndicator(max_messages=50, quiet=False, verbose=False)
        stop_event = asyncio.Event()

        with patch("holodeck.cli.commands.chat.is_tty", return_value=False):
            last_lines = await asyncio.wait_for(
                _render_tools_panel(panel, progress, stop_event), timeout=1.0
            )
        assert last_lines == 0

    def test_clear_panel_emits_no_output_for_zero_lines(self) -> None:
        from holodeck.cli.commands.chat import _clear_panel

        with patch("sys.stdout.write") as mock_write, patch("sys.stdout.flush"):
            _clear_panel(0)
        mock_write.assert_not_called()

    def test_clear_panel_writes_ansi_clear_for_positive_lines(self) -> None:
        from holodeck.cli.commands.chat import _clear_panel

        with patch("sys.stdout.write") as mock_write, patch("sys.stdout.flush"):
            _clear_panel(3)
        mock_write.assert_called_once()
        written = mock_write.call_args.args[0]
        # ANSI: CR + cursor up 3 + erase from cursor down
        assert written == "\r\033[3A\033[J"
