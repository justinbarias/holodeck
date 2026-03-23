"""Unit tests for chat command observability integration.

TDD: These tests verify that observability is properly initialized/shutdown in chat.

Task: T117 - Tests for chat command observability init/shutdown
"""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.models.config import ExecutionConfig
from holodeck.models.observability import ObservabilityConfig


def _make_mock_agent(observability: ObservabilityConfig | None) -> MagicMock:
    """Create a mock agent with the given observability config."""
    mock_agent = MagicMock()
    mock_agent.name = "test-agent"
    mock_agent.observability = observability
    mock_agent.execution = None
    return mock_agent


@pytest.mark.unit
class TestChatObservabilityInit:
    """Tests for observability initialization in chat command."""

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.config.loader.load_agent_with_config")
    def test_initializes_observability_when_enabled(
        self,
        mock_load: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test observability is initialized when agent.observability.enabled=True."""
        mock_agent = _make_mock_agent(ObservabilityConfig(enabled=True))
        mock_load.return_value = (mock_agent, ExecutionConfig(), MagicMock())

        mock_context = MagicMock()
        mock_init_obs.return_value = mock_context
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        mock_init_obs.assert_called_once_with(
            mock_agent.observability,
            mock_agent.name,
            verbose=False,
            quiet=False,
        )

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.config.loader.load_agent_with_config")
    def test_setup_logging_not_called_when_observability_enabled(
        self,
        mock_load: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test setup_logging is NOT called when observability is enabled."""
        mock_agent = _make_mock_agent(ObservabilityConfig(enabled=True))
        mock_load.return_value = (mock_agent, ExecutionConfig(), MagicMock())
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        mock_setup_logging.assert_not_called()

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.config.loader.load_agent_with_config")
    def test_setup_logging_called_when_observability_disabled(
        self,
        mock_load: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test setup_logging IS called when observability is disabled."""
        mock_agent = _make_mock_agent(ObservabilityConfig(enabled=False))
        mock_load.return_value = (mock_agent, ExecutionConfig(), MagicMock())
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        mock_setup_logging.assert_called()
        mock_init_obs.assert_not_called()

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.config.loader.load_agent_with_config")
    def test_setup_logging_called_when_observability_not_configured(
        self,
        mock_load: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test setup_logging IS called when observability is not configured."""
        mock_agent = _make_mock_agent(None)
        mock_load.return_value = (mock_agent, ExecutionConfig(), MagicMock())
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        mock_setup_logging.assert_called()
        mock_init_obs.assert_not_called()


@pytest.mark.unit
class TestChatObservabilityShutdown:
    """Tests for observability shutdown in chat command."""

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.config.loader.load_agent_with_config")
    def test_shutdown_called_on_normal_exit(
        self,
        mock_load: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test shutdown_observability is called on normal exit."""
        mock_agent = _make_mock_agent(ObservabilityConfig(enabled=True))
        mock_load.return_value = (mock_agent, ExecutionConfig(), MagicMock())

        mock_context = MagicMock()
        mock_init_obs.return_value = mock_context
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        mock_shutdown_obs.assert_called_once_with(mock_context)

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.config.loader.load_agent_with_config")
    def test_shutdown_called_on_exception(
        self,
        mock_load: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test shutdown_observability is called even when exception occurs."""
        mock_agent = _make_mock_agent(ObservabilityConfig(enabled=True))
        mock_load.return_value = (mock_agent, ExecutionConfig(), MagicMock())

        mock_context = MagicMock()
        mock_init_obs.return_value = mock_context
        mock_run.side_effect = RuntimeError("Test error")

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"])

        mock_shutdown_obs.assert_called_once_with(mock_context)

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.config.loader.load_agent_with_config")
    def test_shutdown_not_called_when_observability_disabled(
        self,
        mock_load: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test shutdown is NOT called when observability was not initialized."""
        mock_agent = _make_mock_agent(None)
        mock_load.return_value = (mock_agent, ExecutionConfig(), MagicMock())
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        mock_shutdown_obs.assert_not_called()
