"""Unit tests for chat command observability integration.

TDD: These tests verify that observability is properly initialized/shutdown in chat.

Task: T117 - Tests for chat command observability init/shutdown
"""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.models.observability import ObservabilityConfig


@pytest.mark.unit
class TestChatObservabilityInit:
    """Tests for observability initialization in chat command."""

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.cli.commands.chat.ConfigLoader")
    def test_initializes_observability_when_enabled(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test observability is initialized when agent.observability.enabled=True."""
        # Arrange - create agent with observability enabled
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=True)
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader

        mock_context = MagicMock()
        mock_init_obs.return_value = mock_context
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        # Act - use isolated_filesystem to create a temp file
        with runner.isolated_filesystem():
            # Create a dummy agent.yaml so Click's exists=True passes
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")

            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        # Assert - includes verbose/quiet params (chat defaults: quiet=False now)
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
    @patch("holodeck.cli.commands.chat.ConfigLoader")
    def test_setup_logging_not_called_when_observability_enabled(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test setup_logging is NOT called when observability is enabled."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=True)
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        # Assert - setup_logging should NOT be called
        mock_setup_logging.assert_not_called()

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.cli.commands.chat.ConfigLoader")
    def test_setup_logging_called_when_observability_disabled(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test setup_logging IS called when observability is disabled."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=False)
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        # Assert - setup_logging should be called
        mock_setup_logging.assert_called()
        # And observability should NOT be initialized
        mock_init_obs.assert_not_called()

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.cli.commands.chat.ConfigLoader")
    def test_setup_logging_called_when_observability_not_configured(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test setup_logging IS called when observability is not configured."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = None
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        # Assert - setup_logging should be called
        mock_setup_logging.assert_called()
        # And observability should NOT be initialized
        mock_init_obs.assert_not_called()


@pytest.mark.unit
class TestChatObservabilityShutdown:
    """Tests for observability shutdown in chat command."""

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.cli.commands.chat.ConfigLoader")
    def test_shutdown_called_on_normal_exit(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test shutdown_observability is called on normal exit."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=True)
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader

        mock_context = MagicMock()
        mock_init_obs.return_value = mock_context
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        # Assert
        mock_shutdown_obs.assert_called_once_with(mock_context)

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.cli.commands.chat.ConfigLoader")
    def test_shutdown_called_on_exception(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test shutdown_observability is called even when exception occurs."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=True)
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader

        mock_context = MagicMock()
        mock_init_obs.return_value = mock_context
        mock_run.side_effect = RuntimeError("Test error")

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"])

        # Assert - shutdown should still be called
        mock_shutdown_obs.assert_called_once_with(mock_context)

    @patch("holodeck.cli.commands.chat._run_chat_session")
    @patch("holodeck.cli.commands.chat.shutdown_observability")
    @patch("holodeck.cli.commands.chat.initialize_observability")
    @patch("holodeck.cli.commands.chat.setup_logging")
    @patch("holodeck.cli.commands.chat.ConfigLoader")
    def test_shutdown_not_called_when_observability_disabled(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Test shutdown is NOT called when observability was not initialized."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = None
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader
        mock_run.return_value = None

        from click.testing import CliRunner

        from holodeck.cli.commands.chat import chat

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            runner.invoke(chat, ["agent.yaml"], input="exit\n")

        # Assert - shutdown should NOT be called since no context was created
        mock_shutdown_obs.assert_not_called()
