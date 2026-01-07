"""Unit tests for test command observability integration.

TDD: These tests are written FIRST, before implementation.
All tests should FAIL until the test command is updated to integrate observability.

Task: T118 - Tests for test command observability init/shutdown
"""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.models.observability import ObservabilityConfig


@pytest.mark.unit
class TestTestCommandObservabilityInit:
    """Tests for observability initialization in test command."""

    @patch("holodeck.cli.commands.test.shutdown_observability")
    @patch("holodeck.cli.commands.test.initialize_observability")
    @patch("holodeck.cli.commands.test.setup_logging")
    @patch("holodeck.config.loader.ConfigLoader")
    def test_initializes_observability_when_enabled(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
    ) -> None:
        """Test observability is initialized when agent.observability.enabled=True."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=True)
        mock_agent.test_cases = []
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

        from click.testing import CliRunner

        from holodeck.cli.commands.test import test

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor.execute_tests = MagicMock(return_value=MagicMock())
                mock_executor.shutdown = MagicMock()
                mock_executor_cls.return_value = mock_executor
                runner.invoke(test, ["agent.yaml"])

        # Assert - includes verbose/quiet params (test defaults)
        mock_init_obs.assert_called_once_with(
            mock_agent.observability,
            mock_agent.name,
            verbose=False,
            quiet=False,
        )

    @patch("holodeck.cli.commands.test.shutdown_observability")
    @patch("holodeck.cli.commands.test.initialize_observability")
    @patch("holodeck.cli.commands.test.setup_logging")
    @patch("holodeck.config.loader.ConfigLoader")
    def test_setup_logging_not_called_when_observability_enabled(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
    ) -> None:
        """Test setup_logging is NOT called when observability is enabled."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=True)
        mock_agent.test_cases = []
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

        from click.testing import CliRunner

        from holodeck.cli.commands.test import test

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor.execute_tests = MagicMock(return_value=MagicMock())
                mock_executor.shutdown = MagicMock()
                mock_executor_cls.return_value = mock_executor
                runner.invoke(test, ["agent.yaml"])

        # Assert - setup_logging should NOT be called
        mock_setup_logging.assert_not_called()

    @patch("holodeck.cli.commands.test.shutdown_observability")
    @patch("holodeck.cli.commands.test.initialize_observability")
    @patch("holodeck.cli.commands.test.setup_logging")
    @patch("holodeck.config.loader.ConfigLoader")
    def test_setup_logging_called_when_observability_not_configured(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
    ) -> None:
        """Test setup_logging IS called when observability is not configured."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = None
        mock_agent.test_cases = []
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader

        from click.testing import CliRunner

        from holodeck.cli.commands.test import test

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor.execute_tests = MagicMock(return_value=MagicMock())
                mock_executor.shutdown = MagicMock()
                mock_executor_cls.return_value = mock_executor
                runner.invoke(test, ["agent.yaml"])

        # Assert - setup_logging should be called
        mock_setup_logging.assert_called()
        # And observability should NOT be initialized
        mock_init_obs.assert_not_called()


@pytest.mark.unit
class TestTestCommandObservabilityShutdown:
    """Tests for observability shutdown in test command."""

    @patch("holodeck.cli.commands.test.shutdown_observability")
    @patch("holodeck.cli.commands.test.initialize_observability")
    @patch("holodeck.cli.commands.test.setup_logging")
    @patch("holodeck.config.loader.ConfigLoader")
    def test_shutdown_called_after_tests(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
    ) -> None:
        """Test shutdown_observability is called after test execution."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=True)
        mock_agent.test_cases = []
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

        from click.testing import CliRunner

        from holodeck.cli.commands.test import test

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor.execute_tests = MagicMock(return_value=MagicMock())
                mock_executor.shutdown = MagicMock()
                mock_executor_cls.return_value = mock_executor
                runner.invoke(test, ["agent.yaml"])

        # Assert
        mock_shutdown_obs.assert_called_once_with(mock_context)

    @patch("holodeck.cli.commands.test.shutdown_observability")
    @patch("holodeck.cli.commands.test.initialize_observability")
    @patch("holodeck.cli.commands.test.setup_logging")
    @patch("holodeck.config.loader.ConfigLoader")
    def test_shutdown_called_on_exception(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
    ) -> None:
        """Test shutdown_observability is called even when test execution fails."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = ObservabilityConfig(enabled=True)
        mock_agent.test_cases = []
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

        from click.testing import CliRunner

        from holodeck.cli.commands.test import test

        runner = CliRunner()

        # Act - simulate exception during test execution
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor.execute_tests = MagicMock(
                    side_effect=RuntimeError("Test error")
                )
                mock_executor.shutdown = MagicMock()
                mock_executor_cls.return_value = mock_executor
                runner.invoke(test, ["agent.yaml"])

        # Assert - shutdown should still be called
        mock_shutdown_obs.assert_called_once_with(mock_context)

    @patch("holodeck.cli.commands.test.shutdown_observability")
    @patch("holodeck.cli.commands.test.initialize_observability")
    @patch("holodeck.cli.commands.test.setup_logging")
    @patch("holodeck.config.loader.ConfigLoader")
    def test_shutdown_not_called_when_observability_disabled(
        self,
        mock_loader_cls: MagicMock,
        mock_setup_logging: MagicMock,
        mock_init_obs: MagicMock,
        mock_shutdown_obs: MagicMock,
    ) -> None:
        """Test shutdown is NOT called when observability was not initialized."""
        # Arrange
        mock_agent = MagicMock()
        mock_agent.name = "test-agent"
        mock_agent.observability = None
        mock_agent.test_cases = []
        mock_agent.execution = None

        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent
        mock_loader.load_project_config.return_value = None
        mock_loader.load_global_config.return_value = None
        mock_loader.resolve_execution_config.return_value = MagicMock(
            verbose=False, quiet=True, llm_timeout=30
        )
        mock_loader_cls.return_value = mock_loader

        from click.testing import CliRunner

        from holodeck.cli.commands.test import test

        runner = CliRunner()

        # Act
        with runner.isolated_filesystem():
            with open("agent.yaml", "w") as f:
                f.write("name: test\n")
            with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor.execute_tests = MagicMock(return_value=MagicMock())
                mock_executor.shutdown = MagicMock()
                mock_executor_cls.return_value = mock_executor
                runner.invoke(test, ["agent.yaml"])

        # Assert - shutdown should NOT be called since no context was created
        mock_shutdown_obs.assert_not_called()
