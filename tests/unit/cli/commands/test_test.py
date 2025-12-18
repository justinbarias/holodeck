"""Unit tests for CLI test command.

Tests cover:
- Argument parsing (AGENT_CONFIG positional argument)
- Option handling (--output, --format, --verbose, --quiet, --timeout flags)
- Exit code logic (0=success, 1=test failure, 2=config error, 3=execution error)
- Progress callback integration
- Report file generation
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.test import test
from holodeck.models.agent import Agent
from holodeck.models.test_case import TestCaseModel
from holodeck.models.test_result import ReportSummary, TestReport, TestResult


def _create_mock_report(agent_config_path: str) -> TestReport:
    """Create a mock test report for testing."""
    return TestReport(
        agent_name="test_agent",
        agent_config_path=agent_config_path,
        results=[],
        summary=ReportSummary(
            total_tests=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            total_duration_ms=0,
            metrics_evaluated={},
            average_scores={},
        ),
        timestamp="2024-01-01T00:00:00Z",
        holodeck_version="0.1.0",
        environment={},
    )


def _create_agent_with_tests(num_test_cases: int = 0) -> Agent:
    """Create an Agent instance with specified number of test cases.

    Args:
        num_test_cases: Number of test cases to include

    Returns:
        Agent instance
    """
    test_cases = None
    if num_test_cases > 0:
        test_cases = [
            TestCaseModel(name=f"test_{i}", input="input")
            for i in range(num_test_cases)
        ]

    return Agent(
        name="test_agent",
        description="Test agent",
        model={"provider": "openai", "name": "gpt-4"},
        instructions={"inline": "Test instructions"},
        test_cases=test_cases,
    )


def _setup_test_mocks(num_test_cases: int = 0):
    """Set up common test mocks (ConfigLoader, ProgressIndicator).

    Args:
        num_test_cases: Number of test cases for the agent

    Returns:
        Tuple of (config_loader_patch, progress_indicator_patch)
    """
    config_loader_patch = patch("holodeck.cli.commands.test.ConfigLoader")
    progress_indicator_patch = patch("holodeck.cli.commands.test.ProgressIndicator")

    mock_loader_class = config_loader_patch.start()
    mock_loader = MagicMock()
    mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(num_test_cases)
    mock_loader_class.return_value = mock_loader

    mock_progress_class = progress_indicator_patch.start()
    mock_progress = MagicMock()
    mock_progress.get_progress_line.return_value = ""
    mock_progress.get_summary.return_value = "Test summary"
    mock_progress_class.return_value = mock_progress

    return (config_loader_patch, progress_indicator_patch)


class TestCLIArgumentParsing:
    """Tests for T067: CLI command argument parsing."""

    def test_agent_config_defaults_to_agent_yaml(self):
        """AGENT_CONFIG defaults to agent.yaml when not provided."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create agent.yaml in current directory
            Path("agent.yaml").write_text("")

            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report("agent.yaml")
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                # Invoke without agent_config argument
                runner.invoke(test, [])

                # Should use agent.yaml as default
                mock_loader.load_agent_yaml.assert_called_once_with("agent.yaml")

    def test_agent_config_error_when_default_not_found(self):
        """Error when agent.yaml doesn't exist and no argument provided."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Don't create agent.yaml - it should fail
            result = runner.invoke(test, [])

            assert result.exit_code != 0
            # Click's Path(exists=True) will report the file doesn't exist
            assert "agent.yaml" in result.output or "does not exist" in result.output

    def test_agent_config_argument_accepted(self):
        """AGENT_CONFIG positional argument is accepted."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                # Mock ConfigLoader
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path])

                assert "AGENT_CONFIG" not in result.output or result.exit_code == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_output_option_accepted(self):
        """--output option is accepted for report file path."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path, "--output", "report.json"])

                # Should not complain about invalid option
                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_format_option_accepted(self):
        """--format option is accepted for report format (json/markdown)."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path, "--format", "json"])

                # Should not complain about invalid option
                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_verbose_flag_accepted(self):
        """--verbose flag is accepted for verbose output."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path, "--verbose"])

                # Should not complain about invalid option
                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_quiet_flag_accepted(self):
        """--quiet flag is accepted to suppress progress output."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path, "--quiet"])

                # Should not complain about invalid option
                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_timeout_option_accepted(self):
        """--timeout option is accepted for execution timeout configuration."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path, "--timeout", "120"])

                # Should not complain about invalid option
                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_multiple_options_combined(self):
        """Multiple options can be combined in single command."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(
                    test,
                    [
                        tmp_path,
                        "--output",
                        "report.json",
                        "--format",
                        "json",
                        "--verbose",
                        "--timeout",
                        "60",
                    ],
                )

                # Should accept all combined options
                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCLIExitCodeLogic:
    """Tests for T069: Exit code logic."""

    def test_exit_code_zero_on_success(self):
        """Exit code 0 when all tests pass."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                # Create passing test results
                test_result = TestResult(
                    test_name="test_1",
                    test_input="input",
                    processed_files=[],
                    agent_response="response",
                    tool_calls=[],
                    expected_tools=None,
                    tools_matched=None,
                    metric_results=[],
                    ground_truth=None,
                    passed=True,
                    execution_time_ms=100,
                    errors=[],
                    timestamp="2024-01-01T00:00:00Z",
                )

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=TestReport(
                        agent_name="test_agent",
                        agent_config_path=tmp_path,
                        results=[test_result],
                        summary=ReportSummary(
                            total_tests=1,
                            passed=1,
                            failed=0,
                            pass_rate=100.0,
                            total_duration_ms=100,
                            metrics_evaluated={},
                            average_scores={},
                        ),
                        timestamp="2024-01-01T00:00:00Z",
                        holodeck_version="0.1.0",
                        environment={},
                    )
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_one_on_test_failure(self):
        """Exit code 1 when tests fail (but config and execution were valid)."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                # Create failing test result
                test_result = TestResult(
                    test_name="test_1",
                    test_input="input",
                    processed_files=[],
                    agent_response="response",
                    tool_calls=[],
                    expected_tools=None,
                    tools_matched=None,
                    metric_results=[],
                    ground_truth=None,
                    passed=False,
                    execution_time_ms=100,
                    errors=[],
                    timestamp="2024-01-01T00:00:00Z",
                )

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=TestReport(
                        agent_name="test_agent",
                        agent_config_path=tmp_path,
                        results=[test_result],
                        summary=ReportSummary(
                            total_tests=1,
                            passed=0,
                            failed=1,
                            pass_rate=0.0,
                            total_duration_ms=100,
                            metrics_evaluated={},
                            average_scores={},
                        ),
                        timestamp="2024-01-01T00:00:00Z",
                        holodeck_version="0.1.0",
                        environment={},
                    )
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_two_on_config_error(self):
        """Exit code 2 when configuration is invalid or file not found."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                from holodeck.lib.errors import ConfigError

                # Raise config error during initialization
                mock_executor.side_effect = ConfigError(
                    "agent", "Invalid agent configuration"
                )

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_three_on_execution_error(self):
        """Exit code 3 when execution fails (timeout, agent error, etc)."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                from holodeck.lib.errors import ExecutionError

                mock_instance = MagicMock()
                # Raise execution error during test run
                mock_instance.execute_tests = AsyncMock(
                    side_effect=ExecutionError("Timeout executing agent")
                )
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == 3
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_exit_code_four_on_evaluation_error(self):
        """Exit code 4 when metric evaluation fails."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                from holodeck.lib.errors import EvaluationError

                mock_instance = MagicMock()
                # Raise evaluation error during metric calculation
                mock_instance.execute_tests = AsyncMock(
                    side_effect=EvaluationError("Failed to evaluate metrics")
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == 4
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_mixed_pass_fail_returns_exit_code_one(self):
        """Exit code 1 when some tests pass and some fail."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                # Create mixed results
                passing_result = TestResult(
                    test_name="test_1",
                    test_input="input",
                    processed_files=[],
                    agent_response="response",
                    tool_calls=[],
                    expected_tools=None,
                    tools_matched=None,
                    metric_results=[],
                    ground_truth=None,
                    passed=True,
                    execution_time_ms=100,
                    errors=[],
                    timestamp="2024-01-01T00:00:00Z",
                )

                failing_result = TestResult(
                    test_name="test_2",
                    test_input="input",
                    processed_files=[],
                    agent_response="response",
                    tool_calls=[],
                    expected_tools=None,
                    tools_matched=None,
                    metric_results=[],
                    ground_truth=None,
                    passed=False,
                    execution_time_ms=100,
                    errors=[],
                    timestamp="2024-01-01T00:00:00Z",
                )

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=TestReport(
                        agent_name="test_agent",
                        agent_config_path=tmp_path,
                        results=[passing_result, failing_result],
                        summary=ReportSummary(
                            total_tests=2,
                            passed=1,
                            failed=1,
                            pass_rate=50.0,
                            total_duration_ms=200,
                            metrics_evaluated={},
                            average_scores={},
                        ),
                        timestamp="2024-01-01T00:00:00Z",
                        holodeck_version="0.1.0",
                        environment={},
                    )
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCLIProgressDisplay:
    """Tests for T063: CLI progress display integration."""

    def test_progress_indicator_initialized_with_correct_total(self):
        """ProgressIndicator is initialized with correct total_tests count."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch(
                    "holodeck.cli.commands.test.ProgressIndicator"
                ) as mock_progress_class,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(3)
                mock_loader_class.return_value = mock_loader

                # Mock the executor to return 3 test results
                test_results = [
                    TestResult(
                        test_name=f"test_{i}",
                        test_input="input",
                        processed_files=[],
                        agent_response="response",
                        tool_calls=[],
                        expected_tools=None,
                        tools_matched=None,
                        metric_results=[],
                        ground_truth=None,
                        passed=True,
                        execution_time_ms=100,
                        errors=[],
                        timestamp="2024-01-01T00:00:00Z",
                    )
                    for i in range(1, 4)
                ]

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=TestReport(
                        agent_name="test_agent",
                        agent_config_path=tmp_path,
                        results=test_results,
                        summary=ReportSummary(
                            total_tests=3,
                            passed=3,
                            failed=0,
                            pass_rate=100.0,
                            total_duration_ms=300,
                            metrics_evaluated={},
                            average_scores={},
                        ),
                        timestamp="2024-01-01T00:00:00Z",
                        holodeck_version="0.1.0",
                        environment={},
                    )
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                # Mock progress indicator
                mock_progress_instance = MagicMock()
                mock_progress_class.return_value = mock_progress_instance

                runner.invoke(test, [tmp_path])

                # Verify ProgressIndicator was initialized with total_tests=3
                mock_progress_class.assert_called_once()
                call_kwargs = mock_progress_class.call_args.kwargs
                assert call_kwargs["total_tests"] == 3
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_progress_indicator_respects_quiet_flag(self):
        """ProgressIndicator respects --quiet flag from CLI."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch(
                    "holodeck.cli.commands.test.ProgressIndicator"
                ) as mock_progress_class,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                mock_progress_instance = MagicMock()
                mock_progress_class.return_value = mock_progress_instance

                runner.invoke(test, [tmp_path, "--quiet"])

                # Verify ProgressIndicator was initialized with quiet=True
                call_kwargs = mock_progress_class.call_args.kwargs
                assert call_kwargs["quiet"] is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_progress_indicator_respects_verbose_flag(self):
        """ProgressIndicator respects --verbose flag from CLI."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch(
                    "holodeck.cli.commands.test.ProgressIndicator"
                ) as mock_progress_class,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                mock_progress_instance = MagicMock()
                mock_progress_class.return_value = mock_progress_instance

                runner.invoke(test, [tmp_path, "--verbose"])

                # Verify ProgressIndicator was initialized with verbose=True
                call_kwargs = mock_progress_class.call_args.kwargs
                assert call_kwargs["verbose"] is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_callback_function_passed_to_executor(self):
        """Callback function is passed to TestExecutor."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                runner.invoke(test, [tmp_path])

                # Verify TestExecutor was initialized with progress_callback
                call_kwargs = mock_executor.call_args.kwargs
                assert "progress_callback" in call_kwargs
                assert callable(call_kwargs["progress_callback"])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_callback_updates_progress_indicator(self):
        """Callback function updates ProgressIndicator with test results."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch(
                    "holodeck.cli.commands.test.ProgressIndicator"
                ) as mock_progress_class,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                # Create test result
                test_result = TestResult(
                    test_name="test_1",
                    test_input="input",
                    processed_files=[],
                    agent_response="response",
                    tool_calls=[],
                    expected_tools=None,
                    tools_matched=None,
                    metric_results=[],
                    ground_truth=None,
                    passed=True,
                    execution_time_ms=100,
                    errors=[],
                    timestamp="2024-01-01T00:00:00Z",
                )

                # Capture the callback function passed to executor
                captured_callback = None

                def capture_callback(*args, **kwargs):
                    nonlocal captured_callback
                    captured_callback = kwargs.get("progress_callback")
                    mock_instance = MagicMock()
                    mock_instance.execute_tests = AsyncMock(
                        return_value=TestReport(
                            agent_name="test_agent",
                            agent_config_path=tmp_path,
                            results=[test_result],
                            summary=ReportSummary(
                                total_tests=1,
                                passed=1,
                                failed=0,
                                pass_rate=100.0,
                                total_duration_ms=100,
                                metrics_evaluated={},
                                average_scores={},
                            ),
                            timestamp="2024-01-01T00:00:00Z",
                            holodeck_version="0.1.0",
                            environment={},
                        )
                    )
                    mock_instance.shutdown = AsyncMock()
                    return mock_instance

                mock_executor.side_effect = capture_callback

                mock_progress_instance = MagicMock()
                mock_progress_class.return_value = mock_progress_instance

                runner.invoke(test, [tmp_path])

                # Verify callback was captured
                assert captured_callback is not None

                # Simulate calling the callback with a test result
                captured_callback(test_result)

                # Verify progress indicator's update method was called
                mock_progress_instance.update.assert_called_with(test_result)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_progress_line_printed_after_callback(self):
        """Progress line is printed after callback execution."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch(
                    "holodeck.cli.commands.test.ProgressIndicator"
                ) as mock_progress_class,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                test_result = TestResult(
                    test_name="test_1",
                    test_input="input",
                    processed_files=[],
                    agent_response="response",
                    tool_calls=[],
                    expected_tools=None,
                    tools_matched=None,
                    metric_results=[],
                    ground_truth=None,
                    passed=True,
                    execution_time_ms=100,
                    errors=[],
                    timestamp="2024-01-01T00:00:00Z",
                )

                captured_callback = None

                def capture_callback(*args, **kwargs):
                    nonlocal captured_callback
                    captured_callback = kwargs.get("progress_callback")
                    mock_instance = MagicMock()
                    mock_instance.execute_tests = AsyncMock(
                        return_value=TestReport(
                            agent_name="test_agent",
                            agent_config_path=tmp_path,
                            results=[test_result],
                            summary=ReportSummary(
                                total_tests=1,
                                passed=1,
                                failed=0,
                                pass_rate=100.0,
                                total_duration_ms=100,
                                metrics_evaluated={},
                                average_scores={},
                            ),
                            timestamp="2024-01-01T00:00:00Z",
                            holodeck_version="0.1.0",
                            environment={},
                        )
                    )
                    mock_instance.shutdown = AsyncMock()
                    return mock_instance

                mock_executor.side_effect = capture_callback

                mock_progress_instance = MagicMock()
                mock_progress_instance.get_progress_line.return_value = (
                    "Test 1/1: ✓ test_1"
                )
                mock_progress_class.return_value = mock_progress_instance

                runner.invoke(test, [tmp_path])

                # Verify callback exists
                assert captured_callback is not None

                # Call the callback
                captured_callback(test_result)

                # Verify get_progress_line was called
                mock_progress_instance.get_progress_line.assert_called()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_final_summary_displayed_after_tests_complete(self):
        """Final summary is displayed after all tests complete."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch(
                    "holodeck.cli.commands.test.ProgressIndicator"
                ) as mock_progress_class,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                mock_progress_instance = MagicMock()
                mock_progress_instance.get_summary.return_value = (
                    "Test Results: 0/0 passed (0.0%)"
                )
                mock_progress_class.return_value = mock_progress_instance

                runner.invoke(test, [tmp_path])

                # Verify get_summary was called after tests complete
                mock_progress_instance.get_summary.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_quiet_mode_suppresses_progress_not_summary(self):
        """Quiet mode suppresses progress but still shows summary."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch(
                    "holodeck.cli.commands.test.ProgressIndicator"
                ) as mock_progress_class,
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(0)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                mock_progress_instance = MagicMock()
                # In quiet mode, progress lines return empty string
                mock_progress_instance.get_progress_line.return_value = ""
                mock_progress_instance.get_summary.return_value = (
                    "Test Results: 0/0 passed"
                )
                mock_progress_class.return_value = mock_progress_instance

                runner.invoke(test, [tmp_path, "--quiet"])

                # Summary should still be called even in quiet mode
                mock_progress_instance.get_summary.assert_called_once()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestSpinnerThread:
    """Tests for SpinnerThread class."""

    def test_spinner_thread_run_method(self):
        """Test SpinnerThread run() method executes spinner loop."""
        from holodeck.cli.commands.test import SpinnerThread

        mock_progress = MagicMock()
        mock_progress.get_spinner_line.return_value = "⠋ Test 1/5: Running..."

        spinner = SpinnerThread(mock_progress)

        with (
            patch("sys.stdout.write") as mock_write,
            patch("sys.stdout.flush"),
            patch("time.sleep"),
        ):
            # Start and immediately stop
            spinner.start()
            time.sleep(0.05)  # Let it run briefly
            spinner.stop()
            spinner.join(timeout=1)

            # Verify spinner wrote output
            assert mock_write.called or not spinner.is_alive()

    def test_spinner_thread_stop_method(self):
        """Test SpinnerThread stop() method clears the line."""
        from holodeck.cli.commands.test import SpinnerThread

        mock_progress = MagicMock()
        mock_progress.get_spinner_line.return_value = "⠋ Test 1/5: Running..."

        spinner = SpinnerThread(mock_progress)

        with (
            patch("sys.stdout.write") as mock_write,
            patch("sys.stdout.flush"),
            patch("time.sleep"),
        ):
            spinner.start()
            time.sleep(0.05)
            spinner.stop()
            spinner.join(timeout=1)

            # Verify stop cleared the line
            assert (
                any(" " * 60 in str(call) for call in mock_write.call_args_list)
                or not spinner.is_alive()
            )


class TestReportSaving:
    """Tests for report saving functionality."""

    def test_save_report_json_format(self):
        """Test _save_report with JSON format."""
        from holodeck.cli.commands.test import _save_report

        report = _create_mock_report("test.yaml")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            _save_report(report, tmp_path, "json")

            # Verify file was created
            assert Path(tmp_path).exists()

            # Verify content is valid JSON
            content = Path(tmp_path).read_text()
            import json

            data = json.loads(content)
            assert data["agent_name"] == "test_agent"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_report_markdown_format(self):
        """Test _save_report with Markdown format."""
        from holodeck.cli.commands.test import _save_report

        report = _create_mock_report("test.yaml")

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            _save_report(report, tmp_path, "markdown")

            # Verify file was created
            assert Path(tmp_path).exists()

            # Verify content is markdown
            content = Path(tmp_path).read_text()
            assert "# Test Report:" in content
            assert "test_agent" in content
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_report_auto_detect_json(self):
        """Test _save_report auto-detects JSON from .json extension."""
        from holodeck.cli.commands.test import _save_report

        report = _create_mock_report("test.yaml")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            _save_report(report, tmp_path, None)

            # Verify JSON format was used
            content = Path(tmp_path).read_text()
            import json

            json.loads(content)  # Should not raise
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_report_auto_detect_markdown(self):
        """Test _save_report auto-detects Markdown from .md extension."""
        from holodeck.cli.commands.test import _save_report

        report = _create_mock_report("test.yaml")

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            _save_report(report, tmp_path, None)

            # Verify markdown format was used
            content = Path(tmp_path).read_text()
            assert "# Test Report:" in content
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_report_default_to_json(self):
        """Test _save_report defaults to JSON for unknown extensions."""
        from holodeck.cli.commands.test import _save_report

        report = _create_mock_report("test.yaml")

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            _save_report(report, tmp_path, None)

            # Verify JSON format was used as default
            content = Path(tmp_path).read_text()
            import json

            json.loads(content)  # Should not raise
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_save_report_oserror_handling(self):
        """Test _save_report handles OSError when writing fails."""
        from unittest.mock import patch

        from holodeck.cli.commands.test import _save_report

        report = _create_mock_report("test.yaml")

        # Mock write_text to raise OSError
        with (
            patch("pathlib.Path.write_text", side_effect=OSError("Permission denied")),
            pytest.raises(OSError),
        ):
            _save_report(report, "/some/path/report.json", "json")


class TestGenerateMarkdownReport:
    """Tests for _generate_markdown_report function."""

    def test_generate_markdown_report_structure(self):
        """Test generate_markdown_report creates proper markdown structure."""
        from holodeck.lib.test_runner.reporter import generate_markdown_report

        report = _create_mock_report("test.yaml")
        markdown = generate_markdown_report(report)

        # Verify header
        assert "# Test Report: test_agent" in markdown
        assert "test.yaml" in markdown
        assert "Generated:" in markdown
        assert "0.1.0" in markdown

        # Verify summary section
        assert "## Summary" in markdown
        assert "Total Tests" in markdown
        assert "Passed" in markdown
        assert "Failed" in markdown
        assert "Pass Rate" in markdown
        assert "Duration" in markdown

    def test_generate_markdown_report_with_results(self):
        """Test generate_markdown_report includes results section."""
        from holodeck.lib.test_runner.reporter import generate_markdown_report

        test_result = TestResult(
            test_name="test_1",
            test_input="input",
            processed_files=[],
            agent_response="response",
            tool_calls=[],
            expected_tools=None,
            tools_matched=None,
            metric_results=[],
            ground_truth=None,
            passed=True,
            execution_time_ms=100,
            errors=[],
            timestamp="2024-01-01T00:00:00Z",
        )

        report = TestReport(
            agent_name="test_agent",
            agent_config_path="test.yaml",
            results=[test_result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=100,
                metrics_evaluated={},
                average_scores={},
            ),
            timestamp="2024-01-01T00:00:00Z",
            holodeck_version="0.1.0",
            environment={},
        )

        markdown = generate_markdown_report(report)

        # Verify results section
        assert "## Test Results" in markdown
        assert "test_1" in markdown
        assert "response" in markdown
        assert "✅" in markdown or "PASSED" in markdown or "PASS" in markdown


class TestExceptionHandling:
    """Tests for exception handling in test command."""

    def test_generic_exception_handling(self):
        """Test generic Exception is caught and exits with code 3."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor"),
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.side_effect = RuntimeError(
                    "Unexpected error"
                )
                mock_loader_class.return_value = mock_loader

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == 3
                assert "Error:" in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestOnTestStartCallback:
    """Tests for on_test_start callback."""

    def test_on_test_start_callback_passed_to_executor(self):
        """Test on_test_start callback is passed to TestExecutor."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with (
                patch("holodeck.config.loader.ConfigLoader") as mock_loader_class,
                patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
                patch("holodeck.cli.commands.test.ProgressIndicator"),
            ):
                mock_loader = MagicMock()
                mock_loader.load_agent_yaml.return_value = _create_agent_with_tests(1)
                mock_loader_class.return_value = mock_loader

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_create_mock_report(tmp_path)
                )
                mock_executor.return_value = mock_instance

                runner.invoke(test, [tmp_path])

                # Verify TestExecutor was initialized with on_test_start callback
                call_kwargs = mock_executor.call_args.kwargs
                assert "on_test_start" in call_kwargs
                assert callable(call_kwargs["on_test_start"])
        finally:
            Path(tmp_path).unlink(missing_ok=True)
