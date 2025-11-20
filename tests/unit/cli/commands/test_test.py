"""Unit tests for CLI test command.

Tests cover:
- Argument parsing (AGENT_CONFIG positional argument)
- Option handling (--output, --format, --verbose, --quiet, --timeout flags)
- Exit code logic (0=success, 1=test failure, 2=config error, 3=execution error)
- Progress callback integration
- Report file generation
"""

import tempfile
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

    def test_agent_config_positional_argument_required(self):
        """AGENT_CONFIG positional argument is required."""
        runner = CliRunner()
        result = runner.invoke(test, [])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "AGENT_CONFIG" in result.output

    def test_agent_config_argument_accepted(
        self, cli_runner, temp_agent_config, mock_test_command_deps
    ):
        """AGENT_CONFIG positional argument is accepted."""
        result = cli_runner.invoke(test, [str(temp_agent_config)])
        assert "AGENT_CONFIG" not in result.output or result.exit_code == 0

    def test_output_option_accepted(
        self, cli_runner, temp_agent_config, mock_test_command_deps
    ):
        """--output option is accepted for report file path."""
        result = cli_runner.invoke(
            test, [str(temp_agent_config), "--output", "report.json"]
        )
        # Should not complain about invalid option
        assert "no such option" not in result.output.lower()

    @pytest.mark.parametrize(
        "option_args",
        [
            (["--format", "json"], "format option"),
            (["--verbose"], "verbose flag"),
            (["--quiet"], "quiet flag"),
            (["--timeout", "120"], "timeout option"),
        ],
        ids=["format", "verbose", "quiet", "timeout"],
    )
    def test_cli_options_accepted(
        self, cli_runner, temp_agent_config, mock_test_command_deps, option_args
    ):
        """Test that various CLI options are accepted without error."""
        args, description = option_args
        result = cli_runner.invoke(test, [str(temp_agent_config)] + args)
        # Should not complain about invalid option
        assert "no such option" not in result.output.lower()

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
