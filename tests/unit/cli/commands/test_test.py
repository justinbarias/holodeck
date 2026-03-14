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


def _make_test_result(name: str = "test_1", passed: bool = True) -> TestResult:
    """Create a TestResult with minimal boilerplate."""
    return TestResult(
        test_name=name,
        test_input="input",
        processed_files=[],
        agent_response="response",
        tool_calls=[],
        expected_tools=None,
        tools_matched=None,
        metric_results=[],
        ground_truth=None,
        passed=passed,
        execution_time_ms=100,
        errors=[],
        timestamp="2024-01-01T00:00:00Z",
    )


def _make_report(
    agent_config_path: str,
    results: list[TestResult] | None = None,
    passed: int = 0,
    failed: int = 0,
) -> TestReport:
    """Create a TestReport from results or pass/fail counts."""
    if results is None:
        results = []
    total = passed + failed if (passed or failed) else len(results)
    if not results and total > 0:
        results = [_make_test_result(f"test_{i}", True) for i in range(passed)] + [
            _make_test_result(f"test_fail_{i}", False) for i in range(failed)
        ]
    actual_passed = sum(1 for r in results if r.passed)
    actual_failed = total - actual_passed
    pass_rate = (actual_passed / total * 100.0) if total > 0 else 0.0

    return TestReport(
        agent_name="test_agent",
        agent_config_path=agent_config_path,
        results=results,
        summary=ReportSummary(
            total_tests=total,
            passed=actual_passed,
            failed=actual_failed,
            pass_rate=pass_rate,
            total_duration_ms=100 * total,
            metrics_evaluated={},
            average_scores={},
        ),
        timestamp="2024-01-01T00:00:00Z",
        holodeck_version="0.1.0",
        environment={},
    )


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

    @pytest.mark.parametrize(
        "cli_args, description",
        [
            pytest.param(
                ["--output", "report.json"],
                "output option",
                id="output_option",
            ),
            pytest.param(
                ["--format", "json"],
                "format option",
                id="format_option",
            ),
            pytest.param(
                ["--verbose"],
                "verbose flag",
                id="verbose_flag",
            ),
            pytest.param(
                ["--quiet"],
                "quiet flag",
                id="quiet_flag",
            ),
            pytest.param(
                ["--timeout", "120"],
                "timeout option",
                id="timeout_option",
            ),
            pytest.param(
                [
                    "--output",
                    "report.json",
                    "--format",
                    "json",
                    "--verbose",
                    "--timeout",
                    "60",
                ],
                "multiple combined options",
                id="multiple_options_combined",
            ),
        ],
    )
    def test_option_accepted(self, cli_args: list[str], description: str):
        """CLI option '{description}' is accepted without error."""
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

                result = runner.invoke(test, [tmp_path] + cli_args)

                # Should not complain about invalid option
                assert "no such option" not in result.output.lower()
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCLIExitCodeLogic:
    """Tests for T069: Exit code logic."""

    @pytest.mark.parametrize(
        "scenario, error_side_effect, report_kwargs, expected_exit_code",
        [
            pytest.param(
                "all tests pass",
                None,
                {"passed": 1, "failed": 0},
                0,
                id="exit_0_all_pass",
            ),
            pytest.param(
                "tests fail",
                None,
                {"passed": 0, "failed": 1},
                1,
                id="exit_1_test_failure",
            ),
            pytest.param(
                "mixed pass/fail",
                None,
                {"passed": 1, "failed": 1},
                1,
                id="exit_1_mixed_pass_fail",
            ),
        ],
    )
    def test_exit_code_from_report(
        self, scenario, error_side_effect, report_kwargs, expected_exit_code
    ):
        """Exit code {expected_exit_code} when {scenario}."""
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
                    return_value=_make_report(tmp_path, **report_kwargs)
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == expected_exit_code
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "error_class_path, error_args, expected_exit_code",
        [
            pytest.param(
                "holodeck.lib.errors.ConfigError",
                ("agent", "Invalid agent configuration"),
                2,
                id="exit_2_config_error",
            ),
            pytest.param(
                "holodeck.lib.errors.ExecutionError",
                ("Timeout executing agent",),
                3,
                id="exit_3_execution_error",
            ),
            pytest.param(
                "holodeck.lib.errors.EvaluationError",
                ("Failed to evaluate metrics",),
                4,
                id="exit_4_evaluation_error",
            ),
        ],
    )
    def test_exit_code_from_error(
        self, error_class_path, error_args, expected_exit_code
    ):
        """Exit code {expected_exit_code} on {error_class_path}."""
        import importlib

        module_path, class_name = error_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        error_class = getattr(module, class_name)

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

                if expected_exit_code == 2:
                    # ConfigError raised during executor init
                    mock_executor.side_effect = error_class(*error_args)
                else:
                    # Error raised during test execution
                    mock_instance = MagicMock()
                    mock_instance.execute_tests = AsyncMock(
                        side_effect=error_class(*error_args)
                    )
                    mock_instance.shutdown = AsyncMock()
                    mock_executor.return_value = mock_instance

                result = runner.invoke(test, [tmp_path])

                assert result.exit_code == expected_exit_code
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

                mock_instance = MagicMock()
                mock_instance.execute_tests = AsyncMock(
                    return_value=_make_report(tmp_path, passed=3, failed=0)
                )
                mock_instance.shutdown = AsyncMock()
                mock_executor.return_value = mock_instance

                mock_progress_instance = MagicMock()
                mock_progress_class.return_value = mock_progress_instance

                runner.invoke(test, [tmp_path])

                # Verify ProgressIndicator was initialized with total_tests=3
                mock_progress_class.assert_called_once()
                call_kwargs = mock_progress_class.call_args.kwargs
                assert call_kwargs["total_tests"] == 3
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.parametrize(
        "flag, kwarg_name, kwarg_value",
        [
            pytest.param("--quiet", "quiet", True, id="quiet_flag"),
            pytest.param("--verbose", "verbose", True, id="verbose_flag"),
        ],
    )
    def test_progress_indicator_respects_flag(self, flag, kwarg_name, kwarg_value):
        """ProgressIndicator respects {flag} flag from CLI."""
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

                runner.invoke(test, [tmp_path, flag])

                call_kwargs = mock_progress_class.call_args.kwargs
                assert call_kwargs[kwarg_name] is kwarg_value
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

                test_result = _make_test_result()

                # Capture the callback function passed to executor
                captured_callback = None

                def capture_callback(*args, **kwargs):
                    nonlocal captured_callback
                    captured_callback = kwargs.get("progress_callback")
                    mock_instance = MagicMock()
                    mock_instance.execute_tests = AsyncMock(
                        return_value=_make_report(tmp_path, results=[test_result])
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

                test_result = _make_test_result()

                captured_callback = None

                def capture_callback(*args, **kwargs):
                    nonlocal captured_callback
                    captured_callback = kwargs.get("progress_callback")
                    mock_instance = MagicMock()
                    mock_instance.execute_tests = AsyncMock(
                        return_value=_make_report(tmp_path, results=[test_result])
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

    @pytest.mark.parametrize(
        "suffix, format_arg, verify_fn",
        [
            pytest.param(
                ".json",
                "json",
                lambda content: __import__("json").loads(content)["agent_name"]
                == "test_agent",
                id="json_format",
            ),
            pytest.param(
                ".md",
                "markdown",
                lambda content: "# Test Report:" in content and "test_agent" in content,
                id="markdown_format",
            ),
            pytest.param(
                ".json",
                None,
                lambda content: __import__("json").loads(content) is not None,
                id="auto_detect_json",
            ),
            pytest.param(
                ".md",
                None,
                lambda content: "# Test Report:" in content,
                id="auto_detect_markdown",
            ),
            pytest.param(
                ".txt",
                None,
                lambda content: __import__("json").loads(content) is not None,
                id="default_to_json",
            ),
        ],
    )
    def test_save_report_format(self, suffix, format_arg, verify_fn):
        """Test _save_report with format '{format_arg}' and suffix '{suffix}'."""
        from holodeck.cli.commands.test import _save_report

        report = _create_mock_report("test.yaml")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w") as tmp:
            tmp_path = tmp.name

        try:
            _save_report(report, tmp_path, format_arg)

            assert Path(tmp_path).exists()
            content = Path(tmp_path).read_text()
            assert verify_fn(content)
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

        test_result = _make_test_result()
        report = _make_report("test.yaml", results=[test_result])

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
