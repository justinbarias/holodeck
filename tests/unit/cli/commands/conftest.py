"""Shared fixtures for CLI command tests."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.models.agent import Agent
from holodeck.models.test_case import TestCaseModel
from holodeck.models.test_result import ReportSummary, TestReport, TestResult


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_agent_config():
    """Create a temporary agent config file for testing.

    Yields:
        Path to temporary YAML file that is automatically cleaned up.
    """
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as tmp:
        # Write a valid minimal agent config with one test case
        tmp.write(
            """name: test_agent
description: Test agent
model:
  provider: openai
  name: gpt-4
instructions:
  inline: Test instructions
test_cases:
  - name: test_1
    input: test input
"""
        )
        tmp_path = Path(tmp.name)

    yield tmp_path

    # Cleanup
    tmp_path.unlink(missing_ok=True)


@pytest.fixture
def mock_test_result():
    """Create a mock test result factory.

    Returns:
        Callable that creates TestResult instances with configurable pass/fail state.
    """

    def _create_result(passed: bool = True, **kwargs: Any) -> TestResult:
        defaults = {
            "test_name": "test_1",
            "test_input": "input",
            "processed_files": [],
            "agent_response": "response",
            "tool_calls": [],
            "expected_tools": None,
            "tools_matched": None,
            "metric_results": [],
            "ground_truth": None,
            "passed": passed,
            "execution_time_ms": 100,
            "errors": [] if passed else ["Test failed"],
            "timestamp": "2024-01-01T00:00:00Z",
        }
        defaults.update(kwargs)
        return TestResult(**defaults)

    return _create_result


@pytest.fixture
def mock_test_report(mock_test_result):
    """Create a mock test report factory.

    Returns:
        Callable that creates TestReport instances for testing.
    """

    def _create_report(
        agent_config_path: str = "test.yaml",
        passed: int = 0,
        failed: int = 0,
        results: list[TestResult] | None = None,
    ) -> TestReport:
        if results is None:
            results = []
            for i in range(passed):
                results.append(mock_test_result(passed=True, test_name=f"test_{i}"))
            for i in range(failed):
                results.append(
                    mock_test_result(passed=False, test_name=f"test_fail_{i}")
                )

        # Calculate totals from results if provided, otherwise use passed/failed
        if results is not None and (passed == 0 and failed == 0):
            total = len(results)
            passed = sum(1 for r in results if r.passed)
            failed = total - passed
        else:
            total = passed + failed

        pass_rate = (passed / total * 100.0) if total > 0 else 0.0

        return TestReport(
            agent_name="test_agent",
            agent_config_path=agent_config_path,
            results=results,
            summary=ReportSummary(
                total_tests=total,
                passed=passed,
                failed=failed,
                pass_rate=pass_rate,
                total_duration_ms=100 * total,
                metrics_evaluated={},
                average_scores={},
            ),
            timestamp="2024-01-01T00:00:00Z",
            holodeck_version="0.1.0",
            environment={},
        )

    return _create_report


@pytest.fixture
def mock_agent_factory():
    """Create a mock Agent factory for testing.

    Returns:
        Callable that creates Agent instances with specified number of test cases.
    """

    def _create_agent(num_test_cases: int = 0, **kwargs: Any) -> Agent:
        test_cases = None
        if num_test_cases > 0:
            test_cases = [
                TestCaseModel(name=f"test_{i}", input="input")
                for i in range(num_test_cases)
            ]

        defaults = {
            "name": "test_agent",
            "description": "Test agent",
            "model": {"provider": "openai", "name": "gpt-4"},
            "instructions": {"inline": "Test instructions"},
            "test_cases": test_cases,
        }
        defaults.update(kwargs)
        return Agent(**defaults)

    return _create_agent


@pytest.fixture
def mock_config_loader(mock_agent_factory):
    """Mock the ConfigLoader class.

    Yields:
        Tuple of (mock_loader_class, mock_loader_instance) for assertions.
    """
    with patch("holodeck.config.loader.ConfigLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_agent_yaml.return_value = mock_agent_factory(0)
        mock_loader_class.return_value = mock_loader
        yield mock_loader_class, mock_loader


@pytest.fixture
def mock_test_executor(mock_test_report):
    """Mock the TestExecutor class with default behavior.

    The mock can be customized in tests by accessing the instance's
    execute_tests method.

    Yields:
        Tuple of (mock_executor_class, mock_executor_instance) for assertions.
    """
    with patch("holodeck.lib.test_runner.executor.TestExecutor") as mock_executor_class:
        mock_instance = MagicMock()
        mock_instance.execute_tests = AsyncMock(return_value=mock_test_report())
        mock_executor_class.return_value = mock_instance
        yield mock_executor_class, mock_instance


@pytest.fixture
def mock_progress_indicator():
    """Mock the ProgressIndicator class.

    Yields:
        Tuple of (mock_progress_class, mock_progress_instance) for assertions.
    """
    with patch(
        "holodeck.lib.test_runner.progress.ProgressIndicator"
    ) as mock_progress_class:
        mock_progress = MagicMock()
        mock_progress.get_progress_line.return_value = ""
        mock_progress.get_summary.return_value = "Test summary"
        mock_progress_class.return_value = mock_progress
        yield mock_progress_class, mock_progress


@pytest.fixture
def mock_test_command_deps(
    mock_config_loader, mock_test_executor, mock_progress_indicator
):
    """Combined fixture for all test command dependencies.

    Provides all mocks needed for testing the `holodeck test` command.

    Yields:
        Dict with keys: config_loader, executor, progress (each a tuple of
        class and instance mocks).
    """
    yield {
        "config_loader": mock_config_loader,
        "executor": mock_test_executor,
        "progress": mock_progress_indicator,
    }
