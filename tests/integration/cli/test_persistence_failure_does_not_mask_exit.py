"""Integration test: persistence failure logs WARNING but preserves exit code (T020)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.test import test
from holodeck.models.test_result import ReportSummary, TestReport, TestResult


def _make_report(passed: int = 1, failed: int = 0) -> TestReport:
    results = [
        TestResult(
            test_name=f"t{i}",
            test_input="hi",
            agent_response="hi back",
            metric_results=[],
            passed=True,
            execution_time_ms=10,
            timestamp="2026-04-18T14:22:09.812Z",
        )
        for i in range(passed)
    ] + [
        TestResult(
            test_name=f"f{i}",
            test_input="hi",
            agent_response="hi back",
            metric_results=[],
            passed=False,
            execution_time_ms=10,
            timestamp="2026-04-18T14:22:09.812Z",
        )
        for i in range(failed)
    ]
    return TestReport(
        agent_name="my-agent",
        agent_config_path="agent.yaml",
        results=results,
        summary=ReportSummary(
            total_tests=passed + failed,
            passed=passed,
            failed=failed,
            pass_rate=(
                100.0 * passed / (passed + failed) if (passed + failed) else 0.0
            ),
            total_duration_ms=10 * (passed + failed),
        ),
        timestamp="2026-04-18T14:22:09.812Z",
        holodeck_version="0.1.0",
    )


def _write_agent_yaml(path: Path) -> None:
    path.write_text(dedent("""
            name: my-agent
            model:
              provider: openai
              name: gpt-4o
            instructions:
              inline: "You are a helpful assistant."
            test_cases:
              - name: greeting
                input: "Say hi"
            """).strip())


@pytest.mark.integration
class TestPersistenceFailureDoesNotMaskExit:
    @pytest.mark.parametrize(
        "passed, failed, expected_exit",
        [
            (1, 0, 0),
            (0, 1, 1),
        ],
    )
    def test_writer_failure_preserves_exit_code(
        self,
        tmp_path: Path,
        passed: int,
        failed: int,
        expected_exit: int,
    ):
        agent_yaml = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_yaml)

        with (
            patch("holodeck.cli.commands.test.TestExecutor") as mock_executor,
            patch(
                "holodeck.cli.commands.test.write_eval_run",
                side_effect=PermissionError("denied"),
            ),
        ):
            instance = MagicMock()
            instance.execute_tests = AsyncMock(
                return_value=_make_report(passed=passed, failed=failed)
            )
            instance.shutdown = AsyncMock()
            mock_executor.return_value = instance

            runner = CliRunner()
            result = runner.invoke(test, [str(agent_yaml)])

        assert result.exit_code == expected_exit, result.output

        # A single CLI WARNING notice must mention persistence (FR-009).
        combined = result.output
        assert "persist" in combined.lower() or "EvalRun" in combined, combined
        # And the success line must not be emitted.
        assert "EvalRun persisted:" not in combined
