"""Integration tests: `holodeck test` persists an EvalRun (T019, T021, T022).

The TestExecutor is mocked to bypass real LLM calls; everything else
(config loading, env-var resolution, eval-run building, redaction, atomic
write) is real.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.test import test
from holodeck.models.eval_run import EvalRun
from holodeck.models.test_result import ReportSummary, TestReport, TestResult


def _make_report(passed: int = 1) -> TestReport:
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
    ]
    return TestReport(
        agent_name="my-agent",
        agent_config_path="agent.yaml",
        results=results,
        summary=ReportSummary(
            total_tests=passed,
            passed=passed,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=10 * passed,
        ),
        timestamp="2026-04-18T14:22:09.812Z",
        holodeck_version="0.1.0",
    )


def _write_agent_yaml(path: Path, *, with_test_cases: bool = True) -> None:
    contents = dedent(
        """
        name: my-agent
        model:
          provider: openai
          name: gpt-4o
          api_key: ${OPENAI_API_KEY}
        instructions:
          inline: "You are a helpful assistant."
        """
    ).strip()
    if with_test_cases:
        contents += (
            "\n"
            + dedent(
                """
            test_cases:
              - name: greeting
                input: "Say hi"
            """
            ).strip()
        )
    path.write_text(contents)


@pytest.fixture
def patched_executor():
    """Patch TestExecutor so no real LLM call happens; report is configurable."""
    with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor:
        instance = MagicMock()
        instance.execute_tests = AsyncMock()
        instance.shutdown = AsyncMock()
        mock_executor.return_value = instance
        yield instance


@pytest.mark.integration
class TestPersistEvalRun:
    def test_run_creates_eval_run_file(
        self, tmp_path: Path, patched_executor, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-real")
        agent_yaml = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_yaml)
        patched_executor.execute_tests.return_value = _make_report(passed=1)

        runner = CliRunner()
        result = runner.invoke(test, [str(agent_yaml)])
        assert result.exit_code == 0, result.output

        results_dir = tmp_path / "results" / "my-agent"
        files = list(results_dir.glob("*.json"))
        assert len(files) == 1
        # Round-trip via EvalRun.
        run = EvalRun.model_validate_json(files[0].read_text())
        assert run.report.agent_name == "my-agent"

    def test_output_and_persistence_coexist(
        self, tmp_path: Path, patched_executor, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-real")
        agent_yaml = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_yaml)
        patched_executor.execute_tests.return_value = _make_report(passed=1)

        report_md = tmp_path / "report.md"
        runner = CliRunner()
        result = runner.invoke(
            test,
            [str(agent_yaml), "--output", str(report_md), "--format", "markdown"],
        )
        assert result.exit_code == 0, result.output
        # Both artifacts present.
        assert report_md.exists()
        files = list((tmp_path / "results" / "my-agent").glob("*.json"))
        assert len(files) == 1

    def test_no_file_when_no_test_cases(
        self, tmp_path: Path, patched_executor, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-real")
        agent_yaml = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_yaml, with_test_cases=False)
        patched_executor.execute_tests.return_value = _make_report(passed=0)

        runner = CliRunner()
        result = runner.invoke(test, [str(agent_yaml)])
        assert result.exit_code == 0, result.output

        results_dir = tmp_path / "results" / "my-agent"
        if results_dir.exists():
            files = list(results_dir.glob("*.json"))
            assert files == []

    def test_redacts_api_key_from_env(
        self, tmp_path: Path, patched_executor, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-real")
        agent_yaml = tmp_path / "agent.yaml"
        _write_agent_yaml(agent_yaml)
        patched_executor.execute_tests.return_value = _make_report(passed=1)

        runner = CliRunner()
        result = runner.invoke(test, [str(agent_yaml)])
        assert result.exit_code == 0, result.output

        files = list((tmp_path / "results" / "my-agent").glob("*.json"))
        assert len(files) == 1
        contents = files[0].read_text()
        assert "sk-fake-not-real" not in contents
        assert '"api_key": "***"' in contents
