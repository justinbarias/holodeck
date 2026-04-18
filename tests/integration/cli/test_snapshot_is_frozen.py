"""US3 T210: persisted snapshot is frozen against later ``agent.yaml`` edits.

AC5: editing ``agent.yaml`` after a run has been persisted MUST NOT affect the
captured ``EvalRun`` on disk.
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


def _make_report() -> TestReport:
    return TestReport(
        agent_name="snapshot-agent",
        agent_config_path="agent.yaml",
        results=[
            TestResult(
                test_name="smoke",
                test_input="hi",
                agent_response="hi back",
                metric_results=[],
                passed=True,
                execution_time_ms=5,
                timestamp="2026-04-18T14:22:09.812Z",
            )
        ],
        summary=ReportSummary(
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=5,
        ),
        timestamp="2026-04-18T14:22:09.812Z",
        holodeck_version="0.1.0",
    )


def _agent_yaml(temperature: float) -> str:
    return dedent(
        f"""
        name: snapshot-agent
        model:
          provider: openai
          name: gpt-4o
          temperature: {temperature}
        instructions:
          inline: "You are helpful."
        test_cases:
          - name: smoke
            input: "Say hi"
        """
    ).strip()


@pytest.fixture
def patched_executor():
    with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor:
        instance = MagicMock()
        instance.execute_tests = AsyncMock()
        instance.shutdown = AsyncMock()
        mock_executor.return_value = instance
        yield instance


@pytest.mark.integration
class TestSnapshotIsFrozen:
    def test_agent_yaml_edit_does_not_affect_captured_run(
        self, tmp_path: Path, patched_executor, monkeypatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-real")
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(_agent_yaml(temperature=0.7))
        patched_executor.execute_tests.return_value = _make_report()

        runner = CliRunner()
        result = runner.invoke(test, [str(agent_yaml)])
        assert result.exit_code == 0, result.output

        files = list((tmp_path / "results" / "snapshot-agent").glob("*.json"))
        assert len(files) == 1
        captured = files[0]
        run = EvalRun.model_validate_json(captured.read_text())
        assert run.metadata.agent_config.model.temperature == 0.7

        # Now edit agent.yaml (simulating the user changing temperature later).
        agent_yaml.write_text(_agent_yaml(temperature=0.2))

        # Re-load the captured run file; the snapshot MUST be frozen.
        run_after = EvalRun.model_validate_json(captured.read_text())
        assert run_after.metadata.agent_config.model.temperature == 0.7

    def test_snapshot_readable_without_live_agent_yaml(
        self, tmp_path: Path, patched_executor, monkeypatch
    ) -> None:
        """AC6: a reader can reconstruct an Agent from the snapshot alone."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-real")
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(_agent_yaml(temperature=0.55))
        patched_executor.execute_tests.return_value = _make_report()

        runner = CliRunner()
        result = runner.invoke(test, [str(agent_yaml)])
        assert result.exit_code == 0, result.output

        files = list((tmp_path / "results" / "snapshot-agent").glob("*.json"))
        assert len(files) == 1

        # Delete the live agent.yaml — the snapshot must stand alone.
        agent_yaml.unlink()
        run = EvalRun.model_validate_json(files[0].read_text())
        assert run.metadata.agent_config.model.temperature == 0.55
