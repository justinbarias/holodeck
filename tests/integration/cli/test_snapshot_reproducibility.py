"""US3 T211: snapshot is reconstructable into a valid ``Agent`` (AC6).

A reader should be able to load an ``EvalRun`` and revive the exact ``Agent``
configuration that produced it without touching the live ``agent.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.test import test
from holodeck.models.agent import Agent
from holodeck.models.eval_run import EvalRun
from holodeck.models.test_result import ReportSummary, TestReport, TestResult


def _make_report() -> TestReport:
    return TestReport(
        agent_name="repro-agent",
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


_AGENT_YAML = dedent("""
    name: repro-agent
    description: "Reproducible snapshot demo"
    model:
      provider: openai
      name: gpt-4o
      temperature: 0.65
      max_tokens: 512
    instructions:
      inline: "You are a reproducibility test subject."
    tools:
      - name: lookup
        description: "lookup tool"
        type: function
        file: ./t.py
        function: lookup
    evaluations:
      metrics:
        - type: standard
          metric: bleu
          threshold: 0.3
    test_cases:
      - name: smoke
        input: "Say hi"
    """).strip()


@pytest.fixture
def patched_executor():
    with patch("holodeck.cli.commands.test.TestExecutor") as mock_executor:
        instance = MagicMock()
        instance.execute_tests = AsyncMock()
        instance.shutdown = AsyncMock()
        mock_executor.return_value = instance
        yield instance


@pytest.mark.integration
class TestSnapshotReproducibility:
    def test_agent_can_be_reconstructed_from_snapshot_alone(
        self, tmp_path: Path, patched_executor, monkeypatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-real")
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(_AGENT_YAML)
        patched_executor.execute_tests.return_value = _make_report()

        runner = CliRunner()
        result = runner.invoke(test, [str(agent_yaml)])
        assert result.exit_code == 0, result.output

        files = list((tmp_path / "results" / "repro-agent").glob("*.json"))
        assert len(files) == 1

        run = EvalRun.model_validate_json(files[0].read_text())

        # Dump the snapshot to a plain dict and ask Pydantic to rebuild
        # the Agent from it — no live agent.yaml consulted.
        revived = Agent.model_validate(run.metadata.agent_config.model_dump())

        assert revived.name == "repro-agent"
        assert revived.model.temperature == 0.65
        assert revived.model.max_tokens == 512
        assert revived.tools is not None
        assert revived.tools[0].name == "lookup"
        assert revived.evaluations is not None
        assert revived.evaluations.metrics[0].metric == "bleu"  # type: ignore[union-attr]
        assert revived.test_cases is not None
        assert revived.test_cases[0].input == "Say hi"
