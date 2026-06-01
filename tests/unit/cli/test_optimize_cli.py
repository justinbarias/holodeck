"""Unit tests for the `holodeck test optimize` CLI command (T8)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from holodeck.models.test_result import ReportSummary, TestReport, TestResult


def _agent_yaml(with_test_cases: bool) -> str:
    base = """name: opt-agent
model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.3
instructions:
  inline: You are helpful.
evaluations:
  metrics:
    - type: standard
      metric: groundedness
  optimizer:
    loss:
      groundedness: 1.0
    axes:
      numeric:
        - path: model.temperature
          type: float
          range: [0.0, 1.0]
    max_cycles: 1
    numeric_phase:
      max_trials: 3
      patience: 3
    textual_phase:
      max_trials: 1
      patience: 1
"""
    if with_test_cases:
        base += """
test_cases:
  - name: case1
    input: What is 2+2?
    ground_truth: "4"
"""
    return base


def _report() -> TestReport:
    return TestReport(
        agent_name="opt-agent",
        agent_config_path="agent.yaml",
        results=[
            TestResult(
                test_name="case1",
                test_input="What is 2+2?",
                agent_response="4",
                passed=True,
                execution_time_ms=1,
                timestamp="2026-05-31T00:00:00Z",
            )
        ],
        summary=ReportSummary(
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=1,
            metrics_evaluated={"groundedness": 1},
            average_scores={"groundedness": 0.5},
        ),
        timestamp="2026-05-31T00:00:00Z",
        holodeck_version="test",
    )


def _import_cli():
    from holodeck.cli.main import main

    return main


class TestHelp:
    """`holodeck test optimize --help` lists all flags."""

    def test_help_lists_flags(self) -> None:
        runner = CliRunner()
        result = runner.invoke(_import_cli(), ["test", "optimize", "--help"])

        assert result.exit_code == 0
        for flag in (
            "--max-cycles",
            "--numeric-max-trials",
            "--numeric-patience",
            "--textual-max-trials",
            "--textual-patience",
            "--seed",
            "--output-dir",
            "--verbose",
            "--quiet",
        ):
            assert flag in result.output


class TestValidation:
    """Missing/empty test cases exit non-zero with a clear message."""

    def test_no_test_cases_errors(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text(_agent_yaml(with_test_cases=False))

        runner = CliRunner()
        result = runner.invoke(_import_cli(), ["test", "optimize", str(agent_path)])

        assert result.exit_code != 0
        assert "test case" in result.output.lower()


class TestRun:
    """A fixture run streams per-trial losses and writes outputs."""

    def test_streams_losses_and_writes_outputs(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text(_agent_yaml(with_test_cases=True))
        out_dir = tmp_path / "results"

        # Decreasing losses so the numeric phase accepts and streams.
        calls = {"n": 0}

        async def fake_score(agent, path, weights, backend=None):
            value = 0.40 - 0.05 * calls["n"]
            calls["n"] += 1
            return value, _report()

        runner = CliRunner()
        with patch(
            "holodeck.cli.commands.optimize.score",
            new=AsyncMock(side_effect=fake_score),
        ):
            result = runner.invoke(
                _import_cli(),
                [
                    "test",
                    "optimize",
                    str(agent_path),
                    "--output-dir",
                    str(out_dir),
                    "--seed",
                    "1",
                ],
            )

        assert result.exit_code == 0, result.output
        # Per-trial loss lines were streamed.
        assert "loss" in result.output.lower()
        # Outputs were written under a run-id dir.
        run_dirs = list(out_dir.iterdir())
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "best.yaml").exists()
        assert (run_dirs[0] / "trials.jsonl").exists()
        assert (run_dirs[0] / "report.md").exists()
