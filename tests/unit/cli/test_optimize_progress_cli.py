"""CLI transport tests for `--progress` (038 T3).

Asserts the stdout/stderr split: `plain` (default) is unchanged and emits no events;
`json` emits a pure NDJSON stream to stdout (run_started … run_completed) with all human
text and logs on stderr; a fatal failure emits a terminal error event; and the run
artifacts are byte-identical across the two modes (acceptance #3).
"""

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from holodeck.models.test_result import ReportSummary, TestReport, TestResult


def _agent_yaml(with_test_cases: bool = True) -> str:
    base = """name: opt-agent
model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.3
  api_key: test-key
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


def _decreasing_scorer() -> AsyncMock:
    """A scorer whose loss falls by call order, so trials accept deterministically."""
    calls = {"n": 0}

    async def fake_score(agent, path, weights, backend=None):
        value = 0.40 - 0.05 * calls["n"]
        calls["n"] += 1
        return value, _report()

    return AsyncMock(side_effect=fake_score)


def _run(args: list[str]) -> object:
    runner = CliRunner()
    with patch("holodeck.cli.commands.optimize.score", new=_decreasing_scorer()):
        return runner.invoke(_import_cli(), args)


class TestHelp:
    """`--progress` is documented in --help with its default."""

    def test_help_lists_progress(self) -> None:
        result = CliRunner().invoke(_import_cli(), ["test", "optimize", "--help"])
        assert result.exit_code == 0
        assert "--progress" in result.output
        assert "json" in result.output


class TestPlainMode:
    """Default/plain mode emits no events and keeps human output on stdout."""

    def test_plain_emits_no_events(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text(_agent_yaml())
        result = _run(
            [
                "test",
                "optimize",
                str(agent_path),
                "-o",
                str(tmp_path / "r"),
                "--seed",
                "1",
            ]
        )
        assert result.exit_code == 0, result.output
        # No NDJSON event lines on stdout.
        assert "holodeck.optimize.progress" not in result.stdout
        for line in result.stdout.splitlines():
            assert not line.lstrip().startswith('{"schema"')
        # Human per-trial streaming is present.
        assert "loss" in result.stdout.lower()


class TestJsonMode:
    """json mode: pure NDJSON on stdout, human text + logs on stderr."""

    def test_stdout_is_pure_ndjson(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text(_agent_yaml())
        result = _run(
            [
                "test",
                "optimize",
                str(agent_path),
                "-o",
                str(tmp_path / "r"),
                "--seed",
                "1",
                "--progress",
                "json",
            ]
        )
        assert result.exit_code == 0, result.stderr
        lines = result.stdout.splitlines()
        events = [json.loads(line) for line in lines]  # every line parses
        assert all(e["schema"] == "holodeck.optimize.progress/v1" for e in events)
        assert events[0]["event"] == "run_started"
        assert events[-1]["event"] == "run_completed"
        # The grammar opens and closes exactly once.
        assert [e["event"] for e in events].count("run_started") == 1
        assert [e["event"] for e in events].count("run_completed") == 1

    def test_run_completed_carries_artifacts(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text(_agent_yaml())
        result = _run(
            [
                "test",
                "optimize",
                str(agent_path),
                "-o",
                str(tmp_path / "r"),
                "--seed",
                "1",
                "--progress",
                "json",
            ]
        )
        completed = json.loads(result.stdout.splitlines()[-1])
        for key in ("baseline_loss", "best_loss", "accepted", "cycles"):
            assert key in completed
        artifacts = completed["artifacts"]
        assert artifacts["trials_jsonl"].endswith("trials.jsonl")
        assert Path(artifacts["best_yaml"]).exists()
        assert Path(artifacts["trials_jsonl"]).exists()
        assert Path(artifacts["report_md"]).exists()

    def test_human_text_and_logs_on_stderr(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text(_agent_yaml())
        result = _run(
            [
                "test",
                "optimize",
                str(agent_path),
                "-o",
                str(tmp_path / "r"),
                "--seed",
                "1",
                "--progress",
                "json",
            ]
        )
        assert "Optimizing 'opt-agent'" in result.stderr
        assert "Baseline loss" in result.stderr  # INFO log line
        # The human summary is NOT on the machine channel.
        assert "Artifacts written to" not in result.stdout


class TestFatalError:
    """A fatal failure emits a terminal error event under json mode."""

    def test_missing_test_cases_emits_error_event(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text(_agent_yaml(with_test_cases=False))
        result = _run(["test", "optimize", str(agent_path), "--progress", "json"])
        assert result.exit_code != 0
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert events[0]["fatal"] is True
        assert "test case" in events[0]["message"].lower()
        # Human-readable error still goes to stderr.
        assert "Error:" in result.stderr


class TestArtifactsByteIdentical:
    """Artifacts are byte-identical between plain and json runs (acceptance #3)."""

    def test_artifacts_match_across_modes(self, tmp_path: Path) -> None:
        agent_path = tmp_path / "agent.yaml"
        agent_path.write_text(_agent_yaml())

        plain_dir = tmp_path / "plain"
        json_dir = tmp_path / "json"
        _run(["test", "optimize", str(agent_path), "-o", str(plain_dir), "--seed", "1"])
        _run(
            [
                "test",
                "optimize",
                str(agent_path),
                "-o",
                str(json_dir),
                "--seed",
                "1",
                "--progress",
                "json",
            ]
        )

        plain_run = next(plain_dir.iterdir())
        json_run = next(json_dir.iterdir())

        # trials.jsonl and best.yaml carry no run-id, so they must be byte-identical.
        assert (plain_run / "trials.jsonl").read_bytes() == (
            json_run / "trials.jsonl"
        ).read_bytes()
        assert (plain_run / "best.yaml").read_bytes() == (
            json_run / "best.yaml"
        ).read_bytes()

        # report.md embeds the (always-unique) run-id; equal once normalized.
        def _norm(run_dir: Path) -> str:
            text = (run_dir / "report.md").read_text()
            return re.sub(r"\d{8}-\d{6}-[0-9a-f]{6}", "RUNID", text)

        assert _norm(plain_run) == _norm(json_run)
