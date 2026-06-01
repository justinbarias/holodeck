"""Unit tests for optimizer output artifacts (T7)."""

import json
from pathlib import Path

import yaml

from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.optimizer.models import OptimizationResult, TrialRecord
from holodeck.optimizer.output import write_outputs


def _result() -> OptimizationResult:
    best_agent = Agent(
        name="opt-agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4o-mini", temperature=0.7
        ),
        instructions=Instructions(inline="You are a concise expert."),
    )
    trials = [
        TrialRecord(
            trial_id=1,
            cycle=0,
            phase="numeric",
            loss=0.50,
            baseline_loss=0.62,
            accepted=True,
            params={"model.temperature": 0.7},
        ),
        TrialRecord(
            trial_id=2,
            cycle=0,
            phase="textual",
            loss=0.55,
            baseline_loss=0.50,
            accepted=False,
            textual_axis="instructions.inline",
            edit_summary="Tightened the format.",
        ),
    ]
    return OptimizationResult(
        run_id="run-xyz",
        agent_name="opt-agent",
        baseline_loss=0.62,
        best_loss=0.50,
        cycles_run=1,
        accepted_count=1,
        best_agent=best_agent,
        trials=trials,
    )


class TestWriteOutputs:
    """write_outputs writes best.yaml, trials.jsonl, and report.md."""

    def test_writes_three_artifacts(self, tmp_path: Path) -> None:
        run_dir = write_outputs(_result(), tmp_path)

        assert run_dir == tmp_path / "run-xyz"
        assert (run_dir / "best.yaml").exists()
        assert (run_dir / "trials.jsonl").exists()
        assert (run_dir / "report.md").exists()

    def test_best_yaml_is_a_valid_agent(self, tmp_path: Path) -> None:
        run_dir = write_outputs(_result(), tmp_path)

        data = yaml.safe_load((run_dir / "best.yaml").read_text())
        reloaded = Agent.model_validate(data)
        assert reloaded.name == "opt-agent"
        assert reloaded.model.temperature == 0.7
        assert reloaded.instructions.inline == "You are a concise expert."

    def test_trials_jsonl_one_row_per_trial(self, tmp_path: Path) -> None:
        run_dir = write_outputs(_result(), tmp_path)

        lines = (run_dir / "trials.jsonl").read_text().splitlines()
        assert len(lines) == 2
        rows = [json.loads(line) for line in lines]
        assert [r["trial_id"] for r in rows] == [1, 2]
        assert rows[0]["accepted"] is True
        assert rows[1]["phase"] == "textual"

    def test_report_mentions_losses(self, tmp_path: Path) -> None:
        run_dir = write_outputs(_result(), tmp_path)

        report = (run_dir / "report.md").read_text()
        assert "opt-agent" in report
        assert "0.62" in report  # baseline loss
        assert "0.50" in report  # best loss

    def test_does_not_touch_source_agent(self, tmp_path: Path) -> None:
        source = tmp_path / "agent.yaml"
        original_bytes = b"name: original\nmodel: {}\n"
        source.write_bytes(original_bytes)

        write_outputs(_result(), tmp_path / "results")

        assert source.read_bytes() == original_bytes
