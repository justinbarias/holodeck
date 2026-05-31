"""Unit tests for optimizer result models (T1)."""

from holodeck.lib.errors import HoloDeckError, OptimizerError
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.optimizer.models import OptimizationResult, TrialRecord


def _minimal_agent() -> Agent:
    return Agent(
        name="opt-agent",
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="You are helpful."),
    )


class TestOptimizerError:
    """OptimizerError is importable and part of the HoloDeck hierarchy."""

    def test_optimizer_error_is_holodeck_error(self) -> None:
        assert issubclass(OptimizerError, HoloDeckError)
        with_msg = OptimizerError("boom")
        assert str(with_msg) == "boom"


class TestTrialRecord:
    """TrialRecord round-trips through serialization."""

    def test_round_trip(self) -> None:
        rec = TrialRecord(
            trial_id=1,
            cycle=0,
            phase="numeric",
            score=0.55,
            baseline_score=0.40,
            accepted=True,
            params={"model.temperature": 0.7},
            excluded_metrics=["relevance"],
        )
        again = TrialRecord.model_validate(rec.model_dump())
        assert again == rec

    def test_textual_trial_fields(self) -> None:
        rec = TrialRecord(
            trial_id=2,
            cycle=1,
            phase="textual",
            score=0.6,
            baseline_score=0.55,
            accepted=False,
            textual_axis="instructions.inline",
            edit_summary="Tighten the answer format.",
        )
        again = TrialRecord.model_validate(rec.model_dump())
        assert again.phase == "textual"
        assert again.textual_axis == "instructions.inline"


class TestOptimizationResult:
    """OptimizationResult round-trips through serialization."""

    def test_round_trip(self) -> None:
        trial = TrialRecord(
            trial_id=1,
            cycle=0,
            phase="numeric",
            score=0.6,
            baseline_score=0.4,
            accepted=True,
        )
        result = OptimizationResult(
            run_id="run-abc",
            agent_name="opt-agent",
            baseline_score=0.4,
            best_score=0.6,
            cycles_run=1,
            accepted_count=1,
            best_agent=_minimal_agent(),
            trials=[trial],
        )
        again = OptimizationResult.model_validate(result.model_dump())
        assert again.best_score == 0.6
        assert again.baseline_score == 0.4
        assert len(again.trials) == 1
        assert again.best_agent.name == "opt-agent"
