"""Unit tests for the Optuna TPE numeric proposer (T5)."""

import pytest

from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_result import ReportSummary, TestReport, TestResult
from holodeck.optimizer.config import (
    AxesConfig,
    NumericAxis,
    OptimizerConfig,
    PhaseConfig,
)
from holodeck.optimizer.loop import OptimizerLoop
from holodeck.optimizer.proposers.numeric import NumericProposer


def _agent(temperature: float = 0.3) -> Agent:
    return Agent(
        name="opt-agent",
        model=LLMProvider(
            provider=ProviderEnum.OPENAI, name="gpt-4o-mini", temperature=temperature
        ),
        instructions=Instructions(inline="You are helpful."),
    )


def _dummy_report() -> TestReport:
    return TestReport(
        agent_name="opt-agent",
        agent_config_path="agent.yaml",
        results=[
            TestResult(
                test_name="t0",
                test_input="q",
                agent_response="a",
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


class TestSuggestionRanges:
    """Suggestions respect declared ranges and types."""

    @pytest.mark.asyncio
    async def test_float_axis_within_range(self) -> None:
        proposer = NumericProposer(
            axes=[
                NumericAxis(path="model.temperature", type="float", range=[0.0, 1.0])
            ],
            seed=42,
        )
        proposer.begin(_agent(), None)
        for _ in range(5):
            proposal = await proposer.ask()
            assert proposal is not None
            value = proposal.params["model.temperature"]
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
            proposer.tell(proposal, 0.5, False)

    @pytest.mark.asyncio
    async def test_int_axis_within_range(self) -> None:
        proposer = NumericProposer(
            axes=[NumericAxis(path="tools[name=kb].top_k", type="int", range=[1, 10])],
            seed=42,
        )
        proposer.begin(_agent(), None)
        proposal = await proposer.ask()
        assert proposal is not None
        value = proposal.params["tools[name=kb].top_k"]
        assert isinstance(value, int)
        assert 1 <= value <= 10

    @pytest.mark.asyncio
    async def test_categorical_axis_from_choices(self) -> None:
        proposer = NumericProposer(
            axes=[
                NumericAxis(
                    path="tools[name=kb].top_k", type="categorical", range=[3, 5, 8]
                )
            ],
            seed=42,
        )
        proposer.begin(_agent(), None)
        proposal = await proposer.ask()
        assert proposal is not None
        assert proposal.params["tools[name=kb].top_k"] in (3, 5, 8)

    @pytest.mark.asyncio
    async def test_no_axes_returns_none(self) -> None:
        proposer = NumericProposer(axes=[], seed=42)
        proposer.begin(_agent(), None)
        assert await proposer.ask() is None


class TestFreshStudyPerPhase:
    """Each begin() starts a new seeded study (deterministic restart)."""

    @pytest.mark.asyncio
    async def test_begin_resets_study(self) -> None:
        proposer = NumericProposer(
            axes=[
                NumericAxis(path="model.temperature", type="float", range=[0.0, 1.0])
            ],
            seed=7,
        )

        proposer.begin(_agent(), None)
        first = await proposer.ask()
        proposer.tell(first, 0.5, False)

        # A fresh phase with the same seed reproduces the first suggestion.
        proposer.begin(_agent(), None)
        again = await proposer.ask()

        assert first.params == again.params


class TestConvergence:
    """A numeric-only run improves toward a synthetic optimum."""

    @pytest.mark.asyncio
    async def test_improves_toward_optimum(self) -> None:
        # Loss is minimized (== 0) at temperature == 0.7.
        async def scorer(agent: Agent) -> tuple[float, TestReport]:
            temp = agent.model.temperature
            return (temp - 0.7) ** 2, _dummy_report()

        config = OptimizerConfig(
            loss={"groundedness": 1.0},
            axes=AxesConfig(
                numeric=[
                    NumericAxis(
                        path="model.temperature", type="float", range=[0.0, 1.0]
                    )
                ]
            ),
            min_delta=0.0,
            max_cycles=1,
            numeric_phase=PhaseConfig(max_trials=30, patience=30),
            textual_phase=PhaseConfig(max_trials=1, patience=1),
        )
        loop = OptimizerLoop(
            original_agent=_agent(0.0),
            scorer=scorer,
            config=config,
            numeric_proposer=NumericProposer(
                axes=config.axes.numeric, seed=config.seed
            ),
        )

        result = await loop.run()

        # Baseline at temp=0.0 has loss 0.49; optimum has loss 0.0.
        assert result.best_loss < result.baseline_loss
        assert abs(result.best_agent.model.temperature - 0.7) < 0.1
