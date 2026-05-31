"""Unit tests for OptimizerLoop coordinate-descent semantics (T4)."""

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
from holodeck.optimizer.proposers.base import Proposal


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


class StubNumericProposer:
    """Yields a preset list of numeric proposals, resetting each phase."""

    phase = "numeric"

    def __init__(self, param_dicts: list[dict]) -> None:
        self._param_dicts = param_dicts
        self._index = 0

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        self._index = 0

    async def ask(self) -> Proposal | None:
        if self._index >= len(self._param_dicts):
            return None
        params = self._param_dicts[self._index]
        self._index += 1
        return Proposal(params=params)

    def tell(self, proposal: Proposal, score: float, accepted: bool) -> None:
        pass


def _config(
    *,
    min_delta: float = 0.01,
    max_cycles: int = 3,
    numeric_patience: int = 3,
    numeric_max_trials: int = 10,
) -> OptimizerConfig:
    return OptimizerConfig(
        loss={"groundedness": 1.0},
        axes=AxesConfig(
            numeric=[
                NumericAxis(path="model.temperature", type="float", range=[0.0, 1.0])
            ]
        ),
        min_delta=min_delta,
        max_cycles=max_cycles,
        numeric_phase=PhaseConfig(
            max_trials=numeric_max_trials, patience=numeric_patience
        ),
        textual_phase=PhaseConfig(max_trials=5, patience=3),
    )


def _temp_scorer(score_map: dict[float, float]):
    """Score an agent by its model.temperature via a lookup table."""

    async def scorer(agent: Agent) -> tuple[float, TestReport]:
        return score_map[agent.model.temperature], _dummy_report()

    return scorer


class TestCompounding:
    """Accepted improvements advance the baseline (compounding)."""

    @pytest.mark.asyncio
    async def test_baseline_advances_on_each_accept(self) -> None:
        scorer = _temp_scorer({0.3: 0.40, 0.5: 0.50, 0.7: 0.60})
        proposer = StubNumericProposer(
            [{"model.temperature": 0.5}, {"model.temperature": 0.7}]
        )
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=scorer,
            config=_config(),
            numeric_proposer=proposer,
        )

        result = await loop.run()

        assert result.baseline_score == pytest.approx(0.40)
        assert result.best_score == pytest.approx(0.60)
        assert result.best_agent.model.temperature == 0.7
        assert result.accepted_count == 2
        # Cycle 2 is a dry cycle (no further improvement) → stops.
        assert result.cycles_run == 2

    @pytest.mark.asyncio
    async def test_sub_min_delta_rejected(self) -> None:
        scorer = _temp_scorer({0.3: 0.50, 0.5: 0.505})
        proposer = StubNumericProposer([{"model.temperature": 0.5}])
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=scorer,
            config=_config(min_delta=0.05),
            numeric_proposer=proposer,
        )

        result = await loop.run()

        assert result.accepted_count == 0
        assert result.best_score == pytest.approx(0.50)
        assert result.best_agent.model.temperature == 0.3
        assert result.trials[0].accepted is False


class TestStopping:
    """Phase patience, dry cycles, and max_cycles stop the loop."""

    @pytest.mark.asyncio
    async def test_phase_patience_stops_phase(self) -> None:
        # Every proposal scores below baseline → consecutive non-accepts.
        scorer = _temp_scorer({0.3: 0.40, 0.5: 0.10, 0.6: 0.10, 0.7: 0.10, 0.8: 0.10})
        proposer = StubNumericProposer(
            [
                {"model.temperature": 0.5},
                {"model.temperature": 0.6},
                {"model.temperature": 0.7},
                {"model.temperature": 0.8},
            ]
        )
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=scorer,
            config=_config(numeric_patience=2),
            numeric_proposer=proposer,
        )

        result = await loop.run()

        # Stops after 2 consecutive non-accepts; only 2 trials attempted.
        assert len(result.trials) == 2
        assert result.accepted_count == 0
        # First cycle was dry → loop stops at 1 cycle.
        assert result.cycles_run == 1

    @pytest.mark.asyncio
    async def test_dry_cycle_stops_before_max_cycles(self) -> None:
        scorer = _temp_scorer({0.3: 0.40, 0.5: 0.30})
        proposer = StubNumericProposer([{"model.temperature": 0.5}])
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=scorer,
            config=_config(max_cycles=5),
            numeric_proposer=proposer,
        )

        result = await loop.run()

        assert result.cycles_run == 1

    @pytest.mark.asyncio
    async def test_max_cycles_caps_run(self) -> None:
        # Monotonically increasing scorer → every trial accepts forever.
        calls = {"n": 0}

        async def scorer(agent: Agent) -> tuple[float, TestReport]:
            value = 0.40 + 0.10 * calls["n"]
            calls["n"] += 1
            return value, _dummy_report()

        proposer = StubNumericProposer([{"model.temperature": 0.5}])
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=scorer,
            config=_config(max_cycles=3),
            numeric_proposer=proposer,
        )

        result = await loop.run()

        assert result.cycles_run == 3
        assert result.accepted_count == 3


class TestDeterminism:
    """Identical seed + stub produce identical trial sequences."""

    @pytest.mark.asyncio
    async def test_deterministic_given_fixed_stub(self) -> None:
        def build() -> OptimizerLoop:
            return OptimizerLoop(
                original_agent=_agent(0.3),
                scorer=_temp_scorer({0.3: 0.40, 0.5: 0.50, 0.7: 0.60}),
                config=_config(),
                numeric_proposer=StubNumericProposer(
                    [{"model.temperature": 0.5}, {"model.temperature": 0.7}]
                ),
            )

        first = await build().run()
        second = await build().run()

        trace_first = [(t.score, t.accepted, t.params) for t in first.trials]
        trace_second = [(t.score, t.accepted, t.params) for t in second.trials]
        assert trace_first == trace_second
