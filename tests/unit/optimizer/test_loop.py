"""Unit tests for OptimizerLoop coordinate-descent semantics (T4)."""

import json

import pytest

from holodeck.lib.backends.base import ExecutionResult
from holodeck.models.agent import Agent, Instructions
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.test_result import ReportSummary, TestReport, TestResult
from holodeck.optimizer.config import (
    AxesConfig,
    NumericAxis,
    OptimizerConfig,
    PhaseConfig,
    TextualAxis,
)
from holodeck.optimizer.loop import OptimizerLoop
from holodeck.optimizer.proposers.base import Proposal
from holodeck.optimizer.proposers.textual import TextualProposer


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

    def tell(
        self,
        proposal: Proposal,
        score: float,
        accepted: bool,
        report: TestReport | None = None,
    ) -> None:
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


def _temp_scorer(loss_map: dict[float, float]):
    """Map an agent's model.temperature to a loss via a lookup table."""

    async def scorer(agent: Agent) -> tuple[float, TestReport]:
        return loss_map[agent.model.temperature], _dummy_report()

    return scorer


class TestCompounding:
    """Accepted improvements advance the baseline (compounding)."""

    @pytest.mark.asyncio
    async def test_baseline_advances_on_each_accept(self) -> None:
        # Lower loss is better: each accepted candidate undercuts the baseline.
        scorer = _temp_scorer({0.3: 0.60, 0.5: 0.50, 0.7: 0.40})
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

        assert result.baseline_loss == pytest.approx(0.60)
        assert result.best_loss == pytest.approx(0.40)
        assert result.best_agent.model.temperature == 0.7
        assert result.accepted_count == 2
        # Cycle 2 is a dry cycle (no further improvement) → stops.
        assert result.cycles_run == 2

    @pytest.mark.asyncio
    async def test_sub_min_delta_rejected(self) -> None:
        # Loss improves by only 0.04 (< min_delta 0.05) → rejected.
        scorer = _temp_scorer({0.3: 0.50, 0.5: 0.46})
        proposer = StubNumericProposer([{"model.temperature": 0.5}])
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=scorer,
            config=_config(min_delta=0.05),
            numeric_proposer=proposer,
        )

        result = await loop.run()

        assert result.accepted_count == 0
        assert result.best_loss == pytest.approx(0.50)
        assert result.best_agent.model.temperature == 0.3
        assert result.trials[0].accepted is False


class TestStopping:
    """Phase patience, dry cycles, and max_cycles stop the loop."""

    @pytest.mark.asyncio
    async def test_phase_patience_stops_phase(self) -> None:
        # Every proposal has a worse (higher) loss → consecutive non-accepts.
        scorer = _temp_scorer({0.3: 0.40, 0.5: 0.90, 0.6: 0.90, 0.7: 0.90, 0.8: 0.90})
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
        # Candidate loss is worse than baseline → rejected → dry cycle.
        scorer = _temp_scorer({0.3: 0.40, 0.5: 0.50})
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
        # Monotonically decreasing loss → every trial accepts forever.
        calls = {"n": 0}

        async def scorer(agent: Agent) -> tuple[float, TestReport]:
            value = 0.40 - 0.10 * calls["n"]
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


class StubSkippingProposer:
    """Always returns an errored proposal (e.g. unparseable subagent JSON)."""

    phase = "textual"

    def __init__(self, n: int) -> None:
        self._remaining = n

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        pass

    async def ask(self) -> Proposal | None:
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return Proposal(textual_axis="instructions.inline", error="bad JSON")

    def tell(
        self,
        proposal: Proposal,
        score: float,
        accepted: bool,
        report: TestReport | None = None,
    ) -> None:
        pass


class TestSkippedProposals:
    """Errored proposals are recorded as skipped and count toward patience."""

    @pytest.mark.asyncio
    async def test_errored_proposal_skipped_not_applied(self) -> None:
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=_temp_scorer({0.3: 0.40}),
            config=_config(max_cycles=1),
            textual_proposer=StubSkippingProposer(5),
        )

        result = await loop.run()

        # Textual phase patience defaults to 3 → stops after 3 skipped trials.
        assert len(result.trials) == 3
        assert all(t.error == "bad JSON" for t in result.trials)
        assert result.accepted_count == 0
        # Original agent never mutated.
        assert result.best_agent.instructions.inline == "You are helpful."


class RecordingNumericProposer(StubNumericProposer):
    """Captures every ``tell`` call, including the scored report."""

    def __init__(self, param_dicts: list[dict]) -> None:
        super().__init__(param_dicts)
        self.told: list[tuple[float, bool, TestReport | None]] = []

    def tell(
        self,
        proposal: Proposal,
        score: float,
        accepted: bool,
        report: TestReport | None = None,
    ) -> None:
        self.told.append((score, accepted, report))


class RecordingSkippingProposer(StubSkippingProposer):
    """Captures every ``tell`` call for errored (skipped) proposals."""

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.told: list[tuple[float, bool, TestReport | None]] = []

    def tell(
        self,
        proposal: Proposal,
        score: float,
        accepted: bool,
        report: TestReport | None = None,
    ) -> None:
        self.told.append((score, accepted, report))


class TestTellReceivesReport:
    """The loop threads the candidate's scored report into ``tell``."""

    @pytest.mark.asyncio
    async def test_scored_trial_passes_candidate_report(self) -> None:
        # Distinct report objects per scorer call: [0]=baseline, [1]=candidate.
        reports = [_dummy_report(), _dummy_report()]
        calls = {"n": 0}

        async def scorer(agent: Agent) -> tuple[float, TestReport]:
            report = reports[calls["n"]]
            calls["n"] += 1
            return 0.40, report

        proposer = RecordingNumericProposer([{"model.temperature": 0.5}])
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=scorer,
            config=_config(max_cycles=1),
            numeric_proposer=proposer,
        )

        await loop.run()

        assert proposer.told, "tell was never called"
        # The candidate's own report (2nd scorer call) is threaded in — not the
        # baseline report, and not None.
        assert proposer.told[-1][2] is reports[1]

    @pytest.mark.asyncio
    async def test_skipped_trial_passes_none_report(self) -> None:
        proposer = RecordingSkippingProposer(1)
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=_temp_scorer({0.3: 0.40}),
            config=_config(max_cycles=1),
            textual_proposer=proposer,
        )

        await loop.run()

        assert proposer.told, "tell was never called"
        # An errored proposal is never scored, so there is no report to pass.
        assert proposer.told[-1][2] is None


def _subagent(name: str) -> Agent:
    return Agent(
        name=name,
        model=LLMProvider(provider=ProviderEnum.OPENAI, name="gpt-4o-mini"),
        instructions=Instructions(inline="subagent prompt"),
    )


def _textual_config(
    *, max_trials: int = 5, patience: int = 3, min_delta: float = 0.01
) -> OptimizerConfig:
    """A textual-only optimizer config (single instruction axis)."""
    return OptimizerConfig(
        loss={"groundedness": 1.0},
        axes=AxesConfig(textual=[TextualAxis(path="instructions.inline")]),
        min_delta=min_delta,
        max_cycles=1,
        numeric_phase=PhaseConfig(max_trials=1, patience=1),
        textual_phase=PhaseConfig(max_trials=max_trials, patience=patience),
    )


class _CountingInvoker:
    """Applier emits a fresh ``rewrite-N`` each call; Critic is constant."""

    def __init__(self) -> None:
        self.n = 0

    async def __call__(self, agent: Agent, prompt: str) -> ExecutionResult:
        if "critic" in agent.name:
            return ExecutionResult(response='{"gradient": "improve"}')
        self.n += 1
        return ExecutionResult(
            response=json.dumps(
                {"new_text": f"rewrite-{self.n}", "summary": f"s{self.n}"}
            )
        )


class _FlatInvoker:
    """Applier always emits the same text — every attempt scores identically."""

    async def __call__(self, agent: Agent, prompt: str) -> ExecutionResult:
        if "critic" in agent.name:
            return ExecutionResult(response='{"gradient": "x"}')
        return ExecutionResult(response='{"new_text": "flat-rewrite", "summary": "s"}')


def _instruction_scorer():
    """Map ``rewrite-N`` instruction text to a monotonically lower loss."""

    async def scorer(agent: Agent) -> tuple[float, TestReport]:
        text = agent.instructions.inline or ""
        if text.startswith("rewrite-"):
            n = int(text.split("-")[1])
            loss = max(0.0, 0.65 - 0.10 * n)
        elif text == "flat-rewrite":
            loss = 0.70  # always worse than the 0.65 baseline → never accepted
        else:
            loss = 0.65  # baseline ("You are helpful.")
        return loss, _dummy_report()

    return scorer


class TestIterativeTextualPhase:
    """The real TextualProposer iterates a single axis through the loop."""

    def _proposer(self, invoker) -> TextualProposer:
        return TextualProposer(
            axes=[TextualAxis(path="instructions.inline")],
            critic_agent=_subagent("optimizer-critic"),
            applier_agent=_subagent("optimizer-applier"),
            invoker=invoker,
        )

    @pytest.mark.asyncio
    async def test_phase_runs_multiple_refinement_steps(self) -> None:
        # A monotonic gradient: each rewrite undercuts the last → all accepted,
        # the phase runs the full max_trials, and best_agent compounds.
        loop = OptimizerLoop(
            original_agent=_agent(),
            scorer=_instruction_scorer(),
            config=_textual_config(max_trials=5, patience=5),
            textual_proposer=self._proposer(_CountingInvoker()),
        )

        result = await loop.run()

        textual_trials = [t for t in result.trials if t.phase == "textual"]
        assert len(textual_trials) == 5  # max_trials bound the phase, not ask()
        assert all(t.accepted for t in textual_trials)
        assert result.best_agent.instructions.inline == "rewrite-5"
        assert result.best_loss == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_patience_halts_a_flat_chain(self) -> None:
        loop = OptimizerLoop(
            original_agent=_agent(),
            scorer=_instruction_scorer(),
            config=_textual_config(max_trials=10, patience=3),
            textual_proposer=self._proposer(_FlatInvoker()),
        )

        result = await loop.run()

        textual_trials = [t for t in result.trials if t.phase == "textual"]
        # Three consecutive non-accepts trip patience well before max_trials.
        assert len(textual_trials) == 3
        assert result.accepted_count == 0
        # best_agent never regressed — the original instruction is retained.
        assert result.best_agent.instructions.inline == "You are helpful."
        assert result.best_loss == pytest.approx(0.65)

    @pytest.mark.asyncio
    async def test_best_never_regresses_below_single_step(self) -> None:
        # The iterative result must be no worse than the first (single) step.
        loop = OptimizerLoop(
            original_agent=_agent(),
            scorer=_instruction_scorer(),
            config=_textual_config(max_trials=5, patience=5),
            textual_proposer=self._proposer(_CountingInvoker()),
        )

        result = await loop.run()

        first_textual = next(t for t in result.trials if t.phase == "textual")
        assert result.best_loss <= first_textual.loss


class TestProgressCallback:
    """The loop streams each trial to a progress callback."""

    @pytest.mark.asyncio
    async def test_callback_invoked_per_trial(self) -> None:
        seen: list[int] = []
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=_temp_scorer({0.3: 0.60, 0.5: 0.50, 0.7: 0.40}),
            config=_config(),
            numeric_proposer=StubNumericProposer(
                [{"model.temperature": 0.5}, {"model.temperature": 0.7}]
            ),
            progress_callback=lambda t: seen.append(t.trial_id),
        )

        result = await loop.run()

        assert seen == [t.trial_id for t in result.trials]
        assert len(seen) == len(result.trials)


class TestDeterminism:
    """Identical seed + stub produce identical trial sequences."""

    @pytest.mark.asyncio
    async def test_deterministic_given_fixed_stub(self) -> None:
        def build() -> OptimizerLoop:
            return OptimizerLoop(
                original_agent=_agent(0.3),
                scorer=_temp_scorer({0.3: 0.60, 0.5: 0.50, 0.7: 0.40}),
                config=_config(),
                numeric_proposer=StubNumericProposer(
                    [{"model.temperature": 0.5}, {"model.temperature": 0.7}]
                ),
            )

        first = await build().run()
        second = await build().run()

        trace_first = [(t.loss, t.accepted, t.params) for t in first.trials]
        trace_second = [(t.loss, t.accepted, t.params) for t in second.trials]
        assert trace_first == trace_second
