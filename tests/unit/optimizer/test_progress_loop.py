"""Loop-level progress emission tests (038 T2).

Drives ``OptimizerLoop`` with a ``JsonlEmitter`` over a ``StringIO`` and asserts the
emitted NDJSON: the event ordering grammar (FR-003), trial/best_loss semantics, and
field-for-field parity with ``trials.jsonl`` (FR-004).
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path

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
from holodeck.optimizer.output import write_outputs
from holodeck.optimizer.progress import JsonlEmitter
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


class _StubNumericProposer:
    """Yields preset numeric proposals (a None ``params`` + ``error`` marks a skip)."""

    phase = "numeric"

    def __init__(self, proposals: list[Proposal]) -> None:
        self._proposals = proposals
        self._index = 0

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        self._index = 0

    async def ask(self) -> Proposal | None:
        if self._index >= len(self._proposals):
            return None
        proposal = self._proposals[self._index]
        self._index += 1
        return proposal

    def tell(
        self, proposal: Proposal, score: float, accepted: bool, report=None
    ) -> None:
        pass


def _config(*, min_delta: float = 0.01, max_cycles: int = 1) -> OptimizerConfig:
    return OptimizerConfig(
        loss={"groundedness": 1.0},
        axes=AxesConfig(
            numeric=[
                NumericAxis(path="model.temperature", type="float", range=[0.0, 1.0])
            ]
        ),
        min_delta=min_delta,
        max_cycles=max_cycles,
        numeric_phase=PhaseConfig(max_trials=10, patience=10),
        textual_phase=PhaseConfig(max_trials=5, patience=3),
    )


def _temp_scorer(loss_map: dict[float, float]):
    async def scorer(agent: Agent) -> tuple[float, TestReport]:
        return loss_map[agent.model.temperature], _dummy_report()

    return scorer


async def _run_capturing(loop: OptimizerLoop, buf: io.StringIO):
    result = await loop.run()
    events = [json.loads(line) for line in buf.getvalue().splitlines()]
    return result, events


# An accept (0.6 → 0.5) then a reject (0.55) over a single cycle.
def _accept_then_reject_loop(buf: io.StringIO) -> OptimizerLoop:
    proposer = _StubNumericProposer(
        [
            Proposal(params={"model.temperature": 0.5}),
            Proposal(params={"model.temperature": 0.7}),
        ]
    )
    return OptimizerLoop(
        original_agent=_agent(0.3),
        scorer=_temp_scorer({0.3: 0.6, 0.5: 0.5, 0.7: 0.55}),  # type: ignore[arg-type]
        config=_config(),
        numeric_proposer=proposer,
        run_id="run-acc",
        emitter=JsonlEmitter(buf),
        started_at=datetime(2026, 6, 18, 11, 50, 36, tzinfo=timezone.utc),
    )


class TestEventOrdering:
    """The emitted sequence follows the FR-003 grammar."""

    @pytest.mark.asyncio
    async def test_event_type_sequence(self) -> None:
        buf = io.StringIO()
        _, events = await _run_capturing(_accept_then_reject_loop(buf), buf)
        # run_completed is CLI-emitted, so the loop ends at cycle_completed.
        assert [e["event"] for e in events] == [
            "run_started",
            "baseline",
            "cycle_started",
            "phase_started",
            "trial",
            "trial",
            "phase_completed",
            "cycle_completed",
        ]

    @pytest.mark.asyncio
    async def test_run_started_and_baseline_fields(self) -> None:
        buf = io.StringIO()
        await _run_capturing(_accept_then_reject_loop(buf), buf)
        events = [json.loads(x) for x in buf.getvalue().splitlines()]
        run_started = events[0]
        assert run_started["run_id"] == "run-acc"
        assert run_started["agent"] == "opt-agent"
        assert run_started["max_cycles"] == 1
        assert run_started["started_at"] == "2026-06-18T11:50:36Z"
        assert run_started["axes"]["numeric"][0]["path"] == "model.temperature"
        assert run_started["loss_weights"] == {"groundedness": 1.0}
        assert events[1] == {
            "schema": "holodeck.optimize.progress/v1",
            "event": "baseline",
            "loss": 0.6,
        }

    @pytest.mark.asyncio
    async def test_cycle_started_carries_of(self) -> None:
        buf = io.StringIO()
        await _run_capturing(_accept_then_reject_loop(buf), buf)
        cycle_started = next(
            json.loads(x)
            for x in buf.getvalue().splitlines()
            if json.loads(x)["event"] == "cycle_started"
        )
        assert cycle_started == {
            "schema": "holodeck.optimize.progress/v1",
            "event": "cycle_started",
            "cycle": 0,
            "of": 1,
        }


class TestTrialBestLoss:
    """best_loss is the running best AFTER the accept/reject decision."""

    @pytest.mark.asyncio
    async def test_accept_then_reject_best_loss(self) -> None:
        buf = io.StringIO()
        await _run_capturing(_accept_then_reject_loop(buf), buf)
        trials = [
            json.loads(x)
            for x in buf.getvalue().splitlines()
            if json.loads(x)["event"] == "trial"
        ]
        accepted, rejected = trials
        # Accepted trial: pre-trial bar was the baseline; best advances to this loss.
        assert accepted["accepted"] is True
        assert accepted["baseline_loss"] == 0.6
        assert accepted["loss"] == 0.5
        assert accepted["best_loss"] == 0.5
        # Rejected trial: best stays at the previous accept; its own loss is worse.
        assert rejected["accepted"] is False
        assert rejected["baseline_loss"] == 0.5
        assert rejected["loss"] == 0.55
        assert rejected["best_loss"] == 0.5

    @pytest.mark.asyncio
    async def test_phase_and_cycle_completed_counts(self) -> None:
        buf = io.StringIO()
        await _run_capturing(_accept_then_reject_loop(buf), buf)
        events = [json.loads(x) for x in buf.getvalue().splitlines()]
        phase_completed = next(e for e in events if e["event"] == "phase_completed")
        assert phase_completed["trials"] == 2
        assert phase_completed["accepted"] == 1
        cycle_completed = next(e for e in events if e["event"] == "cycle_completed")
        assert cycle_completed["accepted"] == 1
        assert cycle_completed["best_loss"] == 0.5
        assert cycle_completed["stop_reason"] is None


class TestNoAcceptStop:
    """A zero-accept cycle emits cycle_completed with stop_reason=no_accepts."""

    @pytest.mark.asyncio
    async def test_no_accepts_stop_reason(self) -> None:
        buf = io.StringIO()
        proposer = _StubNumericProposer([Proposal(params={"model.temperature": 0.7})])
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=_temp_scorer({0.3: 0.5, 0.7: 0.75}),  # type: ignore[arg-type]
            config=_config(max_cycles=3),
            numeric_proposer=proposer,
            run_id="run-noacc",
            emitter=JsonlEmitter(buf),
        )
        await loop.run()
        cycle_completed = [
            json.loads(x)
            for x in buf.getvalue().splitlines()
            if json.loads(x)["event"] == "cycle_completed"
        ]
        # Stops after the first cycle: no accepts.
        assert len(cycle_completed) == 1
        assert cycle_completed[0]["stop_reason"] == "no_accepts"
        assert cycle_completed[0]["accepted"] == 0


class TestRecoverableError:
    """A proposer error surfaces as a trial event with error != null (FR-008)."""

    @pytest.mark.asyncio
    async def test_skipped_trial_event(self) -> None:
        buf = io.StringIO()
        proposer = _StubNumericProposer([Proposal(error="optuna exhausted")])
        loop = OptimizerLoop(
            original_agent=_agent(0.3),
            scorer=_temp_scorer({0.3: 0.5}),  # type: ignore[arg-type]
            config=_config(),
            numeric_proposer=proposer,
            run_id="run-err",
            emitter=JsonlEmitter(buf),
        )
        await loop.run()
        trial = next(
            json.loads(x)
            for x in buf.getvalue().splitlines()
            if json.loads(x)["event"] == "trial"
        )
        assert trial["error"] == "optuna exhausted"
        assert trial["accepted"] is False


class TestSourceOfTruth:
    """Each trial event equals its trials.jsonl row plus best_loss (FR-004)."""

    @pytest.mark.asyncio
    async def test_trial_events_match_trials_jsonl(self, tmp_path: Path) -> None:
        buf = io.StringIO()
        result, events = await _run_capturing(_accept_then_reject_loop(buf), buf)
        run_dir = write_outputs(result, tmp_path)
        rows = [
            json.loads(line)
            for line in (run_dir / "trials.jsonl").read_text().splitlines()
        ]
        trial_events = [e for e in events if e["event"] == "trial"]
        assert len(trial_events) == len(rows)
        for event, row in zip(trial_events, rows, strict=True):
            stripped = {
                k: v
                for k, v in event.items()
                if k not in ("schema", "event", "best_loss")
            }
            assert stripped == row
