"""Compounding coordinate-descent driver for the optimizer.

Scores the original agent for a baseline loss, then sweeps numeric → textual
phases in cycles. A candidate is accepted iff its loss undercuts the current
best by more than ``min_delta``; on accept the best agent advances so subsequent
proposals compound onto it. A phase stops on its ``patience``/``max_trials``
budget; the whole run stops when a full cycle yields zero accepts or
``max_cycles`` is hit. The original agent is never mutated.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Literal

from holodeck.lib.errors import OptimizerError
from holodeck.models.agent import Agent
from holodeck.models.test_result import TestReport
from holodeck.optimizer.config import OptimizerConfig, PhaseConfig
from holodeck.optimizer.models import OptimizationResult, TrialRecord
from holodeck.optimizer.mutator import apply_axes, apply_textual_edit
from holodeck.optimizer.progress import (
    Baseline,
    CycleCompleted,
    CycleStarted,
    NullEmitter,
    NumericAxisInfo,
    PhaseCompleted,
    PhaseStarted,
    ProgressEmitter,
    RunAxes,
    RunStarted,
    TextualAxisInfo,
    Trial,
)
from holodeck.optimizer.proposers.base import Proposal, Proposer
from holodeck.optimizer.telemetry import OptimizerTelemetry

logger = logging.getLogger(__name__)

# A scorer maps a candidate agent to (scalarized_loss, report); lower is better.
ScorerFn = Callable[[Agent], Awaitable[tuple[float, TestReport]]]


class OptimizerLoop:
    """Runs the compounding coordinate-descent optimization."""

    def __init__(
        self,
        *,
        original_agent: Agent,
        scorer: ScorerFn,
        config: OptimizerConfig,
        numeric_proposer: Proposer | None = None,
        textual_proposer: Proposer | None = None,
        run_id: str = "run",
        progress_callback: Callable[[TrialRecord], None] | None = None,
        emitter: ProgressEmitter | None = None,
        started_at: datetime | None = None,
    ) -> None:
        """Initialize the loop.

        Args:
            original_agent: The agent to optimize (never mutated).
            scorer: Async callable returning ``(loss, report)`` for a candidate.
            config: Optimizer configuration (budgets, min_delta, axes).
            numeric_proposer: Proposer for the numeric phase (skipped if None).
            textual_proposer: Proposer for the textual phase (skipped if None).
            run_id: Identifier recorded on the result.
            progress_callback: Optional callback invoked with each TrialRecord as
                it is produced (used to stream per-trial losses).
            emitter: Structured progress sink for the NDJSON event stream. Defaults
                to a no-op so the loop behaves identically when progress is off.
            started_at: Run start timestamp stamped on ``run_started``; defaults to
                the moment :meth:`run` is entered.
        """
        self.original_agent = original_agent
        self.scorer = scorer
        self.config = config
        self.numeric_proposer = numeric_proposer
        self.textual_proposer = textual_proposer
        self.run_id = run_id
        self.progress_callback = progress_callback
        self._emitter: ProgressEmitter = emitter or NullEmitter()
        self._started_at = started_at

        self._trials: list[TrialRecord] = []
        self._trial_id = 0
        self._accepted_count = 0
        self.best_agent = original_agent
        self.best_loss = float("inf")
        self.best_report: TestReport | None = None
        self.baseline_loss = float("inf")
        # Guarded OTel instrumentation; a strict no-op when observability is off.
        self._telemetry = OptimizerTelemetry()

    async def run(self) -> OptimizationResult:
        """Execute the optimization and return the result."""
        self._emit_run_started()

        with self._telemetry.baseline_span():
            self.best_loss, self.best_report = await self.scorer(self.original_agent)
        self.baseline_loss = self.best_loss
        logger.info("Baseline loss: %.4f", self.baseline_loss)
        self._emitter.emit(Baseline(loss=self.baseline_loss))

        cycles_run = 0
        for cycle in range(self.config.max_cycles):
            self._emitter.emit(CycleStarted(cycle=cycle, of=self.config.max_cycles))
            with self._telemetry.cycle_span(cycle):
                accepts = 0
                if self.numeric_proposer is not None:
                    accepts += await self._run_phase(
                        cycle, self.numeric_proposer, self.config.numeric_phase
                    )
                if self.textual_proposer is not None:
                    accepts += await self._run_phase(
                        cycle, self.textual_proposer, self.config.textual_phase
                    )
                cycles_run += 1
            self._telemetry.record_cycle()
            stop_reason: Literal["no_accepts"] | None = (
                "no_accepts" if accepts == 0 else None
            )
            self._emitter.emit(
                CycleCompleted(
                    cycle=cycle,
                    accepted=accepts,
                    best_loss=self.best_loss,
                    stop_reason=stop_reason,
                )
            )
            if accepts == 0:
                logger.info("Cycle %d produced no accepts — stopping.", cycle)
                break

        self._telemetry.record_improvement(self.baseline_loss - self.best_loss)
        return OptimizationResult(
            run_id=self.run_id,
            agent_name=self.original_agent.name,
            baseline_loss=self.baseline_loss,
            best_loss=self.best_loss,
            cycles_run=cycles_run,
            accepted_count=self._accepted_count,
            best_agent=self.best_agent,
            trials=self._trials,
        )

    def _emit_run_started(self) -> None:
        """Open the progress stream with the run's identity and search space."""
        started = self._started_at or datetime.now(timezone.utc)
        self._emitter.emit(
            RunStarted(
                run_id=self.run_id,
                agent=self.original_agent.name,
                max_cycles=self.config.max_cycles,
                axes=RunAxes(
                    numeric=[
                        NumericAxisInfo(path=a.path, type=a.type, range=a.range)
                        for a in self.config.axes.numeric
                    ],
                    textual=[
                        TextualAxisInfo(path=a.path, max_chars=a.max_chars)
                        for a in self.config.axes.textual
                    ],
                ),
                loss_weights=dict(self.config.loss),
                started_at=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        )

    async def _run_phase(
        self, cycle: int, proposer: Proposer, phase_cfg: PhaseConfig
    ) -> int:
        """Run a single phase, returning the number of accepted improvements."""
        proposer.begin(self.best_agent, self.best_report)
        accepts = 0
        no_improve = 0
        trials_in_phase = 0

        self._emitter.emit(PhaseStarted(cycle=cycle, phase=proposer.phase))
        with self._telemetry.phase_span(proposer.phase, cycle) as phase_span:
            for _ in range(phase_cfg.max_trials):
                if no_improve >= phase_cfg.patience:
                    break
                # Textual proposers run Critic/Applier LLM calls in ask(); wrap
                # those so their GenAI spans nest. Numeric ask() is cheap.
                if proposer.phase == "textual":
                    with self._telemetry.propose_span(proposer.phase, cycle):
                        proposal = await proposer.ask()
                else:
                    proposal = await proposer.ask()
                if proposal is None:
                    break

                self._trial_id += 1
                trials_in_phase += 1

                # The proposer could not produce a usable change — record a
                # skipped trial that counts toward patience, and move on.
                if proposal.error is not None:
                    with self._telemetry.trial_span(
                        trial_id=self._trial_id,
                        cycle=cycle,
                        phase=proposer.phase,
                        baseline_loss=self.best_loss,
                        textual_axis=proposal.textual_axis,
                        error=proposal.error,
                    ):
                        pass
                    self._record(
                        TrialRecord(
                            trial_id=self._trial_id,
                            cycle=cycle,
                            phase=proposer.phase,
                            loss=self.best_loss,
                            baseline_loss=self.best_loss,
                            accepted=False,
                            textual_axis=proposal.textual_axis,
                            error=proposal.error,
                        )
                    )
                    proposer.tell(proposal, self.best_loss, False)
                    self._telemetry.record_skipped_trial(proposer.phase)
                    no_improve += 1
                    logger.info("Trial %d skipped: %s", self._trial_id, proposal.error)
                    continue

                candidate = self._apply(proposal)
                # The pre-trial best is the bar this candidate must beat. Capture it
                # so the persisted ``baseline_loss`` stays pre-trial even after an
                # accept advances ``self.best_loss`` below — the trial event then
                # reports the *post*-decision running best.
                baseline_for_trial = self.best_loss
                with self._telemetry.trial_span(
                    trial_id=self._trial_id,
                    cycle=cycle,
                    phase=proposer.phase,
                    baseline_loss=baseline_for_trial,
                    params=proposal.params,
                    textual_axis=proposal.textual_axis,
                    edit_summary=proposal.edit_summary,
                ) as trial_span:
                    start = time.perf_counter()
                    loss, report = await self.scorer(candidate)
                    duration = time.perf_counter() - start
                    accepted = baseline_for_trial - loss > self.config.min_delta
                    self._telemetry.record_trial_outcome(
                        trial_span, loss=loss, accepted=accepted
                    )
                self._telemetry.record_trial(
                    phase=proposer.phase,
                    loss=loss,
                    duration=duration,
                    accepted=accepted,
                )

                record = TrialRecord(
                    trial_id=self._trial_id,
                    cycle=cycle,
                    phase=proposer.phase,
                    loss=loss,
                    baseline_loss=baseline_for_trial,
                    accepted=accepted,
                    params=proposal.params,
                    textual_axis=proposal.textual_axis,
                    edit_summary=proposal.edit_summary,
                )
                proposer.tell(proposal, loss, accepted, report)

                if accepted:
                    self.best_agent = candidate
                    self.best_loss = loss
                    self.best_report = report
                    self._accepted_count += 1
                    accepts += 1
                    no_improve = 0
                    self._telemetry.record_best_loss(proposer.phase, self.best_loss)
                    logger.info(
                        "Trial %d accepted (%s): loss %.4f",
                        self._trial_id,
                        proposer.phase,
                        loss,
                    )
                else:
                    no_improve += 1

                # Persist + emit after the accept decision so the trial event's
                # best_loss reflects the running best *after* this trial.
                self._record(record)

            self._telemetry.record_phase_accepts(phase_span, accepts)

        self._emitter.emit(
            PhaseCompleted(
                cycle=cycle,
                phase=proposer.phase,
                trials=trials_in_phase,
                accepted=accepts,
            )
        )
        return accepts

    def _record(self, record: TrialRecord) -> None:
        """Append a trial record, notify the callback, and emit the trial event.

        Called after the accept/reject decision so the emitted event's ``best_loss``
        reflects the running best *after* this trial.
        """
        self._trials.append(record)
        if self.progress_callback is not None:
            self.progress_callback(record)
        self._emitter.emit(Trial(best_loss=self.best_loss, **record.model_dump()))

    def _apply(self, proposal: Proposal) -> Agent:
        """Apply a proposal to the current best agent, returning a new candidate."""
        if proposal.params is not None:
            return apply_axes(self.best_agent, proposal.params)
        if proposal.textual_axis is not None and proposal.new_text is not None:
            return apply_textual_edit(
                self.best_agent, proposal.textual_axis, proposal.new_text
            )
        raise OptimizerError(
            "Proposal has neither numeric params nor a complete textual edit."
        )
