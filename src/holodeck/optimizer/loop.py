"""Compounding coordinate-descent driver for the optimizer.

Scores the original agent for a baseline, then sweeps numeric → textual phases
in cycles. A candidate is accepted iff its score beats the current best by more
than ``min_delta``; on accept the best agent advances so subsequent proposals
compound onto it. A phase stops on its ``patience``/``max_trials`` budget; the
whole run stops when a full cycle yields zero accepts or ``max_cycles`` is hit.
The original agent is never mutated.
"""

import logging
from collections.abc import Awaitable, Callable

from holodeck.lib.errors import OptimizerError
from holodeck.models.agent import Agent
from holodeck.models.test_result import TestReport
from holodeck.optimizer.config import OptimizerConfig, PhaseConfig
from holodeck.optimizer.models import OptimizationResult, TrialRecord
from holodeck.optimizer.mutator import apply_axes, apply_textual_edit
from holodeck.optimizer.proposers.base import Proposal, Proposer

logger = logging.getLogger(__name__)

# A scorer maps a candidate agent to (scalarized_score, report).
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
    ) -> None:
        """Initialize the loop.

        Args:
            original_agent: The agent to optimize (never mutated).
            scorer: Async callable returning ``(score, report)`` for a candidate.
            config: Optimizer configuration (budgets, min_delta, axes).
            numeric_proposer: Proposer for the numeric phase (skipped if None).
            textual_proposer: Proposer for the textual phase (skipped if None).
            run_id: Identifier recorded on the result.
            progress_callback: Optional callback invoked with each TrialRecord as
                it is produced (used to stream per-trial scores).
        """
        self.original_agent = original_agent
        self.scorer = scorer
        self.config = config
        self.numeric_proposer = numeric_proposer
        self.textual_proposer = textual_proposer
        self.run_id = run_id
        self.progress_callback = progress_callback

        self._trials: list[TrialRecord] = []
        self._trial_id = 0
        self._accepted_count = 0
        self.best_agent = original_agent
        self.best_score = 0.0
        self.best_report: TestReport | None = None
        self.baseline_score = 0.0

    async def run(self) -> OptimizationResult:
        """Execute the optimization and return the result."""
        self.best_score, self.best_report = await self.scorer(self.original_agent)
        self.baseline_score = self.best_score
        logger.info("Baseline score: %.4f", self.baseline_score)

        cycles_run = 0
        for cycle in range(self.config.max_cycles):
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
            if accepts == 0:
                logger.info("Cycle %d produced no accepts — stopping.", cycle)
                break

        return OptimizationResult(
            run_id=self.run_id,
            agent_name=self.original_agent.name,
            baseline_score=self.baseline_score,
            best_score=self.best_score,
            cycles_run=cycles_run,
            accepted_count=self._accepted_count,
            best_agent=self.best_agent,
            trials=self._trials,
        )

    async def _run_phase(
        self, cycle: int, proposer: Proposer, phase_cfg: PhaseConfig
    ) -> int:
        """Run a single phase, returning the number of accepted improvements."""
        proposer.begin(self.best_agent, self.best_report)
        accepts = 0
        no_improve = 0

        for _ in range(phase_cfg.max_trials):
            if no_improve >= phase_cfg.patience:
                break
            proposal = await proposer.ask()
            if proposal is None:
                break

            self._trial_id += 1

            # The proposer could not produce a usable change — record a skipped
            # trial that counts toward patience, and move on.
            if proposal.error is not None:
                self._record(
                    TrialRecord(
                        trial_id=self._trial_id,
                        cycle=cycle,
                        phase=proposer.phase,
                        score=self.best_score,
                        baseline_score=self.best_score,
                        accepted=False,
                        textual_axis=proposal.textual_axis,
                        error=proposal.error,
                    )
                )
                proposer.tell(proposal, self.best_score, False)
                no_improve += 1
                logger.info("Trial %d skipped: %s", self._trial_id, proposal.error)
                continue

            candidate = self._apply(proposal)
            score, report = await self.scorer(candidate)
            accepted = score - self.best_score > self.config.min_delta

            self._record(
                TrialRecord(
                    trial_id=self._trial_id,
                    cycle=cycle,
                    phase=proposer.phase,
                    score=score,
                    baseline_score=self.best_score,
                    accepted=accepted,
                    params=proposal.params,
                    textual_axis=proposal.textual_axis,
                    edit_summary=proposal.edit_summary,
                )
            )
            proposer.tell(proposal, score, accepted)

            if accepted:
                self.best_agent = candidate
                self.best_score = score
                self.best_report = report
                self._accepted_count += 1
                accepts += 1
                no_improve = 0
                logger.info(
                    "Trial %d accepted (%s): score %.4f",
                    self._trial_id,
                    proposer.phase,
                    score,
                )
            else:
                no_improve += 1

        return accepts

    def _record(self, record: TrialRecord) -> None:
        """Append a trial record and notify the progress callback."""
        self._trials.append(record)
        if self.progress_callback is not None:
            self.progress_callback(record)

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
