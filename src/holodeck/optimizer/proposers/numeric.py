"""Optuna TPE proposer for the numeric phase.

Wraps Optuna's ask-and-tell API: ``begin`` starts a fresh seeded study (a
textual edit changes the objective, so prior observations are stale), ``ask``
suggests a value per declared axis within its range, and ``tell`` feeds the
scored outcome back to the study so TPE can steer subsequent suggestions.
"""

from typing import Literal

import optuna

from holodeck.models.agent import Agent
from holodeck.models.test_result import TestReport
from holodeck.optimizer.config import NumericAxis
from holodeck.optimizer.proposers.base import Proposal

# Optuna is chatty at INFO; keep optimizer output to the loop's own logging.
optuna.logging.set_verbosity(optuna.logging.WARNING)


class NumericProposer:
    """Proposes numeric-axis values via a fresh-per-phase Optuna TPE study."""

    phase: Literal["numeric", "textual"] = "numeric"

    def __init__(self, axes: list[NumericAxis], seed: int) -> None:
        """Initialize the proposer.

        Args:
            axes: Declared numeric axes to tune.
            seed: Seed for the TPE sampler (config-reproducibility).
        """
        self._axes = axes
        self._seed = seed
        self._study: optuna.Study | None = None
        self._pending: optuna.trial.Trial | None = None

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        """Start a fresh seeded study for a new numeric phase."""
        sampler = optuna.samplers.TPESampler(seed=self._seed)
        self._study = optuna.create_study(direction="minimize", sampler=sampler)
        self._pending = None

    async def ask(self) -> Proposal | None:
        """Suggest the next value per axis, or None when there are no axes."""
        if not self._axes or self._study is None:
            return None
        trial = self._study.ask()
        params: dict[str, object] = {}
        for axis in self._axes:
            if axis.type == "float":
                params[axis.path] = trial.suggest_float(
                    axis.path, float(axis.range[0]), float(axis.range[1])
                )
            elif axis.type == "int":
                params[axis.path] = trial.suggest_int(
                    axis.path, int(axis.range[0]), int(axis.range[1])
                )
            else:
                params[axis.path] = trial.suggest_categorical(
                    axis.path, list(axis.range)
                )
        self._pending = trial
        return Proposal(params=params)

    def tell(
        self,
        proposal: Proposal,
        loss: float,
        accepted: bool,
        report: TestReport | None = None,
    ) -> None:
        """Report the candidate's loss back to the study (minimization).

        ``report`` is part of the shared proposer contract but unused here — a
        textual edit, not a per-case report, drives the next numeric suggestion.
        """
        if self._study is not None and self._pending is not None:
            self._study.tell(self._pending, loss)
            self._pending = None
