"""Shared proposer contract for the optimizer phases.

A proposer drives one phase: it proposes candidate changes (``ask``), is told
the scored outcome (``tell``), and is re-initialized at the start of each phase
(``begin``). The loop — not the proposer — applies a ``Proposal`` to the current
best agent, so accepted changes compound onto one another.
"""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from holodeck.models.agent import Agent
from holodeck.models.test_result import TestReport


@dataclass
class Proposal:
    """A single proposed change to the current best agent.

    Exactly one of the numeric or textual fields is populated:

    - Numeric: ``params`` maps axis path → value (applied via ``apply_axes``).
    - Textual: ``textual_axis`` + ``new_text`` (applied via
      ``apply_textual_edit``); ``edit_summary`` is a human-readable note.

    When ``error`` is set the proposer could not produce a usable change (e.g.
    the subagent returned unparseable JSON). The loop records a skipped trial
    that counts toward the phase's patience rather than crashing or applying it.
    """

    params: dict[str, Any] | None = None
    textual_axis: str | None = None
    new_text: str | None = None
    edit_summary: str | None = None
    error: str | None = None


@runtime_checkable
class Proposer(Protocol):
    """Phase proposer protocol consumed by ``OptimizerLoop``."""

    phase: Literal["numeric", "textual"]

    def begin(self, best_agent: Agent, best_report: TestReport | None) -> None:
        """Re-initialize the proposer for a new phase against the current best."""
        ...

    async def ask(self) -> Proposal | None:
        """Return the next proposal, or ``None`` when the phase is exhausted."""
        ...

    def tell(self, proposal: Proposal, loss: float, accepted: bool) -> None:
        """Report the scored loss of a proposal back to the proposer."""
        ...
