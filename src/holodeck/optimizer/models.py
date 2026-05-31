"""Result models for the optimizer: per-trial records and the run summary."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from holodeck.models.agent import Agent


class TrialRecord(BaseModel):
    """One optimizer trial, serialized one-per-line into ``trials.jsonl``.

    Attributes:
        trial_id: Monotonic trial index across the whole run.
        cycle: 0-based coordinate-descent cycle the trial belongs to.
        phase: Which proposer produced the candidate.
        score: Scalarized objective for the candidate.
        baseline_score: Best score the candidate had to beat.
        accepted: Whether the candidate advanced the baseline.
        params: Numeric params applied (numeric phase only).
        textual_axis: Instruction axis rewritten (textual phase only).
        edit_summary: Human-readable summary of the textual edit.
        excluded_metrics: Metrics excluded from the score due to errors.
        error: Reason the trial was skipped, if any.
    """

    model_config = ConfigDict(extra="forbid")

    trial_id: int = Field(..., description="Monotonic trial index across the run.")
    cycle: int = Field(..., ge=0, description="0-based coordinate-descent cycle.")
    phase: Literal["numeric", "textual"] = Field(
        ..., description="Proposer that produced the candidate."
    )
    score: float = Field(..., description="Scalarized objective for the candidate.")
    baseline_score: float = Field(..., description="Score the candidate had to beat.")
    accepted: bool = Field(..., description="Whether the candidate advanced baseline.")
    params: dict[str, Any] | None = Field(
        default=None, description="Numeric params applied (numeric phase only)."
    )
    textual_axis: str | None = Field(
        default=None, description="Instruction axis rewritten (textual phase only)."
    )
    edit_summary: str | None = Field(
        default=None, description="Human-readable summary of the textual edit."
    )
    excluded_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics excluded from the score due to errors.",
    )
    error: str | None = Field(
        default=None, description="Reason the trial was skipped, if any."
    )


class OptimizationResult(BaseModel):
    """Summary of a completed optimization run.

    Attributes:
        run_id: Identifier for the run's output directory.
        agent_name: Name of the optimized agent.
        baseline_score: Scalarized objective of the original agent.
        best_score: Scalarized objective of the best accepted candidate.
        cycles_run: Number of coordinate-descent cycles executed.
        accepted_count: Number of accepted improvements.
        best_agent: The best candidate agent (never the mutated original).
        trials: Per-trial audit trail.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Identifier for the run's output dir.")
    agent_name: str = Field(..., description="Name of the optimized agent.")
    baseline_score: float = Field(..., description="Objective of the original agent.")
    best_score: float = Field(..., description="Objective of the best candidate.")
    cycles_run: int = Field(..., ge=0, description="Cycles executed.")
    accepted_count: int = Field(
        ..., ge=0, description="Number of accepted improvements."
    )
    best_agent: Agent = Field(..., description="The best candidate agent.")
    trials: list[TrialRecord] = Field(
        default_factory=list, description="Per-trial audit trail."
    )
