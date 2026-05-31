"""Configuration models for ``holodeck test optimize``.

Parses the ``evaluations.optimizer`` block of an agent.yaml into a typed config
consumed by the optimizer loop and proposers. Imports only Pydantic and the
stdlib so it can be referenced from ``holodeck.models.evaluation`` without a
circular import.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NumericAxis(BaseModel):
    """A query-time numeric axis the optimizer may tune.

    Attributes:
        path: Axis path (e.g. ``model.temperature`` or ``tools[name=kb].top_k``).
        type: ``float``/``int`` (bounded range) or ``categorical`` (choices).
        range: ``[low, high]`` for float/int; the choices list for categorical.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Axis path into the agent config.")
    type: Literal["float", "int", "categorical"] = Field(
        ..., description="Suggestion strategy for this axis."
    )
    range: list[Any] = Field(
        ...,
        description="[low, high] for float/int; the choices list for categorical.",
    )

    @field_validator("path")
    @classmethod
    def _non_empty_path(cls, v: str) -> str:
        """Reject blank axis paths."""
        if not v or not v.strip():
            raise ValueError("path must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _validate_range(self) -> "NumericAxis":
        """Enforce range shape per axis type."""
        if self.type in ("float", "int"):
            if len(self.range) != 2:
                raise ValueError(
                    f"{self.type} axis '{self.path}' requires range [low, high]"
                )
            low, high = self.range
            if low >= high:
                raise ValueError(
                    f"axis '{self.path}' range low ({low}) must be < high ({high})"
                )
        elif len(self.range) < 1:
            raise ValueError(
                f"categorical axis '{self.path}' requires at least one choice"
            )
        return self


class TextualAxis(BaseModel):
    """A textual axis (an instruction field) the optimizer may rewrite.

    Attributes:
        path: Path to the instruction text (e.g. ``instructions.inline``).
        max_chars: Upper bound on the rewritten text length.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Path to the instruction text to rewrite.")
    max_chars: int = Field(
        default=8000, gt=0, description="Maximum length of the rewritten text."
    )

    @field_validator("path")
    @classmethod
    def _non_empty_path(cls, v: str) -> str:
        """Reject blank axis paths."""
        if not v or not v.strip():
            raise ValueError("path must be a non-empty string")
        return v


class AxesConfig(BaseModel):
    """Declared numeric and textual axes for the optimizer."""

    model_config = ConfigDict(extra="forbid")

    numeric: list[NumericAxis] = Field(
        default_factory=list, description="Numeric axes tuned by the numeric phase."
    )
    textual: list[TextualAxis] = Field(
        default_factory=list, description="Textual axes rewritten by the textual phase."
    )


class PhaseConfig(BaseModel):
    """Per-phase budget controls.

    Attributes:
        max_trials: Hard cap on trials within a single phase.
        patience: Stop the phase after this many consecutive non-accepts.
    """

    model_config = ConfigDict(extra="forbid")

    max_trials: int = Field(..., gt=0, description="Hard cap on trials per phase.")
    patience: int = Field(
        ..., gt=0, description="Consecutive non-accepts before the phase stops."
    )


def _default_numeric_phase() -> PhaseConfig:
    return PhaseConfig(max_trials=10, patience=5)


def _default_textual_phase() -> PhaseConfig:
    return PhaseConfig(max_trials=5, patience=3)


class OptimizerConfig(BaseModel):
    """Typed view of the ``evaluations.optimizer`` block.

    Attributes:
        loss: Per-metric weights for the scalarized objective (non-empty,
            strictly positive).
        axes: Declared numeric and textual axes.
        max_cycles: Maximum numeric→textual cycles before stopping.
        numeric_phase: Numeric-phase budget.
        textual_phase: Textual-phase budget.
        min_delta: Minimum raw score improvement required to accept a candidate.
        seed: Seed for the numeric proposer's study (config-reproducibility).
    """

    model_config = ConfigDict(extra="forbid")

    loss: dict[str, float] = Field(
        ..., description="Per-metric weights for the scalarized objective."
    )
    axes: AxesConfig = Field(
        default_factory=AxesConfig, description="Declared optimizer axes."
    )
    max_cycles: int = Field(
        default=3, gt=0, description="Maximum numeric→textual cycles."
    )
    numeric_phase: PhaseConfig = Field(default_factory=_default_numeric_phase)
    textual_phase: PhaseConfig = Field(default_factory=_default_textual_phase)
    min_delta: float = Field(
        default=0.01,
        ge=0.0,
        description="Minimum raw score improvement required to accept.",
    )
    seed: int = Field(default=42, description="Seed for the numeric study.")

    @field_validator("loss")
    @classmethod
    def _validate_loss(cls, v: dict[str, float]) -> dict[str, float]:
        """Require at least one strictly-positive metric weight."""
        if not v:
            raise ValueError("loss must contain at least one metric weight")
        if any(weight <= 0 for weight in v.values()):
            raise ValueError("loss weights must be strictly positive")
        return v
