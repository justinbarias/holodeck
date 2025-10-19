"""Evaluation models for agent configuration.

This module defines the EvaluationMetric and related models used in agent.yaml
configuration for specifying evaluation criteria and metrics.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from holodeck.models.llm import LLMProvider


class EvaluationMetric(BaseModel):
    """Evaluation metric configuration.

    Represents a single evaluation metric with flexible model configuration,
    including per-metric LLM model overrides.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(..., description="Metric name (e.g., groundedness)")
    threshold: float | None = Field(None, description="Minimum passing score")
    enabled: bool = Field(default=True, description="Whether metric is enabled")
    scale: int | None = Field(None, description="Score scale (e.g., 5 for 1-5 scale)")
    model: LLMProvider | None = Field(
        None, description="LLM model override for this metric"
    )
    fail_on_error: bool = Field(
        default=False, description="Fail test if metric evaluation fails"
    )
    retry_on_failure: int | None = Field(
        None, description="Number of retries on failure (1-3)"
    )
    timeout_ms: int | None = Field(
        None, description="Timeout in milliseconds for LLM calls"
    )
    custom_prompt: str | None = Field(None, description="Custom evaluation prompt")

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate metric is not empty."""
        if not v or not v.strip():
            raise ValueError("metric must be a non-empty string")
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float | None) -> float | None:
        """Validate threshold is numeric if provided."""
        if v is not None and not isinstance(v, int | float):
            raise ValueError("threshold must be numeric")
        return v

    @field_validator("enabled")
    @classmethod
    def validate_enabled(cls, v: bool) -> bool:
        """Validate enabled is boolean."""
        if not isinstance(v, bool):
            raise ValueError("enabled must be boolean")
        return v

    @field_validator("fail_on_error")
    @classmethod
    def validate_fail_on_error(cls, v: bool) -> bool:
        """Validate fail_on_error is boolean."""
        if not isinstance(v, bool):
            raise ValueError("fail_on_error must be boolean")
        return v

    @field_validator("retry_on_failure")
    @classmethod
    def validate_retry_on_failure(cls, v: int | None) -> int | None:
        """Validate retry_on_failure is in valid range."""
        if v is not None and (v < 1 or v > 3):
            raise ValueError("retry_on_failure must be between 1 and 3")
        return v

    @field_validator("timeout_ms")
    @classmethod
    def validate_timeout_ms(cls, v: int | None) -> int | None:
        """Validate timeout_ms is positive."""
        if v is not None and v <= 0:
            raise ValueError("timeout_ms must be positive")
        return v

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: int | None) -> int | None:
        """Validate scale is positive."""
        if v is not None and v <= 0:
            raise ValueError("scale must be positive")
        return v

    @field_validator("custom_prompt")
    @classmethod
    def validate_custom_prompt(cls, v: str | None) -> str | None:
        """Validate custom_prompt is not empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("custom_prompt must be non-empty if provided")
        return v


class EvaluationConfig(BaseModel):
    """Evaluation framework configuration.

    Container for evaluation metrics with optional default model configuration.
    """

    model_config = ConfigDict(extra="forbid")

    model: LLMProvider | None = Field(
        None, description="Default LLM model for all metrics"
    )
    metrics: list[EvaluationMetric] = Field(
        ..., description="List of metrics to evaluate"
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: list[EvaluationMetric]) -> list[EvaluationMetric]:
        """Validate metrics list is not empty."""
        if not v:
            raise ValueError("metrics must have at least one metric")
        return v
