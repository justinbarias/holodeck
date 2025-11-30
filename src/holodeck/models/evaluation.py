"""Evaluation models for agent configuration.

This module defines the EvaluationMetric, GEvalMetric, RAGMetric and related
models used in agent.yaml configuration for specifying evaluation criteria.
"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from holodeck.models.llm import LLMProvider

# Valid evaluation parameter names for GEval metrics
VALID_EVALUATION_PARAMS = frozenset(
    ["input", "actual_output", "expected_output", "context", "retrieval_context"]
)


class RAGMetricType(str, Enum):
    """RAG pipeline evaluation metric types.

    These metrics evaluate the quality of Retrieval-Augmented Generation (RAG)
    pipelines by assessing various aspects of retrieval and response generation.
    """

    FAITHFULNESS = "faithfulness"
    CONTEXTUAL_RELEVANCY = "contextual_relevancy"
    CONTEXTUAL_PRECISION = "contextual_precision"
    CONTEXTUAL_RECALL = "contextual_recall"
    ANSWER_RELEVANCY = "answer_relevancy"


class EvaluationMetric(BaseModel):
    """Evaluation metric configuration.

    Represents a single evaluation metric with flexible model configuration,
    including per-metric LLM model overrides.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["standard"] = Field(
        default="standard",
        description="Discriminator field - 'standard' for built-in metrics",
    )
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


class GEvalMetric(BaseModel):
    """G-Eval custom criteria metric configuration.

    Uses discriminator pattern with type="geval" to distinguish from standard
    EvaluationMetric instances in a discriminated union.

    G-Eval enables custom evaluation criteria defined in natural language,
    using chain-of-thought prompting with LLM-based scoring.

    Example:
        >>> metric = GEvalMetric(
        ...     name="Professionalism",
        ...     criteria="Evaluate if the response uses professional language",
        ...     threshold=0.7
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["geval"] = Field(
        default="geval",
        description="Discriminator field - always 'geval' for GEval metrics",
    )
    name: str = Field(
        ...,
        description="Custom metric identifier (e.g., 'Professionalism', 'Helpfulness')",
    )
    criteria: str = Field(
        ...,
        description="Natural language evaluation criteria",
    )
    evaluation_steps: list[str] | None = Field(
        None,
        description="Explicit evaluation steps (auto-generated from criteria if None)",
    )
    evaluation_params: list[str] = Field(
        default=["actual_output"],
        description="Test case fields to include in evaluation",
    )
    strict_mode: bool = Field(
        default=False,
        description="Binary scoring mode (1.0 or 0.0 only)",
    )
    threshold: float | None = Field(
        None,
        description="Minimum passing score (0.0-1.0)",
    )
    model: LLMProvider | None = Field(
        None,
        description="LLM model override for this metric",
    )
    enabled: bool = Field(
        default=True,
        description="Whether metric is enabled",
    )
    fail_on_error: bool = Field(
        default=False,
        description="Fail test if metric evaluation fails",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty."""
        if not v or not v.strip():
            raise ValueError("name must be a non-empty string")
        return v

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, v: str) -> str:
        """Validate criteria is not empty."""
        if not v or not v.strip():
            raise ValueError("criteria must be a non-empty string")
        return v

    @field_validator("evaluation_params")
    @classmethod
    def validate_evaluation_params(cls, v: list[str]) -> list[str]:
        """Validate evaluation_params contains valid values."""
        if not v:
            raise ValueError("evaluation_params must not be empty")
        invalid_params = set(v) - VALID_EVALUATION_PARAMS
        if invalid_params:
            raise ValueError(
                f"Invalid evaluation_params: {sorted(invalid_params)}. "
                f"Valid options: {sorted(VALID_EVALUATION_PARAMS)}"
            )
        return v

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float | None) -> float | None:
        """Validate threshold is in valid range."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


class RAGMetric(BaseModel):
    """RAG pipeline evaluation metric configuration.

    Uses discriminator pattern with type="rag" to distinguish from standard
    EvaluationMetric and GEvalMetric instances in a discriminated union.

    RAG metrics evaluate the quality of retrieval-augmented generation pipelines:
    - Faithfulness: Detects hallucinations by comparing response to context
    - ContextualRelevancy: Measures relevance of retrieved chunks to query
    - ContextualPrecision: Evaluates ranking quality of retrieved chunks
    - ContextualRecall: Measures retrieval completeness against expected output

    Example:
        >>> metric = RAGMetric(
        ...     metric_type=RAGMetricType.FAITHFULNESS,
        ...     threshold=0.8
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["rag"] = Field(
        default="rag",
        description="Discriminator field - always 'rag' for RAG metrics",
    )
    metric_type: RAGMetricType = Field(
        ...,
        description="RAG metric type (faithfulness, contextual_relevancy, etc.)",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum passing score (0.0-1.0)",
    )
    include_reason: bool = Field(
        default=True,
        description="Include reasoning in evaluation results",
    )
    model: LLMProvider | None = Field(
        None,
        description="LLM model override for this metric",
    )
    enabled: bool = Field(
        default=True,
        description="Whether metric is enabled",
    )
    fail_on_error: bool = Field(
        default=False,
        description="Fail test if metric evaluation fails",
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold is in valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


# Discriminated union type for metrics - uses 'type' field as discriminator
MetricType = Annotated[
    EvaluationMetric | GEvalMetric | RAGMetric,
    Field(discriminator="type"),
]


class EvaluationConfig(BaseModel):
    """Evaluation framework configuration.

    Container for evaluation metrics with optional default model configuration.
    Supports standard EvaluationMetric, GEvalMetric (custom criteria), and
    RAGMetric (RAG pipeline evaluation).
    """

    model_config = ConfigDict(extra="forbid")

    model: LLMProvider | None = Field(
        None, description="Default LLM model for all metrics"
    )
    metrics: list[MetricType] = Field(
        ..., description="List of metrics to evaluate (standard, GEval, or RAG)"
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls, v: list[EvaluationMetric | GEvalMetric | RAGMetric]
    ) -> list[EvaluationMetric | GEvalMetric | RAGMetric]:
        """Validate metrics list is not empty."""
        if not v:
            raise ValueError("metrics must have at least one metric")
        return v
