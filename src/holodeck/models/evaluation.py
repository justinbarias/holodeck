"""Evaluation models for agent configuration.

This module defines the EvaluationMetric, GEvalMetric, RAGMetric and related
models used in agent.yaml configuration for specifying evaluation criteria.
"""

import importlib
import logging
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from holodeck.lib.errors import ConfigError
from holodeck.models.llm import LLMProvider

_logger = logging.getLogger(__name__)

# Literal set of built-in metric names accepted on ``EvaluationMetric.metric``.
# Narrowed from ``str`` in US4 to give config authors early, precise feedback
# when they typo a metric name (see data-model.md §5 and tasks-us4.md T015).
# ``equality`` and ``numeric`` are new in US4; the rest preserve pre-existing
# behaviour so existing fixtures / user YAMLs keep parsing.
BuiltInMetricName = Literal[
    "groundedness",
    "relevance",
    "coherence",
    "fluency",
    "bleu",
    "rouge",
    "meteor",
    "equality",
    "numeric",
]

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
    metric: BuiltInMetricName = Field(
        ...,
        description=(
            "Built-in metric name. One of: groundedness, relevance, coherence, "
            "fluency, bleu, rouge, meteor, equality, numeric."
        ),
    )
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

    # US4 — equality / numeric deterministic evaluator flags (data-model.md §5).
    # Each is meaningful only for its named metric; unused flags are tolerated
    # elsewhere so migrations don't have to touch unrelated YAML.
    case_insensitive: bool = Field(
        default=False,
        description="[equality] Lowercase both sides before compare.",
    )
    strip_whitespace: bool = Field(
        default=False,
        description="[equality] Collapse whitespace + trim before compare.",
    )
    strip_punctuation: bool = Field(
        default=False,
        description="[equality] Remove string.punctuation before compare.",
    )
    absolute_tolerance: float = Field(
        default=1e-6,
        ge=0.0,
        description=(
            "[numeric] Pass when abs(actual - expected) <= absolute_tolerance "
            "(inclusive; FR-018)."
        ),
    )
    relative_tolerance: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "[numeric] Pass when abs(actual - expected) <= "
            "relative_tolerance * abs(expected)."
        ),
    )
    accept_percent: bool = Field(
        default=False,
        description="[numeric] Parse trailing '%' as /100.",
    )
    accept_thousands_separators: bool = Field(
        default=False,
        description="[numeric] Strip ',' / '_' / NBSP before parse.",
    )

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

    @model_validator(mode="after")
    def _warn_model_on_deterministic(self) -> "EvaluationMetric":
        """Warn (don't fail) if ``model`` is set on a deterministic metric.

        ``equality`` / ``numeric`` run entirely in-process with no LLM; an
        ``model`` override is a no-op. We log a warning rather than raise so
        migrations carrying a shared default model config don't break.
        """
        if self.metric in ("equality", "numeric") and self.model is not None:
            _logger.warning(
                "metric %r is deterministic; the 'model' override is ignored.",
                self.metric,
            )
        return self


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


_GRADER_PATH_RE = r"^[\w.]+:[\w_]+$"


class CodeMetric(BaseModel):
    """User-supplied Python grader metric (data-model.md §6).

    Discriminator: ``type: code``. The ``grader`` path is resolved at config
    load time — ``importlib.import_module`` + ``getattr`` — so bad references
    surface as ``ConfigError`` *before* any agent call (FR-025).

    The resolved callable is cached in a private attribute so each turn's
    grader invocation doesn't re-import.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["code"] = Field(
        default="code",
        description="Discriminator field — always 'code' for user graders.",
    )
    grader: str = Field(
        ...,
        pattern=_GRADER_PATH_RE,
        description="Import path 'module.path:callable_name'.",
    )
    threshold: float | None = Field(
        None,
        description="Applied if grader returns a float without explicit passed flag.",
    )
    enabled: bool = Field(default=True, description="Whether metric is enabled")
    fail_on_error: bool = Field(
        default=False,
        description="If true, grader exceptions fail the whole test case.",
    )
    name: str | None = Field(
        None,
        description="Optional display name; defaults to the callable name.",
    )

    # Private attr — the resolved callable, cached after the first import.
    _resolved_callable: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve_grader(self) -> "CodeMetric":
        """Resolve ``grader`` at load time per contracts/code-grader-contract.md §2.

        Raises:
            ConfigError: if the module fails to import, the attribute is
                missing, or the resolved object is not callable.
        """
        module_path, _, callable_name = self.grader.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ConfigError(
                f"evaluations.metrics[{self.grader}]",
                (
                    f"cannot import grader module {module_path!r}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        try:
            fn = getattr(module, callable_name)
        except AttributeError as exc:
            raise ConfigError(
                f"evaluations.metrics[{self.grader}]",
                (
                    f"module {module_path!r} has no attribute "
                    f"{callable_name!r}: {exc}"
                ),
            ) from exc
        if not callable(fn):
            raise ConfigError(
                f"evaluations.metrics[{self.grader}]",
                (
                    f"resolved object {self.grader!r} is not callable "
                    f"(got {type(fn).__name__})"
                ),
            )
        self._resolved_callable = fn
        return self

    @property
    def resolved_callable(self) -> Any:
        """Return the resolved grader callable (cached at load time)."""
        return self._resolved_callable

    @property
    def display_name(self) -> str:
        """Return the ``name`` override, falling back to the callable name."""
        if self.name:
            return self.name
        return self.grader.partition(":")[2]


# Discriminated union type for metrics - uses 'type' field as discriminator
MetricType = Annotated[
    EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric,
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
        cls, v: list[EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric]
    ) -> list[EvaluationMetric | GEvalMetric | RAGMetric | CodeMetric]:
        """Validate metrics list is not empty."""
        if not v:
            raise ValueError("metrics must have at least one metric")
        return v
