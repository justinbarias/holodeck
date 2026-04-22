"""Data models for test execution results.

Defines models for test results, evaluation metrics, and test reports
generated during agent test execution.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from holodeck.models.test_case import FileInput
from holodeck.models.token_usage import TokenUsage


class ProcessedFileInput(BaseModel):
    """Processed file input with extracted content.

    Represents a file that has been processed (converted to markdown)
    and is ready for use in agent prompts.
    """

    model_config = ConfigDict(extra="forbid")

    original: FileInput = Field(..., description="Original file input configuration")
    markdown_content: str = Field(..., description="Markdown-converted file content")
    metadata: dict[str, Any] | None = Field(
        None, description="File metadata (pages, size, format, etc.)"
    )
    cached_path: str | None = Field(None, description="Path to cached processed file")
    processing_time_ms: int | None = Field(
        None, description="Time spent processing file in milliseconds"
    )
    error: str | None = Field(None, description="Error message if processing failed")


class MetricResult(BaseModel):
    """Result of evaluating a single metric.

    Represents the outcome of running an evaluation metric (e.g., groundedness,
    relevance) on a test case response.
    """

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(..., description="Name of the metric evaluated")
    kind: Literal["standard", "rag", "geval", "code"] = Field(
        ...,
        description=(
            "Metric family — 'standard' for NLP metrics, 'rag' for RAG pipeline "
            "metrics, 'geval' for LLM-as-judge G-Eval, 'code' for user-supplied "
            "Python graders. Required; drives dashboard breakdown panels, "
            "metric-trend toggles, and compare-matrix scoring."
        ),
    )
    score: float = Field(..., description="Numeric score from the metric")
    threshold: float | None = Field(None, description="Pass threshold for this metric")
    passed: bool | None = Field(None, description="Whether metric passed threshold")
    scale: str | None = Field(
        None, description="Scale of the score (e.g., '0-1', '0-100')"
    )
    error: str | None = Field(
        None, description="Error message if metric evaluation failed"
    )
    retry_count: int | None = Field(None, description="Number of retries attempted")
    evaluation_time_ms: int | None = Field(
        None, description="Time spent evaluating metric in milliseconds"
    )
    model_used: str | None = Field(
        None, description="LLM model used for AI-powered metrics"
    )
    reasoning: str | None = Field(
        None,
        description="LLM-generated explanation for the score (DeepEval metrics only)",
    )


class ToolInvocation(BaseModel):
    """One completed tool call+result pair captured during a test run.

    Persisted inside `TestResult.tool_invocations` so the dashboard Explorer
    can render `{name, args, result, bytes}` panels with collapse-by-default
    logic on large payloads.

    Named `ToolInvocation` (not `ToolEvent`) to avoid collision with the two
    existing `ToolEvent` classes (`holodeck.models.tool_event` — streaming
    UI events; `holodeck.lib.backends.base` — backend hook events).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Tool name invoked by the agent.")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool input parameters. Empty dict when the tool takes no args.",
    )
    result: Any = Field(
        default=None,
        description=(
            "Tool output — JSON-safe scalar / dict / list / string. "
            "`None` when the call raised before returning."
        ),
    )
    bytes: int = Field(
        ...,
        ge=0,
        description=(
            "`len(json.dumps(result, default=str))`. Used by the dashboard "
            "to collapse large results by default."
        ),
    )
    duration_ms: int | None = Field(
        default=None,
        description=(
            "Tool execution time in milliseconds. `None` when the backend "
            "does not report it (e.g. the SK Plugin path)."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Error message when the invocation failed; `result` is `None`.",
    )


class TurnResult(BaseModel):
    """Per-turn outcome inside a multi-turn test case (data-model.md §8).

    Stored on `TestResult.turns` only for multi-turn test cases. Legacy
    single-turn runs leave `TestResult.turns` as `None`.
    """

    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(..., ge=0, description="0-based position.")
    input: str = Field(..., description="Verbatim turn input.")
    response: str | None = Field(
        None, description="Agent response, or None if the turn erred."
    )
    ground_truth: str | None = Field(None, description="Copied from turn config.")
    expected_tools: list[Any] | None = Field(
        None,
        description=(
            "Serialized config form — bare strings today (US3 widens to objects)."
        ),
    )
    tool_calls: list[str] = Field(
        default_factory=list, description="Tool names (legacy shape)."
    )
    tool_invocations: list[ToolInvocation] = Field(
        default_factory=list, description="Structured tool records."
    )
    tools_matched: bool | None = Field(
        None,
        description="None if this turn asserted no tools.",
    )
    arg_match_details: list[dict[str, Any]] | None = Field(
        None,
        description="Per-assertion arg-matcher records (US3).",
    )
    metric_results: list[MetricResult] = Field(
        default_factory=list, description="Per-turn metric scores."
    )
    passed: bool = Field(
        ..., description="Composition rule in §3 of turn-result-schema."
    )
    execution_time_ms: int = Field(..., ge=0, description="Wall-clock of this turn.")
    token_usage: TokenUsage | None = Field(
        None, description="Per-turn usage (sums into test-case rollup)."
    )
    errors: list[str] = Field(default_factory=list)
    skipped: bool = Field(
        default=False,
        description="True if the session became unrecoverable before this turn.",
    )
    grader_details: dict[str, Any] | None = Field(
        None,
        description=(
            "Reserved for US4: map of metric_name -> grader-specific payload."
        ),
    )


class TestResult(BaseModel):
    """Result of executing a single test case.

    Contains the test input, agent response, tool calls, metric results,
    and overall pass/fail status along with any errors encountered.

    Readers prefer `tool_invocations` (structured `{name, args, result, bytes}`)
    when non-empty; `tool_calls: list[str]` (name-only) remains for back-compat
    with legacy runs predating Phase 2b.
    """

    model_config = ConfigDict(extra="forbid")

    test_name: str | None = Field(None, description="Name of the test case")
    test_input: str = Field(..., description="The test input/prompt")
    processed_files: list[ProcessedFileInput] = Field(
        default_factory=list,
        description="Files processed for this test",
    )
    agent_response: str | None = Field(None, description="Response from the agent")
    tool_calls: list[str] = Field(
        default_factory=list,
        description=(
            "Legacy field — list of tool names called by the agent. "
            "Kept for back-compat; prefer `tool_invocations` for new readers."
        ),
    )
    tool_invocations: list[ToolInvocation] = Field(
        default_factory=list,
        description=(
            "Structured per-call records `{name, args, result, bytes}` "
            "for the dashboard Explorer. Empty list for legacy runs or when "
            "no tools were invoked."
        ),
    )
    expected_tools: list[Any] | None = Field(
        None,
        description=(
            "Expected tools that should be called — bare strings (legacy) or "
            "serialized ExpectedTool objects (US3). Mixed lists allowed."
        ),
    )
    tools_matched: bool | None = Field(
        None,
        description="Whether actual tool calls matched expected tools",
    )
    metric_results: list[MetricResult] = Field(
        default_factory=list,
        description="Results from each evaluated metric",
    )
    ground_truth: str | None = Field(
        None,
        description="Expected output for comparison",
    )
    passed: bool = Field(..., description="Whether test passed all checks")
    execution_time_ms: int = Field(
        ...,
        description="Total time to execute this test in milliseconds",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of errors encountered during execution",
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp of test execution")
    token_usage: TokenUsage | None = Field(
        default=None,
        description=(
            "Token usage reported by the backend via ExecutionResult. "
            "None when the backend did not report any. Consumed by the "
            "dashboard Compare view for cost computation."
        ),
    )
    turns: list[TurnResult] | None = Field(
        default=None,
        description=(
            "Per-turn results for multi-turn test cases (feature 032). "
            "None for legacy single-turn cases."
        ),
    )


class ReportSummary(BaseModel):
    """Summary statistics for a test execution run.

    Aggregates statistics across all test results including pass rates,
    metric scores, and timing information.
    """

    model_config = ConfigDict(extra="forbid")

    total_tests: int = Field(..., description="Total number of tests executed")
    passed: int = Field(..., description="Number of tests that passed")
    failed: int = Field(..., description="Number of tests that failed")
    pass_rate: float = Field(..., description="Percentage of tests passed (0-100)")
    total_duration_ms: int = Field(
        ...,
        description="Total execution time in milliseconds",
    )
    metrics_evaluated: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each metric evaluated across all tests",
    )
    average_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Average score for each metric across all tests",
    )

    @model_validator(mode="after")
    def validate_test_counts(self) -> "ReportSummary":
        """Validate that passed + failed equals total_tests."""
        if self.passed + self.failed != self.total_tests:
            raise ValueError(
                f"passed ({self.passed}) + failed ({self.failed}) "
                f"must equal total_tests ({self.total_tests})"
            )
        return self


class TestReport(BaseModel):
    """Complete test execution report.

    Contains all test results, summary statistics, and metadata about
    the test run including agent name, version, and environment.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str = Field(..., description="Name of the agent being tested")
    agent_config_path: str = Field(
        ...,
        description="Path to the agent configuration file",
    )
    results: list[TestResult] = Field(
        ...,
        description="Individual test results",
    )
    summary: ReportSummary = Field(..., description="Aggregate summary statistics")
    timestamp: str = Field(..., description="ISO 8601 timestamp of report generation")
    holodeck_version: str = Field(..., description="Version of HoloDeck CLI used")
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment information (Python version, OS, etc.)",
    )

    @model_validator(mode="after")
    def validate_results_count(self) -> "TestReport":
        """Validate that summary total_tests matches results count."""
        if self.summary.total_tests != len(self.results):
            raise ValueError(
                f"summary.total_tests ({self.summary.total_tests}) "
                f"must match number of results ({len(self.results)})"
            )
        return self
