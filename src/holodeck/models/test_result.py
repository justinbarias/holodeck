"""Data models for test execution results.

Defines models for test results, evaluation metrics, and test reports
generated during agent test execution.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProcessedFileInput(BaseModel):
    """Processed file input with extracted content.

    Represents a file that has been processed (converted to markdown)
    and is ready for use in agent prompts.
    """

    model_config = ConfigDict(extra="forbid")

    original: str = Field(..., description="Original filename or path")
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
