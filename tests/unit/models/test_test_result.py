"""Unit tests for test result models.

Tests ProcessedFileInput, MetricResult, TestResult, ReportSummary,
and TestReport models.
"""

import pytest
from pydantic import ValidationError

from holodeck.models.test_result import (
    MetricResult,
    ProcessedFileInput,
)


class TestProcessedFileInput:
    """Tests for ProcessedFileInput model."""

    def test_processed_file_input_minimal(self) -> None:
        """Test ProcessedFileInput with minimal required fields."""
        input_obj = ProcessedFileInput(
            original="test.pdf",
            markdown_content="# Document content",
        )

        assert input_obj.original == "test.pdf"
        assert input_obj.markdown_content == "# Document content"
        assert input_obj.metadata is None
        assert input_obj.cached_path is None
        assert input_obj.processing_time_ms is None
        assert input_obj.error is None

    def test_processed_file_input_full(self) -> None:
        """Test ProcessedFileInput with all fields."""
        metadata = {"pages": 10, "language": "en"}
        input_obj = ProcessedFileInput(
            original="report.pdf",
            markdown_content="# Report",
            metadata=metadata,
            cached_path="/cache/report_hash.md",
            processing_time_ms=1500,
            error=None,
        )

        assert input_obj.original == "report.pdf"
        assert input_obj.markdown_content == "# Report"
        assert input_obj.metadata == metadata
        assert input_obj.cached_path == "/cache/report_hash.md"
        assert input_obj.processing_time_ms == 1500
        assert input_obj.error is None

    def test_processed_file_input_with_error(self) -> None:
        """Test ProcessedFileInput with error message."""
        input_obj = ProcessedFileInput(
            original="image.png",
            markdown_content="",
            error="File processing timeout after 30s",
        )

        assert input_obj.original == "image.png"
        assert input_obj.error == "File processing timeout after 30s"

    def test_processed_file_input_metadata_dict(self) -> None:
        """Test ProcessedFileInput accepts arbitrary metadata dict."""
        metadata = {
            "file_size": 5242880,
            "format": "PDF",
            "pages": 15,
            "extracted_tables": 2,
        }
        input_obj = ProcessedFileInput(
            original="data.pdf",
            markdown_content="Content",
            metadata=metadata,
        )

        assert input_obj.metadata == metadata

    def test_processed_file_input_processing_time(self) -> None:
        """Test ProcessedFileInput processing_time_ms field."""
        input_obj = ProcessedFileInput(
            original="test.xlsx",
            markdown_content="Sheet data",
            processing_time_ms=2500,
        )

        assert input_obj.processing_time_ms == 2500

    def test_processed_file_input_forbids_extra_fields(self) -> None:
        """Test that ProcessedFileInput forbids extra fields."""
        with pytest.raises(ValidationError):
            ProcessedFileInput(  # type: ignore
                original="test.pdf",
                markdown_content="Content",
                invalid_field="value",
            )


class TestMetricResult:
    """Tests for MetricResult model."""

    def test_metric_result_basic(self) -> None:
        """Test MetricResult with basic fields."""
        result = MetricResult(
            metric_name="groundedness",
            score=0.85,
        )

        assert result.metric_name == "groundedness"
        assert result.score == 0.85
        assert result.threshold is None
        assert result.passed is None
        assert result.scale is None
        assert result.error is None

    def test_metric_result_full(self) -> None:
        """Test MetricResult with all fields."""
        result = MetricResult(
            metric_name="groundedness",
            score=0.85,
            threshold=0.75,
            passed=True,
            scale="0-1",
            error=None,
            retry_count=0,
            evaluation_time_ms=2000,
            model_used="gpt-4o",
        )

        assert result.metric_name == "groundedness"
        assert result.score == 0.85
        assert result.threshold == 0.75
        assert result.passed is True
        assert result.scale == "0-1"
        assert result.error is None
        assert result.retry_count == 0
        assert result.evaluation_time_ms == 2000
        assert result.model_used == "gpt-4o"

    def test_metric_result_with_error(self) -> None:
        """Test MetricResult with error."""
        result = MetricResult(
            metric_name="relevance",
            score=0.0,
            error="API timeout after 3 retries",
            retry_count=3,
        )

        assert result.metric_name == "relevance"
        assert result.error == "API timeout after 3 retries"
        assert result.retry_count == 3

    def test_metric_result_score_numeric(self) -> None:
        """Test MetricResult score accepts numeric values."""
        result = MetricResult(metric_name="f1_score", score=0.92)
        assert result.score == 0.92

        result = MetricResult(metric_name="bleu", score=0.45)
        assert result.score == 0.45

    def test_metric_result_threshold_comparison(self) -> None:
        """Test MetricResult with threshold for pass/fail."""
        result = MetricResult(
            metric_name="groundedness",
            score=0.85,
            threshold=0.75,
            passed=True,
        )

        assert result.passed is True
        assert result.score >= result.threshold

        result_fail = MetricResult(
            metric_name="groundedness",
            score=0.65,
            threshold=0.75,
            passed=False,
        )

        assert result_fail.passed is False

    def test_metric_result_scale_field(self) -> None:
        """Test MetricResult scale field."""
        result = MetricResult(
            metric_name="test",
            score=0.8,
            scale="0-1",
        )
        assert result.scale == "0-1"

        result = MetricResult(
            metric_name="test",
            score=85,
            scale="0-100",
        )
        assert result.scale == "0-100"

    def test_metric_result_forbids_extra_fields(self) -> None:
        """Test that MetricResult forbids extra fields."""
        with pytest.raises(ValidationError):
            MetricResult(  # type: ignore
                metric_name="test",
                score=0.5,
                invalid_field="value",
            )
