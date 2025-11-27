"""Unit tests for test result models.

Tests ProcessedFileInput, MetricResult, TestResult, ReportSummary,
and TestReport models.
"""

import pytest
from pydantic import ValidationError

from holodeck.models.test_case import FileInput
from holodeck.models.test_result import (
    MetricResult,
    ProcessedFileInput,
    ReportSummary,
    TestReport,
    TestResult,
)


class TestProcessedFileInput:
    """Tests for ProcessedFileInput model."""

    def test_processed_file_input_minimal(self) -> None:
        """Test ProcessedFileInput with minimal required fields."""
        file_input = FileInput(path="test.pdf", type="pdf")
        input_obj = ProcessedFileInput(
            original=file_input,
            markdown_content="# Document content",
        )

        assert input_obj.original == file_input
        assert input_obj.markdown_content == "# Document content"
        assert input_obj.metadata is None
        assert input_obj.cached_path is None
        assert input_obj.processing_time_ms is None
        assert input_obj.error is None

    def test_processed_file_input_full(self) -> None:
        """Test ProcessedFileInput with all fields."""
        file_input = FileInput(path="report.pdf", type="pdf")
        metadata = {"pages": 10, "language": "en"}
        input_obj = ProcessedFileInput(
            original=file_input,
            markdown_content="# Report",
            metadata=metadata,
            cached_path="/cache/report_hash.md",
            processing_time_ms=1500,
            error=None,
        )

        assert input_obj.original == file_input
        assert input_obj.markdown_content == "# Report"
        assert input_obj.metadata == metadata
        assert input_obj.cached_path == "/cache/report_hash.md"
        assert input_obj.processing_time_ms == 1500
        assert input_obj.error is None

    def test_processed_file_input_with_error(self) -> None:
        """Test ProcessedFileInput with error message."""
        file_input = FileInput(path="image.png", type="image")
        input_obj = ProcessedFileInput(
            original=file_input,
            markdown_content="",
            error="File processing timeout after 30s",
        )

        assert input_obj.original == file_input
        assert input_obj.error == "File processing timeout after 30s"

    def test_processed_file_input_metadata_dict(self) -> None:
        """Test ProcessedFileInput accepts arbitrary metadata dict."""
        file_input = FileInput(path="data.pdf", type="pdf")
        metadata = {
            "file_size": 5242880,
            "format": "PDF",
            "pages": 15,
            "extracted_tables": 2,
        }
        input_obj = ProcessedFileInput(
            original=file_input,
            markdown_content="Content",
            metadata=metadata,
        )

        assert input_obj.metadata == metadata

    def test_processed_file_input_processing_time(self) -> None:
        """Test ProcessedFileInput processing_time_ms field."""
        file_input = FileInput(path="test.xlsx", type="excel")
        input_obj = ProcessedFileInput(
            original=file_input,
            markdown_content="Sheet data",
            processing_time_ms=2500,
        )

        assert input_obj.processing_time_ms == 2500

    def test_processed_file_input_forbids_extra_fields(self) -> None:
        """Test that ProcessedFileInput forbids extra fields."""
        file_input = FileInput(path="test.pdf", type="pdf")
        with pytest.raises(ValidationError):
            ProcessedFileInput(  # type: ignore
                original=file_input,
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


class TestTestResult:
    """Tests for TestResult model."""

    def test_test_result_minimal(self) -> None:
        """Test TestResult with minimal required fields."""
        result = TestResult(
            test_input="What are your business hours?",
            passed=True,
            execution_time_ms=3500,
            timestamp="2025-11-01T14:30:00Z",
        )

        assert result.test_input == "What are your business hours?"
        assert result.test_name is None
        assert result.processed_files == []
        assert result.agent_response is None
        assert result.tool_calls == []
        assert result.expected_tools is None
        assert result.tools_matched is None
        assert result.metric_results == []
        assert result.ground_truth is None
        assert result.passed is True
        assert result.execution_time_ms == 3500
        assert result.errors == []
        assert result.timestamp == "2025-11-01T14:30:00Z"

    def test_test_result_full(self) -> None:
        """Test TestResult with all fields."""
        metric = MetricResult(metric_name="groundedness", score=0.85)
        result = TestResult(
            test_name="Business hours query",
            test_input="What are your business hours?",
            agent_response="We're open Monday-Friday 9AM-5PM EST",
            tool_calls=["get_hours"],
            expected_tools=["get_hours"],
            tools_matched=True,
            metric_results=[metric],
            ground_truth="Monday-Friday 9AM-5PM EST",
            passed=True,
            execution_time_ms=3500,
            errors=[],
            timestamp="2025-11-01T14:30:00Z",
        )

        assert result.test_name == "Business hours query"
        assert result.test_input == "What are your business hours?"
        assert result.agent_response == "We're open Monday-Friday 9AM-5PM EST"
        assert result.tool_calls == ["get_hours"]
        assert result.expected_tools == ["get_hours"]
        assert result.tools_matched is True
        assert len(result.metric_results) == 1
        assert result.ground_truth == "Monday-Friday 9AM-5PM EST"
        assert result.passed is True

    def test_test_result_with_errors(self) -> None:
        """Test TestResult with error list."""
        result = TestResult(
            test_input="What is 2+2?",
            passed=False,
            execution_time_ms=5000,
            errors=["LLM API timeout after 60s", "Metric evaluation failed"],
            timestamp="2025-11-01T14:30:00Z",
        )

        assert len(result.errors) == 2
        assert "LLM API timeout after 60s" in result.errors
        assert result.passed is False

    def test_test_result_with_processed_files(self) -> None:
        """Test TestResult with processed files."""
        file_input = FileInput(path="report.pdf", type="pdf")
        processed = ProcessedFileInput(
            original=file_input,
            markdown_content="# Report content",
            processing_time_ms=1500,
        )
        result = TestResult(
            test_input="Summarize the report",
            processed_files=[processed],
            passed=True,
            execution_time_ms=2000,
            timestamp="2025-11-01T14:30:00Z",
        )

        assert len(result.processed_files) == 1
        assert result.processed_files[0].original == file_input

    def test_test_result_tool_mismatch(self) -> None:
        """Test TestResult when tools don't match expected."""
        result = TestResult(
            test_input="Get order status",
            expected_tools=["get_order_status"],
            tool_calls=["search_orders"],
            tools_matched=False,
            passed=False,
            execution_time_ms=3000,
            timestamp="2025-11-01T14:30:00Z",
        )

        assert result.tools_matched is False
        assert result.expected_tools != result.tool_calls

    def test_test_result_forbids_extra_fields(self) -> None:
        """Test that TestResult forbids extra fields."""
        with pytest.raises(ValidationError):
            TestResult(  # type: ignore
                test_input="test",
                passed=True,
                execution_time_ms=100,
                timestamp="2025-11-01T14:30:00Z",
                invalid_field="value",
            )


class TestReportSummary:
    """Tests for ReportSummary model."""

    def test_report_summary_basic(self) -> None:
        """Test ReportSummary with basic fields."""
        summary = ReportSummary(
            total_tests=10,
            passed=9,
            failed=1,
            pass_rate=90.0,
            total_duration_ms=45000,
        )

        assert summary.total_tests == 10
        assert summary.passed == 9
        assert summary.failed == 1
        assert summary.pass_rate == 90.0
        assert summary.total_duration_ms == 45000
        assert summary.metrics_evaluated == {}
        assert summary.average_scores == {}

    def test_report_summary_with_metrics(self) -> None:
        """Test ReportSummary with metric statistics."""
        summary = ReportSummary(
            total_tests=10,
            passed=8,
            failed=2,
            pass_rate=80.0,
            total_duration_ms=50000,
            metrics_evaluated={"groundedness": 10, "relevance": 10},
            average_scores={"groundedness": 0.82, "relevance": 0.75},
        )

        assert summary.metrics_evaluated["groundedness"] == 10
        assert summary.average_scores["groundedness"] == 0.82

    def test_report_summary_all_passed(self) -> None:
        """Test ReportSummary when all tests pass."""
        summary = ReportSummary(
            total_tests=5,
            passed=5,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=20000,
        )

        assert summary.passed == summary.total_tests
        assert summary.failed == 0
        assert summary.pass_rate == 100.0

    def test_report_summary_all_failed(self) -> None:
        """Test ReportSummary when all tests fail."""
        summary = ReportSummary(
            total_tests=5,
            passed=0,
            failed=5,
            pass_rate=0.0,
            total_duration_ms=25000,
        )

        assert summary.passed == 0
        assert summary.failed == summary.total_tests
        assert summary.pass_rate == 0.0

    def test_report_summary_pass_rate_precision(self) -> None:
        """Test ReportSummary with precise pass rate."""
        summary = ReportSummary(
            total_tests=3,
            passed=2,
            failed=1,
            pass_rate=66.66666,
            total_duration_ms=10000,
        )

        assert abs(summary.pass_rate - 66.66666) < 0.0001

    def test_report_summary_forbids_extra_fields(self) -> None:
        """Test that ReportSummary forbids extra fields."""
        with pytest.raises(ValidationError):
            ReportSummary(  # type: ignore
                total_tests=10,
                passed=9,
                failed=1,
                pass_rate=90.0,
                total_duration_ms=45000,
                invalid_field="value",
            )


class TestTestReport:
    """Tests for TestReport model."""

    def test_test_report_minimal(self) -> None:
        """Test TestReport with minimal required fields."""
        test_result = TestResult(
            test_input="test input",
            passed=True,
            execution_time_ms=1000,
            timestamp="2025-11-01T14:30:00Z",
        )
        summary = ReportSummary(
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=1000,
        )
        report = TestReport(
            agent_name="Test Agent",
            agent_config_path="./agent.yaml",
            results=[test_result],
            summary=summary,
            timestamp="2025-11-01T14:35:00Z",
            holodeck_version="0.1.0",
        )

        assert report.agent_name == "Test Agent"
        assert report.agent_config_path == "./agent.yaml"
        assert len(report.results) == 1
        assert report.summary.total_tests == 1
        assert report.timestamp == "2025-11-01T14:35:00Z"
        assert report.holodeck_version == "0.1.0"
        assert report.environment == {}

    def test_test_report_multiple_results(self) -> None:
        """Test TestReport with multiple test results."""
        results = [
            TestResult(
                test_name=f"Test {i}",
                test_input=f"input {i}",
                passed=True,
                execution_time_ms=1000 + i,
                timestamp="2025-11-01T14:30:00Z",
            )
            for i in range(3)
        ]
        summary = ReportSummary(
            total_tests=3,
            passed=3,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=3000,
        )
        report = TestReport(
            agent_name="Test Agent",
            agent_config_path="./agent.yaml",
            results=results,
            summary=summary,
            timestamp="2025-11-01T14:35:00Z",
            holodeck_version="0.1.0",
        )

        assert len(report.results) == 3
        assert report.summary.total_tests == 3

    def test_test_report_with_environment(self) -> None:
        """Test TestReport with environment metadata."""
        test_result = TestResult(
            test_input="test",
            passed=True,
            execution_time_ms=1000,
            timestamp="2025-11-01T14:30:00Z",
        )
        summary = ReportSummary(
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=1000,
        )
        environment = {
            "python_version": "3.10.0",
            "os": "Darwin",
            "platform": "macOS",
        }
        report = TestReport(
            agent_name="Test Agent",
            agent_config_path="./agent.yaml",
            results=[test_result],
            summary=summary,
            timestamp="2025-11-01T14:35:00Z",
            holodeck_version="0.1.0",
            environment=environment,
        )

        assert report.environment == environment
        assert report.environment["python_version"] == "3.10.0"

    def test_test_report_json_serialization(self) -> None:
        """Test TestReport JSON serialization."""
        test_result = TestResult(
            test_name="Test 1",
            test_input="input 1",
            passed=True,
            execution_time_ms=1000,
            timestamp="2025-11-01T14:30:00Z",
        )
        summary = ReportSummary(
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=1000,
        )
        report = TestReport(
            agent_name="Test Agent",
            agent_config_path="./agent.yaml",
            results=[test_result],
            summary=summary,
            timestamp="2025-11-01T14:35:00Z",
            holodeck_version="0.1.0",
        )

        json_str = report.model_dump_json()
        assert "Test Agent" in json_str
        assert "Test 1" in json_str

    def test_test_report_forbids_extra_fields(self) -> None:
        """Test that TestReport forbids extra fields."""
        test_result = TestResult(
            test_input="test",
            passed=True,
            execution_time_ms=1000,
            timestamp="2025-11-01T14:30:00Z",
        )
        summary = ReportSummary(
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=100.0,
            total_duration_ms=1000,
        )
        with pytest.raises(ValidationError):
            TestReport(  # type: ignore
                agent_name="Test Agent",
                agent_config_path="./agent.yaml",
                results=[test_result],
                summary=summary,
                timestamp="2025-11-01T14:35:00Z",
                holodeck_version="0.1.0",
                invalid_field="value",
            )
