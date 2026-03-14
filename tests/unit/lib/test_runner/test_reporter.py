"""Unit tests for report generation.

Tests both markdown and JSON report generation, ensuring comprehensive display
of all TestResult fields including:
- Test details (name, input, timestamp)
- Processed files with metadata
- Agent responses
- Tool usage validation
- Evaluation metrics with scores and thresholds
- Ground truth comparisons
- Error handling
- JSON structure and serialization
"""

import json

import pytest

from holodeck.lib.test_runner.reporter import (
    _format_metrics_table,
    _format_processed_files,
    _format_summary_table,
    _format_test_section,
    _format_tool_usage,
    generate_markdown_report,
)
from holodeck.models.test_case import FileInput
from holodeck.models.test_result import (
    MetricResult,
    ProcessedFileInput,
    ReportSummary,
    TestReport,
    TestResult,
)


@pytest.fixture
def sample_metric_result() -> MetricResult:
    """Create a sample MetricResult for testing."""
    return MetricResult(
        metric_name="groundedness",
        score=0.92,
        threshold=0.8,
        passed=True,
        scale="0-1",
        error=None,
        retry_count=0,
        evaluation_time_ms=1250,
        model_used="gpt-4o-mini",
        reasoning="The response is well-grounded in the source material.",
    )


@pytest.fixture
def sample_metric_result_failed() -> MetricResult:
    """Create a failed MetricResult with error."""
    return MetricResult(
        metric_name="completeness",
        score=0.35,
        threshold=0.7,
        passed=False,
        scale="0-1",
        error="Incomplete response did not address all aspects",
        retry_count=2,
        evaluation_time_ms=2150,
        model_used="gpt-4o-mini",
        reasoning="The response failed to address key aspects of the query.",
    )


@pytest.fixture
def sample_processed_file() -> ProcessedFileInput:
    """Create a sample ProcessedFileInput for testing."""
    return ProcessedFileInput(
        original=FileInput(path="docs/warranty.pdf", type="pdf"),
        markdown_content="# Warranty Policy\n\nAll products include...",
        metadata={"pages": 5, "size_bytes": 245000, "format": "pdf"},
        cached_path="/tmp/cache/warranty_abc123.md",  # noqa: S108
        processing_time_ms=2150,
        error=None,
    )


@pytest.fixture
def sample_test_result_passed(sample_metric_result) -> TestResult:
    """Create a passing TestResult for testing."""
    return TestResult(
        test_name="test_warranty_query",
        test_input="What is the warranty coverage?",
        processed_files=[],
        agent_response="The product includes a 12-month manufacturer warranty.",
        tool_calls=["search_knowledge_base", "retrieve_policy"],
        expected_tools=["search_knowledge_base", "retrieve_policy"],
        tools_matched=True,
        metric_results=[sample_metric_result],
        ground_truth="Product has 12-month warranty.",
        passed=True,
        execution_time_ms=4500,
        errors=[],
        timestamp="2025-11-22T10:30:45Z",
    )


@pytest.fixture
def sample_test_result_failed(sample_metric_result_failed) -> TestResult:
    """Create a failing TestResult for testing."""
    return TestResult(
        test_name="test_refund_policy",
        test_input="Can I get a refund?",
        processed_files=[],
        agent_response="I don't have that information.",
        tool_calls=["search_knowledge_base"],
        expected_tools=["search_knowledge_base", "escalate_to_human"],
        tools_matched=False,
        metric_results=[sample_metric_result_failed],
        ground_truth="Full refunds within 30 days.",
        passed=False,
        execution_time_ms=3200,
        errors=[
            "Agent failed to retrieve policy",
            "Tool mismatch: expected escalate_to_human",
        ],
        timestamp="2025-11-22T10:31:12Z",
    )


@pytest.fixture
def sample_report_summary() -> ReportSummary:
    """Create a sample ReportSummary for testing."""
    return ReportSummary(
        total_tests=2,
        passed=1,
        failed=1,
        pass_rate=50.0,
        total_duration_ms=7700,
        metrics_evaluated={"groundedness": 2, "completeness": 1},
        average_scores={"groundedness": 0.92, "completeness": 0.35},
    )


@pytest.fixture
def sample_test_report(
    sample_test_result_passed, sample_test_result_failed, sample_report_summary
) -> TestReport:
    """Create a comprehensive sample TestReport for testing."""
    return TestReport(
        agent_name="customer-support-agent",
        agent_config_path="agents/customer-support.yaml",
        results=[sample_test_result_passed, sample_test_result_failed],
        summary=sample_report_summary,
        timestamp="2025-11-22T10:33:00Z",
        holodeck_version="0.1.0",
        environment={"python_version": "3.10.0", "os": "Darwin"},
    )


class TestReportStructure:
    """Test overall report structure and formatting."""

    def test_report_contains_all_sections_and_formatting(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that report includes header, summary, test results, and formatting."""
        markdown = generate_markdown_report(sample_test_report)

        # Header with agent name and metadata
        assert "# Test Report: customer-support-agent" in markdown
        assert "agents/customer-support.yaml" in markdown
        assert "0.1.0" in markdown

        # Summary section
        assert "## Summary" in markdown
        assert "Total Tests" in markdown
        assert "Passed" in markdown
        assert "Failed" in markdown
        assert "Pass Rate" in markdown

        # Test results section
        assert "## Test Results" in markdown
        assert "### Test 1:" in markdown or "test_warranty_query" in markdown
        assert "### Test 2:" in markdown or "test_refund_policy" in markdown

        # Markdown formatting
        assert "# Test Report:" in markdown
        assert "### Test" in markdown
        assert "**" in markdown or "*" in markdown


class TestSummaryTable:
    """Test summary statistics table formatting."""

    def test_summary_table_contains_all_fields_and_formatting(
        self, sample_report_summary: ReportSummary
    ) -> None:
        """Test that summary table includes all stats and proper markdown formatting."""
        table = _format_summary_table(sample_report_summary)

        # Content fields
        assert "Total Tests" in table
        assert "2" in table
        assert "Pass Rate" in table
        assert "50" in table or "50.0" in table
        assert "Duration" in table or "Total Duration" in table
        assert "7700" in table

        # Markdown table markers
        assert "|" in table
        assert "-" in table


class TestTestSection:
    """Test individual test result section formatting."""

    def test_passing_test_section_includes_all_details(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test passing test section includes all fields."""
        section = _format_test_section(sample_test_result_passed)

        assert "test_warranty_query" in section
        assert "What is the warranty coverage?" in section
        assert sample_test_result_passed.agent_response in section
        assert "4500" in section or "4.5" in section
        assert "✅" in section or "PASSED" in section or "PASS" in section
        assert "Product has 12-month warranty." in section

    def test_failing_test_section_shows_failure_status(
        self, sample_test_result_failed: TestResult
    ) -> None:
        """Test that failing test shows failure status."""
        section = _format_test_section(sample_test_result_failed)
        assert "❌" in section or "FAILED" in section or "FAIL" in section


class TestProcessedFilesSection:
    """Test processed files section formatting."""

    def test_files_section_empty_list(self) -> None:
        """Test handling of empty processed files list."""
        section = _format_processed_files([])
        assert section == "" or "No files" in section.lower()

    def test_single_file_includes_all_metadata(
        self, sample_processed_file: ProcessedFileInput
    ) -> None:
        """Test single file section includes all fields."""
        section = _format_processed_files([sample_processed_file])

        assert "warranty.pdf" in section or "docs/warranty.pdf" in section
        assert "2150" in section  # processing time
        assert "KB" in section or "239" in section  # size (converted to KB)
        assert "5" in section or "pages" in section.lower()
        assert "pdf" in section.lower()
        assert "/tmp/cache/" in section or "cache" in section.lower()  # noqa: S108

    def test_files_section_multiple_files(
        self, sample_processed_file: ProcessedFileInput
    ) -> None:
        """Test formatting of multiple processed files."""
        file2 = ProcessedFileInput(
            original=FileInput(path="docs/shipping.xlsx", type="excel"),
            markdown_content="| Service | Cost |",
            metadata={"size_bytes": 18500, "format": "xlsx"},
            cached_path="/tmp/cache/shipping.md",  # noqa: S108
            processing_time_ms=1200,
            error=None,
        )
        section = _format_processed_files([sample_processed_file, file2])
        assert "warranty.pdf" in section
        assert "shipping.xlsx" in section


class TestMetricsTable:
    """Test metrics results table formatting."""

    def test_metrics_table_empty(self) -> None:
        """Test handling of empty metrics list."""
        table = _format_metrics_table([])
        assert table == "" or "No metrics" in table.lower()

    def test_passing_metric_includes_all_fields(
        self, sample_metric_result: MetricResult
    ) -> None:
        """Test passing metric displays all fields."""
        table = _format_metrics_table([sample_metric_result])

        assert "groundedness" in table
        assert "0.92" in table
        assert "0.8" in table  # threshold
        assert "✅" in table or "PASS" in table
        assert "gpt-4o-mini" in table
        assert "Metric Details" in table
        assert "well-grounded" in table

    def test_failing_metric_includes_error_and_retry(
        self, sample_metric_result_failed: MetricResult
    ) -> None:
        """Test that failing metric displays status, error, and retry count."""
        table = _format_metrics_table([sample_metric_result_failed])

        assert "❌" in table or "FAIL" in table
        assert "Incomplete response" in table
        assert "2" in table  # retry_count

    def test_metrics_table_multiple_metrics(
        self,
        sample_metric_result: MetricResult,
        sample_metric_result_failed: MetricResult,
    ) -> None:
        """Test formatting of multiple metric results."""
        table = _format_metrics_table(
            [sample_metric_result, sample_metric_result_failed]
        )
        assert "groundedness" in table
        assert "completeness" in table

    def test_metrics_table_reasoning_none(self) -> None:
        """Test that metric details section handles None reasoning gracefully."""
        metric = MetricResult(
            metric_name="relevance",
            score=0.85,
            threshold=0.7,
            passed=True,
            scale="0-1",
            error=None,
            retry_count=0,
            evaluation_time_ms=500,
            model_used="gpt-4o",
            reasoning=None,
        )
        table = _format_metrics_table([metric])
        assert "Metric Details" in table  # Section header still present
        # Metric name should appear in details section
        assert "**relevance**" in table
        # No blockquote reasoning should appear (since reasoning is None)
        lines = table.split("\n")
        detail_lines = [line for line in lines if line.startswith(">")]
        # No reasoning blockquotes when reasoning is None
        assert len(detail_lines) == 0

    def test_metrics_table_reasoning_full_display(self) -> None:
        """Test that full reasoning is displayed without truncation."""
        long_reasoning = "A" * 150  # 150 characters
        metric = MetricResult(
            metric_name="coherence",
            score=0.9,
            threshold=0.7,
            passed=True,
            scale="0-1",
            error=None,
            retry_count=0,
            evaluation_time_ms=800,
            model_used="gpt-4o",
            reasoning=long_reasoning,
        )
        table = _format_metrics_table([metric])
        # Full reasoning should be displayed (no truncation in new format)
        assert "A" * 150 in table
        # Should be in blockquote format
        assert "> " + "A" * 150 in table


class TestToolUsageSection:
    """Test tool usage validation section formatting."""

    def test_tool_usage_no_tools(self) -> None:
        """Test handling when no tools are called."""
        result = TestResult(
            test_name="test_simple",
            test_input="Hello",
            agent_response="Hi",
            tool_calls=[],
            expected_tools=None,
            tools_matched=None,
            passed=True,
            execution_time_ms=100,
            timestamp="2025-11-22T10:30:00Z",
        )
        section = _format_tool_usage(result)
        assert section == "" or "No tools" in section.lower()

    def test_tool_usage_matched_shows_tools_and_status(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that matched tools section shows all tools and match status."""
        section = _format_tool_usage(sample_test_result_passed)

        assert "search_knowledge_base" in section
        assert "retrieve_policy" in section
        assert "✅" in section or "matched" in section.lower()

    def test_tool_usage_mismatch_shows_status_and_missing(
        self, sample_test_result_failed: TestResult
    ) -> None:
        """Test that mismatch shows failure status and missing tools."""
        section = _format_tool_usage(sample_test_result_failed)

        assert (
            "❌" in section
            or "mismatch" in section.lower()
            or "did not match" in section.lower()
        )
        assert "escalate_to_human" in section or "missing" in section.lower()


class TestErrorHandling:
    """Test error display in reports."""

    def test_errors_section_empty(self, sample_test_result_passed: TestResult) -> None:
        """Test handling when no errors present."""
        section = _format_test_section(sample_test_result_passed)
        # Should either not include errors section or indicate no errors
        if "Error" in section or "error" in section:
            assert "No errors" in section or "error" in section.lower()

    def test_errors_section_displays_errors(
        self, sample_test_result_failed: TestResult
    ) -> None:
        """Test that errors are displayed in report."""
        section = _format_test_section(sample_test_result_failed)
        assert "Agent failed to retrieve policy" in section
        assert "Tool mismatch" in section


class TestComprehensiveReport:
    """Test comprehensive report generation with all fields."""

    def test_report_includes_all_details_metrics_tools_and_status(
        self, sample_test_report: TestReport
    ) -> None:
        """Test comprehensive report includes all sections."""
        markdown = generate_markdown_report(sample_test_report)

        # Passing test details
        assert "test_warranty_query" in markdown
        assert "What is the warranty coverage?" in markdown
        assert "12-month manufacturer warranty" in markdown
        assert "Product has 12-month warranty." in markdown  # ground truth

        # Failing test details
        assert "test_refund_policy" in markdown
        assert "Can I get a refund?" in markdown
        assert "I don't have that information." in markdown
        assert "Full refunds within 30 days." in markdown  # ground truth

        # Metrics from all tests
        assert "groundedness" in markdown
        assert "completeness" in markdown
        assert "0.92" in markdown
        assert "0.35" in markdown

        # Tool usage validation
        assert "search_knowledge_base" in markdown
        assert "retrieve_policy" in markdown
        assert "escalate_to_human" in markdown

        # Pass/fail distinction
        assert ("✅" in markdown or "PASSED" in markdown) or "PASS" in markdown
        assert ("❌" in markdown or "FAILED" in markdown) or "FAIL" in markdown


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_report_with_zero_metrics(self) -> None:
        """Test report generation when tests have no metrics."""
        result = TestResult(
            test_name="test_no_metrics",
            test_input="Test input",
            agent_response="Response",
            metric_results=[],
            passed=True,
            execution_time_ms=100,
            timestamp="2025-11-22T10:30:00Z",
        )
        report = TestReport(
            agent_name="test-agent",
            agent_config_path="test.yaml",
            results=[result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=100,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        markdown = generate_markdown_report(report)
        assert "test-agent" in markdown
        assert "test_no_metrics" in markdown

    def test_report_with_no_ground_truth(self) -> None:
        """Test report when ground truth is not provided."""
        result = TestResult(
            test_name="test_no_truth",
            test_input="Test input",
            agent_response="Response",
            ground_truth=None,
            passed=True,
            execution_time_ms=100,
            timestamp="2025-11-22T10:30:00Z",
        )
        report = TestReport(
            agent_name="test-agent",
            agent_config_path="test.yaml",
            results=[result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=100,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        markdown = generate_markdown_report(report)
        assert "test-agent" in markdown

    def test_report_with_long_responses(self) -> None:
        """Test report generation with very long agent responses."""
        long_response = "This is a very long response. " * 50
        result = TestResult(
            test_name="test_long_response",
            test_input="Test input",
            agent_response=long_response,
            passed=True,
            execution_time_ms=100,
            timestamp="2025-11-22T10:30:00Z",
        )
        report = TestReport(
            agent_name="test-agent",
            agent_config_path="test.yaml",
            results=[result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=100,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        markdown = generate_markdown_report(report)
        assert long_response in markdown

    def test_report_with_special_characters(self) -> None:
        """Test report generation with special characters in responses."""
        result = TestResult(
            test_name="test_special_chars",
            test_input="What are the prices? $$$",
            agent_response="Prices: $99.99, €89.99, ¥10,000 | Special: 50% off!",
            passed=True,
            execution_time_ms=100,
            timestamp="2025-11-22T10:30:00Z",
        )
        report = TestReport(
            agent_name="test-agent",
            agent_config_path="test.yaml",
            results=[result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=100,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        markdown = generate_markdown_report(report)
        # Should handle special characters without errors
        assert "test-agent" in markdown


class TestJSONReportGeneration:
    """Test JSON report generation and serialization."""

    def test_json_report_structure_metadata_and_values(
        self, sample_test_report: TestReport
    ) -> None:
        """Test JSON report structure and all fields."""
        json_str = sample_test_report.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Top-level keys
        for key in [
            "agent_name",
            "agent_config_path",
            "results",
            "summary",
            "timestamp",
            "holodeck_version",
            "environment",
        ]:
            assert key in data

        # Agent metadata values
        assert data["agent_name"] == "customer-support-agent"
        assert data["agent_config_path"] == "agents/customer-support.yaml"
        assert data["holodeck_version"] == "0.1.0"

        # Environment info
        assert data["environment"]["python_version"] == "3.10.0"
        assert data["environment"]["os"] == "Darwin"

        # Results array
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2

        # Test result fields
        result = data["results"][0]
        for field in [
            "test_name",
            "test_input",
            "agent_response",
            "tool_calls",
            "expected_tools",
            "tools_matched",
            "metric_results",
            "ground_truth",
            "passed",
            "execution_time_ms",
            "timestamp",
            "errors",
            "processed_files",
        ]:
            assert field in result

        # Metric result fields
        metrics = result["metric_results"]
        assert len(metrics) > 0
        metric = metrics[0]
        for field in [
            "metric_name",
            "score",
            "threshold",
            "passed",
            "scale",
            "error",
            "retry_count",
            "evaluation_time_ms",
            "model_used",
        ]:
            assert field in metric

        # Summary fields and values
        summary = data["summary"]
        for field in [
            "total_tests",
            "passed",
            "failed",
            "pass_rate",
            "total_duration_ms",
            "metrics_evaluated",
            "average_scores",
        ]:
            assert field in summary
        assert summary["total_tests"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 50.0
        assert summary["total_duration_ms"] == 7700

    def test_json_report_processed_files(self, sample_test_report: TestReport) -> None:
        """Test that processed files are included in JSON."""
        result = TestResult(
            test_name="test_with_file",
            test_input="Process this document",
            processed_files=[
                ProcessedFileInput(
                    original=FileInput(path="docs/test.pdf", type="pdf"),
                    markdown_content="Test content",
                    metadata={"pages": 5, "size_bytes": 100000},
                    cached_path=".cache/processed/test.md",
                    processing_time_ms=1000,
                    error=None,
                )
            ],
            agent_response="Document processed",
            passed=True,
            execution_time_ms=2000,
            timestamp="2025-11-22T10:30:00Z",
        )
        report = TestReport(
            agent_name="test-agent",
            agent_config_path="test.yaml",
            results=[result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=2000,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        json_str = report.model_dump_json(indent=2)
        data = json.loads(json_str)

        files = data["results"][0]["processed_files"]
        assert len(files) == 1
        assert files[0]["original"]["path"] == "docs/test.pdf"
        assert files[0]["original"]["type"] == "pdf"
        assert files[0]["markdown_content"] == "Test content"
        assert files[0]["cached_path"] == ".cache/processed/test.md"
        assert files[0]["processing_time_ms"] == 1000

    def test_json_report_null_values(self) -> None:
        """Test that null values are properly handled in JSON."""
        result = TestResult(
            test_name="test_minimal",
            test_input="Test",
            agent_response=None,
            expected_tools=None,
            tools_matched=None,
            ground_truth=None,
            passed=True,
            execution_time_ms=100,
            timestamp="2025-11-22T10:30:00Z",
        )
        report = TestReport(
            agent_name="test-agent",
            agent_config_path="test.yaml",
            results=[result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=100,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        json_str = report.model_dump_json(indent=2)
        data = json.loads(json_str)

        result_data = data["results"][0]
        assert result_data["agent_response"] is None
        assert result_data["expected_tools"] is None
        assert result_data["tools_matched"] is None
        assert result_data["ground_truth"] is None

    def test_json_report_empty_errors_list(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that empty error lists are preserved in JSON."""
        json_str = sample_test_report.model_dump_json(indent=2)
        data = json.loads(json_str)

        passed_result = data["results"][0]
        assert "errors" in passed_result
        assert isinstance(passed_result["errors"], list)
        assert len(passed_result["errors"]) == 0

    def test_json_report_errors_in_failed_test(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that errors are properly serialized in failed tests."""
        json_str = sample_test_report.model_dump_json(indent=2)
        data = json.loads(json_str)

        failed_result = data["results"][1]
        assert "errors" in failed_result
        assert isinstance(failed_result["errors"], list)
        assert len(failed_result["errors"]) == 2
        assert any("Agent failed" in err for err in failed_result["errors"])

    def test_json_report_special_characters(self) -> None:
        """Test that special characters are properly escaped in JSON."""
        result = TestResult(
            test_name="test_special",
            test_input='What costs "$99.99"?',
            agent_response='Prices: €89.99, ¥10,000 | Special: 50% off! "Amazing!"',
            passed=True,
            execution_time_ms=100,
            timestamp="2025-11-22T10:30:00Z",
        )
        report = TestReport(
            agent_name="test-agent",
            agent_config_path="test.yaml",
            results=[result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=100,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        json_str = report.model_dump_json(indent=2)

        # Should be valid JSON despite special characters
        data = json.loads(json_str)
        assert "$99.99" in data["results"][0]["test_input"]
        assert "50% off" in data["results"][0]["agent_response"]

    def test_json_report_numeric_precision(self) -> None:
        """Test that numeric values maintain precision in JSON."""
        metric = MetricResult(
            metric_name="precision_test",
            score=0.9234567890,
            threshold=0.75,
            passed=True,
            scale="0-1",
            evaluation_time_ms=1234,
        )
        result = TestResult(
            test_name="test_precision",
            test_input="Test",
            metric_results=[metric],
            passed=True,
            execution_time_ms=5000,
            timestamp="2025-11-22T10:30:00Z",
        )
        report = TestReport(
            agent_name="test-agent",
            agent_config_path="test.yaml",
            results=[result],
            summary=ReportSummary(
                total_tests=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                total_duration_ms=5000,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        json_str = report.model_dump_json(indent=2)
        data = json.loads(json_str)

        metric_data = data["results"][0]["metric_results"][0]
        assert metric_data["score"] == pytest.approx(0.9234567890, rel=1e-9)
        assert metric_data["threshold"] == 0.75
        assert metric_data["evaluation_time_ms"] == 1234

    def test_json_report_valid_json_format(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that generated JSON is valid and properly formatted."""
        json_str = sample_test_report.model_dump_json(indent=2)

        # Should parse without errors
        data = json.loads(json_str)
        assert isinstance(data, dict)

        # Should re-serialize consistently
        json_str_2 = json.dumps(data, indent=2)
        data_2 = json.loads(json_str_2)
        assert data == data_2

    def test_json_report_round_trip_serialization(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that JSON can be parsed back into TestReport without data loss."""
        json_str = sample_test_report.model_dump_json(indent=2)
        data = json.loads(json_str)

        # Reconstruct TestReport from JSON data
        restored_report = TestReport(**data)

        # Verify key data is preserved
        assert restored_report.agent_name == sample_test_report.agent_name
        assert restored_report.agent_config_path == sample_test_report.agent_config_path
        assert len(restored_report.results) == len(sample_test_report.results)
        assert (
            restored_report.summary.total_tests
            == sample_test_report.summary.total_tests
        )
        assert restored_report.holodeck_version == sample_test_report.holodeck_version

    def test_json_report_empty_report(self) -> None:
        """Test JSON generation for minimal/empty report."""
        report = TestReport(
            agent_name="minimal-agent",
            agent_config_path="minimal.yaml",
            results=[],
            summary=ReportSummary(
                total_tests=0,
                passed=0,
                failed=0,
                pass_rate=0.0,
                total_duration_ms=0,
            ),
            timestamp="2025-11-22T10:30:00Z",
            holodeck_version="0.1.0",
        )
        json_str = report.model_dump_json(indent=2)
        data = json.loads(json_str)

        assert data["agent_name"] == "minimal-agent"
        assert len(data["results"]) == 0
        assert data["summary"]["total_tests"] == 0
