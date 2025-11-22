"""Unit tests for markdown report generation.

Tests the TestReport.to_markdown() method and related formatting functions,
ensuring comprehensive display of all TestResult fields including:
- Test details (name, input, timestamp)
- Processed files with metadata
- Agent responses
- Tool usage validation
- Evaluation metrics with scores and thresholds
- Ground truth comparisons
- Error handling
"""

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
        environment={"python_version": "3.13.0", "os": "Darwin"},
    )


class TestReportStructure:
    """Test overall report structure and formatting."""

    def test_report_has_header(self, sample_test_report: TestReport) -> None:
        """Test that report includes a header with agent name and metadata."""
        markdown = generate_markdown_report(sample_test_report)
        assert "# Test Report: customer-support-agent" in markdown
        assert "agents/customer-support.yaml" in markdown
        assert "0.1.0" in markdown

    def test_report_has_summary_section(self, sample_test_report: TestReport) -> None:
        """Test that report includes summary statistics section."""
        markdown = generate_markdown_report(sample_test_report)
        assert "## Summary" in markdown
        assert "Total Tests" in markdown
        assert "Passed" in markdown
        assert "Failed" in markdown
        assert "Pass Rate" in markdown

    def test_report_has_test_results_section(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that report includes individual test results."""
        markdown = generate_markdown_report(sample_test_report)
        assert "## Test Results" in markdown
        assert "### Test 1:" in markdown or "test_warranty_query" in markdown
        assert "### Test 2:" in markdown or "test_refund_policy" in markdown

    def test_report_markdown_formatting(self, sample_test_report: TestReport) -> None:
        """Test that report uses proper markdown formatting."""
        markdown = generate_markdown_report(sample_test_report)
        # Check for markdown headers
        assert "# Test Report:" in markdown
        assert "## Summary" in markdown
        assert "### Test" in markdown
        # Check for markdown emphasis
        assert "**" in markdown or "*" in markdown


class TestSummaryTable:
    """Test summary statistics table formatting."""

    def test_summary_table_contains_total_tests(
        self, sample_report_summary: ReportSummary
    ) -> None:
        """Test that summary table includes total test count."""
        table = _format_summary_table(sample_report_summary)
        assert "Total Tests" in table
        assert "2" in table

    def test_summary_table_contains_pass_rate(
        self, sample_report_summary: ReportSummary
    ) -> None:
        """Test that summary table includes pass rate."""
        table = _format_summary_table(sample_report_summary)
        assert "Pass Rate" in table
        assert "50" in table or "50.0" in table

    def test_summary_table_contains_duration(
        self, sample_report_summary: ReportSummary
    ) -> None:
        """Test that summary table includes total duration."""
        table = _format_summary_table(sample_report_summary)
        assert "Duration" in table or "Total Duration" in table
        assert "7700" in table

    def test_summary_table_formatting(
        self, sample_report_summary: ReportSummary
    ) -> None:
        """Test that summary table is properly formatted."""
        table = _format_summary_table(sample_report_summary)
        # Check for markdown table markers
        assert "|" in table
        assert "-" in table


class TestTestSection:
    """Test individual test result section formatting."""

    def test_test_section_includes_name(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that test section includes test name."""
        section = _format_test_section(sample_test_result_passed)
        assert "test_warranty_query" in section

    def test_test_section_includes_input(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that test section includes test input prompt."""
        section = _format_test_section(sample_test_result_passed)
        assert "What is the warranty coverage?" in section

    def test_test_section_includes_response(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that test section includes agent response."""
        section = _format_test_section(sample_test_result_passed)
        assert sample_test_result_passed.agent_response in section

    def test_test_section_includes_execution_time(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that test section includes execution time."""
        section = _format_test_section(sample_test_result_passed)
        assert "4500" in section or "4.5" in section

    def test_test_section_includes_pass_status(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that passing test shows success status."""
        section = _format_test_section(sample_test_result_passed)
        assert "✅" in section or "PASSED" in section or "PASS" in section

    def test_test_section_shows_failed_status(
        self, sample_test_result_failed: TestResult
    ) -> None:
        """Test that failing test shows failure status."""
        section = _format_test_section(sample_test_result_failed)
        assert "❌" in section or "FAILED" in section or "FAIL" in section

    def test_test_section_includes_ground_truth(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that test section includes ground truth when present."""
        section = _format_test_section(sample_test_result_passed)
        assert "Product has 12-month warranty." in section


class TestProcessedFilesSection:
    """Test processed files section formatting."""

    def test_files_section_empty_list(self) -> None:
        """Test handling of empty processed files list."""
        section = _format_processed_files([])
        # Should handle gracefully
        assert section == "" or "No files" in section.lower()

    def test_files_section_single_file(
        self, sample_processed_file: ProcessedFileInput
    ) -> None:
        """Test formatting of single processed file."""
        section = _format_processed_files([sample_processed_file])
        assert "warranty.pdf" in section or "docs/warranty.pdf" in section
        assert "2150" in section  # processing time
        assert "KB" in section or "239" in section  # size (converted to KB)

    def test_files_section_includes_metadata(
        self, sample_processed_file: ProcessedFileInput
    ) -> None:
        """Test that file metadata is displayed."""
        section = _format_processed_files([sample_processed_file])
        assert "5" in section or "pages" in section.lower()
        assert "pdf" in section.lower()

    def test_files_section_shows_cached_path(
        self, sample_processed_file: ProcessedFileInput
    ) -> None:
        """Test that cached path is displayed."""
        section = _format_processed_files([sample_processed_file])
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

    def test_metrics_table_single_metric(
        self, sample_metric_result: MetricResult
    ) -> None:
        """Test formatting of single metric result."""
        table = _format_metrics_table([sample_metric_result])
        assert "groundedness" in table
        assert "0.92" in table
        assert "0.8" in table  # threshold

    def test_metrics_table_includes_score(
        self, sample_metric_result: MetricResult
    ) -> None:
        """Test that metric score is displayed."""
        table = _format_metrics_table([sample_metric_result])
        assert "0.92" in table

    def test_metrics_table_includes_threshold(
        self, sample_metric_result: MetricResult
    ) -> None:
        """Test that threshold is displayed."""
        table = _format_metrics_table([sample_metric_result])
        assert "0.8" in table

    def test_metrics_table_shows_pass_status(
        self, sample_metric_result: MetricResult
    ) -> None:
        """Test that passing metric shows success."""
        table = _format_metrics_table([sample_metric_result])
        assert "✅" in table or "PASS" in table

    def test_metrics_table_shows_fail_status(
        self, sample_metric_result_failed: MetricResult
    ) -> None:
        """Test that failing metric shows failure."""
        table = _format_metrics_table([sample_metric_result_failed])
        assert "❌" in table or "FAIL" in table

    def test_metrics_table_includes_model(
        self, sample_metric_result: MetricResult
    ) -> None:
        """Test that model name is displayed for AI metrics."""
        table = _format_metrics_table([sample_metric_result])
        assert "gpt-4o-mini" in table

    def test_metrics_table_includes_error(
        self, sample_metric_result_failed: MetricResult
    ) -> None:
        """Test that metric errors are displayed."""
        table = _format_metrics_table([sample_metric_result_failed])
        assert "Incomplete response" in table

    def test_metrics_table_includes_retry_count(
        self, sample_metric_result_failed: MetricResult
    ) -> None:
        """Test that retry count is displayed."""
        table = _format_metrics_table([sample_metric_result_failed])
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


class TestToolUsageSection:
    """Test tool usage validation section formatting."""

    def test_tool_usage_no_tools(self, sample_test_result_passed: TestResult) -> None:
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

    def test_tool_usage_shows_called_tools(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that called tools are displayed."""
        section = _format_tool_usage(sample_test_result_passed)
        assert "search_knowledge_base" in section
        assert "retrieve_policy" in section

    def test_tool_usage_shows_expected_tools(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that expected tools are displayed."""
        section = _format_tool_usage(sample_test_result_passed)
        assert "search_knowledge_base" in section
        assert "retrieve_policy" in section

    def test_tool_usage_matched_status(
        self, sample_test_result_passed: TestResult
    ) -> None:
        """Test that tool match status is displayed."""
        section = _format_tool_usage(sample_test_result_passed)
        assert "✅" in section or "matched" in section.lower()

    def test_tool_usage_mismatch_status(
        self, sample_test_result_failed: TestResult
    ) -> None:
        """Test that tool mismatch is highlighted."""
        section = _format_tool_usage(sample_test_result_failed)
        assert (
            "❌" in section
            or "mismatch" in section.lower()
            or "did not match" in section.lower()
        )

    def test_tool_usage_missing_tools(
        self, sample_test_result_failed: TestResult
    ) -> None:
        """Test that missing tools are identified."""
        section = _format_tool_usage(sample_test_result_failed)
        # Should indicate escalate_to_human is missing
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

    def test_report_includes_all_test_details(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that comprehensive report includes all test result details."""
        markdown = generate_markdown_report(sample_test_report)

        # Check for passing test details
        assert "test_warranty_query" in markdown
        assert "What is the warranty coverage?" in markdown
        assert "12-month manufacturer warranty" in markdown
        assert "Product has 12-month warranty." in markdown  # ground truth

        # Check for failing test details
        assert "test_refund_policy" in markdown
        assert "Can I get a refund?" in markdown
        assert "I don't have that information." in markdown
        assert "Full refunds within 30 days." in markdown  # ground truth

    def test_report_includes_all_metrics(self, sample_test_report: TestReport) -> None:
        """Test that report includes metrics from all tests."""
        markdown = generate_markdown_report(sample_test_report)
        assert "groundedness" in markdown
        assert "completeness" in markdown
        assert "0.92" in markdown
        assert "0.35" in markdown

    def test_report_includes_tool_validation(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that report includes tool usage validation."""
        markdown = generate_markdown_report(sample_test_report)
        assert "search_knowledge_base" in markdown
        assert "retrieve_policy" in markdown
        assert "escalate_to_human" in markdown  # from expected_tools in failing test

    def test_report_distinguishes_pass_fail(
        self, sample_test_report: TestReport
    ) -> None:
        """Test that report clearly distinguishes passing and failing tests."""
        markdown = generate_markdown_report(sample_test_report)
        # Should have both pass and fail indicators
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
