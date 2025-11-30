"""Markdown report generation for test execution results.

Generates comprehensive markdown reports that display all TestResult details
including processed files, metric results, tool calls, agent responses,
and execution details.
"""

from holodeck.models.test_result import (
    MetricResult,
    ProcessedFileInput,
    ReportSummary,
    TestReport,
    TestResult,
)


def generate_markdown_report(report: TestReport) -> str:
    """Generate a comprehensive markdown report from test results.

    Creates a formatted markdown document containing:
    - Report header with agent name and metadata
    - Summary statistics table
    - Detailed test result sections with all fields

    Parameters:
        report: The TestReport containing all test results and summary data.

    Returns:
        A formatted markdown string ready for display or file output.
    """
    lines: list[str] = []

    # Header
    lines.append(f"# Test Report: {report.agent_name}\n")
    lines.append(f"**Configuration:** `{report.agent_config_path}`")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**HoloDeck Version:** {report.holodeck_version}")

    if report.environment:
        env_parts = []
        if "python_version" in report.environment:
            env_parts.append(report.environment["python_version"])
        if "os" in report.environment:
            env_parts.append(report.environment["os"])
        if env_parts:
            lines.append(f"**Environment:** {' on '.join(env_parts)}")

    lines.append("")

    # Summary section
    lines.append("## Summary\n")
    lines.append(_format_summary_table(report.summary))
    lines.append("")

    # Average metric scores (if available)
    if report.summary.average_scores:
        lines.append("### Average Metric Scores\n")
        score_lines = ["| Metric | Average Score | Scale |"]
        score_lines.append("|--------|----------------|-------|")
        for metric, avg_score in report.summary.average_scores.items():
            score_lines.append(f"| {metric} | {avg_score:.2f} | 0-1 |")
        lines.append("\n".join(score_lines))
        lines.append("")

    lines.append("---\n")

    # Test results
    lines.append("## Test Results\n")
    for result in report.results:
        lines.append(_format_test_section(result))
        lines.append("")

    # Final summary
    lines.append("---\n")
    lines.append("## Report Summary\n")
    status_emoji_pass = "✅" if report.summary.passed > 0 else ""
    status_emoji_fail = "❌" if report.summary.failed > 0 else ""
    lines.append(
        f"{status_emoji_pass} **{report.summary.passed} tests passed** | "
        f"{status_emoji_fail} **{report.summary.failed} tests failed** | "
        f"**Pass Rate: {report.summary.pass_rate:.2f}%**\n"
    )

    return "\n".join(lines)


def _format_summary_table(summary: ReportSummary) -> str:
    """Format summary statistics as a markdown table.

    Parameters:
        summary: The ReportSummary containing aggregate statistics.

    Returns:
        A formatted markdown table string.
    """
    duration_s = summary.total_duration_ms / 1000.0
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Tests | {summary.total_tests} |",
        f"| Passed | {summary.passed} |",
        f"| Failed | {summary.failed} |",
        f"| Pass Rate | {summary.pass_rate:.2f}% |",
        f"| Total Duration | {summary.total_duration_ms}ms ({duration_s:.1f}s) |",
    ]
    return "\n".join(lines)


def _format_test_section(result: TestResult) -> str:
    """Format a single test result as a detailed markdown section.

    Parameters:
        result: The TestResult to format.

    Returns:
        A formatted markdown section string.
    """
    lines: list[str] = []

    # Test header with status
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    test_name = result.test_name or "Unnamed Test"
    lines.append(f"### Test: {test_name} {status}\n")

    # Test input
    lines.append("**Input:**")
    lines.append("```")
    lines.append(result.test_input)
    lines.append("```\n")

    # Execution time
    lines.append(f"**Execution Time:** {result.execution_time_ms}ms\n")

    # Processed files section
    if result.processed_files:
        lines.append("#### Processed Files\n")
        files_section = _format_processed_files(result.processed_files)
        if files_section:
            lines.append(files_section)
            lines.append("")

    # Agent response
    lines.append("#### Agent Response\n")
    if result.agent_response:
        lines.append("> " + result.agent_response.replace("\n", "\n> "))
    else:
        lines.append("> (No response)")
    lines.append("")

    # Tool usage section
    if result.tool_calls or result.expected_tools:
        lines.append("#### Tool Usage\n")
        tool_section = _format_tool_usage(result)
        if tool_section:
            lines.append(tool_section)
            lines.append("")

    # Metrics section
    if result.metric_results:
        lines.append("#### Evaluation Metrics\n")
        metrics_section = _format_metrics_table(result.metric_results)
        if metrics_section:
            lines.append(metrics_section)
            lines.append("")

    # Ground truth
    if result.ground_truth:
        lines.append("#### Ground Truth Comparison\n")
        lines.append("**Expected:**")
        lines.append("```")
        lines.append(result.ground_truth)
        lines.append("```\n")

    # Errors section
    if result.errors:
        lines.append("#### Errors\n")
        for error in result.errors:
            lines.append(f"- ❌ {error}")
        lines.append("")

    return "\n".join(lines)


def _format_processed_files(files: list[ProcessedFileInput]) -> str:
    """Format processed files as a markdown table with metadata.

    Parameters:
        files: List of ProcessedFileInput objects.

    Returns:
        A formatted markdown table string, or empty string if no files.
    """
    if not files:
        return ""

    lines = [
        "| File | Type | Format | Processing Time | Status |",
        "|------|------|--------|-----------------|--------|",
    ]

    for file_input in files:
        file_path = file_input.original.path
        file_type = file_input.original.type
        format_info = file_type.upper()

        # Add metadata to format
        if file_input.metadata:
            size_bytes = file_input.metadata.get("size_bytes", 0)
            size_kb = size_bytes / 1024.0
            format_info = f"{file_type.upper()} ({size_kb:.1f} KB)"

            if "pages" in file_input.metadata and file_input.metadata["pages"]:
                pages = file_input.metadata["pages"]
                format_info = f"{file_type.upper()} ({pages} pages, {size_kb:.1f} KB)"
            elif "sheet" in file_input.metadata:
                sheet = file_input.metadata["sheet"]
                format_info = f"{file_type.upper()} ({sheet}, {size_kb:.1f} KB)"

        processing_time = (
            f"{file_input.processing_time_ms}ms"
            if file_input.processing_time_ms
            else ""
        )
        status = "✅ Success" if not file_input.error else f"❌ {file_input.error}"

        table_row = (
            f"| {file_path} | {file_type} | {format_info} | "
            f"{processing_time} | {status} |"
        )
        lines.append(table_row)

    # Add file metadata details below table
    lines.append("")
    lines.append("**File Metadata:**")
    for file_input in files:
        lines.append(f"- **{file_input.original.path}**")
        if file_input.metadata:
            if "sheet" in file_input.metadata:
                lines.append(f"  - Sheet: `{file_input.metadata['sheet']}`")
            if "pages" in file_input.metadata:
                lines.append(f"  - Pages: {file_input.metadata['pages']}")
            if "size_bytes" in file_input.metadata:
                size_kb = file_input.metadata["size_bytes"] / 1024.0
                lines.append(f"  - Size: {size_kb:.1f} KB")
        if file_input.cached_path:
            lines.append(f"  - Cached: `{file_input.cached_path}`")

    return "\n".join(lines)


def _format_metrics_table(metrics: list[MetricResult]) -> str:
    """Format metric results as a compact summary table with detailed subsections.

    Displays a scannable summary table followed by detailed reasoning for each
    metric in dedicated subsections. This preserves full reasoning without
    truncation while keeping the summary readable.

    Parameters:
        metrics: List of MetricResult objects.

    Returns:
        A formatted markdown string with summary table and detail sections.
    """
    if not metrics:
        return ""

    # Compact summary table
    lines = [
        "| Metric | Score | Threshold | Status | Model | Eval Time |",
        "|--------|-------|-----------|--------|-------|-----------|",
    ]

    for metric in metrics:
        scale = metric.scale or "0-1"
        score_display = f"{metric.score}/{scale.split('-')[1]}"
        threshold = f"{metric.threshold}" if metric.threshold is not None else "-"
        status = "✅ PASS" if metric.passed else "❌ FAIL"
        model = metric.model_used or "-"
        eval_time = (
            f"{metric.evaluation_time_ms}ms" if metric.evaluation_time_ms else "-"
        )

        lines.append(
            f"| {metric.metric_name} | {score_display} | {threshold} | "
            f"{status} | {model} | {eval_time} |"
        )

    # Detailed subsections with full reasoning
    lines.append("")
    lines.append("##### Metric Details\n")

    for metric in metrics:
        scale = metric.scale or "0-1"
        score_display = f"{metric.score}/{scale.split('-')[1]}"
        status_emoji = "✅" if metric.passed else "❌"

        lines.append(f"**{metric.metric_name}** — {score_display} {status_emoji}")

        if metric.reasoning:
            # Format reasoning as blockquote, preserving line breaks
            reasoning_lines = metric.reasoning.strip().split("\n")
            for reasoning_line in reasoning_lines:
                lines.append(f"> {reasoning_line}")

        if metric.error:
            lines.append(f"> ⚠️ **Error:** {metric.error}")

        if metric.retry_count and metric.retry_count > 0:
            lines.append(f"> *(Retries: {metric.retry_count})*")

        lines.append("")  # Blank line between metrics

    return "\n".join(lines)


def _format_tool_usage(result: TestResult) -> str:
    """Format tool usage and validation as markdown.

    Parameters:
        result: The TestResult containing tool information.

    Returns:
        A formatted markdown string showing tool usage.
    """
    if not result.tool_calls and not result.expected_tools:
        return ""

    lines: list[str] = []

    if result.tool_calls:
        tools_str = ", ".join(f"`{tool}`" for tool in result.tool_calls)
        lines.append(f"**Tools Called:** {tools_str}")

    if result.expected_tools:
        expected_str = ", ".join(f"`{tool}`" for tool in result.expected_tools)
        lines.append(f"**Expected Tools:** {expected_str}")

    if result.tools_matched is not None:
        if result.tools_matched:
            lines.append("**Match Status:** ✅ Tools matched expected")
        else:
            lines.append("**Match Status:** ❌ Tools did not match expected")
            # Show missing tools
            if result.tool_calls and result.expected_tools:
                called_set = set(result.tool_calls)
                expected_set = set(result.expected_tools)
                missing = expected_set - called_set
                extra = called_set - expected_set
                if missing:
                    missing_str = ", ".join(f"`{tool}`" for tool in missing)
                    lines.append(f"  - Missing: {missing_str}")
                if extra:
                    extra_str = ", ".join(f"`{tool}`" for tool in extra)
                    lines.append(f"  - Extra: {extra_str}")

    return "\n".join(lines)
