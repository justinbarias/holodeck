"""Progress indicators for test execution.

Provides real-time progress display with TTY detection for interactive
environments and CI/CD-compatible plain text output.
"""

import sys
from datetime import datetime

from holodeck.models.test_result import MetricResult, TestResult


class ProgressIndicator:
    """Displays progress during test execution with TTY-aware formatting.

    Detects whether stdout is a terminal (TTY) and adjusts output accordingly:
    - TTY (interactive): Colored symbols, spinners, ANSI formatting
    - Non-TTY (CI/CD): Plain text, compatible with log aggregation systems

    Attributes:
        total_tests: Total number of tests to execute
        current_test: Number of tests completed so far
        passed: Number of tests that passed
        failed: Number of tests that failed
        is_tty: Whether stdout is a terminal
        quiet: Suppress progress output (only show summary)
        verbose: Show detailed output including timing
    """

    # ANSI color codes
    _COLOR_GREEN = "\033[92m"
    _COLOR_RED = "\033[91m"
    _COLOR_RESET = "\033[0m"

    # Spinner characters for long-running tests
    _SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(
        self,
        total_tests: int,
        quiet: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize progress indicator.

        Args:
            total_tests: Total number of tests to execute
            quiet: If True, suppress progress output (only show summary)
            verbose: If True, show detailed output with timing information
        """
        self.total_tests = total_tests
        self.current_test = 0
        self.passed = 0
        self.failed = 0
        self.quiet = quiet
        self.verbose = verbose
        self.test_results: list[TestResult] = []
        self.start_time = datetime.now()
        self._spinner_index = 0

    @property
    def is_tty(self) -> bool:
        """Check if stdout is connected to a terminal.

        Returns:
            True if stdout is a TTY (interactive terminal), False otherwise
        """
        return sys.stdout.isatty()

    def update(self, result: "TestResult") -> None:
        """Update progress with a completed test result.

        Args:
            result: TestResult instance from a completed test
        """
        self.current_test += 1
        self.test_results.append(result)

        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    def _colorize(self, text: str, color: str) -> str:
        """Apply ANSI color codes to text if in TTY mode.

        Args:
            text: Text to colorize
            color: ANSI color code

        Returns:
            Colorized text if in TTY, plain text otherwise
        """
        if not self.is_tty:
            return text
        return f"{color}{text}{self._COLOR_RESET}"

    def _get_spinner_char(self) -> str:
        """Get current spinner character and advance rotation.

        Returns:
            Current spinner character
        """
        if not self.is_tty:
            return ""
        char = self._SPINNER_CHARS[self._spinner_index % len(self._SPINNER_CHARS)]
        self._spinner_index += 1
        return char

    def start_test(self, test_name: str) -> None:
        """Mark a test as started.

        Args:
            test_name: Name of the test starting
        """
        self.current_test_name = test_name

    def get_spinner_line(self) -> str:
        """Get current spinner line for running test.

        Returns:
            Formatted spinner string (e.g. "⠋ Test 1/5: Running...")
        """
        if not self.is_tty or self.quiet:
            return ""

        spinner = self._get_spinner_char()
        next_test = self.current_test + 1

        # Ensure we don't exceed total tests in display
        if next_test > self.total_tests:
            next_test = self.total_tests

        return f"{spinner} Test {next_test}/{self.total_tests}: Running..."

    def _is_long_running(self, execution_time_ms: int | None) -> bool:
        """Check if test execution time exceeds long-running threshold.

        Args:
            execution_time_ms: Execution time in milliseconds

        Returns:
            True if execution time >= 5 seconds
        """
        if execution_time_ms is None:
            return False
        return execution_time_ms >= 5000

    def _get_pass_symbol(self) -> str:
        """Get appropriate pass symbol based on environment.

        Returns:
            Colored checkmark for TTY, PASS for plain text
        """
        if self.is_tty:
            return self._colorize("\u2713", self._COLOR_GREEN)  # ✓ checkmark
        return "PASS"

    def _get_fail_symbol(self) -> str:
        """Get appropriate fail symbol based on environment.

        Returns:
            Colored X mark for TTY, FAIL for plain text
        """
        if self.is_tty:
            return self._colorize("\u2717", self._COLOR_RED)  # ✗ X mark
        return "FAIL"

    def _should_show_elapsed_time(self, execution_time_ms: int | None) -> bool:
        """Check if elapsed time should be displayed.

        Args:
            execution_time_ms: Execution time in milliseconds

        Returns:
            True if elapsed time should be shown (>1 second), False otherwise
        """
        if execution_time_ms is None:
            return False
        return execution_time_ms >= 1000

    def _format_metric_symbol(self, passed: bool | None) -> str:
        """Get symbol for metric pass/fail status.

        Args:
            passed: Whether the metric passed its threshold (None if no threshold)

        Returns:
            Colored symbol for TTY, plain text for non-TTY
        """
        if passed is None:
            # No threshold defined, show neutral indicator
            return self._colorize("-", self._COLOR_RESET) if self.is_tty else "-"
        elif passed:
            return (
                self._colorize("\u2713", self._COLOR_GREEN) if self.is_tty else "PASS"
            )
        else:
            return self._colorize("\u2717", self._COLOR_RED) if self.is_tty else "FAIL"

    def _format_metric_score(self, metric: MetricResult) -> str:
        """Format metric score with threshold information.

        Args:
            metric: MetricResult to format

        Returns:
            Formatted score string (e.g., "0.85 (threshold: 0.80)")
        """
        if metric.error:
            return self._colorize(f"ERROR: {metric.error}", self._COLOR_RED)

        score_str = f"{metric.score:.2f}"
        if metric.threshold is not None:
            score_str += f" (threshold: {metric.threshold:.2f})"
        return score_str

    def _format_test_status(self, result: "TestResult") -> str:
        """Format a single test result status line.

        Args:
            result: TestResult to format

        Returns:
            Formatted status string
        """
        symbol = self._get_pass_symbol() if result.passed else self._get_fail_symbol()
        status = symbol

        if result.test_name:
            status += f" {result.test_name}"

        # Show elapsed time for long tests (>1s) or when in verbose mode
        if result.execution_time_ms:
            if self._should_show_elapsed_time(result.execution_time_ms):
                # Convert ms to seconds with 2 decimal places
                elapsed_seconds = result.execution_time_ms / 1000.0
                status += f" ({elapsed_seconds:.2f}s)"
            elif self.verbose:
                # In verbose mode, always show timing
                status += f" ({result.execution_time_ms}ms)"

        return status

    def get_progress_line(self) -> str:
        """Get current progress display line.

        Returns:
            Progress string showing current test count and status
            Empty string if quiet mode is enabled
        """
        if self.quiet and self.current_test < self.total_tests:
            return ""

        if self.current_test == 0:
            return ""

        # Get the last test result
        last_result = self.test_results[-1]

        # Format: "Test X/Y: [symbol] TestName"
        progress = f"Test {self.current_test}/{self.total_tests}"

        if self.is_tty:
            status = self._format_test_status(last_result)
            return f"{progress}: {status}"
        else:
            # Plain text format for CI/CD
            status = self._format_test_status(last_result)
            return f"[{progress}] {status}"

    def get_summary(self) -> str:
        """Get summary statistics for all completed tests.

        Returns:
            Formatted summary string with pass/fail counts and rate
        """
        if self.total_tests == 0:
            return "No tests to execute"

        # Calculate pass rate
        pass_rate = (self.passed / self.total_tests) * 100

        # Format summary
        summary_lines: list[str] = []
        summary_lines.append("")
        summary_lines.append("=" * 60)

        if self.is_tty:
            # TTY: Use colored symbols
            if self.failed == 0:
                pass_symbol = self._colorize("\u2713", self._COLOR_GREEN)  # ✓
            else:
                pass_symbol = self._colorize("\u26a0", self._COLOR_RED)  # ⚠
            summary_lines.append(
                f"{pass_symbol} Test Results: {self.passed}/{self.total_tests} passed "
                f"({pass_rate:.1f}%)"
            )
        else:
            # Plain text
            summary_lines.append(
                f"Test Results: {self.passed}/{self.total_tests} passed "
                f"({pass_rate:.1f}%)"
            )

        if self.failed > 0:
            summary_lines.append(f"  Failed: {self.failed}")

        # Add timing if available
        if hasattr(self, "start_time") and self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            summary_lines.append(f"  Duration: {elapsed:.2f}s")

        # Verbose mode: show per-test details with metrics
        if self.verbose and self.test_results:
            summary_lines.append("")
            summary_lines.append("Test Details:")
            for i, result in enumerate(self.test_results, 1):
                if result.passed:
                    check = self._colorize("\u2713", self._COLOR_GREEN)  # ✓
                else:
                    check = self._colorize("\u2717", self._COLOR_RED)  # ✗
                name = result.test_name or f"Test {i}"
                timing = (
                    f" ({result.execution_time_ms}ms)"
                    if result.execution_time_ms
                    else ""
                )
                summary_lines.append(f"  {check} {name}{timing}")

                # Display metric results for each test
                if result.metric_results:
                    for metric in result.metric_results:
                        metric_symbol = self._format_metric_symbol(metric.passed)
                        score_str = self._format_metric_score(metric)
                        summary_lines.append(
                            f"      {metric_symbol} {metric.metric_name}: {score_str}"
                        )
                        # Show reasoning if available (DeepEval metrics only)
                        if metric.reasoning:
                            summary_lines.append(
                                f"        Reasoning: {metric.reasoning}"
                            )

        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)

    def __str__(self) -> str:
        """String representation of progress indicator.

        Returns:
            Current progress line
        """
        return self.get_progress_line()
