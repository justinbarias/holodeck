"""Progress indicators for test execution.

Provides real-time progress display with TTY detection for interactive
environments and CI/CD-compatible plain text output.
"""

import sys
from datetime import datetime

from holodeck.models.test_result import TestResult


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

    def _get_pass_symbol(self) -> str:
        """Get appropriate pass symbol based on environment.

        Returns:
            checkmark for TTY, PASS for plain text
        """
        if self.is_tty:
            return "\u2713"  # ✓ checkmark
        return "PASS"

    def _get_fail_symbol(self) -> str:
        """Get appropriate fail symbol based on environment.

        Returns:
            X mark for TTY, FAIL for plain text
        """
        if self.is_tty:
            return "\u2717"  # ✗ X mark
        return "FAIL"

    def _format_test_status(self, result: "TestResult") -> str:
        """Format a single test result status line.

        Args:
            result: TestResult to format

        Returns:
            Formatted status string
        """
        symbol = self._get_pass_symbol() if result.passed else self._get_fail_symbol()
        status = symbol

        if self.verbose and result.execution_time_ms:
            status += f" ({result.execution_time_ms}ms)"

        if result.test_name:
            status += f" {result.test_name}"

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
            # TTY: Use symbols
            pass_symbol = "\u2713" if self.failed == 0 else "\u26a0"  # ✓ or ⚠
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

        # Verbose mode: show per-test details
        if self.verbose and self.test_results:
            summary_lines.append("")
            summary_lines.append("Test Details:")
            for i, result in enumerate(self.test_results, 1):
                check = "\u2713" if result.passed else "\u2717"  # ✓ or ✗
                name = result.test_name or f"Test {i}"
                timing = (
                    f" ({result.execution_time_ms}ms)"
                    if result.execution_time_ms
                    else ""
                )
                summary_lines.append(f"  {check} {name}{timing}")

        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)

    def __str__(self) -> str:
        """String representation of progress indicator.

        Returns:
            Current progress line
        """
        return self.get_progress_line()
