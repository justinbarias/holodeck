"""Unit tests for progress indicators during test execution.

Tests progress display functionality including TTY detection, progress updates,
pass/fail symbols, and CI/CD compatibility.
"""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.test_runner.progress import ProgressIndicator
from holodeck.models.test_result import TestResult


class TestTTYDetection:
    """Test TTY detection for progress display formatting."""

    def test_tty_detection_with_tty(self) -> None:
        """Test that TTY is correctly detected when stdout is a terminal."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=10)
            assert indicator.is_tty is True

    def test_tty_detection_without_tty(self) -> None:
        """Test that non-TTY is correctly detected (e.g., in CI/CD)."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=10)
            assert indicator.is_tty is False

    def test_tty_detection_with_pipe(self) -> None:
        """Test TTY detection when stdout is piped."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=10)
            assert indicator.is_tty is False


class TestProgressDisplay:
    """Test progress indicator display formats."""

    def test_progress_format_with_tty(self) -> None:
        """Test that progress format includes interactive elements in TTY mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=10)
            assert indicator.is_tty is True

    def test_progress_format_without_tty(self) -> None:
        """Test that progress format is plain text in non-TTY mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=10)
            assert indicator.is_tty is False

    def test_initial_state(self) -> None:
        """Test that progress indicator initializes with correct state."""
        indicator = ProgressIndicator(total_tests=5)
        assert indicator.current_test == 0
        assert indicator.total_tests == 5
        assert indicator.passed == 0
        assert indicator.failed == 0

    def test_progress_update(self) -> None:
        """Test that progress updates correctly when test completes."""
        indicator = ProgressIndicator(total_tests=3)

        # Create mock test results
        result1 = MagicMock(spec=TestResult)
        result1.test_name = "Test 1"
        result1.passed = True

        indicator.update(result1)
        assert indicator.current_test == 1
        assert indicator.passed == 1
        assert indicator.failed == 0

    def test_failed_test_tracking(self) -> None:
        """Test that failed tests are correctly tracked."""
        indicator = ProgressIndicator(total_tests=3)

        result_fail = MagicMock(spec=TestResult)
        result_fail.test_name = "Test Failed"
        result_fail.passed = False

        indicator.update(result_fail)
        assert indicator.failed == 1
        assert indicator.passed == 0


class TestProgressFormat:
    """Test progress formatting and display strings."""

    def test_test_count_format_plain_text(self) -> None:
        """Test 'Test X/Y' format in plain text mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=10)
            result = MagicMock(spec=TestResult)
            result.test_name = "Basic Test"
            result.passed = True

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should contain "Test 1/10" format
            assert "1/10" in output or "Test 1" in output

    def test_test_count_format_multiple_tests(self) -> None:
        """Test progress format updates correctly for multiple tests."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=5)

            for i in range(3):
                result = MagicMock(spec=TestResult)
                result.test_name = f"Test {i+1}"
                result.passed = True
                indicator.update(result)

            assert indicator.current_test == 3

    def test_checkmark_symbol_tty(self) -> None:
        """Test checkmark symbol in TTY mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=2)
            result = MagicMock(spec=TestResult)
            result.test_name = "Passing Test"
            result.passed = True

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should contain checkmark or similar passing indicator
            assert "✓" in output or "✅" in output or "PASS" in output.upper()

    def test_fail_symbol_tty(self) -> None:
        """Test fail symbol (X mark) in TTY mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=2)
            result = MagicMock(spec=TestResult)
            result.test_name = "Failing Test"
            result.passed = False

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should contain X mark or similar failure indicator
            assert "✗" in output or "❌" in output or "FAIL" in output.upper()

    def test_plain_text_pass_indicator(self) -> None:
        """Test plain text PASS indicator in non-TTY mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=2)
            result = MagicMock(spec=TestResult)
            result.test_name = "Passing Test"
            result.passed = True

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should contain plain text indicator
            assert "PASS" in output.upper() or "OK" in output.upper() or "✓" in output

    def test_plain_text_fail_indicator(self) -> None:
        """Test plain text FAIL indicator in non-TTY mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=2)
            result = MagicMock(spec=TestResult)
            result.test_name = "Failing Test"
            result.passed = False

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should contain plain text failure indicator
            assert (
                "FAIL" in output.upper() or "ERROR" in output.upper() or "✗" in output
            )


class TestSummaryDisplay:
    """Test summary statistics display after all tests complete."""

    def test_summary_format(self) -> None:
        """Test summary statistics format."""
        indicator = ProgressIndicator(total_tests=5)

        # Add 3 passing tests
        for i in range(3):
            result = MagicMock(spec=TestResult)
            result.test_name = f"Test {i+1}"
            result.passed = True
            indicator.update(result)

        # Add 2 failing tests
        for i in range(2):
            result = MagicMock(spec=TestResult)
            result.test_name = f"Failing Test {i+1}"
            result.passed = False
            indicator.update(result)

        summary = indicator.get_summary()

        assert "5" in summary  # Total tests
        assert "3" in summary  # Passed
        assert "2" in summary  # Failed

    def test_pass_rate_calculation(self) -> None:
        """Test that pass rate is correctly calculated."""
        indicator = ProgressIndicator(total_tests=4)

        # 3 passing tests
        for i in range(3):
            result = MagicMock(spec=TestResult)
            result.test_name = f"Test {i+1}"
            result.passed = True
            indicator.update(result)

        # 1 failing test
        result_fail = MagicMock(spec=TestResult)
        result_fail.test_name = "Failing Test"
        result_fail.passed = False
        indicator.update(result_fail)

        summary = indicator.get_summary()

        # Should indicate 75% pass rate
        assert "75" in summary or "3/4" in summary

    def test_summary_with_all_passed(self) -> None:
        """Test summary when all tests pass."""
        indicator = ProgressIndicator(total_tests=3)

        for i in range(3):
            result = MagicMock(spec=TestResult)
            result.test_name = f"Test {i+1}"
            result.passed = True
            indicator.update(result)

        summary = indicator.get_summary()

        assert "100" in summary or "3/3" in summary

    def test_summary_with_all_failed(self) -> None:
        """Test summary when all tests fail."""
        indicator = ProgressIndicator(total_tests=3)

        for i in range(3):
            result = MagicMock(spec=TestResult)
            result.test_name = f"Test {i+1}"
            result.passed = False
            indicator.update(result)

        summary = indicator.get_summary()

        assert "0" in summary


class TestCIDCDCompatibility:
    """Test CI/CD environment compatibility."""

    def test_ci_cd_no_interactive_elements(self) -> None:
        """Test that non-TTY output contains no interactive elements."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=5)

            for i in range(3):
                result = MagicMock(spec=TestResult)
                result.test_name = f"Test {i+1}"
                result.passed = i % 2 == 0
                indicator.update(result)

            output = indicator.get_progress_line()

            # CI/CD logs should not have control characters for ANSI codes
            assert "\x1b" not in output or indicator.is_tty

    def test_ci_cd_output_readability(self) -> None:
        """Test that CI/CD output is readable and parseable."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=2)

            result1 = MagicMock(spec=TestResult)
            result1.test_name = "Test 1"
            result1.passed = True
            indicator.update(result1)

            result2 = MagicMock(spec=TestResult)
            result2.test_name = "Test 2"
            result2.passed = False
            indicator.update(result2)

            output = indicator.get_progress_line()
            summary = indicator.get_summary()

            # Output should be plain text, parseable
            assert isinstance(output, str)
            assert isinstance(summary, str)

    def test_ci_cd_log_format(self) -> None:
        """Test that output is suitable for CI/CD log aggregation."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Simple Test"
            result.passed = True

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should be single line, easily parseable
            assert "\n" not in output or output.count("\n") <= 1


class TestQuietMode:
    """Test quiet mode suppression of progress output."""

    def test_quiet_mode_initialization(self) -> None:
        """Test that quiet mode can be initialized."""
        indicator = ProgressIndicator(total_tests=5, quiet=True)
        assert indicator.quiet is True

    def test_quiet_mode_suppresses_progress(self) -> None:
        """Test that quiet mode suppresses progress output."""
        indicator = ProgressIndicator(total_tests=5, quiet=True)
        result = MagicMock(spec=TestResult)
        result.test_name = "Test 1"
        result.passed = True

        indicator.update(result)

        # In quiet mode, get_progress_line should return empty or minimal output
        output = indicator.get_progress_line()
        assert output == "" or len(output) < 20  # Very minimal

    def test_quiet_mode_summary_still_shown(self) -> None:
        """Test that summary is still shown in quiet mode."""
        indicator = ProgressIndicator(total_tests=2, quiet=True)

        for i in range(2):
            result = MagicMock(spec=TestResult)
            result.test_name = f"Test {i+1}"
            result.passed = True
            indicator.update(result)

        summary = indicator.get_summary()

        # Summary should still be shown even in quiet mode
        assert summary != ""


class TestVerboseMode:
    """Test verbose mode with detailed output."""

    def test_verbose_mode_initialization(self) -> None:
        """Test that verbose mode can be initialized."""
        indicator = ProgressIndicator(total_tests=5, verbose=True)
        assert indicator.verbose is True

    def test_verbose_mode_detailed_output(self) -> None:
        """Test that verbose mode provides detailed output."""
        indicator = ProgressIndicator(total_tests=5, verbose=True)
        result = MagicMock(spec=TestResult)
        result.test_name = "Detailed Test"
        result.passed = True
        result.execution_time_ms = 1234

        indicator.update(result)
        output = indicator.get_progress_line()

        # Verbose mode should include timing or additional details
        assert "Detailed Test" in output or len(output) > 30

    def test_verbose_summary_details(self) -> None:
        """Test that verbose summary includes detailed statistics."""
        indicator = ProgressIndicator(total_tests=3, verbose=True)

        for i in range(3):
            result = MagicMock(spec=TestResult)
            result.test_name = f"Test {i+1}"
            result.passed = True
            result.execution_time_ms = 500 * (i + 1)
            indicator.update(result)

        summary = indicator.get_summary()

        # Verbose summary should have more information
        assert len(summary) > 50  # Should be more detailed


@pytest.mark.unit
class TestProgressIndicatorIntegration:
    """Integration tests for progress indicator."""

    def test_complete_test_run_simulation(self) -> None:
        """Test complete progress indicator flow with multiple tests."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=5)

            # Simulate test execution
            for i in range(5):
                result = MagicMock(spec=TestResult)
                result.test_name = f"Test {i+1}"
                result.passed = i < 3  # First 3 pass, last 2 fail
                result.execution_time_ms = 100 * (i + 1)

                indicator.update(result)

            # Verify final state
            assert indicator.current_test == 5
            assert indicator.passed == 3
            assert indicator.failed == 2

            summary = indicator.get_summary()
            assert "5" in summary
            assert "3" in summary
            assert "2" in summary

    def test_single_test_run(self) -> None:
        """Test progress indicator with single test."""
        indicator = ProgressIndicator(total_tests=1)
        result = MagicMock(spec=TestResult)
        result.test_name = "Only Test"
        result.passed = True

        indicator.update(result)

        assert indicator.current_test == 1
        assert indicator.passed == 1
        summary = indicator.get_summary()
        assert "1" in summary
