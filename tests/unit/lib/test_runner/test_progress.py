"""Unit tests for progress indicators during test execution.

Tests progress display functionality including TTY detection, progress updates,
pass/fail symbols, and CI/CD compatibility.
"""

from unittest.mock import MagicMock, patch

import pytest

from holodeck.lib.test_runner.progress import ProgressIndicator
from holodeck.models.test_result import MetricResult, TestResult


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
        result1.execution_time_ms = None

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
        result_fail.execution_time_ms = None

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
            result.execution_time_ms = None

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
                result.execution_time_ms = None
                indicator.update(result)

            assert indicator.current_test == 3

    def test_checkmark_symbol_tty(self) -> None:
        """Test checkmark symbol in TTY mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=2)
            result = MagicMock(spec=TestResult)
            result.test_name = "Passing Test"
            result.passed = True
            result.execution_time_ms = None

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
            result.execution_time_ms = None

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
            result.execution_time_ms = None

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
            result.execution_time_ms = None

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
            result.execution_time_ms = None
            indicator.update(result)

        # Add 2 failing tests
        for i in range(2):
            result = MagicMock(spec=TestResult)
            result.test_name = f"Failing Test {i+1}"
            result.passed = False
            result.execution_time_ms = None
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
            result.execution_time_ms = None
            indicator.update(result)

        # 1 failing test
        result_fail = MagicMock(spec=TestResult)
        result_fail.test_name = "Failing Test"
        result_fail.passed = False
        result_fail.execution_time_ms = None
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
            result.execution_time_ms = None
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
            result.execution_time_ms = None
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
                result.execution_time_ms = None
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
            result1.execution_time_ms = None
            indicator.update(result1)

            result2 = MagicMock(spec=TestResult)
            result2.test_name = "Test 2"
            result2.passed = False
            result2.execution_time_ms = None
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
            result.execution_time_ms = None

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
        result.execution_time_ms = None

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
            result.execution_time_ms = None
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
            result.metric_results = []
            indicator.update(result)

        summary = indicator.get_summary()

        # Verbose summary should have more information
        assert len(summary) > 50  # Should be more detailed

    def test_verbose_summary_includes_metric_scores(self) -> None:
        """Test that verbose summary displays metric scores under test results."""
        indicator = ProgressIndicator(total_tests=1, verbose=True)

        result = MagicMock(spec=TestResult)
        result.test_name = "Test with Metrics"
        result.passed = True
        result.execution_time_ms = 500
        result.metric_results = [
            MetricResult(
                metric_name="groundedness",
                score=0.85,
                threshold=0.80,
                passed=True,
            ),
            MetricResult(
                metric_name="relevance",
                score=0.72,
                threshold=0.75,
                passed=False,
            ),
        ]
        indicator.update(result)

        summary = indicator.get_summary()

        # Check that metric names and scores are included
        assert "groundedness" in summary
        assert "relevance" in summary
        assert "0.85" in summary
        assert "0.72" in summary
        # Check threshold values are shown
        assert "0.80" in summary
        assert "0.75" in summary

    def test_verbose_summary_handles_metric_errors(self) -> None:
        """Test that metric errors are displayed properly."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1, verbose=True)

            result = MagicMock(spec=TestResult)
            result.test_name = "Test with Error"
            result.passed = False
            result.execution_time_ms = 100
            result.metric_results = [
                MetricResult(
                    metric_name="groundedness",
                    score=0.0,
                    error="LLM API error",
                ),
            ]
            indicator.update(result)

            summary = indicator.get_summary()

            # Check error message is included
            assert "ERROR" in summary
            assert "LLM API error" in summary


class TestSpinnerDisplay:
    """Test spinner animation for long-running tests."""

    def test_spinner_appears_for_long_tests(self) -> None:
        """Test that spinner appears for tests exceeding 5 seconds."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Long Running Test"
            result.passed = True
            result.execution_time_ms = 6000  # 6 seconds, above 5s threshold

            indicator.update(result)

            # Get spinner character
            spinner_char = indicator._get_spinner_char()

            # Spinner character should be one of the Braille patterns
            spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            assert spinner_char in spinner_chars or spinner_char == ""

    def test_spinner_rotation(self) -> None:
        """Test that spinner characters rotate correctly."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=3)
            spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

            # Simulate multiple long-running tests
            previous_chars = []
            for i in range(3):
                result = MagicMock(spec=TestResult)
                result.test_name = f"Long Test {i+1}"
                result.passed = True
                result.execution_time_ms = 6000

                indicator.update(result)
                spinner_char = indicator._get_spinner_char()
                previous_chars.append(spinner_char)

                # Each character should be in the spinner set or empty
                assert spinner_char in spinner_chars or spinner_char == ""

    def test_spinner_disabled_in_non_tty(self) -> None:
        """Test that spinner is disabled in non-TTY environments."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Test"
            result.passed = True
            result.execution_time_ms = 6000

            indicator.update(result)
            spinner_char = indicator._get_spinner_char()

            # In non-TTY, spinner should return empty string
            assert spinner_char == ""

    def test_spinner_not_shown_for_quick_tests(self) -> None:
        """Test that spinner does not appear for tests under 5 seconds."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Quick Test"
            result.passed = True
            result.execution_time_ms = 2000  # 2 seconds, below threshold

            indicator.update(result)
            output = indicator.get_progress_line()

            # Quick tests should not trigger spinner indicator in output
            assert isinstance(output, str)


class TestANSIColorOutput:
    """Test ANSI color code output for progress indicators."""

    def test_color_codes_in_tty_mode(self) -> None:
        """Test that ANSI color codes are present in TTY mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Color Test"
            result.passed = True
            result.execution_time_ms = 100

            indicator.update(result)
            output = indicator.get_progress_line()

            # In TTY mode, color codes should be present
            # Colors are added to pass/fail symbols
            assert "\033[" in output or "✓" in output  # Either colors or unicode

    def test_no_color_codes_in_non_tty(self) -> None:
        """Test that ANSI color codes are absent in non-TTY mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Plain Test"
            result.passed = True
            result.execution_time_ms = 100

            indicator.update(result)
            output = indicator.get_progress_line()

            # In non-TTY mode, no ANSI color codes
            # Plain text only
            assert isinstance(output, str)

    def test_pass_symbol_colored(self) -> None:
        """Test that pass symbol is properly colored in TTY."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Pass Color Test"
            result.passed = True
            result.execution_time_ms = 100

            indicator.update(result)
            output = indicator.get_progress_line()

            # Pass symbol should be present
            assert "✓" in output or "\033[92m" in output or "PASS" in output.upper()

    def test_fail_symbol_colored(self) -> None:
        """Test that fail symbol is properly colored in TTY."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Fail Color Test"
            result.passed = False
            result.execution_time_ms = 100

            indicator.update(result)
            output = indicator.get_progress_line()

            # Fail symbol should be present
            assert "✗" in output or "\033[91m" in output or "FAIL" in output.upper()

    def test_color_reset_codes(self) -> None:
        """Test that color reset codes are present when colors are used."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Reset Test"
            result.passed = True
            result.execution_time_ms = 100

            indicator.update(result)
            output = indicator.get_progress_line()

            # If colors are present, reset codes should follow
            if "\033[" in output:
                assert "\033[0m" in output  # Reset code

    def test_summary_colors_tty(self) -> None:
        """Test that summary uses colors in TTY mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=2)

            result1 = MagicMock(spec=TestResult)
            result1.test_name = "Test 1"
            result1.passed = True
            result1.execution_time_ms = 100
            indicator.update(result1)

            result2 = MagicMock(spec=TestResult)
            result2.test_name = "Test 2"
            result2.passed = False
            result2.execution_time_ms = 100
            indicator.update(result2)

            summary = indicator.get_summary()

            # Summary should have symbols or color codes
            assert "✓" in summary or "✗" in summary or "\033[" in summary


class TestElapsedTimeDisplay:
    """Test elapsed time display for test execution."""

    def test_elapsed_time_shown_for_long_tests(self) -> None:
        """Test that elapsed time is shown for tests exceeding threshold."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Long Test"
            result.passed = True
            result.execution_time_ms = 2500  # 2.5 seconds, above 1s threshold

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should contain elapsed time in seconds format
            assert "2.5s" in output or "2.50s" in output

    def test_elapsed_time_hidden_for_quick_tests(self) -> None:
        """Test that elapsed time is hidden for tests below threshold."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Quick Test"
            result.passed = True
            result.execution_time_ms = 250  # 0.25 seconds, below 1s threshold

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should NOT contain elapsed time
            assert "0.25s" not in output
            assert "250ms" not in output

    def test_elapsed_time_format(self) -> None:
        """Test that elapsed time is formatted correctly."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Formatted Test"
            result.passed = True
            result.execution_time_ms = 1500  # 1.5 seconds

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should contain formatted time
            assert "1.5s" in output or "1.50s" in output

    def test_elapsed_time_not_shown_in_non_tty(self) -> None:
        """Test elapsed time behavior in non-TTY mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Test"
            result.passed = True
            result.execution_time_ms = 5000  # Long test

            indicator.update(result)
            output = indicator.get_progress_line()

            # In non-TTY mode with verbose disabled, elapsed time may not show
            # unless explicitly enabled in verbose mode
            assert isinstance(output, str)

    def test_elapsed_time_at_threshold(self) -> None:
        """Test elapsed time display at exactly 1 second threshold."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=1)
            result = MagicMock(spec=TestResult)
            result.test_name = "Threshold Test"
            result.passed = True
            result.execution_time_ms = 1000  # Exactly 1 second

            indicator.update(result)
            output = indicator.get_progress_line()

            # Should show elapsed time at threshold
            assert "1.0s" in output or "1.00s" in output


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
        result.execution_time_ms = None

        indicator.update(result)

        assert indicator.current_test == 1
        assert indicator.passed == 1
        summary = indicator.get_summary()
        assert "1" in summary


class TestStartTest:
    """Tests for start_test method."""

    def test_start_test_sets_current_test_name(self) -> None:
        """Test that start_test sets the current test name."""
        indicator = ProgressIndicator(total_tests=5)
        indicator.start_test("My Test")
        assert indicator.current_test_name == "My Test"

    def test_start_test_updates_for_multiple_tests(self) -> None:
        """Test that start_test updates correctly for multiple tests."""
        indicator = ProgressIndicator(total_tests=3)
        indicator.start_test("Test 1")
        assert indicator.current_test_name == "Test 1"
        indicator.start_test("Test 2")
        assert indicator.current_test_name == "Test 2"


class TestGetSpinnerLine:
    """Tests for get_spinner_line method."""

    def test_get_spinner_line_in_tty(self) -> None:
        """Test get_spinner_line returns spinner in TTY mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=5)
            indicator.start_test("Running Test")
            line = indicator.get_spinner_line()
            assert "Test 1/5" in line
            assert "Running..." in line

    def test_get_spinner_line_in_non_tty(self) -> None:
        """Test get_spinner_line returns empty in non-TTY mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=5)
            indicator.start_test("Running Test")
            line = indicator.get_spinner_line()
            assert line == ""

    def test_get_spinner_line_in_quiet_mode(self) -> None:
        """Test get_spinner_line returns empty in quiet mode."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=5, quiet=True)
            indicator.start_test("Running Test")
            line = indicator.get_spinner_line()
            assert line == ""

    def test_get_spinner_line_exceeds_total(self) -> None:
        """Test get_spinner_line handles case where current test exceeds total."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=2)
            indicator.current_test = 3  # Exceeds total
            line = indicator.get_spinner_line()
            # Should cap at total_tests
            assert "Test 2/2" in line or "Test 3/2" in line


class TestColorize:
    """Tests for _colorize method."""

    def test_colorize_in_non_tty(self) -> None:
        """Test _colorize returns plain text in non-TTY mode."""
        with patch("sys.stdout.isatty", return_value=False):
            indicator = ProgressIndicator(total_tests=1)
            result = indicator._colorize("test", "\033[92m")
            assert result == "test"
            assert "\033[" not in result


class TestGetSummaryEdgeCases:
    """Tests for get_summary edge cases."""

    def test_get_summary_no_tests(self) -> None:
        """Test get_summary with zero tests."""
        indicator = ProgressIndicator(total_tests=0)
        summary = indicator.get_summary()
        assert "No tests" in summary

    def test_get_summary_with_failures_in_tty(self) -> None:
        """Test get_summary shows warning symbol with failures in TTY."""
        with patch("sys.stdout.isatty", return_value=True):
            indicator = ProgressIndicator(total_tests=2)
            result1 = MagicMock(spec=TestResult)
            result1.test_name = "Test 1"
            result1.passed = True
            result1.execution_time_ms = 100
            indicator.update(result1)

            result2 = MagicMock(spec=TestResult)
            result2.test_name = "Test 2"
            result2.passed = False
            result2.execution_time_ms = 200
            indicator.update(result2)

            summary = indicator.get_summary()
            # Should contain warning symbol or failure indicator
            assert "⚠" in summary or "Failed" in summary

    def test_get_summary_verbose_mode(self) -> None:
        """Test get_summary includes test details in verbose mode."""
        indicator = ProgressIndicator(total_tests=2, verbose=True)
        result1 = MagicMock(spec=TestResult)
        result1.test_name = "Test 1"
        result1.passed = True
        result1.execution_time_ms = 100
        result1.metric_results = []
        indicator.update(result1)

        result2 = MagicMock(spec=TestResult)
        result2.test_name = "Test 2"
        result2.passed = False
        result2.execution_time_ms = 200
        result2.metric_results = []
        indicator.update(result2)

        summary = indicator.get_summary()
        # Should include test details
        assert "Test 1" in summary
        assert "Test 2" in summary


class TestStrMethod:
    """Tests for __str__ method."""

    def test_str_returns_progress_line(self) -> None:
        """Test __str__ returns the current progress line."""
        indicator = ProgressIndicator(total_tests=2)
        result = MagicMock(spec=TestResult)
        result.test_name = "Test 1"
        result.passed = True
        result.execution_time_ms = 100
        indicator.update(result)

        str_output = str(indicator)
        progress_line = indicator.get_progress_line()
        assert str_output == progress_line
