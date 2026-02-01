"""Unit tests for holodeck.lib.ui.colors module."""

import pytest

from holodeck.lib.ui.colors import ANSIColors, colorize


@pytest.mark.unit
class TestANSIColors:
    """Tests for ANSIColors class."""

    def test_green_is_correct_escape_code(self) -> None:
        """Test GREEN contains correct ANSI escape code."""
        assert ANSIColors.GREEN == "\033[92m"

    def test_red_is_correct_escape_code(self) -> None:
        """Test RED contains correct ANSI escape code."""
        assert ANSIColors.RED == "\033[91m"

    def test_yellow_is_correct_escape_code(self) -> None:
        """Test YELLOW contains correct ANSI escape code."""
        assert ANSIColors.YELLOW == "\033[93m"

    def test_reset_is_correct_escape_code(self) -> None:
        """Test RESET contains correct ANSI escape code."""
        assert ANSIColors.RESET == "\033[0m"


@pytest.mark.unit
class TestColorize:
    """Tests for colorize function."""

    def test_colorize_applies_color_when_tty(self) -> None:
        """Test colorize wraps text with color codes when force_tty=True."""
        result = colorize("test", ANSIColors.GREEN, force_tty=True)
        assert result == f"{ANSIColors.GREEN}test{ANSIColors.RESET}"

    def test_colorize_returns_plain_text_when_not_tty(self) -> None:
        """Test colorize returns plain text when force_tty=False."""
        result = colorize("test", ANSIColors.GREEN, force_tty=False)
        assert result == "test"

    def test_colorize_with_red_color(self) -> None:
        """Test colorize works with RED color."""
        result = colorize("error", ANSIColors.RED, force_tty=True)
        assert result == f"{ANSIColors.RED}error{ANSIColors.RESET}"

    def test_colorize_with_yellow_color(self) -> None:
        """Test colorize works with YELLOW color."""
        result = colorize("warning", ANSIColors.YELLOW, force_tty=True)
        assert result == f"{ANSIColors.YELLOW}warning{ANSIColors.RESET}"

    def test_colorize_preserves_empty_string(self) -> None:
        """Test colorize handles empty string correctly."""
        result_tty = colorize("", ANSIColors.GREEN, force_tty=True)
        result_no_tty = colorize("", ANSIColors.GREEN, force_tty=False)
        assert result_tty == f"{ANSIColors.GREEN}{ANSIColors.RESET}"
        assert result_no_tty == ""
