"""Unit tests for holodeck.lib.ui.spinner module."""

import pytest

from holodeck.lib.ui.spinner import SpinnerMixin


class ConcreteSpinner(SpinnerMixin):
    """Concrete implementation of SpinnerMixin for testing."""

    def __init__(self) -> None:
        self._spinner_index = 0


@pytest.mark.unit
class TestSpinnerMixin:
    """Tests for SpinnerMixin class."""

    def test_spinner_chars_are_braille_characters(self) -> None:
        """Test SPINNER_CHARS contains expected braille characters."""
        assert len(SpinnerMixin.SPINNER_CHARS) == 10
        # Verify first and last are braille characters
        assert SpinnerMixin.SPINNER_CHARS[0] == "\u280b"  # ⠋
        assert SpinnerMixin.SPINNER_CHARS[-1] == "\u280f"  # ⠏

    def test_get_spinner_char_returns_first_char_initially(self) -> None:
        """Test get_spinner_char returns first character on initial call."""
        spinner = ConcreteSpinner()
        char = spinner.get_spinner_char()
        assert char == SpinnerMixin.SPINNER_CHARS[0]

    def test_get_spinner_char_advances_index(self) -> None:
        """Test get_spinner_char increments spinner index."""
        spinner = ConcreteSpinner()
        spinner.get_spinner_char()
        assert spinner._spinner_index == 1
        spinner.get_spinner_char()
        assert spinner._spinner_index == 2

    def test_get_spinner_char_cycles_through_all_chars(self) -> None:
        """Test get_spinner_char returns all characters in sequence."""
        spinner = ConcreteSpinner()
        chars = [spinner.get_spinner_char() for _ in range(10)]
        assert chars == SpinnerMixin.SPINNER_CHARS

    def test_get_spinner_char_wraps_around(self) -> None:
        """Test get_spinner_char wraps to first character after last."""
        spinner = ConcreteSpinner()
        # Get all 10 chars
        for _ in range(10):
            spinner.get_spinner_char()
        # Next should wrap to first
        char = spinner.get_spinner_char()
        assert char == SpinnerMixin.SPINNER_CHARS[0]
