"""Unit tests for holodeck.lib.ui.terminal module."""

from unittest.mock import patch

import pytest

from holodeck.lib.ui.terminal import is_tty


@pytest.mark.unit
class TestIsTty:
    """Tests for is_tty function."""

    def test_is_tty_returns_true_when_stdout_is_tty(self) -> None:
        """Test is_tty returns True when stdout.isatty() returns True."""
        with patch("sys.stdout.isatty", return_value=True):
            assert is_tty() is True

    def test_is_tty_returns_false_when_stdout_is_not_tty(self) -> None:
        """Test is_tty returns False when stdout.isatty() returns False."""
        with patch("sys.stdout.isatty", return_value=False):
            assert is_tty() is False
