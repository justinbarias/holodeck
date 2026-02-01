"""UI utilities for terminal-based progress display.

This module provides shared utilities for terminal interaction, including:
- TTY detection for adaptive output formatting
- Spinner animation for progress indication
- ANSI color support with graceful degradation

These utilities are used by both chat and test_runner modules.
"""

from holodeck.lib.ui.colors import ANSIColors, colorize
from holodeck.lib.ui.spinner import SpinnerMixin
from holodeck.lib.ui.terminal import is_tty

__all__ = [
    "ANSIColors",
    "SpinnerMixin",
    "colorize",
    "is_tty",
]
