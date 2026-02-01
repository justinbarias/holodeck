"""ANSI color utilities for terminal output.

Provides color constants and helper functions for colorized terminal output
with graceful degradation in non-TTY environments.
"""

from holodeck.lib.ui.terminal import is_tty


class ANSIColors:
    """ANSI color escape codes for terminal output.

    Provides standard ANSI color codes that can be used with the colorize()
    function or applied directly to strings.

    Attributes:
        GREEN: Bright green color (for success indicators).
        RED: Bright red color (for failure indicators).
        YELLOW: Bright yellow color (for warnings).
        RESET: Reset code to restore default terminal color.
    """

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


def colorize(text: str, color: str, force_tty: bool | None = None) -> str:
    """Apply ANSI color codes to text if in TTY mode.

    Wraps text with the specified color code and reset sequence,
    but only if stdout is connected to a terminal. This ensures
    clean output in CI/CD logs and file redirects.

    Args:
        text: Text to colorize.
        color: ANSI color code to apply (e.g., ANSIColors.GREEN).
        force_tty: Override TTY detection (for testing). None uses auto-detection.

    Returns:
        Colorized text if in TTY mode, plain text otherwise.
    """
    use_colors = force_tty if force_tty is not None else is_tty()
    if not use_colors:
        return text
    return f"{color}{text}{ANSIColors.RESET}"
