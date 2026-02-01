"""Terminal detection utilities.

Provides functions for detecting terminal capabilities and output modes.
"""

import sys


def is_tty() -> bool:
    """Check if stdout is connected to a terminal.

    Used to determine whether to use rich formatting (colors, spinners)
    or plain text output suitable for CI/CD logs.

    Returns:
        True if stdout is a TTY (interactive terminal), False otherwise.
    """
    return sys.stdout.isatty()
