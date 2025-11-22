"""Validation utilities for chat runtime."""

from __future__ import annotations

import re

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


class ValidationPipeline:
    """Extensible validation pipeline for user input."""

    def __init__(self, max_length: int = 10_000) -> None:
        """Initialize the pipeline with a max length constraint."""
        self.max_length = max_length

    def validate(self, message: str | None) -> tuple[bool, str | None]:
        """Validate a message and return (is_valid, error_message)."""
        if message is None:
            return False, "Message cannot be empty."

        stripped = message.strip()
        if not stripped:
            return False, "Message cannot be empty."

        if len(stripped) > self.max_length:
            return False, "Message exceeds 10,000 characters."

        if CONTROL_CHAR_RE.search(stripped):
            return False, "Message contains control characters."

        try:
            stripped.encode("utf-8")
        except UnicodeEncodeError:
            return False, "Message must be valid UTF-8."

        return True, None


def sanitize_tool_output(output: str, max_length: int = 5_000) -> str:
    """Remove control/ANSI sequences and truncate long outputs."""
    cleaned = ANSI_ESCAPE_RE.sub("", output)
    cleaned = CONTROL_CHAR_RE.sub("", cleaned)

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "... (output truncated)"

    return cleaned
