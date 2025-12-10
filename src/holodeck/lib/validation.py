"""Validation utilities for HoloDeck.

This module provides shared validation functions and constants used across
the codebase, including agent name validation, chat input validation, and
tool output sanitization.
"""

from __future__ import annotations

import re

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# Agent name validation constants
AGENT_NAME_MAX_LENGTH = 64
AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def validate_agent_name(name: str) -> str:
    """Validate agent name format.

    Agent names must:
    - Not be empty
    - Be 64 characters or less
    - Start with a letter (a-z, A-Z)
    - Contain only alphanumeric characters, hyphens, and underscores

    Args:
        name: The agent name to validate

    Returns:
        The validated agent name (unchanged if valid)

    Raises:
        ValueError: If agent name is invalid
    """
    if not name:
        raise ValueError("Agent name cannot be empty")
    if len(name) > AGENT_NAME_MAX_LENGTH:
        raise ValueError(
            f"Agent name must be {AGENT_NAME_MAX_LENGTH} characters or less"
        )
    if not name[0].isalpha():
        raise ValueError("Agent name must start with a letter")
    if not AGENT_NAME_PATTERN.match(name):
        raise ValueError(
            "Agent name must start with a letter and contain only "
            "alphanumeric characters, hyphens, and underscores"
        )
    return name


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
