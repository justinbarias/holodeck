"""Tests for validation pipeline and sanitizers."""

from __future__ import annotations

import re

from holodeck.lib.validation import ValidationPipeline, sanitize_tool_output


class TestValidationPipeline:
    """Runtime validation of user input."""

    def test_empty_message_rejected(self) -> None:
        """Whitespace-only messages fail validation."""
        pipeline = ValidationPipeline()
        is_valid, error = pipeline.validate("   ")
        assert not is_valid
        assert error
        assert "empty" in error.lower()

    def test_size_limit_enforced(self) -> None:
        """Messages over the limit are rejected."""
        pipeline = ValidationPipeline(max_length=10_000)
        is_valid, error = pipeline.validate("a" * 10_001)
        assert not is_valid
        assert "10,000" in (error or "")

    def test_control_characters_rejected(self) -> None:
        """Control characters cause validation failure."""
        pipeline = ValidationPipeline()
        is_valid, error = pipeline.validate("hello\x00world")
        assert not is_valid
        assert "control" in (error or "").lower()

    def test_invalid_utf8_rejected(self) -> None:
        """Lone surrogates are rejected as invalid UTF-8."""
        pipeline = ValidationPipeline()
        bad_message = "hi \udce2"
        is_valid, error = pipeline.validate(bad_message)
        assert not is_valid
        assert "utf-8" in (error or "").lower()

    def test_valid_message_passes(self) -> None:
        """Normal messages pass."""
        pipeline = ValidationPipeline()
        is_valid, error = pipeline.validate("hello world")
        assert is_valid
        assert error is None


class TestSanitizeToolOutput:
    """Sanitization for tool outputs."""

    def test_strips_ansi_and_truncates(self) -> None:
        """ANSI sequences removed and long output truncated."""
        raw = "\x1b[31mRED\x1b[0m" + "a" * 60
        cleaned = sanitize_tool_output(raw, max_length=30)
        assert "\x1b" not in cleaned
        assert len(cleaned) <= 30 + len("... (output truncated)")
        assert cleaned.endswith("... (output truncated)")

    def test_control_chars_removed(self) -> None:
        """Control characters removed from output."""
        raw = "ok\x00done"
        cleaned = sanitize_tool_output(raw)
        assert not re.search(r"\\x00", cleaned)
