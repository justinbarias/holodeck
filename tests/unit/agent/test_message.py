"""Unit tests for message validation."""

from __future__ import annotations

import pytest

from holodeck.chat.message import MessageValidator


class TestMessageValidator:
    """Unit tests for MessageValidator."""

    def test_validator_custom_max_length(self) -> None:
        """Validator accepts custom max_length."""
        validator = MessageValidator(max_length=5000)
        assert validator is not None

    @pytest.mark.parametrize(
        "message, error_substr",
        [
            pytest.param("", "empty", id="empty_string"),
            pytest.param("   \t\n  ", "empty", id="whitespace_only"),
            pytest.param(None, None, id="none_value"),
            pytest.param("hello\x00world", "control", id="control_char_null_embedded"),
            pytest.param("test\x00message", None, id="null_byte"),
        ],
    )
    def test_invalid_message_rejected(
        self, message: str | None, error_substr: str | None
    ) -> None:
        """Invalid messages fail validation."""
        validator = MessageValidator()
        is_valid, error = validator.validate(message)  # type: ignore[arg-type]
        assert not is_valid
        assert error is not None
        if error_substr is not None:
            assert error_substr in error.lower()

    def test_message_exceeds_size_limit(self) -> None:
        """Messages exceeding max_length are rejected."""
        validator = MessageValidator(max_length=100)
        is_valid, error = validator.validate("a" * 101)
        assert not is_valid
        assert error is not None
        assert "exceed" in error.lower() or "limit" in error.lower()

    def test_message_at_size_limit_accepted(self) -> None:
        """Messages at exactly max_length are accepted."""
        validator = MessageValidator(max_length=100)
        is_valid, error = validator.validate("a" * 100)
        assert is_valid
        assert error is None

    def test_default_10k_limit(self) -> None:
        """Default max_length is 10,000 characters."""
        validator = MessageValidator()
        is_valid, error = validator.validate("a" * 10_001)
        assert not is_valid
        assert error is not None

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param("hello\tworld", id="tab_character"),
            pytest.param("hello\nworld", id="newline_character"),
            pytest.param("Hello 世界 🌍", id="unicode"),
            pytest.param("hello world", id="simple_message"),
            pytest.param("  hello world  ", id="leading_trailing_whitespace"),
        ],
    )
    def test_valid_message_accepted(self, message: str) -> None:
        """Valid messages pass validation."""
        validator = MessageValidator()
        is_valid, error = validator.validate(message)
        assert is_valid
        assert error is None

    def test_return_tuple_structure(self) -> None:
        """Validate returns (bool, str|None) tuple."""
        validator = MessageValidator()
        result = validator.validate("test")
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_valid, error = result
        assert isinstance(is_valid, bool)
        assert error is None or isinstance(error, str)
