"""Unit tests for message validation."""

from __future__ import annotations

from holodeck.chat.message import MessageValidator


class TestMessageValidator:
    """Unit tests for MessageValidator."""

    def test_validator_initialization(self) -> None:
        """Validator initializes with default max_length."""
        validator = MessageValidator()
        assert validator is not None

    def test_validator_custom_max_length(self) -> None:
        """Validator accepts custom max_length."""
        validator = MessageValidator(max_length=5000)
        assert validator is not None

    def test_empty_message_rejected(self) -> None:
        """Empty string messages fail validation."""
        validator = MessageValidator()
        is_valid, error = validator.validate("")
        assert not is_valid
        assert error is not None
        assert "empty" in error.lower()

    def test_whitespace_only_message_rejected(self) -> None:
        """Whitespace-only messages fail validation."""
        validator = MessageValidator()
        is_valid, error = validator.validate("   \t\n  ")
        assert not is_valid
        assert error is not None
        assert "empty" in error.lower()

    def test_none_message_rejected(self) -> None:
        """None messages fail validation."""
        validator = MessageValidator()
        is_valid, error = validator.validate(None)  # type: ignore
        assert not is_valid
        assert error is not None

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

    def test_control_characters_rejected(self) -> None:
        """Messages with control characters are rejected."""
        validator = MessageValidator()
        is_valid, error = validator.validate("hello\x00world")
        assert not is_valid
        assert error is not None
        assert "control" in error.lower()

    def test_null_byte_rejected(self) -> None:
        """Messages with null bytes are rejected."""
        validator = MessageValidator()
        is_valid, error = validator.validate("test\x00message")
        assert not is_valid

    def test_tab_character_allowed(self) -> None:
        """Tab characters (outside message) are handled correctly."""
        validator = MessageValidator()
        # Tabs in the middle of content should be stripped at edges
        is_valid, error = validator.validate("hello\tworld")
        assert is_valid
        assert error is None

    def test_newline_in_message_allowed(self) -> None:
        """Newlines within message content are allowed."""
        validator = MessageValidator()
        is_valid, error = validator.validate("hello\nworld")
        assert is_valid
        assert error is None

    def test_unicode_message_accepted(self) -> None:
        """Valid UTF-8 Unicode messages are accepted."""
        validator = MessageValidator()
        is_valid, error = validator.validate("Hello 世界 🌍")
        assert is_valid
        assert error is None

    def test_valid_simple_message(self) -> None:
        """Normal messages pass validation."""
        validator = MessageValidator()
        is_valid, error = validator.validate("hello world")
        assert is_valid
        assert error is None

    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Validation strips leading/trailing whitespace."""
        validator = MessageValidator()
        # Message with leading/trailing whitespace should be valid
        # if the trimmed content is non-empty
        is_valid, error = validator.validate("  hello world  ")
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
