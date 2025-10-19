"""Tests for custom exception hierarchy in holodeck.lib.errors."""

from holodeck.lib.errors import (
    ConfigError,
    FileNotFoundError,
    HoloDeckError,
    ValidationError,
)


class TestHoloDeckError:
    """Tests for base HoloDeckError exception."""

    def test_holodeck_error_creates_with_message(self) -> None:
        """Test that HoloDeckError can be created with a message."""
        error = HoloDeckError("Test error message")
        assert str(error) == "Test error message"

    def test_holodeck_error_is_exception(self) -> None:
        """Test that HoloDeckError is an Exception subclass."""
        error = HoloDeckError("Test")
        assert isinstance(error, Exception)

    def test_holodeck_error_preserves_message(self) -> None:
        """Test that HoloDeckError preserves the original message."""
        msg = "Detailed error description"
        error = HoloDeckError(msg)
        assert error.args[0] == msg


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_config_error_formats_message_with_field(self) -> None:
        """Test that ConfigError formats messages with field information."""
        error = ConfigError("name", "Field 'name' is required")
        assert "name" in str(error)
        assert "required" in str(error).lower()

    def test_config_error_is_holodeck_error(self) -> None:
        """Test that ConfigError is an HoloDeckError subclass."""
        error = ConfigError("test_field", "Test message")
        assert isinstance(error, HoloDeckError)

    def test_config_error_includes_field_name(self) -> None:
        """Test that ConfigError includes field name in error message."""
        field = "temperature"
        error = ConfigError(field, "Invalid value")
        assert field in str(error)

    def test_config_error_with_multiline_message(self) -> None:
        """Test ConfigError handles multiline messages."""
        msg = "Line 1\nLine 2\nLine 3"
        error = ConfigError("field", msg)
        assert "Line 1" in str(error)
        assert "Line 2" in str(error)


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_with_field_details(self) -> None:
        """Test ValidationError includes field details."""
        error = ValidationError(
            field="model.temperature",
            message="Value must be between 0 and 2",
            expected="float (0-2.0)",
            actual="3.5",
        )
        error_str = str(error)
        assert "model.temperature" in error_str
        assert "0-2" in error_str or "0 and 2" in error_str

    def test_validation_error_is_holodeck_error(self) -> None:
        """Test that ValidationError is an HoloDeckError subclass."""
        error = ValidationError(
            field="test",
            message="Test message",
            expected="str",
            actual="int",
        )
        assert isinstance(error, HoloDeckError)

    def test_validation_error_formats_nested_field(self) -> None:
        """Test ValidationError formats nested field paths."""
        error = ValidationError(
            field="tools[0].parameters.weight",
            message="Required field missing",
            expected="float",
            actual="null",
        )
        assert "tools[0].parameters.weight" in str(error)

    def test_validation_error_with_expected_actual(self) -> None:
        """Test ValidationError includes expected vs actual values."""
        expected = "one of: vectorstore, function, mcp, prompt"
        actual = "database"
        error = ValidationError(
            field="tool.type",
            message="Invalid tool type",
            expected=expected,
            actual=actual,
        )
        error_str = str(error)
        assert "Expected" in error_str or expected in error_str
        assert actual in error_str


class TestFileNotFoundError:
    """Tests for FileNotFoundError exception."""

    def test_file_not_found_error_with_path(self) -> None:
        """Test FileNotFoundError includes file path."""
        path = "/path/to/agent.yaml"
        error = FileNotFoundError(path, "Agent configuration file not found")
        assert path in str(error)

    def test_file_not_found_error_is_holodeck_error(self) -> None:
        """Test that FileNotFoundError is an HoloDeckError subclass."""
        error = FileNotFoundError("missing.yaml", "Not found")
        assert isinstance(error, HoloDeckError)

    def test_file_not_found_error_suggests_path(self) -> None:
        """Test FileNotFoundError includes suggestion for path."""
        path = "./missing_instructions.md"
        error = FileNotFoundError(
            path,
            f"File not found: {path}. Please check the file path is correct.",
        )
        assert path in str(error)
        assert "check" in str(error).lower() or "path" in str(error).lower()

    def test_file_not_found_error_with_suggestion(self) -> None:
        """Test FileNotFoundError can include helpful suggestion."""
        path = "data/faqs.md"
        suggestion = "Relative paths should be relative to agent.yaml directory"
        error = FileNotFoundError(path, f"Not found: {path}. {suggestion}")
        error_str = str(error)
        assert path in error_str
        assert "relative" in error_str.lower() or "agent.yaml" in error_str
