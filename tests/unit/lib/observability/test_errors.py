"""Unit tests for observability custom exceptions.

TDD: These tests are written FIRST, before implementation.
All tests should FAIL until the exceptions are implemented.

Task: T030 - Unit tests for ObservabilityError, ObservabilityConfigError
"""

import pytest

from holodeck.lib.errors import HoloDeckError


@pytest.mark.unit
class TestObservabilityError:
    """Tests for ObservabilityError exception."""

    def test_is_holodeck_error_subclass(self) -> None:
        """Test that ObservabilityError inherits from HoloDeckError."""
        from holodeck.lib.observability.errors import ObservabilityError

        error = ObservabilityError("Test error")
        assert isinstance(error, HoloDeckError)
        assert isinstance(error, Exception)

    def test_stores_message_attribute(self) -> None:
        """Test that ObservabilityError stores message attribute."""
        from holodeck.lib.observability.errors import ObservabilityError

        msg = "Something went wrong with telemetry"
        error = ObservabilityError(msg)
        assert error.message == msg

    def test_string_representation(self) -> None:
        """Test ObservabilityError string contains message."""
        from holodeck.lib.observability.errors import ObservabilityError

        msg = "Failed to initialize tracer"
        error = ObservabilityError(msg)
        assert msg in str(error)

    def test_can_be_raised_and_caught(self) -> None:
        """Test ObservabilityError can be raised and caught."""
        from holodeck.lib.observability.errors import ObservabilityError

        with pytest.raises(ObservabilityError) as exc_info:
            raise ObservabilityError("Test raise")

        assert "Test raise" in str(exc_info.value)

    def test_can_be_caught_as_holodeck_error(self) -> None:
        """Test ObservabilityError can be caught as HoloDeckError."""
        from holodeck.lib.observability.errors import ObservabilityError

        with pytest.raises(HoloDeckError):
            raise ObservabilityError("Test")


@pytest.mark.unit
class TestObservabilityConfigError:
    """Tests for ObservabilityConfigError exception."""

    def test_is_observability_error_subclass(self) -> None:
        """Test that ObservabilityConfigError inherits from ObservabilityError."""
        from holodeck.lib.observability.errors import (
            ObservabilityConfigError,
            ObservabilityError,
        )

        error = ObservabilityConfigError("field", "message")
        assert isinstance(error, ObservabilityError)
        assert isinstance(error, HoloDeckError)

    def test_stores_field_attribute(self) -> None:
        """Test that ObservabilityConfigError stores field attribute."""
        from holodeck.lib.observability.errors import ObservabilityConfigError

        error = ObservabilityConfigError("exporters.otlp.endpoint", "Invalid URL")
        assert error.field == "exporters.otlp.endpoint"

    def test_stores_message_attribute(self) -> None:
        """Test that ObservabilityConfigError stores message attribute."""
        from holodeck.lib.observability.errors import ObservabilityConfigError

        error = ObservabilityConfigError("field", "Invalid value")
        assert "Invalid value" in error.message

    def test_string_includes_field_and_message(self) -> None:
        """Test error string includes both field and message."""
        from holodeck.lib.observability.errors import ObservabilityConfigError

        error = ObservabilityConfigError(
            "traces.sample_rate", "Must be between 0 and 1"
        )
        error_str = str(error)
        assert "traces.sample_rate" in error_str
        assert "Must be between 0 and 1" in error_str

    def test_with_nested_field_path(self) -> None:
        """Test with deeply nested field path."""
        from holodeck.lib.observability.errors import ObservabilityConfigError

        field = "exporters.azure_monitor.connection_string"
        msg = "Required when enabled"
        error = ObservabilityConfigError(field, msg)
        assert field in str(error)
        assert msg in str(error)

    def test_can_be_caught_as_observability_error(self) -> None:
        """Test ObservabilityConfigError can be caught as ObservabilityError."""
        from holodeck.lib.observability.errors import (
            ObservabilityConfigError,
            ObservabilityError,
        )

        with pytest.raises(ObservabilityError):
            raise ObservabilityConfigError("field", "message")

    def test_can_be_caught_as_holodeck_error(self) -> None:
        """Test ObservabilityConfigError can be caught as HoloDeckError."""
        from holodeck.lib.observability.errors import ObservabilityConfigError

        with pytest.raises(HoloDeckError):
            raise ObservabilityConfigError("field", "message")
