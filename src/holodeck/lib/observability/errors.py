"""Custom exceptions for observability module.

Follows the HoloDeck error hierarchy pattern from holodeck.lib.errors.

Task: T039 - Implement ObservabilityError, ObservabilityConfigError exceptions
"""

from holodeck.lib.errors import HoloDeckError


class ObservabilityError(HoloDeckError):
    """Base exception for all observability-related errors.

    All observability-specific exceptions inherit from this class,
    enabling centralized exception handling for telemetry operations.

    Attributes:
        message: Human-readable error message
    """

    def __init__(self, message: str) -> None:
        """Initialize ObservabilityError with message.

        Args:
            message: Descriptive error message
        """
        self.message = message
        super().__init__(message)


class ObservabilityConfigError(ObservabilityError):
    """Exception raised for observability configuration errors.

    Raised when observability configuration is invalid or incomplete,
    such as missing required fields or invalid exporter settings.

    Attributes:
        field: The configuration field that caused the error
        message: Human-readable error message
    """

    def __init__(self, field: str, message: str) -> None:
        """Initialize ObservabilityConfigError with field and message.

        Args:
            field: Configuration field name where error occurred
            message: Descriptive error message
        """
        self.field = field
        full_message = f"Observability configuration error in '{field}': {message}"
        super().__init__(full_message)


__all__ = ["ObservabilityError", "ObservabilityConfigError"]
