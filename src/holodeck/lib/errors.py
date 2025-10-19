"""Custom exception hierarchy for HoloDeck configuration and operations."""


class HoloDeckError(Exception):
    """Base exception for all HoloDeck errors.

    All HoloDeck-specific exceptions inherit from this class, enabling
    centralized exception handling and error tracking.
    """

    pass


class ConfigError(HoloDeckError):
    """Exception raised for configuration errors.

    This exception is raised when configuration loading or parsing fails.
    It includes field-specific information to help users identify and fix
    configuration issues.

    Attributes:
        field: The configuration field that caused the error
        message: Human-readable error message describing the issue
    """

    def __init__(self, field: str, message: str) -> None:
        """Initialize ConfigError with field and message.

        Args:
            field: Configuration field name where error occurred
            message: Descriptive error message
        """
        self.field = field
        self.message = message
        super().__init__(f"Configuration error in '{field}': {message}")


class ValidationError(HoloDeckError):
    """Exception raised for validation errors during configuration parsing.

    Provides detailed information about what was expected versus what was received,
    enabling users to quickly understand and fix validation issues.

    Attributes:
        field: The field that failed validation
        message: Description of the validation failure
        expected: Human description of expected value/type
        actual: The actual value that failed validation
    """

    def __init__(
        self,
        field: str,
        message: str,
        expected: str,
        actual: str,
    ) -> None:
        """Initialize ValidationError with detailed information.

        Args:
            field: Field that failed validation (can use dot notation for nested fields)
            message: Description of what went wrong
            expected: Human-readable description of expected value
            actual: The actual value that failed
        """
        self.field = field
        self.message = message
        self.expected = expected
        self.actual = actual
        full_message = (
            f"Validation error in '{field}': {message}\n"
            f"  Expected: {expected}\n"
            f"  Got: {actual}"
        )
        super().__init__(full_message)


class FileNotFoundError(HoloDeckError):
    """Exception raised when a configuration file is not found.

    Includes the file path and helpful suggestions for resolving the issue.

    Attributes:
        path: Path to the file that was not found
        message: Human-readable error message
    """

    def __init__(self, path: str, message: str) -> None:
        """Initialize FileNotFoundError with path and message.

        Args:
            path: Path to the file that was not found
            message: Descriptive error message, optionally with suggestions
        """
        self.path = path
        self.message = message
        super().__init__(f"File not found: {path}\n{message}")
