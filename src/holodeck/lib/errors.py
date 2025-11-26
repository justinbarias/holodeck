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


class ExecutionError(HoloDeckError):
    """Exception raised when test execution fails.

    Covers timeout, agent invocation errors, and other runtime failures
    during test execution.

    Attributes:
        message: Human-readable error message
    """

    pass


class AgentInitializationError(HoloDeckError):
    """Exception raised when an agent fails to initialize."""

    def __init__(self, agent_name: str, message: str) -> None:
        """Create an initialization error with context."""
        self.agent_name = agent_name
        self.message = message
        super().__init__(f"Agent '{agent_name}' failed to initialize: {message}")


class ChatValidationError(HoloDeckError):
    """Exception raised when chat input validation fails."""

    def __init__(self, message: str) -> None:
        """Create a validation error for chat messages."""
        self.message = message
        super().__init__(message)


class ChatSessionError(HoloDeckError):
    """Exception raised for chat session lifecycle failures."""

    def __init__(self, message: str) -> None:
        """Create a chat session error."""
        self.message = message
        super().__init__(message)


class EvaluationError(HoloDeckError):
    """Exception raised when metric evaluation fails.

    Covers failures in evaluator initialization or metric calculation.

    Attributes:
        message: Human-readable error message
    """

    pass


class AgentFactoryError(HoloDeckError):
    """Exception raised during agent factory operations.

    Base exception for errors during agent initialization and execution.

    Attributes:
        message: Human-readable error message
    """

    pass


class OllamaConnectionError(AgentFactoryError):
    """Error raised when Ollama endpoint is unreachable.

    Provides actionable guidance for resolving connectivity issues with
    local or remote Ollama servers.

    Attributes:
        endpoint: The Ollama endpoint URL that failed
        message: Human-readable error message with resolution guidance
    """

    def __init__(self, endpoint: str, original_error: Exception | None = None) -> None:
        """Initialize OllamaConnectionError with endpoint and optional cause.

        Args:
            endpoint: The Ollama endpoint URL that failed to connect
            original_error: The underlying exception that caused the connection failure
        """
        self.endpoint = endpoint
        message = (
            f"Failed to connect to Ollama endpoint at {endpoint}.\n"
            f"Ensure Ollama is running: ollama serve\n"
            f"Check endpoint URL is correct and accessible."
        )
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class OllamaModelNotFoundError(AgentFactoryError):
    """Error raised when requested Ollama model is not found.

    Provides specific resolution steps for pulling missing models.

    Attributes:
        model_name: The model that was not found
        endpoint: The Ollama endpoint that was queried
        message: Human-readable error message with resolution guidance
    """

    def __init__(self, model_name: str, endpoint: str) -> None:
        """Initialize OllamaModelNotFoundError with model and endpoint details.

        Args:
            model_name: The name of the model that was not found
            endpoint: The Ollama endpoint URL that was queried
        """
        self.model_name = model_name
        self.endpoint = endpoint
        message = (
            f"Model '{model_name}' not found on Ollama endpoint {endpoint}.\n"
            f"Pull the model first: ollama pull {model_name}\n"
            f"List available models: ollama list"
        )
        super().__init__(message)
