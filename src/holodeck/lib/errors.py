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


# MCP Registry Exceptions


class RegistryConnectionError(HoloDeckError):
    """Exception raised when connection to MCP registry fails.

    Raised for network timeouts, DNS failures, and connection refused errors.

    Attributes:
        url: The registry URL that failed
        message: Human-readable error message with resolution guidance
    """

    def __init__(self, url: str, original_error: Exception | None = None) -> None:
        """Initialize RegistryConnectionError with URL and optional cause.

        Args:
            url: The registry URL that failed to connect
            original_error: The underlying exception that caused the failure
        """
        self.url = url
        message = (
            f"Failed to connect to MCP registry at {url}.\n"
            f"Check your network connection and try again."
        )
        if original_error:
            message += f"\nOriginal error: {original_error}"
        super().__init__(message)


class RegistryAPIError(HoloDeckError):
    """Exception raised when MCP registry returns an error response.

    Raised for HTTP error status codes from the registry API.

    Attributes:
        url: The registry URL that returned the error
        status_code: HTTP status code returned
        message: Human-readable error message
    """

    def __init__(self, url: str, status_code: int, detail: str | None = None) -> None:
        """Initialize RegistryAPIError with response details.

        Args:
            url: The registry URL that returned the error
            status_code: HTTP status code
            detail: Optional error detail from response body
        """
        self.url = url
        self.status_code = status_code

        if status_code == 429:
            message = "Rate limited by MCP registry. Please wait and try again."
        elif status_code >= 500:
            message = "MCP registry service error. Try again later."
        else:
            message = f"MCP registry error (HTTP {status_code})"

        if detail:
            message += f": {detail}"

        super().__init__(message)


class ServerNotFoundError(HoloDeckError):
    """Exception raised when requested MCP server is not found.

    Can be raised in two contexts:
    1. Server not found in MCP registry (during search/add)
    2. Server not found in local configuration (during remove)

    Attributes:
        server_name: The server name that was not found
        location: Optional location context (e.g., "agent.yaml", "global configuration")
        message: Human-readable error message
    """

    def __init__(self, server_name: str, location: str | None = None) -> None:
        """Initialize ServerNotFoundError with server name.

        Args:
            server_name: The name of the server that was not found
            location: Optional location where server was expected
        """
        self.server_name = server_name
        self.location = location
        if location:
            message = f"Server '{server_name}' not found in {location}."
        else:
            message = f"Server '{server_name}' not found in MCP registry."
        super().__init__(message)


class DuplicateServerError(HoloDeckError):
    """Exception raised when attempting to add an MCP server that already exists.

    Raised when a server with the same registry_name (for registry servers) or
    name (for manual servers) is already configured.

    Attributes:
        server_name: The short name of the server being added
        registry_name: The full registry name of the server being added
        existing_registry_name: The registry name of the existing server
    """

    def __init__(
        self,
        server_name: str,
        registry_name: str | None = None,
        existing_registry_name: str | None = None,
    ) -> None:
        """Initialize DuplicateServerError with server details.

        Args:
            server_name: The short name of the server being added
            registry_name: The full registry name of the server being added
            existing_registry_name: The registry name of the existing server
        """
        self.server_name = server_name
        self.registry_name = registry_name
        self.existing_registry_name = existing_registry_name

        if registry_name and registry_name == existing_registry_name:
            # Exact duplicate (same registry server)
            message = (
                f"Server '{registry_name}' is already configured. "
                f"Use --version to install a different version, "
                f"or remove the existing server first."
            )
        elif existing_registry_name:
            # Name conflict between different registry servers
            message = (
                f"A server named '{server_name}' already exists "
                f"(from '{existing_registry_name}'). "
                f"Use --name to specify a different name."
            )
        else:
            # Manual server with same name
            message = (
                f"A server named '{server_name}' already exists. "
                f"Use --name to specify a different name, "
                f"or remove the existing server first."
            )

        super().__init__(message)


class RecordPathError(HoloDeckError):
    """Exception raised when navigating a record path in JSON structure fails.

    Raised when a configured record_path cannot be resolved in the source data.
    Provides helpful information about available keys at the point of failure.

    Attributes:
        path: The record_path that failed to navigate
        available_keys: List of available keys at the failure point
        message: Detailed error message describing the failure
    """

    def __init__(
        self,
        path: str,
        available_keys: list[str],
        message: str,
    ) -> None:
        """Initialize RecordPathError with navigation details.

        Args:
            path: The record_path that failed (e.g., "data.items")
            available_keys: Keys available at the point of failure
            message: Detailed description of what went wrong
        """
        self.path = path
        self.available_keys = available_keys
        self.message = message
        super().__init__(
            f"Failed to navigate path '{path}': {message}. "
            f"Available keys: {available_keys}"
        )
