"""Exception classes for HoloDeck CLI operations.

These exceptions are raised when CLI operations fail with specific,
actionable error conditions that users can understand and resolve.
"""


class CLIError(Exception):
    """Base exception for all CLI errors.

    This is the parent class for all exceptions raised by the CLI module.
    Users can catch this to handle any CLI error generically.
    """

    pass


class ValidationError(CLIError):
    """Raised when user input validation fails.

    This exception is raised when:
    - Project name is invalid (special characters, leading digits, etc.)
    - Template choice doesn't exist
    - Directory permissions are insufficient
    - Input constraints are violated

    Attributes:
        message: Description of the validation failure
    """

    pass


class InitError(CLIError):
    """Raised when project initialization fails.

    This exception is raised when:
    - Directory creation fails
    - File writing fails
    - Cleanup fails after partial creation
    - Unexpected errors occur during initialization

    Attributes:
        message: Description of the initialization failure
    """

    pass


class TemplateError(CLIError):
    """Raised when template processing fails.

    This exception is raised when:
    - Template manifest is malformed or missing
    - Jinja2 rendering fails
    - Generated YAML doesn't validate against schema
    - Template variables are missing or invalid

    Attributes:
        message: Description of the template failure
    """

    pass


class ChatConfigError(CLIError):
    """Raised when chat command cannot load agent configuration."""

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        """Initialize the error with a human-readable message."""
        self.message = message
        super().__init__(message)


class ChatAgentInitError(CLIError):
    """Raised when agent initialization fails for chat sessions."""

    exit_code: int = 2

    def __init__(self, message: str) -> None:
        """Initialize the error with a human-readable message."""
        self.message = message
        super().__init__(message)


class ChatRuntimeError(CLIError):
    """Raised for runtime chat failures that should exit the CLI."""

    exit_code: int = 1

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        """Initialize the error with optional exit code override."""
        self.exit_code = exit_code if exit_code is not None else self.exit_code
        self.message = message
        super().__init__(message)


class ChatValidationError(CLIError):
    """Raised for recoverable user input validation errors during chat."""

    exit_code: int = 0

    def __init__(self, message: str) -> None:
        """Initialize the error with a human-readable message."""
        self.message = message
        super().__init__(message)
