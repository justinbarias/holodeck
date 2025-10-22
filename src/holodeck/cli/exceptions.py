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
