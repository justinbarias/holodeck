"""Exception classes for HoloDeck library operations.

These exceptions are raised when library operations fail with specific,
actionable error conditions that users can understand and resolve.
"""


class HoloDeckError(Exception):
    """Base exception for all HoloDeck library errors.

    This is the parent class for all exceptions raised by the holodeck library.
    Users can catch this to handle any library error generically.
    """

    pass


class ValidationError(HoloDeckError):
    """Raised when validation fails.

    This exception is raised when:
    - Input validation fails
    - Schema validation fails
    - Configuration is invalid

    Attributes:
        message: Description of the validation failure
    """

    pass


class InitError(HoloDeckError):
    """Raised when initialization fails.

    This exception is raised when:
    - Project initialization fails
    - Directory creation fails
    - File writing fails
    - Template rendering fails

    Attributes:
        message: Description of the initialization failure
    """

    pass


class TemplateError(HoloDeckError):
    """Raised when template processing fails.

    This exception is raised when:
    - Template manifest is malformed or missing
    - Jinja2 rendering fails
    - Generated content doesn't validate

    Attributes:
        message: Description of the template failure
    """

    pass
