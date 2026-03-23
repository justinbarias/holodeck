"""Validation utilities for HoloDeck configuration."""

from pydantic import ValidationError as PydanticValidationError


def flatten_pydantic_errors(exc: PydanticValidationError) -> list[str]:
    """Flatten Pydantic ValidationError into human-readable messages.

    Converts Pydantic's nested error structure into a flat list of
    user-friendly error messages that include field names and descriptions.

    Args:
        exc: Pydantic ValidationError exception

    Returns:
        List of human-readable error messages, one per field error

    Example:
        >>> from pydantic import BaseModel, ValidationError
        >>> class Model(BaseModel):
        ...     name: str
        >>> try:
        ...     Model(name=123)
        ... except ValidationError as e:
        ...     msgs = flatten_pydantic_errors(e)
        ...     # msgs contains human-readable descriptions
    """
    errors: list[str] = []

    for error in exc.errors():
        loc = error.get("loc", ())
        field_path = ".".join(str(item) for item in loc) if loc else "unknown"

        msg = error.get("msg", "Unknown error")
        error_type = error.get("type", "")

        if error_type == "value_error":
            input_val = error.get("input")
            formatted = f"Field '{field_path}': {msg} (received: {input_val!r})"
        else:
            formatted = f"Field '{field_path}': {msg}"

        errors.append(formatted)

    return errors if errors else ["Validation failed with unknown error"]
