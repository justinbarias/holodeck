"""Validation utilities for HoloDeck configuration."""

from typing import Any

from pydantic import ValidationError as PydanticValidationError


def normalize_errors(errors: list[str]) -> list[str]:
    """Convert raw error messages to human-readable format.

    Processes error messages to be more user-friendly and actionable,
    removing technical jargon where possible.

    Args:
        errors: List of error message strings

    Returns:
        List of normalized, human-readable error messages
    """
    normalized: list[str] = []

    for error in errors:
        # Remove common technical prefixes
        msg = error
        if msg.startswith("value_error"):
            msg = msg.replace("value_error", "").strip()
        if msg.startswith("type_error"):
            msg = msg.replace("type_error", "").strip()

        # Improve message readability
        if msg:
            normalized.append(msg)

    return normalized if normalized else ["An unknown validation error occurred"]


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
        # Extract location (field path)
        loc = error.get("loc", ())
        field_path = ".".join(str(item) for item in loc) if loc else "unknown"

        # Extract error message
        msg = error.get("msg", "Unknown error")
        error_type = error.get("type", "")

        # Format the error message
        if error_type == "value_error":
            # For value errors, include what was provided
            input_val = error.get("input")
            formatted = f"Field '{field_path}': {msg} (received: {input_val!r})"
        else:
            formatted = f"Field '{field_path}': {msg}"

        errors.append(formatted)

    return errors if errors else ["Validation failed with unknown error"]


def validate_field_exists(data: dict[str, Any], field: str, field_type: type) -> None:
    """Validate that a required field exists and has correct type.

    Args:
        data: Dictionary to validate
        field: Field name to check
        field_type: Expected type for the field

    Raises:
        ValueError: If field is missing or has wrong type
    """
    if field not in data:
        raise ValueError(f"Required field '{field}' is missing")
    if not isinstance(data[field], field_type):
        raise ValueError(
            f"Field '{field}' must be {field_type.__name__}, "
            f"got {type(data[field]).__name__}"
        )


def validate_mutually_exclusive(data: dict[str, Any], fields: list[str]) -> None:
    """Validate that exactly one of the given fields is present.

    Args:
        data: Dictionary to validate
        fields: List of mutually exclusive field names

    Raises:
        ValueError: If not exactly one field is present
    """
    present = [f for f in fields if f in data and data[f] is not None]
    if len(present) == 0:
        raise ValueError(f"Exactly one of {fields} must be provided")
    if len(present) > 1:
        raise ValueError(f"Only one of {fields} can be provided, got {present}")


def validate_range(
    value: float, min_val: float, max_val: float, name: str = "value"
) -> None:
    """Validate that a numeric value is within a range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the field for error messages

    Raises:
        ValueError: If value is outside the range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_enum(value: str, allowed: list[str], name: str = "value") -> None:
    """Validate that a string is one of allowed values.

    Args:
        value: Value to validate
        allowed: List of allowed values
        name: Name of the field for error messages

    Raises:
        ValueError: If value is not in allowed list
    """
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got '{value}'")


def validate_path_exists(path: str, description: str = "file") -> None:
    """Validate that a file or directory exists.

    Args:
        path: Path to validate
        description: Description of path for error messages

    Raises:
        ValueError: If path does not exist
    """
    from pathlib import Path

    if not Path(path).exists():
        raise ValueError(f"Path does not exist: {path}")
