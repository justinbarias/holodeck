"""Tests for CLI-specific exception classes.

Tests verify that custom exception classes for the init command
can be created, raised, and caught properly.
"""

import pytest

from holodeck.cli.exceptions import InitError, TemplateError, ValidationError


@pytest.mark.unit
@pytest.mark.parametrize(
    "exception_class",
    [ValidationError, InitError, TemplateError],
    ids=["ValidationError", "InitError", "TemplateError"],
)
def test_exception_exists_and_raisable(
    exception_class: type[Exception],
) -> None:
    """Test that exception can be imported, exists, and can be raised/caught."""
    # Test existence
    assert exception_class is not None

    # Test can be raised and caught
    with pytest.raises(exception_class):
        raise exception_class(f"Test {exception_class.__name__}")


@pytest.mark.unit
def test_exception_message_preserved() -> None:
    """Test that exception messages are preserved."""
    from holodeck.cli.exceptions import ValidationError

    message = "Invalid project name: test-123"
    try:
        raise ValidationError(message)
    except ValidationError as e:
        assert str(e) == message


@pytest.mark.unit
def test_exceptions_inherit_from_exception() -> None:
    """Test that all custom exceptions inherit from Exception."""
    from holodeck.cli.exceptions import InitError, TemplateError, ValidationError

    assert issubclass(ValidationError, Exception)
    assert issubclass(InitError, Exception)
    assert issubclass(TemplateError, Exception)
