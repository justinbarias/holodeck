"""Tests for CLI-specific exception classes.

Tests verify that custom exception classes for the init command
can be created, raised, and caught properly.
"""

import pytest


@pytest.mark.unit
def test_validation_error_exists() -> None:
    """Test that ValidationError can be imported from cli.exceptions."""
    from holodeck.cli.exceptions import ValidationError

    assert ValidationError is not None


@pytest.mark.unit
def test_init_error_exists() -> None:
    """Test that InitError can be imported from cli.exceptions."""
    from holodeck.cli.exceptions import InitError

    assert InitError is not None


@pytest.mark.unit
def test_template_error_exists() -> None:
    """Test that TemplateError can be imported from cli.exceptions."""
    from holodeck.cli.exceptions import TemplateError

    assert TemplateError is not None


@pytest.mark.unit
def test_validation_error_can_be_raised_and_caught() -> None:
    """Test that ValidationError can be raised and caught."""
    from holodeck.cli.exceptions import ValidationError

    with pytest.raises(ValidationError):
        raise ValidationError("Test validation error")


@pytest.mark.unit
def test_init_error_can_be_raised_and_caught() -> None:
    """Test that InitError can be raised and caught."""
    from holodeck.cli.exceptions import InitError

    with pytest.raises(InitError):
        raise InitError("Test init error")


@pytest.mark.unit
def test_template_error_can_be_raised_and_caught() -> None:
    """Test that TemplateError can be raised and caught."""
    from holodeck.cli.exceptions import TemplateError

    with pytest.raises(TemplateError):
        raise TemplateError("Test template error")


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
