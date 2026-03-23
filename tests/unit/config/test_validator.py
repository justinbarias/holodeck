"""Tests for validation utility functions."""

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from holodeck.config.validator import flatten_pydantic_errors


class SampleModel(BaseModel):
    """Simple test model for validation testing."""

    name: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)


class TestFlattenPydanticErrors:
    """Tests for flatten_pydantic_errors() function."""

    def test_flatten_pydantic_errors_with_simple_error(self) -> None:
        """Test flattening simple Pydantic validation error."""
        try:
            SampleModel(name="", temperature=0.5)  # noqa: F821
        except PydanticValidationError as e:
            result = flatten_pydantic_errors(e)
            assert isinstance(result, list)
            assert len(result) > 0
            result_str = " ".join(result)
            assert "name" in result_str

    def test_flatten_pydantic_errors_with_nested_error(self) -> None:
        """Test flattening nested Pydantic validation errors."""
        try:
            SampleModel(name="test", temperature=3.5)  # temperature too high
        except PydanticValidationError as e:
            result = flatten_pydantic_errors(e)
            assert isinstance(result, list)
            result_str = " ".join(result)
            assert "temperature" in result_str

    def test_flatten_pydantic_errors_returns_list_of_strings(self) -> None:
        """Test that flatten_pydantic_errors returns list of strings."""
        try:
            SampleModel(name="", temperature=0.5)  # noqa: F821
        except PydanticValidationError as e:
            result = flatten_pydantic_errors(e)
            assert isinstance(result, list)
            for item in result:
                assert isinstance(item, str)

    def test_flatten_pydantic_errors_with_multiple_errors(self) -> None:
        """Test flattening multiple Pydantic validation errors."""
        try:
            SampleModel(name="", temperature=-1.0)  # noqa: F821
        except PydanticValidationError as e:
            result = flatten_pydantic_errors(e)
            assert isinstance(result, list)
            assert len(result) >= 2

    def test_flatten_pydantic_errors_actionable_messages(self) -> None:
        """Test that flattened errors contain actionable information."""
        try:
            SampleModel(name="", temperature=0.5)  # Empty name is invalid
        except PydanticValidationError as e:
            result = flatten_pydantic_errors(e)
            result_str = " ".join(result)
            assert "name" in result_str
