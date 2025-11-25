"""Tests for validation utility functions."""

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from holodeck.config.validator import (
    ConfigValidator,
    flatten_pydantic_errors,
    normalize_errors,
)


class SampleModel(BaseModel):
    """Simple test model for validation testing."""

    name: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)


class TestNormalizeErrors:
    """Tests for normalize_errors() function."""

    def test_normalize_errors_returns_list(self) -> None:
        """Test that normalize_errors returns a list of normalized messages."""
        errors = ["Error 1", "Error 2"]
        result = normalize_errors(errors)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_normalize_errors_human_readable_format(self) -> None:
        """Test that normalized errors are human-readable."""
        errors = ["Field 'name' is required"]
        result = normalize_errors(errors)
        for msg in result:
            assert isinstance(msg, str)
            assert len(msg) > 0

    def test_normalize_errors_empty_list(self) -> None:
        """Test normalize_errors with empty list."""
        result = normalize_errors([])
        assert isinstance(result, list)

    def test_normalize_errors_with_field_info(self) -> None:
        """Test normalize_errors includes field information."""
        errors = ["name: Field is required", "temperature: Must be between 0 and 2"]
        result = normalize_errors(errors)
        result_str = " ".join(result)
        assert "name" in result_str.lower() or "required" in result_str.lower()


class TestFlattenPydanticErrors:
    """Tests for flatten_pydantic_errors() function."""

    def test_flatten_pydantic_errors_with_simple_error(self) -> None:
        """Test flattening simple Pydantic validation error."""
        try:
            # Create a validation error with missing required field
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
            # Create a validation error with out-of-range value
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


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_validate_vectorstore_config_valid(self) -> None:
        """Test validating a valid vectorstore configuration."""
        config = {
            "type": "vectorstore",
            "source": "/path/to/data.txt",
            "top_k": 5,
            "min_similarity_score": 0.5,
        }
        # We need to create a temporary file for the test
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            config["source"] = tmp_path
            errors = ConfigValidator.validate_vectorstore_config(config)
            assert isinstance(errors, list)
            assert len(errors) == 0
        finally:
            os.unlink(tmp_path)

    def test_validate_vectorstore_config_invalid_type(self) -> None:
        """Test that invalid type generates error."""
        config = {
            "type": "function",
            "source": "/path/to/data.txt",
        }
        errors = ConfigValidator.validate_vectorstore_config(config)
        assert len(errors) > 0
        assert any("type" in error.lower() for error in errors)

    def test_validate_vectorstore_config_missing_source(self) -> None:
        """Test that missing source generates error."""
        config = {
            "type": "vectorstore",
        }
        errors = ConfigValidator.validate_vectorstore_config(config)
        assert len(errors) > 0
        assert any("source" in error.lower() for error in errors)

    def test_validate_vectorstore_config_empty_source(self) -> None:
        """Test that empty source generates error."""
        config = {
            "type": "vectorstore",
            "source": "",
        }
        errors = ConfigValidator.validate_vectorstore_config(config)
        assert len(errors) > 0
        assert any("source" in error.lower() for error in errors)

    def test_validate_vectorstore_config_invalid_top_k(self) -> None:
        """Test that invalid top_k generates error."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            config = {
                "type": "vectorstore",
                "source": tmp_path,
                "top_k": 0,
            }
            errors = ConfigValidator.validate_vectorstore_config(config)
            assert len(errors) > 0
            assert any("top_k" in error.lower() for error in errors)
        finally:
            os.unlink(tmp_path)

    def test_validate_vectorstore_config_top_k_exceeds_maximum(self) -> None:
        """Test that top_k > 100 generates error."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            config = {
                "type": "vectorstore",
                "source": tmp_path,
                "top_k": 101,
            }
            errors = ConfigValidator.validate_vectorstore_config(config)
            assert len(errors) > 0
            assert any("top_k" in error.lower() for error in errors)
        finally:
            os.unlink(tmp_path)

    def test_validate_vectorstore_config_invalid_min_similarity_score(self) -> None:
        """Test that invalid min_similarity_score generates error."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            config = {
                "type": "vectorstore",
                "source": tmp_path,
                "min_similarity_score": 1.5,
            }
            errors = ConfigValidator.validate_vectorstore_config(config)
            assert len(errors) > 0
            assert any("similarity" in error.lower() for error in errors)
        finally:
            os.unlink(tmp_path)

    def test_validate_database_config_valid_redis_hashset(self) -> None:
        """Test validating a valid Redis Hashset database configuration."""
        config = {
            "provider": "redis-hashset",
            "connection_string": "redis://localhost:6379",
        }
        errors = ConfigValidator.validate_database_config(config)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_database_config_valid_postgres(self) -> None:
        """Test validating a valid PostgreSQL database configuration."""
        config = {
            "provider": "postgres",
            "connection_string": "postgresql://user:pass@localhost/db",
        }
        errors = ConfigValidator.validate_database_config(config)
        assert len(errors) == 0

    def test_validate_database_config_valid_qdrant(self) -> None:
        """Test validating a valid Qdrant database configuration."""
        config = {
            "provider": "qdrant",
            "url": "http://localhost:6333",
        }
        errors = ConfigValidator.validate_database_config(config)
        assert len(errors) == 0

    def test_validate_database_config_valid_in_memory(self) -> None:
        """Test validating in-memory vector store configuration."""
        config = {
            "provider": "in-memory",
        }
        errors = ConfigValidator.validate_database_config(config)
        assert len(errors) == 0

    def test_validate_database_config_invalid_provider(self) -> None:
        """Test that invalid provider generates error."""
        config = {
            "provider": "invalid-provider",
            "connection_string": "connection://string",
        }
        errors = ConfigValidator.validate_database_config(config)
        assert len(errors) > 0
        assert any("provider" in error.lower() for error in errors)

    def test_validate_database_config_missing_provider(self) -> None:
        """Test that missing provider generates error."""
        config = {
            "connection_string": "redis://localhost:6379",
        }
        errors = ConfigValidator.validate_database_config(config)
        assert len(errors) > 0
        assert any("provider" in error.lower() for error in errors)

    def test_validate_vectorstore_with_database_config(self) -> None:
        """Test validating vectorstore config with nested database config."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            config = {
                "type": "vectorstore",
                "source": tmp_path,
                "top_k": 10,
                "database": {
                    "provider": "redis-hashset",
                    "connection_string": "redis://localhost:6379",
                },
            }
            errors = ConfigValidator.validate_vectorstore_config(config)
            assert isinstance(errors, list)
            assert len(errors) == 0
        finally:
            os.unlink(tmp_path)
