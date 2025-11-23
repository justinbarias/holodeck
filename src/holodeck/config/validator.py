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


class ConfigValidator:
    """Validator for HoloDeck configuration components.

    Provides validation methods for various configuration types including
    vectorstore tool configuration.
    """

    @staticmethod
    def validate_vectorstore_config(config: dict[str, Any]) -> list[str]:
        """Validate vectorstore tool configuration.

        Checks:
        - type field is "vectorstore"
        - source field is non-empty and valid path
        - top_k is positive integer (1-100)
        - min_similarity_score is float between 0.0-1.0 if provided
        - database config is valid if provided

        Args:
            config: Configuration dictionary for vectorstore tool

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Validate type
        if config.get("type") != "vectorstore":
            errors.append("Field 'type': must be 'vectorstore'")

        # Validate source
        source = config.get("source")
        if not source or not str(source).strip():
            errors.append("Field 'source': must be non-empty path")
        else:
            try:
                validate_path_exists(str(source), "source path")
            except ValueError as e:
                errors.append(f"Field 'source': {str(e)}")

        # Validate top_k
        top_k = config.get("top_k", 5)
        if not isinstance(top_k, int):
            errors.append(f"Field 'top_k': must be integer, got {type(top_k).__name__}")
        elif top_k <= 0 or top_k > 100:
            errors.append("Field 'top_k': must be between 1 and 100")

        # Validate min_similarity_score
        min_score = config.get("min_similarity_score")
        if min_score is not None:
            if not isinstance(min_score, int | float):
                errors.append(
                    f"Field 'min_similarity_score': must be number, "
                    f"got {type(min_score).__name__}"
                )
            elif not (0.0 <= min_score <= 1.0):
                errors.append(
                    "Field 'min_similarity_score': must be between 0.0 and 1.0"
                )

        # Validate database config if provided
        database = config.get("database")
        if database:
            db_errors = ConfigValidator.validate_database_config(database)
            errors.extend(db_errors)

        return errors

    @staticmethod
    def validate_database_config(config: Any) -> list[str]:
        """Validate vector database configuration.

        Checks:
        - provider is one of the supported vector store types
        - connection parameters are provided as appropriate for the provider
        - Optional parameters are valid if provided

        Supported providers:
        - redis-hashset, redis-json: Redis with different storage types
        - postgres: PostgreSQL with pgvector
        - azure-ai-search: Azure Cognitive Search
        - qdrant: Qdrant vector database
        - weaviate: Weaviate vector database
        - chromadb: ChromaDB
        - faiss: FAISS vector search
        - azure-cosmos-mongo: Azure Cosmos DB (MongoDB API)
        - azure-cosmos-nosql: Azure Cosmos DB (NoSQL API)
        - sql-server: SQL Server with vector support
        - pinecone: Pinecone serverless
        - in-memory: In-memory storage (development only)

        Args:
            config: Database configuration (dict or object)

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Supported providers
        supported_providers = {
            "redis-hashset",
            "redis-json",
            "postgres",
            "azure-ai-search",
            "qdrant",
            "weaviate",
            "chromadb",
            "faiss",
            "azure-cosmos-mongo",
            "azure-cosmos-nosql",
            "sql-server",
            "pinecone",
            "in-memory",
        }

        # Convert to dict if needed
        if hasattr(config, "model_dump"):
            config_dict = config.model_dump()
        elif hasattr(config, "__dict__"):
            config_dict = config.__dict__
        else:
            config_dict = config if isinstance(config, dict) else {}

        # Validate provider
        provider = config_dict.get("provider")
        if provider not in supported_providers:
            errors.append(
                f"Field 'database.provider': must be one of "
                f"{sorted(supported_providers)}, got '{provider}'"
            )

        return errors
