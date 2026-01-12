"""Unit tests for project initialization utilities.

Tests the helper functions in project_init.py module.
"""

import pytest

from holodeck.cli.utils.project_init import (
    get_model_for_provider,
    get_provider_api_key_env_var,
    get_provider_endpoint_env_var,
    get_vectorstore_config,
    get_vectorstore_endpoint,
)


class TestGetModelForProvider:
    """Tests for the get_model_for_provider() function."""

    def test_get_model_for_ollama(self) -> None:
        """Test returns correct model for Ollama provider."""
        result = get_model_for_provider("ollama")
        assert result == "gpt-oss:20b"

    def test_get_model_for_openai(self) -> None:
        """Test returns correct model for OpenAI provider."""
        result = get_model_for_provider("openai")
        assert result == "gpt-4o"

    def test_get_model_for_azure_openai(self) -> None:
        """Test returns correct model for Azure OpenAI provider."""
        result = get_model_for_provider("azure_openai")
        assert result == "gpt-4o"

    def test_get_model_for_anthropic(self) -> None:
        """Test returns correct model for Anthropic provider."""
        result = get_model_for_provider("anthropic")
        assert result == "claude-3-5-sonnet-20241022"

    def test_get_model_for_unknown_provider(self) -> None:
        """Test returns fallback model for unknown provider."""
        result = get_model_for_provider("unknown")
        assert result == "gpt-oss:20b"


class TestGetProviderApiKeyEnvVar:
    """Tests for the get_provider_api_key_env_var() function."""

    def test_get_api_key_env_var_for_ollama(self) -> None:
        """Test returns None for Ollama (no API key required)."""
        result = get_provider_api_key_env_var("ollama")
        assert result is None

    def test_get_api_key_env_var_for_openai(self) -> None:
        """Test returns correct env var for OpenAI."""
        result = get_provider_api_key_env_var("openai")
        assert result == "OPENAI_API_KEY"

    def test_get_api_key_env_var_for_azure_openai(self) -> None:
        """Test returns correct env var for Azure OpenAI."""
        result = get_provider_api_key_env_var("azure_openai")
        assert result == "AZURE_OPENAI_API_KEY"

    def test_get_api_key_env_var_for_anthropic(self) -> None:
        """Test returns correct env var for Anthropic."""
        result = get_provider_api_key_env_var("anthropic")
        assert result == "ANTHROPIC_API_KEY"

    def test_get_api_key_env_var_for_unknown_provider(self) -> None:
        """Test returns None for unknown provider."""
        result = get_provider_api_key_env_var("unknown")
        assert result is None


class TestGetProviderEndpointEnvVar:
    """Tests for the get_provider_endpoint_env_var() function."""

    def test_get_endpoint_env_var_for_ollama(self) -> None:
        """Test returns None for Ollama (no endpoint required)."""
        result = get_provider_endpoint_env_var("ollama")
        assert result is None

    def test_get_endpoint_env_var_for_openai(self) -> None:
        """Test returns None for OpenAI (no custom endpoint)."""
        result = get_provider_endpoint_env_var("openai")
        assert result is None

    def test_get_endpoint_env_var_for_azure_openai(self) -> None:
        """Test returns correct env var for Azure OpenAI."""
        result = get_provider_endpoint_env_var("azure_openai")
        assert result == "AZURE_OPENAI_ENDPOINT"

    def test_get_endpoint_env_var_for_anthropic(self) -> None:
        """Test returns None for Anthropic (no custom endpoint)."""
        result = get_provider_endpoint_env_var("anthropic")
        assert result is None

    def test_get_endpoint_env_var_for_unknown_provider(self) -> None:
        """Test returns None for unknown provider."""
        result = get_provider_endpoint_env_var("unknown")
        assert result is None


class TestGetVectorstoreEndpoint:
    """Tests for the get_vectorstore_endpoint() function."""

    def test_get_endpoint_for_chromadb(self) -> None:
        """Test returns correct endpoint for ChromaDB."""
        result = get_vectorstore_endpoint("chromadb")
        assert result == "http://localhost:8000"

    def test_get_endpoint_for_qdrant(self) -> None:
        """Test returns correct endpoint for Qdrant."""
        result = get_vectorstore_endpoint("qdrant")
        assert result == "http://localhost:6333"

    def test_get_endpoint_for_in_memory(self) -> None:
        """Test returns None for in-memory store."""
        result = get_vectorstore_endpoint("in-memory")
        assert result is None

    def test_get_endpoint_for_unknown_store(self) -> None:
        """Test returns None for unknown vector store."""
        result = get_vectorstore_endpoint("unknown")
        assert result is None


class TestGetVectorstoreConfig:
    """Tests for the get_vectorstore_config() function."""

    def test_get_config_for_chromadb(self) -> None:
        """Test returns correct configuration for ChromaDB."""
        result = get_vectorstore_config("chromadb")

        assert result["provider"] == "chromadb"
        assert result["display_name"] == "ChromaDB"
        assert result["endpoint"] == "http://localhost:8000"
        assert result["is_ephemeral"] is False
        assert result["requires_connection"] is True

    def test_get_config_for_qdrant(self) -> None:
        """Test returns correct configuration for Qdrant."""
        result = get_vectorstore_config("qdrant")

        assert result["provider"] == "qdrant"
        assert result["display_name"] == "Qdrant"
        assert result["endpoint"] == "http://localhost:6333"
        assert result["is_ephemeral"] is False
        assert result["requires_connection"] is True

    def test_get_config_for_in_memory(self) -> None:
        """Test returns correct configuration for in-memory store."""
        result = get_vectorstore_config("in-memory")

        assert result["provider"] == "in-memory"
        assert result["display_name"] == "In-Memory"
        assert result["endpoint"] is None
        assert result["is_ephemeral"] is True
        assert result["requires_connection"] is False

    def test_get_config_for_unknown_store(self) -> None:
        """Test returns fallback configuration for unknown store."""
        result = get_vectorstore_config("unknown")

        assert result["provider"] == "unknown"
        assert result["display_name"] == "unknown"
        assert result["endpoint"] is None
        assert result["is_ephemeral"] is False
        assert result["requires_connection"] is True
