"""Tests for default configuration templates."""

from holodeck.config.defaults import (
    EMBEDDING_MODEL_DIMENSIONS,
    get_embedding_dimensions,
)


class TestEmbeddingModelDimensions:
    """Tests for embedding model dimension mapping and resolution."""

    def test_embedding_model_dimensions_constant_exists(self) -> None:
        """Test that EMBEDDING_MODEL_DIMENSIONS constant is defined."""
        assert isinstance(EMBEDDING_MODEL_DIMENSIONS, dict)
        assert len(EMBEDDING_MODEL_DIMENSIONS) > 0

    def test_known_openai_models_in_mapping(self) -> None:
        """Test that known OpenAI models are in the mapping."""
        assert "text-embedding-3-small" in EMBEDDING_MODEL_DIMENSIONS
        assert EMBEDDING_MODEL_DIMENSIONS["text-embedding-3-small"] == 1536
        assert "text-embedding-3-large" in EMBEDDING_MODEL_DIMENSIONS
        assert EMBEDDING_MODEL_DIMENSIONS["text-embedding-3-large"] == 3072
        assert "text-embedding-ada-002" in EMBEDDING_MODEL_DIMENSIONS
        assert EMBEDDING_MODEL_DIMENSIONS["text-embedding-ada-002"] == 1536

    def test_known_ollama_models_in_mapping(self) -> None:
        """Test that known Ollama models are in the mapping."""
        assert "nomic-embed-text:latest" in EMBEDDING_MODEL_DIMENSIONS
        assert EMBEDDING_MODEL_DIMENSIONS["nomic-embed-text:latest"] == 768
        assert "mxbai-embed-large" in EMBEDDING_MODEL_DIMENSIONS
        assert EMBEDDING_MODEL_DIMENSIONS["mxbai-embed-large"] == 1024


class TestGetEmbeddingDimensions:
    """Tests for get_embedding_dimensions function."""

    def test_known_openai_model_returns_correct_dimension(self) -> None:
        """Test known OpenAI model returns correct dimensions."""
        dims = get_embedding_dimensions("text-embedding-3-small", provider="openai")
        assert dims == 1536

        dims = get_embedding_dimensions("text-embedding-3-large", provider="openai")
        assert dims == 3072

    def test_known_ollama_model_returns_correct_dimension(self) -> None:
        """Test known Ollama model returns correct dimensions."""
        dims = get_embedding_dimensions("nomic-embed-text:latest", provider="ollama")
        assert dims == 768

        dims = get_embedding_dimensions("mxbai-embed-large", provider="ollama")
        assert dims == 1024

    def test_unknown_model_openai_provider_returns_1536(self) -> None:
        """Test unknown model with OpenAI provider defaults to 1536."""
        dims = get_embedding_dimensions("unknown-model", provider="openai")
        assert dims == 1536

    def test_unknown_model_azure_provider_returns_1536(self) -> None:
        """Test unknown model with Azure provider defaults to 1536."""
        dims = get_embedding_dimensions("unknown-model", provider="azure_openai")
        assert dims == 1536

    def test_unknown_model_ollama_provider_returns_768(self) -> None:
        """Test unknown model with Ollama provider defaults to 768."""
        dims = get_embedding_dimensions("unknown-model", provider="ollama")
        assert dims == 768

    def test_none_model_openai_provider_returns_1536(self) -> None:
        """Test None model with OpenAI provider defaults to 1536."""
        dims = get_embedding_dimensions(None, provider="openai")
        assert dims == 1536

    def test_none_model_ollama_provider_returns_768(self) -> None:
        """Test None model with Ollama provider defaults to 768."""
        dims = get_embedding_dimensions(None, provider="ollama")
        assert dims == 768

    def test_default_provider_is_openai(self) -> None:
        """Test that default provider is OpenAI."""
        dims = get_embedding_dimensions("text-embedding-3-small")
        assert dims == 1536

    def test_model_lookup_is_case_sensitive(self) -> None:
        """Test that model name lookup is case-sensitive."""
        # Correct case should work
        dims = get_embedding_dimensions("nomic-embed-text:latest", provider="ollama")
        assert dims == 768

        # Wrong case should fall back to default
        dims = get_embedding_dimensions("Nomic-Embed-Text:Latest", provider="ollama")
        assert dims == 768  # Falls back to Ollama default
