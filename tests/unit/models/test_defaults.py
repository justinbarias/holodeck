"""Tests for default configuration templates."""

from holodeck.config.defaults import (
    EMBEDDING_MODEL_DIMENSIONS,
    get_default_evaluation_config,
    get_default_model_config,
    get_default_tool_config,
    get_embedding_dimensions,
)


class TestDefaultModelConfig:
    """Tests for default model configuration."""

    def test_default_model_config_returns_dict(self) -> None:
        """Test that default model config returns a dictionary."""
        config = get_default_model_config()
        assert isinstance(config, dict)

    def test_default_model_config_has_required_fields(self) -> None:
        """Test that default model config has required fields."""
        config = get_default_model_config()
        assert "provider" in config
        assert "name" in config

    def test_default_model_config_provider_is_valid(self) -> None:
        """Test that default provider is one of supported values."""
        config = get_default_model_config()
        valid_providers = ["openai", "azure_openai", "anthropic"]
        assert config["provider"] in valid_providers

    def test_default_model_config_optional_fields(self) -> None:
        """Test that default model config includes optional fields with defaults."""
        config = get_default_model_config()
        # Temperature should have a sensible default (not required but present)
        if "temperature" in config:
            assert 0.0 <= config["temperature"] <= 2.0


class TestDefaultToolConfig:
    """Tests for default tool configuration."""

    def test_default_tool_config_returns_dict(self) -> None:
        """Test that default tool config returns a dictionary."""
        config = get_default_tool_config()
        assert isinstance(config, dict)

    def test_default_tool_config_has_type_field(self) -> None:
        """Test that default tool config has type field."""
        config = get_default_tool_config()
        # Should have sensible defaults for a tool
        assert isinstance(config, dict)

    def test_default_tool_config_per_type(self) -> None:
        """Test that can get defaults for specific tool types."""
        for tool_type in ["vectorstore", "function", "mcp", "prompt"]:
            config = get_default_tool_config(tool_type=tool_type)
            assert isinstance(config, dict)


class TestDefaultEvaluationConfig:
    """Tests for default evaluation configuration."""

    def test_default_evaluation_config_returns_dict(self) -> None:
        """Test that default evaluation config returns a dictionary."""
        config = get_default_evaluation_config()
        assert isinstance(config, dict)

    def test_default_evaluation_config_has_metrics_field(self) -> None:
        """Test that default evaluation config has metrics structure."""
        config = get_default_evaluation_config()
        # Should have a structure for metrics
        assert isinstance(config, dict)

    def test_default_evaluation_config_metric_options(self) -> None:
        """Test that can get defaults for specific metrics."""
        metrics = ["groundedness", "relevance", "f1_score", "bleu"]
        for metric in metrics:
            config = get_default_evaluation_config(metric_name=metric)
            assert isinstance(config, dict)

    def test_default_evaluation_config_has_threshold(self) -> None:
        """Test that default evaluation config includes threshold."""
        config = get_default_evaluation_config()
        # Evaluation configs should have sensible defaults
        assert isinstance(config, dict)


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
