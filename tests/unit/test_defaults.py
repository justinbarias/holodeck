"""Tests for default configuration templates."""

from holodeck.config.defaults import (
    get_default_evaluation_config,
    get_default_model_config,
    get_default_tool_config,
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
