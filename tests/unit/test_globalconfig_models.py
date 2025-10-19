"""Tests for GlobalConfig model in holodeck.models.config."""

import pytest
from pydantic import ValidationError

from holodeck.models.config import DeploymentConfig, GlobalConfig, VectorstoreConfig
from holodeck.models.llm import LLMProvider, ProviderEnum


class TestVectorstoreConfig:
    """Tests for VectorstoreConfig model."""

    def test_vectorstore_config_valid(self) -> None:
        """Test creating a valid VectorstoreConfig."""
        config = VectorstoreConfig(
            provider="postgres",
            connection_string="postgresql://localhost/holodeck",
        )
        assert config.provider == "postgres"
        assert config.connection_string == "postgresql://localhost/holodeck"

    def test_vectorstore_config_provider_required(self) -> None:
        """Test that provider is required."""
        with pytest.raises(ValidationError):
            VectorstoreConfig(connection_string="connection")

    def test_vectorstore_config_connection_string_required(self) -> None:
        """Test that connection_string is required."""
        with pytest.raises(ValidationError):
            VectorstoreConfig(provider="postgres")

    def test_vectorstore_config_options_optional(self) -> None:
        """Test that options are optional."""
        config = VectorstoreConfig(
            provider="redis",
            connection_string="redis://localhost",
        )
        assert config.options is None or isinstance(config.options, dict)

    def test_vectorstore_config_with_options(self) -> None:
        """Test VectorstoreConfig with options."""
        config = VectorstoreConfig(
            provider="postgres",
            connection_string="postgresql://localhost/db",
            options={"pool_size": 10, "timeout": 30},
        )
        assert config.options == {"pool_size": 10, "timeout": 30}


class TestDeploymentConfig:
    """Tests for DeploymentConfig model."""

    def test_deployment_config_valid(self) -> None:
        """Test creating a valid DeploymentConfig."""
        config = DeploymentConfig(
            type="docker",
        )
        assert config.type == "docker"

    def test_deployment_config_type_required(self) -> None:
        """Test that type is required."""
        with pytest.raises(ValidationError):
            DeploymentConfig()

    def test_deployment_config_settings_optional(self) -> None:
        """Test that settings are optional."""
        config = DeploymentConfig(type="docker")
        assert config.settings is None or isinstance(config.settings, dict)

    def test_deployment_config_with_settings(self) -> None:
        """Test DeploymentConfig with settings."""
        config = DeploymentConfig(
            type="kubernetes",
            settings={"namespace": "default", "replicas": 3},
        )
        assert config.settings == {"namespace": "default", "replicas": 3}


class TestGlobalConfig:
    """Tests for GlobalConfig model."""

    def test_global_config_empty_creation(self) -> None:
        """Test creating an empty GlobalConfig (all optional)."""
        config = GlobalConfig()
        assert config.providers is None or isinstance(config.providers, dict)
        assert config.vectorstores is None or isinstance(config.vectorstores, dict)
        assert config.deployment is None

    def test_global_config_providers_optional(self) -> None:
        """Test that providers dict is optional."""
        config = GlobalConfig()
        assert config.providers is None or isinstance(config.providers, dict)

    def test_global_config_with_providers(self) -> None:
        """Test GlobalConfig with providers."""
        provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        config = GlobalConfig(providers={"default": provider})
        assert "default" in config.providers
        assert config.providers["default"].provider == ProviderEnum.OPENAI

    def test_global_config_vectorstores_optional(self) -> None:
        """Test that vectorstores dict is optional."""
        config = GlobalConfig()
        assert config.vectorstores is None or isinstance(config.vectorstores, dict)

    def test_global_config_with_vectorstores(self) -> None:
        """Test GlobalConfig with vectorstores."""
        vectorstore = VectorstoreConfig(
            provider="postgres",
            connection_string="postgresql://localhost/holodeck",
        )
        config = GlobalConfig(vectorstores={"knowledge_base": vectorstore})
        assert "knowledge_base" in config.vectorstores
        assert config.vectorstores["knowledge_base"].provider == "postgres"

    def test_global_config_deployment_optional(self) -> None:
        """Test that deployment is optional."""
        config = GlobalConfig()
        assert config.deployment is None

    def test_global_config_with_deployment(self) -> None:
        """Test GlobalConfig with deployment config."""
        deployment = DeploymentConfig(type="docker")
        config = GlobalConfig(deployment=deployment)
        assert config.deployment is not None
        assert config.deployment.type == "docker"

    def test_global_config_all_fields(self) -> None:
        """Test GlobalConfig with all fields."""
        provider = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-3-opus",
        )
        vectorstore = VectorstoreConfig(
            provider="redis",
            connection_string="redis://localhost",
        )
        deployment = DeploymentConfig(
            type="kubernetes",
            settings={"namespace": "holodeck"},
        )

        config = GlobalConfig(
            providers={"anthropic": provider},
            vectorstores={"cache": vectorstore},
            deployment=deployment,
        )

        assert len(config.providers) == 1
        assert len(config.vectorstores) == 1
        assert config.deployment.type == "kubernetes"

    def test_global_config_multiple_providers(self) -> None:
        """Test GlobalConfig with multiple providers."""
        provider1 = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4o",
        )
        provider2 = LLMProvider(
            provider=ProviderEnum.ANTHROPIC,
            name="claude-3-opus",
        )
        config = GlobalConfig(
            providers={
                "openai": provider1,
                "anthropic": provider2,
            }
        )
        assert len(config.providers) == 2
        assert config.providers["openai"].provider == ProviderEnum.OPENAI
        assert config.providers["anthropic"].provider == ProviderEnum.ANTHROPIC

    def test_global_config_multiple_vectorstores(self) -> None:
        """Test GlobalConfig with multiple vectorstores."""
        vs1 = VectorstoreConfig(
            provider="postgres",
            connection_string="postgresql://localhost/db1",
        )
        vs2 = VectorstoreConfig(
            provider="redis",
            connection_string="redis://localhost",
        )
        config = GlobalConfig(
            vectorstores={
                "postgres_db": vs1,
                "redis_cache": vs2,
            }
        )
        assert len(config.vectorstores) == 2
        assert config.vectorstores["postgres_db"].provider == "postgres"
        assert config.vectorstores["redis_cache"].provider == "redis"
