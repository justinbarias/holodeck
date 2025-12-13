"""Configuration manager for HoloDeck.

This module implements the ConfigManager class which handles configuration
operations such as creation, path resolution, and file writing.
"""

from pathlib import Path

import yaml

from holodeck.models.config import (
    DeploymentConfig,
    ExecutionConfig,
    GlobalConfig,
    VectorstoreConfig,
)
from holodeck.models.llm import LLMProvider, ProviderEnum


class ConfigManager:
    """Manager for configuration operations to improve testability."""

    @staticmethod
    def create_default_config() -> GlobalConfig:
        """Create a default GlobalConfig with sample settings."""
        # Create a default LLM provider
        default_provider = LLMProvider(
            provider=ProviderEnum.OPENAI,
            name="gpt-4",
            temperature=0.3,
            max_tokens=1000,
            api_key="your-openai-api-key-here",
            endpoint=None,
        )

        # Create a default vectorstore config
        default_vectorstore = VectorstoreConfig(
            provider="postgres",
            connection_string="postgresql://user:password@localhost:5432/vectorstore",
            options={"sslmode": "prefer"},
        )

        # Create a default execution config
        default_execution = ExecutionConfig(
            file_timeout=30,
            llm_timeout=30,
            download_timeout=30,
            cache_enabled=True,
            cache_dir=".cache",
            verbose=False,
            quiet=False,
        )

        # Create a default deployment config
        default_deployment = DeploymentConfig(
            type="docker", settings={"registry": "docker.io"}
        )

        # Create the global config
        return GlobalConfig(
            providers={"openai": default_provider},
            vectorstores={"postgres": default_vectorstore},
            execution=default_execution,
            deployment=default_deployment,
            mcp_servers=None,
        )

    @staticmethod
    def get_config_path(global_config: bool, project_config: bool) -> tuple[Path, str]:
        """Determine the configuration file path and type.

        Args:
            global_config: Whether to use global configuration.
            project_config: Whether to use project configuration.

        Returns:
            Tuple of (config_path, config_type_name)
        """
        if global_config:
            return Path.home() / ".holodeck" / "config.yaml", "global"
        else:
            # Default to project config if neither or project specified
            return Path.cwd() / "config.yaml", "project"

    @staticmethod
    def generate_config_content(config: GlobalConfig) -> str:
        """Generate YAML content for the configuration."""
        config_dict = config.model_dump(exclude_unset=True, mode="json")
        return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    @staticmethod
    def write_config(path: Path, content: str) -> None:
        """Write configuration content to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
