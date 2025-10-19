"""Configuration loader for HoloDeck agents.

This module provides the ConfigLoader class for loading, parsing, and validating
agent configuration from YAML files.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from holodeck.config.env_loader import substitute_env_vars
from holodeck.config.validator import flatten_pydantic_errors
from holodeck.lib.errors import ConfigError, FileNotFoundError
from holodeck.models.agent import Agent


class ConfigLoader:
    """Loads and validates agent configuration from YAML files.

    This class handles:
    - Parsing YAML files into Python dictionaries
    - Loading global configuration from ~/.holodeck/config.yaml
    - Merging configurations with proper precedence
    - Resolving file references (instructions, tools)
    - Converting validation errors into human-readable messages
    - Environment variable substitution
    """

    def __init__(self) -> None:
        """Initialize the ConfigLoader."""
        pass

    def parse_yaml(self, file_path: str) -> dict[str, Any] | None:
        """Parse a YAML file and return its contents as a dictionary.

        Args:
            file_path: Path to the YAML file to parse

        Returns:
            Dictionary containing parsed YAML content, or None if file is empty

        Raises:
            FileNotFoundError: If the file does not exist
            ConfigError: If YAML parsing fails
        """
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(
                file_path,
                f"Configuration file not found at {file_path}. "
                f"Please ensure the file exists at this path.",
            )

        try:
            with open(path, encoding="utf-8") as f:
                content = yaml.safe_load(f)
                return content if content is not None else {}
        except yaml.YAMLError as e:
            raise ConfigError(
                "yaml_parse",
                f"Failed to parse YAML file {file_path}: {str(e)}",
            ) from e

    def load_agent_yaml(self, file_path: str) -> Agent:
        """Load and validate an agent configuration from YAML.

        This method:
        1. Parses the YAML file
        2. Applies environment variable substitution
        3. Merges with global configuration if available
        4. Validates against Agent schema
        5. Returns an Agent instance

        Args:
            file_path: Path to agent.yaml file

        Returns:
            Validated Agent instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ConfigError: If YAML parsing fails
            ValidationError: If configuration is invalid
        """
        # Parse the agent YAML file
        agent_yaml_content = self.parse_yaml(file_path)

        # Apply environment variable substitution
        yaml_str = yaml.dump(agent_yaml_content)
        substituted_yaml = substitute_env_vars(yaml_str)
        agent_config = yaml.safe_load(substituted_yaml)

        # Load and merge global config
        global_config = self.load_global_config()
        merged_config = self.merge_configs(agent_config, global_config)

        # Validate against Agent schema
        try:
            agent = Agent(**merged_config)
            return agent
        except PydanticValidationError as e:
            # Convert Pydantic errors to human-readable messages
            error_messages = flatten_pydantic_errors(e)
            error_text = "\n".join(error_messages)
            raise ConfigError(
                "agent_validation",
                f"Invalid agent configuration in {file_path}:\n{error_text}",
            ) from e

    def load_global_config(self) -> dict[str, Any]:
        """Load global configuration from ~/.holodeck/config.yaml.

        Returns:
            Dictionary containing global configuration, or empty dict if
            file doesn't exist

        Raises:
            ConfigError: If YAML parsing fails (but not if file is missing)
        """
        home_dir = Path.home()
        global_config_path = home_dir / ".holodeck" / "config.yaml"

        if not global_config_path.exists():
            return {}

        try:
            with open(global_config_path, encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if content is None:
                    return {}

                # Apply environment variable substitution to global config
                config_str = yaml.dump(content)
                substituted = substitute_env_vars(config_str)
                return yaml.safe_load(substituted) or {}
        except yaml.YAMLError as e:
            raise ConfigError(
                "global_config_parse",
                f"Failed to parse global config at {global_config_path}: {str(e)}",
            ) from e

    def merge_configs(
        self, agent_config: dict[str, Any], global_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge agent config with global config using proper precedence.

        Precedence (highest to lowest):
        1. agent.yaml explicit settings
        2. Environment variables (already substituted)
        3. ~/.holodeck/config.yaml global settings

        Args:
            agent_config: Configuration from agent.yaml
            global_config: Configuration from ~/.holodeck/config.yaml

        Returns:
            Merged configuration dictionary
        """
        # For now, agent config is the primary source
        # Global config is kept separate as it may contain provider configs
        # and other infrastructure settings not directly used by Agent model
        # The merging of global settings would happen at a higher level
        # when actually using the agent (e.g., for LLM provider setup)

        # Return agent config as-is (it's validated and complete)
        # Global config would be used separately for system configuration
        return agent_config if agent_config else {}

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
        """Deep merge override dict into base dict.

        Args:
            base: Base dictionary to merge into (modified in-place)
            override: Dictionary with values to override
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigLoader._deep_merge(base[key], value)
            else:
                base[key] = value

    def resolve_file_path(self, file_path: str, base_dir: str) -> str:
        """Resolve a file path relative to base directory.

        This method handles:
        - Absolute paths: returned as-is
        - Relative paths: resolved relative to base_dir
        - File existence verification

        Args:
            file_path: Path to resolve (absolute or relative)
            base_dir: Base directory for relative path resolution

        Returns:
            Absolute path to the file

        Raises:
            FileNotFoundError: If the resolved file doesn't exist
        """
        path = Path(file_path)

        # If path is absolute, use it directly
        if path.is_absolute():
            resolved = path
        else:
            # Resolve relative to base directory
            resolved = (Path(base_dir) / file_path).resolve()

        # Verify file exists
        if not resolved.exists():
            raise FileNotFoundError(
                str(resolved),
                f"Referenced file not found: {resolved}\n"
                f"Please ensure the file exists at this path.",
            )

        return str(resolved)

    def load_instructions(self, agent_yaml_path: str, agent: Agent) -> str | None:
        """Load instruction content from file or return inline content.

        Args:
            agent_yaml_path: Path to the agent.yaml file
            agent: Agent instance with instructions

        Returns:
            Instruction content string, or None if not defined

        Raises:
            FileNotFoundError: If instruction file doesn't exist
        """
        if agent.instructions.inline:
            return agent.instructions.inline

        if agent.instructions.file:
            base_dir = str(Path(agent_yaml_path).parent)
            file_path = self.resolve_file_path(agent.instructions.file, base_dir)
            with open(file_path, encoding="utf-8") as f:
                return f.read()

        return None
