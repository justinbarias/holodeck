"""Configuration loader for HoloDeck agents.

This module provides the ConfigLoader class for loading, parsing, and validating
agent configuration from YAML files.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from holodeck.config.env_loader import substitute_env_vars
from holodeck.config.validator import flatten_pydantic_errors
from holodeck.lib.errors import ConfigError, FileNotFoundError
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig, GlobalConfig

logger = logging.getLogger(__name__)

# Environment variable to field name mapping
ENV_VAR_MAP = {
    "file_timeout": "HOLODECK_FILE_TIMEOUT",
    "llm_timeout": "HOLODECK_LLM_TIMEOUT",
    "download_timeout": "HOLODECK_DOWNLOAD_TIMEOUT",
    "cache_enabled": "HOLODECK_CACHE_ENABLED",
    "cache_dir": "HOLODECK_CACHE_DIR",
    "verbose": "HOLODECK_VERBOSE",
    "quiet": "HOLODECK_QUIET",
}


def _parse_env_value(field_name: str, value: str) -> Any:
    """Parse environment variable value to appropriate type.

    Args:
        field_name: Name of the field (used to determine type)
        value: String value from environment variable

    Returns:
        Parsed value in correct type (int, bool, or str)

    Raises:
        ValueError: If value cannot be parsed
    """
    if field_name in ("file_timeout", "llm_timeout", "download_timeout"):
        return int(value)
    elif field_name in ("cache_enabled", "verbose", "quiet"):
        return value.lower() in ("true", "1", "yes", "on")
    else:
        return value


def _get_env_value(field_name: str, env_vars: dict[str, str]) -> Any | None:
    """Get environment variable value for a field.

    Args:
        field_name: Name of field to get
        env_vars: Dictionary of environment variables

    Returns:
        Parsed value or None if not found or invalid
    """
    env_var_name = ENV_VAR_MAP.get(field_name)
    if not env_var_name or env_var_name not in env_vars:
        return None

    try:
        return _parse_env_value(field_name, env_vars[env_var_name])
    except (ValueError, KeyError):
        return None


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
        3. Loads project config (if available) with fallback to global config
        4. Merges configurations with proper precedence
        5. Validates against Agent schema
        6. Returns an Agent instance

        Configuration precedence (highest to lowest):
        1. agent.yaml explicit settings
        2. Environment variables
        3. Project-level config.yaml/config.yml
        4. Global ~/.holodeck/config.yaml/config.yml

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

        # Load project config, fallback to global config
        agent_dir = str(Path(file_path).parent)
        config = self.load_project_config(agent_dir)
        if config is None:
            config = self.load_global_config()

        # Merge configurations with proper precedence
        merged_config = self.merge_configs(agent_config, config)

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

    def load_global_config(self) -> GlobalConfig | None:
        """Load global configuration from ~/.holodeck/config.yml|config.yaml.

        Searches for config files with the following precedence:
        1. ~/.holodeck/config.yml (preferred)
        2. ~/.holodeck/config.yaml (fallback)

        Returns:
            GlobalConfig instance containing global configuration, or None if
            no config file exists or is empty

        Raises:
            ConfigError: If YAML parsing fails or validation fails
        """
        home_dir = Path.home()
        holodeck_dir = home_dir / ".holodeck"
        return self._load_config_file(
            holodeck_dir, "global_config", "global configuration"
        )

    def load_project_config(self, project_dir: str) -> GlobalConfig | None:
        """Load project-level configuration from config.yml|config.yaml.

        Searches for config files with the following precedence:
        1. config.yml (preferred)
        2. config.yaml (fallback)

        Args:
            project_dir: Path to project root directory

        Returns:
            GlobalConfig instance containing project configuration, or None if
            no config file exists or is empty

        Raises:
            ConfigError: If YAML parsing fails or validation fails
        """
        project_path = Path(project_dir)
        return self._load_config_file(
            project_path, "project_config", "project configuration"
        )

    def _load_config_file(
        self, config_dir: Path, error_code: str, config_name: str
    ) -> GlobalConfig | None:
        """Load configuration file from directory with .yml/.yaml preference.

        Private helper method to load global or project configuration files.

        Args:
            config_dir: Directory to search for config files
            error_code: Error code prefix (e.g., "global_config", "project_config")
            config_name: Human-readable config name for error messages

        Returns:
            GlobalConfig instance containing configuration, or None if
            no config file exists or is empty

        Raises:
            ConfigError: If YAML parsing fails or validation fails
        """
        # Check for both .yml and .yaml with .yml preference
        yml_path = config_dir / "config.yml"
        yaml_path = config_dir / "config.yaml"

        # Determine which file to use
        config_path = None
        if yml_path.exists():
            config_path = yml_path
            # Log info if both files exist
            if yaml_path.exists():
                logger.info(
                    f"Both {yml_path} and {yaml_path} exist. "
                    f"Using {yml_path} (prefer .yml extension)."
                )
        elif yaml_path.exists():
            config_path = yaml_path

        # If no config file found, return None
        if config_path is None:
            return None

        try:
            with open(config_path, encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if content is None:
                    return None

                # Apply environment variable substitution
                config_str = yaml.dump(content)
                substituted = substitute_env_vars(config_str)
                config_dict = yaml.safe_load(substituted)

                if not config_dict:
                    return None

                # Validate and create GlobalConfig instance
                try:
                    return GlobalConfig(**config_dict)
                except PydanticValidationError as e:
                    error_messages = flatten_pydantic_errors(e)
                    error_text = "\n".join(error_messages)
                    raise ConfigError(
                        f"{error_code}_validation",
                        f"Invalid {config_name} in {config_path}:\n{error_text}",
                    ) from e
        except yaml.YAMLError as e:
            raise ConfigError(
                f"{error_code}_parse",
                f"Failed to parse {config_name} at {config_path}: {str(e)}",
            ) from e

    def merge_configs(
        self, agent_config: dict[str, Any], global_config: GlobalConfig | None
    ) -> dict[str, Any]:
        """Merge agent config with global config using proper precedence.

        Precedence (highest to lowest):
        1. agent.yaml explicit settings
        2. Environment variables (already substituted)
        3. ~/.holodeck/config.yaml global settings

        Merges global LLM provider configs into:
        - agent model: when a provider's name matches agent_config.model.provider
        - evaluation model: when a provider's name matches evaluations.model.provider

        Keys don't get overwritten if they already exist in the agent config.

        Args:
            agent_config: Configuration from agent.yaml
            global_config: GlobalConfig instance from ~/.holodeck/config.yaml

        Returns:
            Merged configuration dictionary
        """
        # Return early if missing required data
        if not agent_config or "model" not in agent_config:
            return agent_config if agent_config else {}

        if not global_config or not global_config.providers:
            return agent_config

        # Get the provider types from agent config
        agent_model_provider = agent_config["model"].get("provider")
        if not agent_model_provider:
            return agent_config

        # Find matching provider in global config and merge to agent model
        for provider in global_config.providers.values():
            if provider.provider == agent_model_provider:
                # Convert provider to dict and merge non-conflicting keys
                provider_dict = provider.model_dump(exclude_unset=True)
                for key, value in provider_dict.items():
                    if key not in agent_config["model"]:
                        agent_config["model"][key] = value
                break

        # Also merge global provider config to evaluation model if it exists
        if (
            "evaluations" in agent_config
            and isinstance(agent_config["evaluations"], dict)
            and "model" in agent_config["evaluations"]
            and isinstance(agent_config["evaluations"]["model"], dict)
        ):
            eval_model: dict[str, Any] = agent_config["evaluations"]["model"]
            eval_model_provider = eval_model.get("provider")
            if eval_model_provider:
                for provider in global_config.providers.values():
                    if provider.provider == eval_model_provider:
                        # Convert provider to dict and merge non-conflicting keys
                        provider_dict = provider.model_dump(exclude_unset=True)
                        for key, value in provider_dict.items():
                            if key not in eval_model:
                                eval_model[key] = value
                        break

        return agent_config

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

    def resolve_execution_config(
        self,
        cli_config: ExecutionConfig | None,
        yaml_config: ExecutionConfig | None,
        defaults: dict[str, Any],
    ) -> ExecutionConfig:
        """Resolve execution configuration with priority hierarchy.

        Configuration priority (highest to lowest):
        1. CLI flags (cli_config)
        2. agent.yaml execution section (yaml_config)
        3. Environment variables (HOLODECK_* vars)
        4. Built-in defaults

        Args:
            cli_config: Execution config from CLI flags (optional)
            yaml_config: Execution config from agent.yaml (optional)
            defaults: Dictionary of default values

        Returns:
            Resolved ExecutionConfig with all fields populated
        """
        resolved: dict[str, Any] = {}
        env_vars = dict(os.environ)

        # List of all configuration fields
        fields = [
            "file_timeout",
            "llm_timeout",
            "download_timeout",
            "cache_enabled",
            "cache_dir",
            "verbose",
            "quiet",
        ]

        for field in fields:
            # Priority 1: CLI flag
            if cli_config and getattr(cli_config, field, None) is not None:
                resolved[field] = getattr(cli_config, field)
            # Priority 2: agent.yaml execution section
            elif yaml_config and getattr(yaml_config, field, None) is not None:
                resolved[field] = getattr(yaml_config, field)
            # Priority 3: Environment variable
            elif (env_value := _get_env_value(field, env_vars)) is not None:
                resolved[field] = env_value
            # Priority 4: Built-in default
            else:
                resolved[field] = defaults.get(field)

        return ExecutionConfig(**resolved)
