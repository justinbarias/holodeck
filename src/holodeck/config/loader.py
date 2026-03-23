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
from holodeck.lib.errors import (
    ConfigError,
    DuplicateServerError,
    FileNotFoundError,
    ServerNotFoundError,
)
from holodeck.models.agent import Agent
from holodeck.models.config import ExecutionConfig, GlobalConfig, VectorstoreConfig
from holodeck.models.tool import DatabaseConfig, MCPTool

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


def _get_env_value(
    field_name: str, env_vars: os._Environ[str] | dict[str, str]
) -> Any | None:
    """Get environment variable value for a field.

    Args:
        field_name: Name of field to get
        env_vars: Environment variables mapping

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


# Provider alias mapping — only non-identity entries needed.
# Unknown keys pass through unchanged via .get(key, key).
_PROVIDER_ALIASES: dict[str, str] = {"postgresql": "postgres"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Deep merge override dict into base dict (in-place).

    For nested dicts, merging is recursive.
    For other types, override completely replaces base.

    Args:
        base: Base dictionary to merge into (modified in-place)
        override: Dictionary with values to override
    """
    for key, override_value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(override_value, dict)
        ):
            _deep_merge(base[key], override_value)
        else:
            base[key] = override_value


def _read_yaml_with_env_substitution(path: Path) -> dict[str, Any] | None:
    """Read a YAML file with environment variable substitution.

    Reads raw text, substitutes env vars, then parses YAML once.
    Avoids the lossy dict→YAML→regex→dict roundtrip.

    Args:
        path: Path to YAML file

    Returns:
        Parsed dictionary or None if empty

    Raises:
        OSError: If file cannot be read
        yaml.YAMLError: If YAML parsing fails
        ConfigError: If env var substitution fails
    """
    raw_text = path.read_text(encoding="utf-8")
    substituted = substitute_env_vars(raw_text)
    content = yaml.safe_load(substituted)
    return content if content else None


def _convert_vectorstore_to_database_config(
    vectorstore_config: VectorstoreConfig,
) -> DatabaseConfig:
    """Convert VectorstoreConfig to DatabaseConfig.

    Args:
        vectorstore_config: Global vectorstore configuration

    Returns:
        DatabaseConfig suitable for VectorstoreTool
    """
    mapped_provider = _PROVIDER_ALIASES.get(
        vectorstore_config.provider.lower(),
        vectorstore_config.provider,
    )

    config_dict: dict[str, Any] = {
        "provider": mapped_provider,
        "connection_string": vectorstore_config.connection_string,
    }

    if vectorstore_config.options:
        config_dict.update(vectorstore_config.options)

    return DatabaseConfig(**config_dict)


def _merge_provider_into_model(
    model_dict: dict[str, Any],
    provider_key: str,
    providers: dict[str, Any],
) -> None:
    """Merge a global provider entry into a model dict (non-conflicting keys only).

    Lookup strategy: try dict key first, fall back to .provider field match.

    Args:
        model_dict: Model configuration dict (modified in-place)
        provider_key: Provider identifier from model.provider
        providers: GlobalConfig.providers mapping
    """
    # Try by key first
    provider_entry = providers.get(provider_key)
    if provider_entry is None:
        # Fallback: match by .provider field (backwards compat)
        for p in providers.values():
            if p.provider == provider_key:
                provider_entry = p
                break
    if provider_entry is not None:
        provider_dict = provider_entry.model_dump(exclude_unset=True)
        for key, value in provider_dict.items():
            if key not in model_dict:
                model_dict[key] = value


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
        """Initialize the ConfigLoader with empty caches."""
        self._user_config_loaded = False
        self._user_config: GlobalConfig | None = None
        self._project_configs: dict[str, GlobalConfig | None] = {}

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

        try:
            with open(path, encoding="utf-8") as f:
                content = yaml.safe_load(f)
                return content if content is not None else {}
        except OSError as e:
            raise FileNotFoundError(
                file_path,
                f"Configuration file not found at {file_path}. "
                f"Please ensure the file exists at this path.",
            ) from e
        except yaml.YAMLError as e:
            raise ConfigError(
                "yaml_parse",
                f"Failed to parse YAML file {file_path}: {str(e)}",
            ) from e

    def load_agent_yaml(self, file_path: str) -> Agent:
        """Load and validate an agent configuration from YAML.

        This method:
        1. Parses the YAML file with env var substitution (single pass)
        2. Loads and merges user + project configs (project overrides user)
        3. Merges global config into agent config
        4. Validates against Agent schema
        5. Returns an Agent instance

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
        path = Path(file_path)
        try:
            agent_config = _read_yaml_with_env_substitution(path)
        except OSError as e:
            raise FileNotFoundError(
                file_path,
                f"Configuration file not found at {file_path}. "
                f"Please ensure the file exists at this path.",
            ) from e
        except yaml.YAMLError as e:
            raise ConfigError(
                "yaml_parse",
                f"Failed to parse YAML file {file_path}: {str(e)}",
            ) from e

        if not agent_config:
            agent_config = {}

        # Load and deep-merge user + project configs (project overrides user)
        agent_dir = str(path.parent)
        user_config = self.load_global_config()
        project_config = self.load_project_config(agent_dir)
        config = self._merge_global_configs(user_config, project_config)

        # Merge configurations with proper precedence
        merged_config = self.merge_configs(agent_config, config)

        # Validate against Agent schema
        try:
            agent = Agent(**merged_config)
            return agent
        except PydanticValidationError as e:
            error_messages = flatten_pydantic_errors(e)
            error_text = "\n".join(error_messages)
            raise ConfigError(
                "agent_validation",
                f"Invalid agent configuration in {file_path}:\n{error_text}",
            ) from e

    def load_global_config(self) -> GlobalConfig | None:
        """Load global configuration from ~/.holodeck/config.yml|config.yaml.

        Results are cached after first load.

        Returns:
            GlobalConfig instance, or None if no config file exists

        Raises:
            ConfigError: If YAML parsing fails or validation fails
        """
        if self._user_config_loaded:
            return self._user_config

        home_dir = Path.home()
        holodeck_dir = home_dir / ".holodeck"
        result = self._load_config_file(
            holodeck_dir, "global_config", "global configuration"
        )
        self._user_config = result
        self._user_config_loaded = True
        return result

    def load_project_config(self, project_dir: str) -> GlobalConfig | None:
        """Load project-level configuration from config.yml|config.yaml.

        Results are cached per project_dir after first load.

        Args:
            project_dir: Path to project root directory

        Returns:
            GlobalConfig instance, or None if no config file exists

        Raises:
            ConfigError: If YAML parsing fails or validation fails
        """
        if project_dir in self._project_configs:
            return self._project_configs[project_dir]

        project_path = Path(project_dir)
        result = self._load_config_file(
            project_path, "project_config", "project configuration"
        )
        self._project_configs[project_dir] = result
        return result

    def _load_config_file(
        self, config_dir: Path, error_code: str, config_name: str
    ) -> GlobalConfig | None:
        """Load configuration file from directory with .yml/.yaml preference.

        Args:
            config_dir: Directory to search for config files
            error_code: Error code prefix for error messages
            config_name: Human-readable config name for error messages

        Returns:
            GlobalConfig instance, or None if no config file exists

        Raises:
            ConfigError: If YAML parsing fails or validation fails
        """
        yml_path = config_dir / "config.yml"
        yaml_path = config_dir / "config.yaml"

        config_path = None
        if yml_path.exists():
            config_path = yml_path
            if yaml_path.exists():
                logger.info(
                    f"Both {yml_path} and {yaml_path} exist. "
                    f"Using {yml_path} (prefer .yml extension)."
                )
        elif yaml_path.exists():
            config_path = yaml_path

        if config_path is None:
            return None

        try:
            config_dict = _read_yaml_with_env_substitution(config_path)
            if not config_dict:
                return None

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

    def _merge_global_configs(
        self,
        user_config: GlobalConfig | None,
        project_config: GlobalConfig | None,
    ) -> GlobalConfig | None:
        """Deep-merge user and project global configs (project overrides user).

        Args:
            user_config: User-level config from ~/.holodeck/
            project_config: Project-level config from project dir

        Returns:
            Merged GlobalConfig, or None if neither exists
        """
        if user_config is None and project_config is None:
            return None
        if user_config is None:
            return project_config
        if project_config is None:
            return user_config

        base = user_config.model_dump()
        override = project_config.model_dump()
        _deep_merge(base, override)
        return GlobalConfig(**base)

    def merge_configs(
        self, agent_config: dict[str, Any], global_config: GlobalConfig | None
    ) -> dict[str, Any]:
        """Merge agent config with global config using proper precedence.

        Precedence (highest to lowest):
        1. agent.yaml explicit settings
        2. Environment variables (already substituted)
        3. Global settings (merged user + project)

        Merges:
        - Global LLM provider configs into agent model and evaluation model
        - Global vectorstore configs into tool database fields (by name reference)

        Keys don't get overwritten if they already exist in the agent config.

        Args:
            agent_config: Configuration from agent.yaml
            global_config: GlobalConfig instance (merged user + project)

        Returns:
            Merged configuration dictionary
        """
        if not agent_config:
            return {}

        if not global_config:
            return agent_config

        # Merge LLM provider configs (dict key lookup with .provider fallback)
        if "model" in agent_config and global_config.providers:
            agent_model_provider = agent_config["model"].get("provider")
            if agent_model_provider:
                _merge_provider_into_model(
                    agent_config["model"],
                    agent_model_provider,
                    global_config.providers,
                )

            # Also merge global provider config to evaluation model
            if (
                "evaluations" in agent_config
                and isinstance(agent_config["evaluations"], dict)
                and "model" in agent_config["evaluations"]
                and isinstance(agent_config["evaluations"]["model"], dict)
            ):
                eval_model: dict[str, Any] = agent_config["evaluations"]["model"]
                eval_model_provider = eval_model.get("provider")
                if eval_model_provider:
                    _merge_provider_into_model(
                        eval_model,
                        eval_model_provider,
                        global_config.providers,
                    )

        # Resolve vectorstore references in tools
        if (
            global_config.vectorstores
            and "tools" in agent_config
            and isinstance(agent_config["tools"], list)
        ):
            self._resolve_vectorstore_references(
                agent_config["tools"], global_config.vectorstores
            )

        # Merge global MCP servers into agent tools
        if global_config.mcp_servers and len(global_config.mcp_servers) > 0:
            tools_missing = "tools" not in agent_config
            tools_invalid = not isinstance(agent_config.get("tools"), list)
            if tools_missing or tools_invalid:
                agent_config["tools"] = []

            self._merge_mcp_servers(agent_config["tools"], global_config.mcp_servers)

        # Merge global deployment config (agent.yaml takes precedence)
        if global_config.deployment:
            if "deployment" not in agent_config:
                agent_config["deployment"] = global_config.deployment.model_dump(
                    exclude_unset=True
                )
            else:
                global_deploy = global_config.deployment.model_dump(exclude_unset=True)
                agent_deploy = agent_config["deployment"]
                if isinstance(agent_deploy, dict):
                    _deep_merge(global_deploy, agent_deploy)
                    agent_config["deployment"] = global_deploy

        return agent_config

    def _resolve_vectorstore_references(
        self,
        tools: list[Any],
        vectorstores: dict[str, VectorstoreConfig],
    ) -> None:
        """Resolve string database references in vectorstore tools.

        Args:
            tools: List of tool configurations (modified in-place)
            vectorstores: Named vectorstore configurations from global config
        """
        for tool in tools:
            if not isinstance(tool, dict):
                continue

            if tool.get("type") != "vectorstore":
                continue

            database = tool.get("database")

            if database is None or isinstance(database, dict):
                continue

            if isinstance(database, str):
                if database not in vectorstores:
                    logger.warning(
                        f"Vectorstore tool references unknown database '{database}'. "
                        f"Available: {list(vectorstores.keys())}. "
                        f"Falling back to in-memory storage."
                    )
                    tool["database"] = None
                    continue

                vectorstore_config = vectorstores[database]
                database_config = _convert_vectorstore_to_database_config(
                    vectorstore_config
                )
                tool["database"] = database_config.model_dump()
                logger.debug(
                    f"Resolved vectorstore reference '{database}' "
                    f"to provider '{database_config.provider}'"
                )

    def _merge_mcp_servers(
        self,
        tools: list[Any],
        global_mcp_servers: list[MCPTool],
    ) -> None:
        """Merge global MCP servers into agent tools list.

        Agent-level MCP tools with the same name take precedence over global ones.

        Args:
            tools: Agent's tools list (modified in-place)
            global_mcp_servers: Global MCP servers from GlobalConfig
        """
        existing_names: set[str] = set()
        for tool in tools:
            if isinstance(tool, dict) and "name" in tool:
                existing_names.add(tool["name"])

        for mcp_server in global_mcp_servers:
            if mcp_server.name in existing_names:
                logger.debug(
                    f"Skipping global MCP server '{mcp_server.name}' - "
                    f"agent has tool with same name (agent takes precedence)"
                )
                continue

            tool_dict = mcp_server.model_dump(
                exclude_unset=True, exclude_none=True, mode="json"
            )
            tools.append(tool_dict)
            logger.debug(
                f"Merged global MCP server '{mcp_server.name}' into agent tools"
            )

    def resolve_file_path(self, file_path: str, base_dir: str) -> str:
        """Resolve a file path relative to base directory.

        Args:
            file_path: Path to resolve (absolute or relative)
            base_dir: Base directory for relative path resolution

        Returns:
            Absolute path to the file

        Raises:
            FileNotFoundError: If the resolved file doesn't exist
        """
        path = Path(file_path)

        if path.is_absolute():
            resolved = path
        else:
            resolved = (Path(base_dir) / file_path).resolve()

        if not resolved.exists():
            raise FileNotFoundError(
                str(resolved),
                f"Referenced file not found: {resolved}\n"
                f"Please ensure the file exists at this path.",
            )

        return str(resolved)

    def resolve_execution_config(
        self,
        cli_config: ExecutionConfig | None,
        yaml_config: ExecutionConfig | None,
        project_config: ExecutionConfig | None,
        user_config: ExecutionConfig | None,
        defaults: dict[str, Any],
    ) -> ExecutionConfig:
        """Resolve execution configuration with priority hierarchy.

        Configuration priority (highest to lowest):
        1. CLI flags (cli_config)
        2. agent.yaml execution section (yaml_config)
        3. Project config execution section (project_config from ./config.yaml)
        4. User config execution section (user_config from ~/.holodeck/config.yaml)
        5. Environment variables (HOLODECK_* vars)
        6. Built-in defaults

        Args:
            cli_config: Execution config from CLI flags (optional)
            yaml_config: Execution config from agent.yaml (optional)
            project_config: Execution config from project config.yaml (optional)
            user_config: Execution config from ~/.holodeck/config.yaml (optional)
            defaults: Dictionary of default values

        Returns:
            Resolved ExecutionConfig with all fields populated
        """
        resolved: dict[str, Any] = {}

        fields = list(ExecutionConfig.model_fields.keys())

        for field in fields:
            # Priority 1: CLI flag
            if cli_config and getattr(cli_config, field, None) is not None:
                resolved[field] = getattr(cli_config, field)
            # Priority 2: agent.yaml execution section
            elif yaml_config and getattr(yaml_config, field, None) is not None:
                resolved[field] = getattr(yaml_config, field)
            # Priority 3: Project config execution section
            elif project_config and getattr(project_config, field, None) is not None:
                resolved[field] = getattr(project_config, field)
            # Priority 4: User config execution section (~/.holodeck/)
            elif user_config and getattr(user_config, field, None) is not None:
                resolved[field] = getattr(user_config, field)
            # Priority 5: Environment variable
            elif (env_value := _get_env_value(field, os.environ)) is not None:
                resolved[field] = env_value
            # Priority 6: Built-in default
            else:
                resolved[field] = defaults.get(field)

        return ExecutionConfig(**resolved)


def load_agent_with_config(
    agent_config_path: str,
    cli_config: ExecutionConfig | None = None,
) -> tuple[Agent, ExecutionConfig, ConfigLoader]:
    """Load agent and resolve execution config in one call.

    Encapsulates the config-loading boilerplate shared across CLI commands:
    - Creates ConfigLoader, loads agent.yaml
    - Sets agent_base_dir context variable
    - Resolves execution config with full priority hierarchy
    - Returns (agent, resolved_config, loader)

    Args:
        agent_config_path: Path to agent.yaml file
        cli_config: Optional execution config from CLI flags

    Returns:
        Tuple of (Agent, resolved ExecutionConfig, ConfigLoader)
    """
    from holodeck.config.context import agent_base_dir
    from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG

    loader = ConfigLoader()
    agent = loader.load_agent_yaml(agent_config_path)

    # Set the base directory context for resolving relative paths in tools
    agent_dir = str(Path(agent_config_path).parent.resolve())
    agent_base_dir.set(agent_dir)

    # Resolve execution config (CLI > agent.yaml > project > user > env > defaults)
    # Configs are already cached from load_agent_yaml, no duplicate I/O
    project_config = loader.load_project_config(agent_dir)
    project_execution = project_config.execution if project_config else None
    user_config = loader.load_global_config()
    user_execution = user_config.execution if user_config else None

    resolved_config = loader.resolve_execution_config(
        cli_config=cli_config,
        yaml_config=agent.execution,
        project_config=project_execution,
        user_config=user_execution,
        defaults=DEFAULT_EXECUTION_CONFIG,
    )

    return agent, resolved_config, loader


# --- MCP Server Helper Functions ---


def save_global_config(
    config: GlobalConfig,
    path: Path | None = None,
) -> Path:
    """Save GlobalConfig to ~/.holodeck/config.yaml.

    Creates the ~/.holodeck/ directory if it doesn't exist.
    Preserves existing fields when updating.

    Args:
        config: GlobalConfig instance to save
        path: Optional custom path (defaults to ~/.holodeck/config.yaml)

    Returns:
        Path where the configuration was saved

    Raises:
        ConfigError: If file write fails
    """
    if path is None:
        path = Path.home() / ".holodeck" / "config.yaml"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump(
            exclude_unset=True, exclude_none=True, mode="json"
        )

        yaml_content = yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

        path.write_text(yaml_content, encoding="utf-8")
        logger.debug(f"Saved global configuration to {path}")
        return path

    except OSError as e:
        raise ConfigError(
            "global_config_write",
            f"Failed to write global configuration to {path}: {e}",
        ) from e


def _check_mcp_duplicate(
    tools: list[dict[str, Any]],
    new_tool: MCPTool,
) -> None:
    """Check for duplicate MCP servers in tools list.

    Args:
        tools: List of existing tool configurations
        new_tool: The MCPTool being added

    Raises:
        DuplicateServerError: If duplicate or conflict detected
    """
    for tool in tools:
        if tool.get("type") != "mcp":
            continue

        existing_name = tool.get("name")
        existing_registry_name = tool.get("registry_name")

        if (
            new_tool.registry_name
            and existing_registry_name
            and new_tool.registry_name == existing_registry_name
        ):
            raise DuplicateServerError(
                server_name=new_tool.name,
                registry_name=new_tool.registry_name,
                existing_registry_name=existing_registry_name,
            )

        if existing_name == new_tool.name:
            raise DuplicateServerError(
                server_name=new_tool.name,
                registry_name=new_tool.registry_name,
                existing_registry_name=existing_registry_name,
            )


def add_mcp_server_to_agent(
    agent_path: Path,
    mcp_tool: MCPTool,
) -> None:
    """Add an MCP server to agent.yaml tools list.

    Args:
        agent_path: Path to agent.yaml file
        mcp_tool: MCPTool configuration to add

    Raises:
        FileNotFoundError: If agent.yaml doesn't exist
        DuplicateServerError: If server already configured
        ConfigError: If YAML parsing or writing fails
    """
    loader = ConfigLoader()

    try:
        agent_config = loader.parse_yaml(str(agent_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(agent_path),
            "No agent.yaml found. Use --agent to specify a file "
            "or -g for global install.",
        ) from e

    if agent_config is None:
        agent_config = {}

    if "tools" not in agent_config:
        agent_config["tools"] = []

    _check_mcp_duplicate(agent_config["tools"], mcp_tool)

    tool_dict = mcp_tool.model_dump(exclude_unset=True, exclude_none=True, mode="json")

    agent_config["tools"].append(tool_dict)

    try:
        yaml_content = yaml.dump(
            agent_config,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        agent_path.write_text(yaml_content, encoding="utf-8")
        logger.debug(f"Added MCP server '{mcp_tool.name}' to {agent_path}")

    except OSError as e:
        raise ConfigError(
            "agent_config_write",
            f"Failed to write agent configuration to {agent_path}: {e}",
        ) from e


def add_mcp_server_to_global(
    mcp_tool: MCPTool,
    global_path: Path | None = None,
) -> Path:
    """Add an MCP server to global config mcp_servers list.

    Args:
        mcp_tool: MCPTool configuration to add
        global_path: Optional custom path (defaults to ~/.holodeck/config.yaml)

    Returns:
        Path where the configuration was saved

    Raises:
        DuplicateServerError: If server already configured
        ConfigError: If YAML parsing or writing fails
    """
    if global_path is None:
        global_path = Path.home() / ".holodeck" / "config.yaml"

    loader = ConfigLoader()
    global_config = loader.load_global_config()

    if global_config is None:
        global_config = GlobalConfig(
            providers=None,
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=None,
        )

    if global_config.mcp_servers is None:
        global_config.mcp_servers = []

    existing_tools = [t.model_dump(mode="json") for t in global_config.mcp_servers]

    _check_mcp_duplicate(existing_tools, mcp_tool)

    global_config.mcp_servers.append(mcp_tool)

    return save_global_config(global_config, global_path)


def remove_mcp_server_from_agent(
    agent_path: Path,
    server_name: str,
) -> None:
    """Remove an MCP server from agent.yaml tools list.

    Args:
        agent_path: Path to agent.yaml file
        server_name: Name of the MCP server to remove

    Raises:
        FileNotFoundError: If agent.yaml doesn't exist
        ServerNotFoundError: If server not found in configuration
        ConfigError: If YAML parsing or writing fails
    """
    loader = ConfigLoader()

    try:
        agent_config = loader.parse_yaml(str(agent_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(agent_path),
            f"Agent file not found: {agent_path}",
        ) from e

    if agent_config is None:
        agent_config = {}

    tools = agent_config.get("tools", [])

    original_len = len(tools)
    tools = [
        tool
        for tool in tools
        if not (tool.get("type") == "mcp" and tool.get("name") == server_name)
    ]

    if len(tools) == original_len:
        raise ServerNotFoundError(server_name, str(agent_path))

    agent_config["tools"] = tools

    try:
        yaml_content = yaml.dump(
            agent_config,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        agent_path.write_text(yaml_content, encoding="utf-8")
        logger.debug(f"Removed MCP server '{server_name}' from {agent_path}")

    except OSError as e:
        raise ConfigError(
            "agent_config_write",
            f"Failed to write agent configuration to {agent_path}: {e}",
        ) from e


def remove_mcp_server_from_global(
    server_name: str,
    global_path: Path | None = None,
) -> Path:
    """Remove an MCP server from global config mcp_servers list.

    Args:
        server_name: Name of the MCP server to remove
        global_path: Optional custom path (defaults to ~/.holodeck/config.yaml)

    Returns:
        Path where the configuration was saved

    Raises:
        ServerNotFoundError: If server not found in configuration
        ConfigError: If YAML parsing or writing fails
    """
    if global_path is None:
        global_path = Path.home() / ".holodeck" / "config.yaml"

    loader = ConfigLoader()
    if global_path.exists():
        raw_config = loader.parse_yaml(str(global_path))
        if raw_config is None:
            raise ServerNotFoundError(server_name, "global configuration")

        mcp_servers_raw = [
            s for s in raw_config.get("mcp_servers", []) if s.get("type") == "mcp"
        ]

        original_len = len(mcp_servers_raw)
        mcp_servers_raw = [s for s in mcp_servers_raw if s.get("name") != server_name]

        if len(mcp_servers_raw) == original_len:
            raise ServerNotFoundError(server_name, "global configuration")

        mcp_servers = (
            [MCPTool(**s) for s in mcp_servers_raw] if mcp_servers_raw else None
        )

        global_config = GlobalConfig(
            providers=raw_config.get("providers"),
            vectorstores=raw_config.get("vectorstores"),
            execution=raw_config.get("execution"),
            deployment=raw_config.get("deployment"),
            mcp_servers=mcp_servers,
        )
    else:
        raise ServerNotFoundError(server_name, "global configuration")

    return save_global_config(global_config, global_path)


def get_mcp_servers_from_agent(agent_path: Path) -> list[MCPTool]:
    """Get all MCP servers from agent.yaml tools list.

    Args:
        agent_path: Path to agent.yaml file

    Returns:
        List of MCPTool objects from agent config (empty list if no MCP tools)

    Raises:
        FileNotFoundError: If agent file doesn't exist
        ConfigError: If agent config is invalid YAML
    """
    loader = ConfigLoader()

    try:
        agent_config = loader.parse_yaml(str(agent_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(agent_path),
            f"Agent file not found: {agent_path}",
        ) from e

    if agent_config is None:
        return []

    tools = agent_config.get("tools", [])
    if not tools:
        return []

    mcp_servers: list[MCPTool] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "mcp":
            continue

        try:
            mcp_tool = MCPTool(**tool)
            mcp_servers.append(mcp_tool)
        except PydanticValidationError as e:
            logger.warning(
                f"Failed to parse MCP tool '{tool.get('name', 'unknown')}': {e}"
            )
            continue

    return mcp_servers


def get_mcp_servers_from_global(global_path: Path | None = None) -> list[MCPTool]:
    """Get all MCP servers from global config.

    Args:
        global_path: Optional path to global config (default: ~/.holodeck/config.yaml)

    Returns:
        List of MCPTool objects from global config (empty list if no config or servers)
    """
    loader = ConfigLoader()

    global_config = loader.load_global_config()

    if global_config is None:
        return []

    if global_config.mcp_servers is None:
        return []

    return global_config.mcp_servers
