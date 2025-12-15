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


# Provider name mapping from GlobalConfig.vectorstores to DatabaseConfig.provider
_PROVIDER_MAPPING: dict[str, str] = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "qdrant": "qdrant",
    "weaviate": "weaviate",
    "chromadb": "chromadb",
    "faiss": "faiss",
    "pinecone": "pinecone",
    "azure-ai-search": "azure-ai-search",
    "azure-cosmos-mongo": "azure-cosmos-mongo",
    "azure-cosmos-nosql": "azure-cosmos-nosql",
    "sql-server": "sql-server",
    "in-memory": "in-memory",
}


def _convert_vectorstore_to_database_config(
    vectorstore_config: VectorstoreConfig,
) -> DatabaseConfig:
    """Convert VectorstoreConfig to DatabaseConfig.

    VectorstoreConfig (global) has:
    - provider: str (e.g., "redis", "postgres")
    - connection_string: str
    - options: dict[str, Any] | None

    DatabaseConfig (tool) has:
    - provider: Literal[...] with specific provider names
    - connection_string: str | None
    - Extra fields allowed via ConfigDict(extra="allow")

    Args:
        vectorstore_config: Global vectorstore configuration

    Returns:
        DatabaseConfig suitable for VectorstoreTool
    """
    # Map global provider names to DatabaseConfig provider literals
    mapped_provider = _PROVIDER_MAPPING.get(
        vectorstore_config.provider.lower(),
        vectorstore_config.provider,
    )

    # Build DatabaseConfig with options merged as extra fields
    config_dict: dict[str, Any] = {
        "provider": mapped_provider,
        "connection_string": vectorstore_config.connection_string,
    }

    # Merge options as extra fields (DatabaseConfig allows extra)
    if vectorstore_config.options:
        config_dict.update(vectorstore_config.options)

    return DatabaseConfig(**config_dict)


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

        Merges:
        - Global LLM provider configs into agent model and evaluation model
        - Global vectorstore configs into tool database fields (by name reference)

        Keys don't get overwritten if they already exist in the agent config.

        Args:
            agent_config: Configuration from agent.yaml
            global_config: GlobalConfig instance from ~/.holodeck/config.yaml

        Returns:
            Merged configuration dictionary
        """
        # Return early if missing required data
        if not agent_config:
            return {}

        if not global_config:
            return agent_config

        # Merge LLM provider configs
        if "model" in agent_config and global_config.providers:
            agent_model_provider = agent_config["model"].get("provider")
            if agent_model_provider:
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
                            provider_dict = provider.model_dump(exclude_unset=True)
                            for key, value in provider_dict.items():
                                if key not in eval_model:
                                    eval_model[key] = value
                            break

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
            # Initialize tools list if missing or invalid
            tools_missing = "tools" not in agent_config
            tools_invalid = not isinstance(agent_config.get("tools"), list)
            if tools_missing or tools_invalid:
                agent_config["tools"] = []

            self._merge_mcp_servers(agent_config["tools"], global_config.mcp_servers)

        return agent_config

    def _resolve_vectorstore_references(
        self,
        tools: list[Any],
        vectorstores: dict[str, VectorstoreConfig],
    ) -> None:
        """Resolve string database references in vectorstore tools.

        For each vectorstore tool with a string database field, look up the
        named vectorstore in global config and convert to DatabaseConfig.

        Args:
            tools: List of tool configurations (modified in-place)
            vectorstores: Named vectorstore configurations from global config
        """
        for tool in tools:
            # Skip non-dict tools (shouldn't happen, but be defensive)
            if not isinstance(tool, dict):
                continue

            if tool.get("type") != "vectorstore":
                continue

            database = tool.get("database")

            # Skip if database is None or already a dict (DatabaseConfig)
            if database is None or isinstance(database, dict):
                continue

            # database is a string reference
            if isinstance(database, str):
                if database not in vectorstores:
                    logger.warning(
                        f"Vectorstore tool references unknown database '{database}'. "
                        f"Available: {list(vectorstores.keys())}. "
                        f"Falling back to in-memory storage."
                    )
                    tool["database"] = None
                    continue

                # Convert VectorstoreConfig to DatabaseConfig dict
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
        Global MCP servers are appended to the tools list only if no agent-level
        tool has the same name.

        Args:
            tools: Agent's tools list (modified in-place)
            global_mcp_servers: Global MCP servers from GlobalConfig
        """
        # Build set of existing tool names (all tools, not just MCP)
        existing_names: set[str] = set()
        for tool in tools:
            if isinstance(tool, dict) and "name" in tool:
                existing_names.add(tool["name"])

        # Append global MCP servers that don't conflict with existing tools
        for mcp_server in global_mcp_servers:
            if mcp_server.name in existing_names:
                logger.debug(
                    f"Skipping global MCP server '{mcp_server.name}' - "
                    f"agent has tool with same name (agent takes precedence)"
                )
                continue

            # Convert MCPTool to dict for YAML compatibility
            tool_dict = mcp_server.model_dump(
                exclude_unset=True, exclude_none=True, mode="json"
            )
            tools.append(tool_dict)
            logger.debug(
                f"Merged global MCP server '{mcp_server.name}' into agent tools"
            )

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
            # Priority 3: Project config execution section
            elif project_config and getattr(project_config, field, None) is not None:
                resolved[field] = getattr(project_config, field)
            # Priority 4: User config execution section (~/.holodeck/)
            elif user_config and getattr(user_config, field, None) is not None:
                resolved[field] = getattr(user_config, field)
            # Priority 5: Environment variable
            elif (env_value := _get_env_value(field, env_vars)) is not None:
                resolved[field] = env_value
            # Priority 6: Built-in default
            else:
                resolved[field] = defaults.get(field)

        return ExecutionConfig(**resolved)

    def resolve_vectorstore_database_config(
        self, agent_yaml_path: str, vectorstore_name: str | None = None
    ) -> dict[str, Any] | None:
        """Resolve vector database configuration with proper precedence.

        Configuration precedence (highest to lowest):
        1. Tool-specific database config in agent.yaml
        2. Project-level vectorstore config (.holodeck/config.yaml or config.yaml)
        3. User-level vectorstore config (~/.holodeck/config.yaml)
        4. In-memory fallback (if no config found)

        This allows tools to inherit database configuration from project or
        user-level settings while allowing per-tool overrides.

        Args:
            agent_yaml_path: Path to agent.yaml for resolving project config
            vectorstore_name: Optional name of specific vectorstore to resolve.
                If None, returns the first available vectorstore.

        Returns:
            Dictionary with database configuration (provider, connection_string, etc.)
            or None if using in-memory fallback

        Raises:
            ConfigError: If configuration is invalid
        """
        # Load project and user configs
        agent_dir = str(Path(agent_yaml_path).parent)
        project_config = self.load_project_config(agent_dir)
        user_config = self.load_global_config()

        # Get vectorstores from project config first, then user config
        vectorstores: dict[str, VectorstoreConfig] | None = None
        if project_config and project_config.vectorstores:
            vectorstores = project_config.vectorstores
        elif user_config and user_config.vectorstores:
            vectorstores = user_config.vectorstores

        if not vectorstores:
            return None

        # If specific name requested, look it up
        if vectorstore_name:
            if vectorstore_name in vectorstores:
                vectorstore = vectorstores[vectorstore_name]
                return _convert_vectorstore_to_database_config(vectorstore).model_dump()
            return None

        # Return first vectorstore as default (backward compatibility)
        first_key = next(iter(vectorstores.keys()))
        vectorstore = vectorstores[first_key]
        return _convert_vectorstore_to_database_config(vectorstore).model_dump()


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
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to YAML-safe dict (exclude unset/None for clean output)
        # Note: We don't use exclude_defaults=True because the 'type' field
        # in MCP tools is essential for discriminated union parsing
        config_dict = config.model_dump(
            exclude_unset=True, exclude_none=True, mode="json"
        )

        # Write YAML with readable formatting
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

    Raises DuplicateServerError if:
    1. Same registry_name (exact duplicate from registry)
    2. Same name with no registry_name (manual duplicate)
    3. Same name with different registry_name (conflict warning via exception)

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

        # Case 1: Exact registry duplicate
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

        # Case 2: Name conflict (same name, different or no registry)
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

    Loads the agent configuration, appends the MCP tool, and saves.
    Checks for duplicate servers before adding.

    Args:
        agent_path: Path to agent.yaml file
        mcp_tool: MCPTool configuration to add

    Raises:
        FileNotFoundError: If agent.yaml doesn't exist
        DuplicateServerError: If server already configured
        ConfigError: If YAML parsing or writing fails
    """
    loader = ConfigLoader()

    # Load existing agent config
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

    # Initialize tools list if missing
    if "tools" not in agent_config:
        agent_config["tools"] = []

    # Check for duplicates
    _check_mcp_duplicate(agent_config["tools"], mcp_tool)

    # Convert MCPTool to dict for YAML (exclude defaults/None for clean output)
    # Note: We don't use exclude_defaults=True because the 'type' field is
    # essential for discriminated union parsing and must always be included
    tool_dict = mcp_tool.model_dump(exclude_unset=True, exclude_none=True, mode="json")

    # Append to tools list
    agent_config["tools"].append(tool_dict)

    # Write back to YAML
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

    Loads or creates global configuration, adds the MCP server to
    the mcp_servers list, and saves.

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

    # Load existing global config or create new one
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

    # Initialize mcp_servers list if None
    if global_config.mcp_servers is None:
        global_config.mcp_servers = []

    # Convert existing mcp_servers to dicts for duplicate check
    existing_tools = [t.model_dump(mode="json") for t in global_config.mcp_servers]

    # Check for duplicates
    _check_mcp_duplicate(existing_tools, mcp_tool)

    # Add new server
    global_config.mcp_servers.append(mcp_tool)

    # Save updated config
    return save_global_config(global_config, global_path)


def remove_mcp_server_from_agent(
    agent_path: Path,
    server_name: str,
) -> None:
    """Remove an MCP server from agent.yaml tools list.

    Loads the agent configuration, finds and removes the MCP tool
    by name, and saves the updated configuration.

    Args:
        agent_path: Path to agent.yaml file
        server_name: Name of the MCP server to remove

    Raises:
        FileNotFoundError: If agent.yaml doesn't exist
        ServerNotFoundError: If server not found in configuration
        ConfigError: If YAML parsing or writing fails
    """
    loader = ConfigLoader()

    # Load existing agent config
    try:
        agent_config = loader.parse_yaml(str(agent_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            str(agent_path),
            f"Agent file not found: {agent_path}",
        ) from e

    if agent_config is None:
        agent_config = {}

    # Get tools list (empty if missing)
    tools = agent_config.get("tools", [])

    # Find and remove the MCP server by name
    original_len = len(tools)
    tools = [
        tool
        for tool in tools
        if not (tool.get("type") == "mcp" and tool.get("name") == server_name)
    ]

    # Check if anything was removed
    if len(tools) == original_len:
        raise ServerNotFoundError(server_name, str(agent_path))

    # Update tools list
    agent_config["tools"] = tools

    # Write back to YAML
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

    Loads the global configuration, finds and removes the MCP server
    by name, and saves the updated configuration.

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

    # Load existing global config from the specified path
    loader = ConfigLoader()
    if global_path.exists():
        # Load from custom path
        raw_config = loader.parse_yaml(str(global_path))
        if raw_config is None:
            raise ServerNotFoundError(server_name, "global configuration")

        # Filter MCP servers directly from raw data
        mcp_servers_raw = [
            s for s in raw_config.get("mcp_servers", []) if s.get("type") == "mcp"
        ]

        # Find and remove the server by name
        original_len = len(mcp_servers_raw)
        mcp_servers_raw = [s for s in mcp_servers_raw if s.get("name") != server_name]

        # Check if anything was removed
        if len(mcp_servers_raw) == original_len:
            raise ServerNotFoundError(server_name, "global configuration")

        # Parse to Pydantic models only when needed
        mcp_servers = (
            [MCPTool(**s) for s in mcp_servers_raw] if mcp_servers_raw else None
        )

        # Create GlobalConfig with updated mcp_servers
        global_config = GlobalConfig(
            providers=raw_config.get("providers"),
            vectorstores=raw_config.get("vectorstores"),
            execution=raw_config.get("execution"),
            deployment=raw_config.get("deployment"),
            mcp_servers=mcp_servers,
        )
    else:
        # No config file exists
        raise ServerNotFoundError(server_name, "global configuration")

    # Save updated config
    return save_global_config(global_config, global_path)


def get_mcp_servers_from_agent(agent_path: Path) -> list[MCPTool]:
    """Get all MCP servers from agent.yaml tools list.

    Loads the agent configuration and filters tools to return only
    those with type='mcp'.

    Args:
        agent_path: Path to agent.yaml file

    Returns:
        List of MCPTool objects from agent config (empty list if no MCP tools)

    Raises:
        FileNotFoundError: If agent file doesn't exist
        ConfigError: If agent config is invalid YAML
    """
    loader = ConfigLoader()

    # Load agent config (raises FileNotFoundError if missing)
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

    # Filter and convert MCP tools
    mcp_servers: list[MCPTool] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "mcp":
            continue

        try:
            mcp_tool = MCPTool(**tool)
            mcp_servers.append(mcp_tool)
        except Exception as e:
            logger.warning(
                f"Failed to parse MCP tool '{tool.get('name', 'unknown')}': {e}"
            )
            continue

    return mcp_servers


def get_mcp_servers_from_global(global_path: Path | None = None) -> list[MCPTool]:
    """Get all MCP servers from global config.

    Loads the global configuration from ~/.holodeck/config.yaml and returns
    the mcp_servers list.

    Args:
        global_path: Optional path to global config (default: ~/.holodeck/config.yaml)

    Returns:
        List of MCPTool objects from global config (empty list if no config or servers)
    """
    loader = ConfigLoader()

    # Load global config (returns None if not found)
    global_config = loader.load_global_config()

    if global_config is None:
        return []

    if global_config.mcp_servers is None:
        return []

    return global_config.mcp_servers
