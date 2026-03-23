# Configuration Loading and Management API

This section documents the HoloDeck configuration system, including YAML loading,
validation, environment variable substitution, schema validation, and configuration management.

## Overview

The configuration system is organized across seven modules:

| Module | Purpose |
|--------|---------|
| `loader` | YAML parsing, global/project config loading, merging, and MCP server helpers |
| `env_loader` | Environment variable substitution (`${VAR}` pattern) and `.env` file loading |
| `validator` | Pydantic error flattening for human-readable validation messages |
| `defaults` | Built-in default constants and embedding dimension lookup |
| `context` | Request-scoped `ContextVar` for agent base directory |
| `manager` | Configuration file creation, path resolution, and YAML generation |
| `schema` | JSON Schema validation for LLM response formats |

---

## ConfigLoader

The main entry point for loading HoloDeck agent configurations from YAML.

::: holodeck.config.loader.ConfigLoader
    options:
      docstring_style: google
      show_source: true
      members:
        - __init__
        - parse_yaml
        - load_agent_yaml
        - load_global_config
        - load_project_config
        - merge_configs
        - resolve_file_path
        - resolve_execution_config

## Module-Level Functions (loader)

### load_agent_with_config

Convenience function that creates a `ConfigLoader`, loads the agent, sets the
`agent_base_dir` context variable, and resolves the execution config in one call.

::: holodeck.config.loader.load_agent_with_config
    options:
      docstring_style: google
      show_source: true

### save_global_config

::: holodeck.config.loader.save_global_config
    options:
      docstring_style: google
      show_source: true

### MCP Server Helpers

Functions for adding, removing, and listing MCP servers in agent and global configs.

::: holodeck.config.loader.add_mcp_server_to_agent
    options:
      docstring_style: google
      show_source: true

::: holodeck.config.loader.add_mcp_server_to_global
    options:
      docstring_style: google
      show_source: true

::: holodeck.config.loader.remove_mcp_server_from_agent
    options:
      docstring_style: google
      show_source: true

::: holodeck.config.loader.remove_mcp_server_from_global
    options:
      docstring_style: google
      show_source: true

::: holodeck.config.loader.get_mcp_servers_from_agent
    options:
      docstring_style: google
      show_source: true

::: holodeck.config.loader.get_mcp_servers_from_global
    options:
      docstring_style: google
      show_source: true

---

## Environment Variable Utilities

Support for dynamic configuration using environment variables with the `${VAR_NAME}` pattern.

::: holodeck.config.env_loader.substitute_env_vars
    options:
      docstring_style: google
      show_source: true

::: holodeck.config.env_loader.get_env_var
    options:
      docstring_style: google
      show_source: true

::: holodeck.config.env_loader.load_env_file
    options:
      docstring_style: google
      show_source: true

---

## Configuration Validation

Utility for converting Pydantic validation errors into user-friendly messages.

::: holodeck.config.validator.flatten_pydantic_errors
    options:
      docstring_style: google
      show_source: true

---

## Default Configuration

Built-in default constants and embedding dimension resolution.

### Constants

| Constant | Type | Description |
|----------|------|-------------|
| `OLLAMA_DEFAULTS` | `dict` | Default Ollama provider settings (endpoint, temperature, max_tokens, top_p, api_key) |
| `OLLAMA_EMBEDDING_DEFAULTS` | `dict` | Default Ollama embedding model (`nomic-embed-text:latest`) |
| `DEFAULT_EXECUTION_CONFIG` | `dict` | Default execution settings (timeouts, cache, verbosity) |
| `EMBEDDING_MODEL_DIMENSIONS` | `dict` | Known embedding model dimension mappings (OpenAI and Ollama models) |

### get_embedding_dimensions

::: holodeck.config.defaults.get_embedding_dimensions
    options:
      docstring_style: google
      show_source: true

---

## Configuration Context

Request-scoped context variable for passing the agent base directory through async
call stacks without explicit parameter threading.

### agent_base_dir

```python
from contextvars import ContextVar

agent_base_dir: ContextVar[str | None] = ContextVar("agent_base_dir", default=None)
```

Set at CLI entry points (e.g., `holodeck test`, `holodeck chat`) to the parent
directory of `agent.yaml`. Read downstream by tools and resolvers that need to
resolve relative file paths.

**Usage:**

```python
# At CLI entry point:
from holodeck.config.context import agent_base_dir
agent_base_dir.set(str(Path(agent_yaml_path).parent))

# Anywhere downstream:
base_dir = agent_base_dir.get()  # Returns str | None
```

---

## ConfigManager

Manager class for configuration file operations: creating defaults, resolving
paths, generating YAML, and writing config files.

::: holodeck.config.manager.ConfigManager
    options:
      docstring_style: google
      show_source: true
      members:
        - create_default_config
        - get_config_path
        - generate_config_content
        - write_config

---

## Schema Validation

JSON Schema validation for LLM response formats, aligned with OpenAI structured
output requirements. Only Basic JSON Schema keywords are supported.

### ALLOWED_KEYWORDS

```python
ALLOWED_KEYWORDS = {
    "type", "properties", "required", "additionalProperties",
    "items", "enum", "default", "description", "minimum", "maximum",
}
```

### SchemaValidator

::: holodeck.config.schema.SchemaValidator
    options:
      docstring_style: google
      show_source: true
      members:
        - validate_schema
        - load_schema_from_file

---

## Related Documentation

- [Data Models](models.md): Configuration model definitions
- [CLI Commands](cli.md): CLI API reference
- [YAML Schema](../guides/agent-configuration.md): Agent configuration YAML format
