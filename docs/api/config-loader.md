# Configuration Loading and Management API

This section documents the HoloDeck configuration system, including YAML loading,
validation, environment variable substitution, and default configuration management.

## Overview

The configuration system is built on three pillars:

1. **Loading**: Parse YAML agent configuration files
2. **Validation**: Validate against Pydantic models with detailed error messages
3. **Merging**: Combine default settings, user config, and environment overrides

## ConfigLoader

The main entry point for loading HoloDeck agent configurations.

::: holodeck.config.loader.ConfigLoader
    options:
      docstring_style: google
      show_source: true
      members:
        - load
        - load_raw
        - validate

## Environment Variable Utilities

Support for dynamic configuration using environment variables with `${VAR_NAME}` pattern.

::: holodeck.config.env_loader.substitute_env_vars
    options:
      docstring_style: google

::: holodeck.config.env_loader.get_env_var
    options:
      docstring_style: google

::: holodeck.config.env_loader.load_env_file
    options:
      docstring_style: google

## Configuration Validation

Schema validation and error normalization utilities for configuration validation.

::: holodeck.config.validator.normalize_errors
    options:
      docstring_style: google

::: holodeck.config.validator.flatten_pydantic_errors
    options:
      docstring_style: google

::: holodeck.config.validator.validate_field_exists
    options:
      docstring_style: google

::: holodeck.config.validator.validate_mutually_exclusive
    options:
      docstring_style: google

::: holodeck.config.validator.validate_range
    options:
      docstring_style: google

::: holodeck.config.validator.validate_enum
    options:
      docstring_style: google

::: holodeck.config.validator.validate_path_exists
    options:
      docstring_style: google

## Default Configuration

Utilities for generating default configuration templates for common components.

::: holodeck.config.defaults.get_default_model_config
    options:
      docstring_style: google

::: holodeck.config.defaults.get_default_tool_config
    options:
      docstring_style: google

::: holodeck.config.defaults.get_default_evaluation_config
    options:
      docstring_style: google

## Configuration Merging

::: holodeck.config.merge.ConfigMerger
    options:
      docstring_style: google
      show_source: true

## Related Documentation

- [Data Models](models.md): Configuration model definitions
- [CLI Commands](cli.md): CLI API reference
- [YAML Schema](../guides/agent-configuration.md): Agent configuration YAML format
