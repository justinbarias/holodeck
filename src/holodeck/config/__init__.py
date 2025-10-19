"""Configuration loading, validation, and management for HoloDeck agents.

This package provides tools for loading, parsing, validating, and managing
HoloDeck agent configurations from YAML files.

Main components:
- ConfigLoader: Load and validate agent.yaml files
- Environment variable substitution (${VAR_NAME} pattern)
- Validation utilities for configuration data
- Default configuration templates
"""

from holodeck.config.env_loader import get_env_var, load_env_file, substitute_env_vars
from holodeck.config.loader import ConfigLoader

__all__ = [
    "ConfigLoader",
    "substitute_env_vars",
    "get_env_var",
    "load_env_file",
]
