"""Configuration merging logic for HoloDeck.

This module provides utilities for merging configurations at different levels:
- User-level (~/.holodeck/config.yml)
- Project-level (config.yml in project root)
- Agent-level (agent.yaml)

Precedence (highest to lowest):
1. Agent-level configuration (explicit agent settings always win)
2. Project-level configuration (project overrides user)
3. User-level configuration (global defaults)
"""

import logging
from typing import Any

from holodeck.models.config import GlobalConfig

logger = logging.getLogger(__name__)


class ConfigMerger:
    """Merges configurations with proper precedence and inheritance rules."""

    @staticmethod
    def merge_global_configs(
        user_config: GlobalConfig | None, project_config: GlobalConfig | None
    ) -> GlobalConfig | None:
        """Merge user-level and project-level global configurations.

        Project-level config overrides user-level config when both are present.

        Args:
            user_config: Global configuration from ~/.holodeck/config.yml|yaml
            project_config: Global configuration from project root config.yml|yaml

        Returns:
            Merged GlobalConfig instance, or None if neither config exists
        """
        if user_config is None and project_config is None:
            return None

        if user_config is None:
            return project_config

        if project_config is None:
            return user_config

        # Merge project config (override) into user config (base)
        user_dict = user_config.model_dump()
        project_dict = project_config.model_dump()

        merged_dict = ConfigMerger._deep_merge_dicts(user_dict, project_dict)
        return GlobalConfig(**merged_dict)

    @staticmethod
    def merge_agent_with_global(
        agent_config: dict[str, Any], global_config: GlobalConfig | None
    ) -> dict[str, Any]:
        """Merge agent configuration with global configuration.

        Agent-level settings take precedence. When inherit_global is False,
        only agent settings are used.

        Args:
            agent_config: Agent configuration from agent.yaml
            global_config: Merged global configuration (user + project level)

        Returns:
            Merged configuration dict with agent settings taking precedence
        """
        # If inherit_global is explicitly false, return agent config as-is
        if agent_config.get("inherit_global") is False:
            logger.info("inherit_global set to false; using only agent configuration")
            # Remove the inherit_global flag from the config
            merged = dict(agent_config)
            merged.pop("inherit_global", None)
            return merged

        # If no global config or agent has explicit config, use agent config
        if global_config is None:
            merged = dict(agent_config)
            merged.pop("inherit_global", None)
            return merged

        # Merge global config (base) with agent config (override)
        global_dict = global_config.model_dump()
        agent_dict = dict(agent_config)
        agent_dict.pop("inherit_global", None)

        # For agent-level properties (response_format, tools, etc),
        # agent config completely overrides global
        merged_dict = ConfigMerger._deep_merge_dicts(global_dict, agent_dict)

        return merged_dict

    @staticmethod
    def _deep_merge_dicts(
        base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge override dict into base dict.

        Values from override take precedence. For nested dicts, merging is
        recursive. For other types, override completely replaces base.

        Args:
            base: Base dictionary (lowest precedence)
            override: Override dictionary (highest precedence)

        Returns:
            Merged dictionary with override taking precedence
        """
        result = dict(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dicts
                result[key] = ConfigMerger._deep_merge_dicts(result[key], value)
            else:
                # Override completely replaces base value
                result[key] = value

        return result
