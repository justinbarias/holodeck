"""Tests for configuration inheritance and precedence (T009).

Tests for user-level → project-level → agent precedence,
inherit_global flag, configuration merging, and overrides.
"""

from pathlib import Path
from typing import Any

import yaml

from holodeck.config.loader import ConfigLoader
from holodeck.config.merge import ConfigMerger
from holodeck.models.config import GlobalConfig


class TestConfigurationPrecedence:
    """Tests for configuration precedence hierarchy."""

    def test_user_level_precedence(self, temp_dir: Path, monkeypatch: Any) -> None:
        """Test that user-level config is the base level."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # User-level config
        user_config = {
            "providers": {
                "openai": {"provider": "openai", "name": "gpt-4o", "temperature": 0.7}
            }
        }
        (holodeck_dir / "config.yml").write_text(yaml.dump(user_config))

        loader = ConfigLoader()
        user_config_obj = loader.load_global_config()

        assert user_config_obj is not None
        assert user_config_obj.providers["openai"].temperature == 0.7

    def test_project_overrides_user_level(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that project-level config overrides user-level."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # User-level config
        user_config = {
            "providers": {
                "openai": {"provider": "openai", "name": "gpt-4o", "temperature": 0.7}
            }
        }
        (holodeck_dir / "config.yml").write_text(yaml.dump(user_config))

        # Project-level config (override temperature)
        project_config = {
            "providers": {
                "openai": {"provider": "openai", "name": "gpt-4o", "temperature": 0.3}
            }
        }
        (temp_dir / "config.yml").write_text(yaml.dump(project_config))

        loader = ConfigLoader()
        user_config_obj = loader.load_global_config()
        project_config_obj = loader.load_project_config(str(temp_dir))

        # Merge configs with proper precedence
        merged = ConfigMerger.merge_global_configs(user_config_obj, project_config_obj)

        assert merged is not None
        assert merged.providers["openai"].temperature == 0.3

    def test_agent_overrides_global(self, temp_dir: Path, monkeypatch: Any) -> None:
        """Test that agent-level config overrides global settings."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # Global config
        global_config = {
            "providers": {
                "openai": {"provider": "openai", "name": "gpt-4o", "temperature": 0.7}
            }
        }
        (holodeck_dir / "config.yml").write_text(yaml.dump(global_config))

        # Agent config with override
        agent_config = {
            "name": "test-agent",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "temperature": 0.1,  # Override
            },
            "instructions": {"inline": "Test instructions"},
        }

        loader = ConfigLoader()
        global_config_obj = loader.load_global_config()
        merged = ConfigMerger.merge_agent_with_global(agent_config, global_config_obj)

        # Agent config should override global
        assert merged["model"]["temperature"] == 0.1

    def test_inherit_global_false_disables_inheritance(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that inherit_global: false disables inheritance."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # Global config
        global_config = {
            "providers": {
                "openai": {"provider": "openai", "name": "gpt-4o", "temperature": 0.7}
            }
        }
        (holodeck_dir / "config.yml").write_text(yaml.dump(global_config))

        # Agent config with inherit_global: false
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o", "temperature": 0.9},
            "instructions": {"inline": "Test instructions"},
            "inherit_global": False,
        }

        loader = ConfigLoader()
        global_config_obj = loader.load_global_config()
        merged = ConfigMerger.merge_agent_with_global(agent_config, global_config_obj)

        # Should use only agent config, not global
        assert merged["model"]["temperature"] == 0.9
        assert "inherit_global" not in merged


class TestConfigurationMerging:
    """Tests for configuration merging with proper override semantics."""

    def test_deep_merge_nested_objects(self) -> None:
        """Test that nested objects are merged with project override."""
        user_config = GlobalConfig(
            providers={
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                }
            }
        )

        project_config = GlobalConfig(
            providers={
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "max_tokens": 2000,
                }
            }
        )

        merged = ConfigMerger.merge_global_configs(user_config, project_config)

        assert merged is not None
        # Project config overrides user config
        assert merged.providers["openai"].max_tokens == 2000
        assert merged.providers["openai"].name == "gpt-4o"

    def test_agent_config_completely_replaces_provider(self) -> None:
        """Test that agent config completely replaces provider settings."""
        global_config = GlobalConfig(
            providers={
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                }
            }
        )

        # Agent with different provider temperature (complete override)
        agent_config = {
            "name": "test-agent",
            "model": {
                "provider": "openai",
                "name": "gpt-4-turbo",
                "temperature": 0.1,
            },
            "instructions": {"inline": "Test instructions"},
        }

        merged = ConfigMerger.merge_agent_with_global(agent_config, global_config)

        # Agent override should completely replace
        assert merged["model"]["temperature"] == 0.1
        assert merged["model"]["name"] == "gpt-4-turbo"

    def test_merge_with_none_global_config(self) -> None:
        """Test merging when global config is None."""
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test instructions"},
        }

        merged = ConfigMerger.merge_agent_with_global(agent_config, None)

        assert merged["name"] == "test-agent"
        assert merged["model"]["provider"] == "openai"

    def test_merge_inherits_global_when_agent_not_specified(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that agent inherits global settings when not explicitly specified."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # Global config with settings
        global_config = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "temperature": 0.7,
                }
            }
        }
        (holodeck_dir / "config.yml").write_text(yaml.dump(global_config))

        # Agent config without model settings (should inherit)
        agent_config_dict = {
            "name": "test-agent",
            "instructions": {"inline": "Test instructions"},
        }

        loader = ConfigLoader()
        global_config_obj = loader.load_global_config()

        # Merge should apply global provider settings
        merged = ConfigMerger.merge_agent_with_global(
            agent_config_dict, global_config_obj
        )

        # Should have inherited global provider
        assert merged["providers"] is not None
        assert "openai" in merged["providers"]
