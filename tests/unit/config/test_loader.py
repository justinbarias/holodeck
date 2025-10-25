"""Tests for configuration file discovery (T008).

Tests for user-level and project-level configuration file discovery,
file extension handling, and graceful missing file handling.
"""

from pathlib import Path
from typing import Any

import yaml

from holodeck.config.loader import ConfigLoader
from holodeck.models.config import GlobalConfig


class TestUserLevelConfigDiscovery:
    """Tests for user-level config discovery at ~/.holodeck/config.yml|yaml."""

    def test_load_user_config_yml_preferred(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that .yml is preferred over .yaml for user config."""
        # Set up home directory
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # Create both .yml and .yaml files
        config_content = {
            "providers": {"openai": {"provider": "openai", "name": "gpt-4o"}}
        }
        yml_file = holodeck_dir / "config.yml"
        yaml_file = holodeck_dir / "config.yaml"

        yml_file.write_text(yaml.dump(config_content))
        yaml_file.write_text(
            yaml.dump(
                {"providers": {"openai": {"provider": "openai", "name": "gpt-4-turbo"}}}
            )
        )

        loader = ConfigLoader()
        result = loader.load_global_config()

        # Should load from .yml, not .yaml
        assert isinstance(result, GlobalConfig)
        assert result.providers is not None
        assert result.providers["openai"].name == "gpt-4o"

    def test_load_user_config_yml_file(self, temp_dir: Path, monkeypatch: Any) -> None:
        """Test loading user config from ~/.holodeck/config.yml."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        config_content = {
            "providers": {"openai": {"provider": "openai", "name": "gpt-4o"}}
        }
        yml_file = holodeck_dir / "config.yml"
        yml_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        result = loader.load_global_config()

        assert isinstance(result, GlobalConfig)
        assert result.providers is not None
        assert "openai" in result.providers

    def test_load_user_config_yaml_fallback(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test loading user config from ~/.holodeck/config.yaml as fallback."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        config_content = {
            "providers": {"openai": {"provider": "openai", "name": "gpt-4o"}}
        }
        yaml_file = holodeck_dir / "config.yaml"
        yaml_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        result = loader.load_global_config()

        assert isinstance(result, GlobalConfig)
        assert result.providers is not None

    def test_missing_user_config_returns_none(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that missing user config returns None gracefully."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        loader = ConfigLoader()
        result = loader.load_global_config()

        assert result is None

    def test_both_user_config_files_exist_warning(
        self, temp_dir: Path, monkeypatch: Any, caplog: Any
    ) -> None:
        """Test that warning is logged when both .yml and .yaml exist."""
        import logging

        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        config_content = {
            "providers": {"openai": {"provider": "openai", "name": "gpt-4o"}}
        }
        yml_file = holodeck_dir / "config.yml"
        yaml_file = holodeck_dir / "config.yaml"

        yml_file.write_text(yaml.dump(config_content))
        yaml_file.write_text(yaml.dump(config_content))

        with caplog.at_level(logging.INFO):
            loader = ConfigLoader()
            result = loader.load_global_config()

        assert isinstance(result, GlobalConfig)
        # Check that info message about preference was logged
        assert any("prefer" in record.message.lower() for record in caplog.records)


class TestProjectLevelConfigDiscovery:
    """Tests for project-level config discovery at project root."""

    def test_load_project_config_yml_file(self, temp_dir: Path) -> None:
        """Test loading project config from config.yml in project root."""
        config_content = {
            "providers": {"openai": {"provider": "openai", "name": "gpt-4o"}}
        }
        config_file = temp_dir / "config.yml"
        config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        result = loader.load_project_config(str(temp_dir))

        assert isinstance(result, GlobalConfig)
        assert result.providers is not None

    def test_load_project_config_yaml_fallback(self, temp_dir: Path) -> None:
        """Test loading project config from config.yaml as fallback."""
        config_content = {
            "providers": {"openai": {"provider": "openai", "name": "gpt-4o"}}
        }
        config_file = temp_dir / "config.yaml"
        config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        result = loader.load_project_config(str(temp_dir))

        assert isinstance(result, GlobalConfig)
        assert result.providers is not None

    def test_project_config_yml_preferred(self, temp_dir: Path) -> None:
        """Test that .yml is preferred over .yaml for project config."""
        config_content = {
            "providers": {"openai": {"provider": "openai", "name": "gpt-4o"}}
        }
        yml_file = temp_dir / "config.yml"
        yaml_file = temp_dir / "config.yaml"

        yml_file.write_text(yaml.dump(config_content))
        yaml_file.write_text(
            yaml.dump(
                {"providers": {"openai": {"provider": "openai", "name": "gpt-4-turbo"}}}
            )
        )

        loader = ConfigLoader()
        result = loader.load_project_config(str(temp_dir))

        # Should load from .yml, not .yaml
        assert isinstance(result, GlobalConfig)
        assert result.providers is not None
        assert result.providers["openai"].name == "gpt-4o"

    def test_missing_project_config_returns_none(self, temp_dir: Path) -> None:
        """Test that missing project config returns None gracefully."""
        loader = ConfigLoader()
        result = loader.load_project_config(str(temp_dir))

        assert result is None

    def test_project_config_file_not_found_error(self) -> None:
        """Test that invalid project directory raises appropriate error."""
        loader = ConfigLoader()
        # Non-existent directory should still return None gracefully
        result = loader.load_project_config("/nonexistent/path")
        assert result is None
