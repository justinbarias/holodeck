"""Tests for configuration file discovery (T008).

Tests for user-level and project-level configuration file discovery,
file extension handling, and graceful missing file handling.
"""

from pathlib import Path
from typing import Any

import yaml

from holodeck.config.defaults import DEFAULT_EXECUTION_CONFIG
from holodeck.config.loader import ConfigLoader
from holodeck.models.config import ExecutionConfig, GlobalConfig


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


class TestExecutionConfigResolution:
    """Tests for ExecutionConfig resolution with priority hierarchy."""

    def test_cli_overrides_all(self) -> None:
        """CLI flags take highest priority over YAML, env, and defaults."""
        cli_config = ExecutionConfig(
            file_timeout=100,
            llm_timeout=200,
            download_timeout=150,
            cache_enabled=False,
            cache_dir="/custom/cache",
            verbose=True,
            quiet=False,
        )

        yaml_config = ExecutionConfig(
            file_timeout=50,
            llm_timeout=80,
            download_timeout=60,
            cache_enabled=True,
            cache_dir="/yaml/cache",
            verbose=False,
        )

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 100  # CLI
        assert resolved.llm_timeout == 200  # CLI
        assert resolved.download_timeout == 150  # CLI
        assert resolved.cache_enabled is False  # CLI
        assert resolved.cache_dir == "/custom/cache"  # CLI
        assert resolved.verbose is True  # CLI

    def test_yaml_overrides_env_and_defaults(self, monkeypatch: Any) -> None:
        """YAML config takes priority over env vars and defaults."""
        # Clear env vars to ensure they don't interfere
        monkeypatch.delenv("HOLODECK_FILE_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_LLM_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_DOWNLOAD_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_CACHE_DIR", raising=False)

        cli_config = None

        yaml_config = ExecutionConfig(
            file_timeout=50,
            llm_timeout=80,
            download_timeout=60,
            cache_dir="/yaml/cache",
        )

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 50  # YAML
        assert resolved.llm_timeout == 80  # YAML
        assert resolved.download_timeout == 60  # YAML
        assert resolved.cache_dir == "/yaml/cache"  # YAML
        # Others from defaults
        assert resolved.cache_enabled is True  # defaults
        assert resolved.verbose is False  # defaults

    def test_env_overrides_defaults(self, monkeypatch: Any) -> None:
        """Environment variables take priority over built-in defaults."""
        monkeypatch.setenv("HOLODECK_FILE_TIMEOUT", "25")
        monkeypatch.setenv("HOLODECK_LLM_TIMEOUT", "40")
        monkeypatch.setenv("HOLODECK_DOWNLOAD_TIMEOUT", "30")
        monkeypatch.setenv("HOLODECK_CACHE_ENABLED", "false")
        monkeypatch.setenv("HOLODECK_CACHE_DIR", "/env/cache")
        monkeypatch.setenv("HOLODECK_VERBOSE", "true")
        monkeypatch.setenv("HOLODECK_QUIET", "false")

        cli_config = None
        yaml_config = None

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 25  # env
        assert resolved.llm_timeout == 40  # env
        assert resolved.download_timeout == 30  # env
        assert resolved.cache_enabled is False  # env
        assert resolved.cache_dir == "/env/cache"  # env
        assert resolved.verbose is True  # env
        assert resolved.quiet is False  # env

    def test_all_defaults_used(self, monkeypatch: Any) -> None:
        """All fields use built-in defaults when nothing specified."""
        # Clear all env vars
        monkeypatch.delenv("HOLODECK_FILE_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_LLM_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_DOWNLOAD_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_CACHE_ENABLED", raising=False)
        monkeypatch.delenv("HOLODECK_CACHE_DIR", raising=False)
        monkeypatch.delenv("HOLODECK_VERBOSE", raising=False)
        monkeypatch.delenv("HOLODECK_QUIET", raising=False)

        cli_config = None
        yaml_config = None

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 30  # default
        assert resolved.llm_timeout == 60  # default
        assert resolved.download_timeout == 30  # default
        assert resolved.cache_enabled is True  # default
        assert resolved.cache_dir == ".holodeck/cache"  # default
        assert resolved.verbose is False  # default
        assert resolved.quiet is False  # default

    def test_partial_cli_merges_with_yaml(self, monkeypatch: Any) -> None:
        """CLI config merges with YAML for unspecified fields."""
        monkeypatch.setenv("HOLODECK_VERBOSE", "true")

        cli_config = ExecutionConfig(
            file_timeout=100,
            # Other fields unspecified (None)
        )

        yaml_config = ExecutionConfig(
            llm_timeout=80,
            download_timeout=60,
            cache_dir="/yaml/cache",
        )

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 100  # CLI
        assert resolved.llm_timeout == 80  # YAML
        assert resolved.download_timeout == 60  # YAML
        assert resolved.cache_dir == "/yaml/cache"  # YAML
        assert resolved.verbose is True  # env
        assert resolved.cache_enabled is True  # default

    def test_env_var_type_conversion(self, monkeypatch: Any) -> None:
        """Environment variables are converted to correct types."""
        monkeypatch.setenv("HOLODECK_FILE_TIMEOUT", "45")
        monkeypatch.setenv("HOLODECK_CACHE_ENABLED", "false")
        monkeypatch.setenv("HOLODECK_VERBOSE", "true")

        cli_config = None
        yaml_config = None

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 45
        assert isinstance(resolved.file_timeout, int)
        assert resolved.cache_enabled is False
        assert isinstance(resolved.cache_enabled, bool)
        assert resolved.verbose is True
        assert isinstance(resolved.verbose, bool)

    def test_invalid_env_var_uses_yaml_or_default(self, monkeypatch: Any) -> None:
        """Invalid environment variables are skipped, falling back to YAML/defaults."""
        monkeypatch.setenv("HOLODECK_FILE_TIMEOUT", "invalid_number")
        monkeypatch.setenv("HOLODECK_LLM_TIMEOUT", "75")

        cli_config = None

        yaml_config = ExecutionConfig(
            file_timeout=50,
        )

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 50  # YAML (env invalid, skipped)
        assert resolved.llm_timeout == 75  # env (valid)
