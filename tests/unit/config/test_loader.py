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
from holodeck.models.tool import CommandType, MCPTool, TransportType


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

        with caplog.at_level(logging.INFO, logger="holodeck.config.loader"):
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
            project_config=None,
            user_config=None,
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
            project_config=None,
            user_config=None,
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
            project_config=None,
            user_config=None,
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
            project_config=None,
            user_config=None,
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
            project_config=None,
            user_config=None,
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
            project_config=None,
            user_config=None,
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
            project_config=None,
            user_config=None,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 50  # YAML (env invalid, skipped)
        assert resolved.llm_timeout == 75  # env (valid)

    def test_project_config_overrides_user_config(self, monkeypatch: Any) -> None:
        """Project config takes priority over user config."""
        # Clear env vars
        monkeypatch.delenv("HOLODECK_FILE_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_LLM_TIMEOUT", raising=False)

        project_config = ExecutionConfig(
            file_timeout=45,
            llm_timeout=90,
        )

        user_config = ExecutionConfig(
            file_timeout=35,
            llm_timeout=70,
            download_timeout=40,
        )

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=None,
            yaml_config=None,
            project_config=project_config,
            user_config=user_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 45  # project
        assert resolved.llm_timeout == 90  # project
        assert resolved.download_timeout == 40  # user (not in project)

    def test_user_config_overrides_env_and_defaults(self, monkeypatch: Any) -> None:
        """User config takes priority over env vars and defaults."""
        monkeypatch.setenv("HOLODECK_FILE_TIMEOUT", "25")
        monkeypatch.delenv("HOLODECK_LLM_TIMEOUT", raising=False)

        user_config = ExecutionConfig(
            file_timeout=35,
            llm_timeout=70,
        )

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=None,
            yaml_config=None,
            project_config=None,
            user_config=user_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 35  # user (not env)
        assert resolved.llm_timeout == 70  # user

    def test_yaml_overrides_project_and_user_config(self, monkeypatch: Any) -> None:
        """YAML config takes priority over project and user config."""
        # Clear env vars
        monkeypatch.delenv("HOLODECK_FILE_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_LLM_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_DOWNLOAD_TIMEOUT", raising=False)

        yaml_config = ExecutionConfig(
            file_timeout=100,
        )

        project_config = ExecutionConfig(
            file_timeout=45,
            llm_timeout=90,
        )

        user_config = ExecutionConfig(
            file_timeout=35,
            llm_timeout=70,
            download_timeout=40,
        )

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=None,
            yaml_config=yaml_config,
            project_config=project_config,
            user_config=user_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 100  # yaml
        assert resolved.llm_timeout == 90  # project (not in yaml)
        assert resolved.download_timeout == 40  # user (not in yaml or project)

    def test_full_priority_hierarchy(self, monkeypatch: Any) -> None:
        """Test complete priority: CLI > YAML > project > user > env > default."""
        monkeypatch.setenv("HOLODECK_CACHE_DIR", "/env/cache")
        monkeypatch.delenv("HOLODECK_FILE_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_LLM_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_DOWNLOAD_TIMEOUT", raising=False)
        monkeypatch.delenv("HOLODECK_CACHE_ENABLED", raising=False)
        monkeypatch.delenv("HOLODECK_VERBOSE", raising=False)
        monkeypatch.delenv("HOLODECK_QUIET", raising=False)

        cli_config = ExecutionConfig(file_timeout=200)
        yaml_config = ExecutionConfig(llm_timeout=150)
        project_config = ExecutionConfig(download_timeout=120)
        user_config = ExecutionConfig(cache_enabled=False)
        # env: cache_dir="/env/cache"
        # default: verbose=False, quiet=False

        config_loader = ConfigLoader()
        resolved = config_loader.resolve_execution_config(
            cli_config=cli_config,
            yaml_config=yaml_config,
            project_config=project_config,
            user_config=user_config,
            defaults=DEFAULT_EXECUTION_CONFIG,
        )

        assert resolved.file_timeout == 200  # CLI
        assert resolved.llm_timeout == 150  # YAML
        assert resolved.download_timeout == 120  # project
        assert resolved.cache_enabled is False  # user
        assert resolved.cache_dir == "/env/cache"  # env
        assert resolved.verbose is False  # default
        assert resolved.quiet is False  # default


class TestParseYaml:
    """Tests for parse_yaml method."""

    def test_parse_yaml_returns_dict(self, temp_dir: Path) -> None:
        """Test parse_yaml returns dictionary from valid YAML file."""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("name: test\nvalue: 123")

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert result == {"name": "test", "value": 123}

    def test_parse_yaml_empty_file_returns_empty_dict(self, temp_dir: Path) -> None:
        """Test parse_yaml returns empty dict for empty file."""
        yaml_file = temp_dir / "empty.yaml"
        yaml_file.write_text("")

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert result == {}

    def test_parse_yaml_file_not_found(self, temp_dir: Path) -> None:
        """Test parse_yaml raises FileNotFoundError for missing file."""
        import pytest

        from holodeck.lib.errors import FileNotFoundError

        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.parse_yaml(str(temp_dir / "nonexistent.yaml"))

        assert "not found" in str(exc_info.value).lower()

    def test_parse_yaml_invalid_yaml(self, temp_dir: Path) -> None:
        """Test parse_yaml raises ConfigError for invalid YAML."""
        import pytest

        from holodeck.lib.errors import ConfigError

        yaml_file = temp_dir / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.parse_yaml(str(yaml_file))

        assert "yaml_parse" in str(exc_info.value)


class TestLoadAgentYaml:
    """Tests for load_agent_yaml method."""

    def test_load_agent_yaml_basic(self, temp_dir: Path) -> None:
        """Test loading a valid agent.yaml file."""
        agent_yaml = temp_dir / "agent.yaml"
        agent_content = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are a helpful assistant."},
        }
        agent_yaml.write_text(yaml.dump(agent_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))

        assert agent.name == "test-agent"
        assert agent.model.provider.value == "openai"
        assert agent.instructions.inline == "You are a helpful assistant."

    def test_load_agent_yaml_with_env_substitution(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that environment variables are substituted in agent.yaml."""
        monkeypatch.setenv("TEST_API_KEY", "sk-test-12345")

        agent_yaml = temp_dir / "agent.yaml"
        agent_content = {
            "name": "test-agent",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "api_key": "${TEST_API_KEY}",
            },
            "instructions": {"inline": "Test instructions."},
        }
        agent_yaml.write_text(yaml.dump(agent_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))

        assert agent.model.api_key == "sk-test-12345"

    def test_load_agent_yaml_with_global_config_merge(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that global config is merged into agent config."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # Create global config with API key (name is required)
        global_config = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "api_key": "global-api-key",
                }
            }
        }
        (holodeck_dir / "config.yml").write_text(yaml.dump(global_config))

        # Create agent yaml without API key
        agent_yaml = temp_dir / "agent.yaml"
        agent_content = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test instructions."},
        }
        agent_yaml.write_text(yaml.dump(agent_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))

        # API key should be merged from global config
        assert agent.model.api_key == "global-api-key"

    def test_load_agent_yaml_validation_error(self, temp_dir: Path) -> None:
        """Test that invalid agent config raises ConfigError."""
        import pytest

        from holodeck.lib.errors import ConfigError

        agent_yaml = temp_dir / "agent.yaml"
        # Missing required 'model' field
        agent_content = {
            "name": "test-agent",
            "instructions": {"inline": "Test"},
        }
        agent_yaml.write_text(yaml.dump(agent_content))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_agent_yaml(str(agent_yaml))

        assert "agent_validation" in str(exc_info.value)


class TestLoadConfigFileEdgeCases:
    """Tests for _load_config_file edge cases."""

    def test_config_file_empty_content_returns_none(self, temp_dir: Path) -> None:
        """Test that empty config file returns None."""
        config_file = temp_dir / "config.yml"
        config_file.write_text("")

        loader = ConfigLoader()
        result = loader.load_project_config(str(temp_dir))

        assert result is None

    def test_config_file_only_whitespace_returns_none(self, temp_dir: Path) -> None:
        """Test that config file with only whitespace returns None."""
        config_file = temp_dir / "config.yml"
        config_file.write_text("   \n\n  \n")

        loader = ConfigLoader()
        result = loader.load_project_config(str(temp_dir))

        assert result is None

    def test_config_file_yaml_parse_error(self, temp_dir: Path) -> None:
        """Test that YAML parse error raises ConfigError."""
        import pytest

        from holodeck.lib.errors import ConfigError

        config_file = temp_dir / "config.yml"
        config_file.write_text("invalid: yaml: [broken")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_project_config(str(temp_dir))

        assert "project_config_parse" in str(exc_info.value)

    def test_config_file_validation_error(self, temp_dir: Path) -> None:
        """Test that invalid config content raises ConfigError."""
        import pytest

        from holodeck.lib.errors import ConfigError

        config_file = temp_dir / "config.yml"
        # Invalid provider structure
        config_file.write_text(
            yaml.dump(
                {
                    "providers": {
                        "openai": {
                            "provider": "invalid_provider_type",
                            "name": "test",
                        }
                    }
                }
            )
        )

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_project_config(str(temp_dir))

        assert "validation" in str(exc_info.value)


class TestMergeConfigsWithProviders:
    """Tests for merge_configs with LLM provider configurations."""

    def test_merge_configs_empty_agent_config(self) -> None:
        """Test merge_configs with empty agent config returns empty dict."""
        loader = ConfigLoader()
        result = loader.merge_configs({}, None)

        assert result == {}

    def test_merge_configs_no_global_config(self) -> None:
        """Test merge_configs without global config returns agent config as-is."""
        agent_config = {"name": "test", "model": {"provider": "openai"}}

        loader = ConfigLoader()
        result = loader.merge_configs(agent_config, None)

        assert result == agent_config

    def test_merge_configs_provider_api_key(self, temp_dir: Path) -> None:
        """Test that provider API key is merged from global config."""
        from holodeck.models.config import GlobalConfig

        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
        }

        global_config = GlobalConfig(
            providers={
                "my_openai": {
                    "provider": "openai",
                    "api_key": "sk-global-key",
                    "name": "gpt-4o-mini",
                }
            }
        )

        loader = ConfigLoader()
        result = loader.merge_configs(agent_config, global_config)

        # API key should be merged (agent doesn't have it)
        assert result["model"]["api_key"] == "sk-global-key"
        # Name should NOT be overwritten (agent has it)
        assert result["model"]["name"] == "gpt-4o"

    def test_merge_configs_evaluation_model_provider(self) -> None:
        """Test that evaluation model provider config is merged."""
        from holodeck.models.config import GlobalConfig

        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "evaluations": {
                "model": {"provider": "openai", "name": "gpt-4o-mini"},
            },
        }

        global_config = GlobalConfig(
            providers={
                "my_openai": {
                    "provider": "openai",
                    "api_key": "sk-eval-key",
                    "name": "gpt-4o",
                }
            }
        )

        loader = ConfigLoader()
        result = loader.merge_configs(agent_config, global_config)

        # API key should be merged into evaluation model
        assert result["evaluations"]["model"]["api_key"] == "sk-eval-key"


class TestResolveVectorstoreReferences:
    """Tests for _resolve_vectorstore_references method."""

    def test_resolve_vectorstore_string_reference(self) -> None:
        """Test resolving string database reference to DatabaseConfig."""
        from holodeck.models.config import VectorstoreConfig

        tools = [
            {
                "name": "knowledge_base",
                "type": "vectorstore",
                "database": "postgres_store",
                "source": "data/",
            }
        ]

        vectorstores = {
            "postgres_store": VectorstoreConfig(
                provider="postgres",
                connection_string="postgresql://localhost:5432/db",
            )
        }

        loader = ConfigLoader()
        loader._resolve_vectorstore_references(tools, vectorstores)

        # Database should be resolved to dict
        assert isinstance(tools[0]["database"], dict)
        assert tools[0]["database"]["provider"] == "postgres"
        assert (
            tools[0]["database"]["connection_string"]
            == "postgresql://localhost:5432/db"
        )

    def test_resolve_vectorstore_unknown_reference_logs_warning(
        self, caplog: Any
    ) -> None:
        """Test that unknown vectorstore reference logs warning."""
        import logging

        from holodeck.models.config import VectorstoreConfig

        tools = [
            {
                "name": "knowledge_base",
                "type": "vectorstore",
                "database": "unknown_store",
                "source": "data/",
            }
        ]

        vectorstores = {
            "postgres_store": VectorstoreConfig(
                provider="postgres",
                connection_string="postgresql://localhost:5432/db",
            )
        }

        with caplog.at_level(logging.WARNING, logger="holodeck.config.loader"):
            loader = ConfigLoader()
            loader._resolve_vectorstore_references(tools, vectorstores)

        # Database should be set to None (fallback)
        assert tools[0]["database"] is None
        # Warning should be logged
        assert any("unknown_store" in record.message for record in caplog.records)

    def test_resolve_vectorstore_dict_database_unchanged(self) -> None:
        """Test that dict database config is left unchanged."""
        from holodeck.models.config import VectorstoreConfig

        tools = [
            {
                "name": "knowledge_base",
                "type": "vectorstore",
                "database": {"provider": "postgres", "connection_string": "local"},
                "source": "data/",
            }
        ]

        vectorstores = {
            "postgres_store": VectorstoreConfig(
                provider="postgres",
                connection_string="postgresql://localhost:5432/db",
            )
        }

        loader = ConfigLoader()
        loader._resolve_vectorstore_references(tools, vectorstores)

        # Database should be unchanged
        assert tools[0]["database"]["provider"] == "postgres"

    def test_resolve_vectorstore_none_database_unchanged(self) -> None:
        """Test that None database is left unchanged."""
        from holodeck.models.config import VectorstoreConfig

        tools = [
            {
                "name": "knowledge_base",
                "type": "vectorstore",
                "database": None,
                "source": "data/",
            }
        ]

        vectorstores = {
            "postgres_store": VectorstoreConfig(
                provider="postgres",
                connection_string="postgresql://localhost:5432/db",
            )
        }

        loader = ConfigLoader()
        loader._resolve_vectorstore_references(tools, vectorstores)

        # Database should still be None
        assert tools[0]["database"] is None

    def test_resolve_vectorstore_skips_non_vectorstore_tools(self) -> None:
        """Test that non-vectorstore tools are skipped."""
        from holodeck.models.config import VectorstoreConfig

        tools = [
            {
                "name": "my_function",
                "type": "function",
                "file": "tools/my_func.py",
                "function": "run",
            }
        ]

        vectorstores = {
            "postgres_store": VectorstoreConfig(
                provider="postgres",
                connection_string="postgresql://localhost:5432/db",
            )
        }

        loader = ConfigLoader()
        # Should not raise, just skip
        loader._resolve_vectorstore_references(tools, vectorstores)

        # Tool should be unchanged
        assert tools[0]["type"] == "function"


class TestConvertVectorstoreToDatabaseConfig:
    """Tests for _convert_vectorstore_to_database_config function."""

    def test_convert_postgres_provider(self) -> None:
        """Test converting postgres provider."""
        from holodeck.config.loader import _convert_vectorstore_to_database_config
        from holodeck.models.config import VectorstoreConfig

        vs_config = VectorstoreConfig(
            provider="postgres",
            connection_string="postgresql://localhost:5432/db",
        )

        result = _convert_vectorstore_to_database_config(vs_config)

        assert result.provider == "postgres"
        assert result.connection_string == "postgresql://localhost:5432/db"

    def test_convert_chromadb_provider(self) -> None:
        """Test converting chromadb provider."""
        from holodeck.config.loader import _convert_vectorstore_to_database_config
        from holodeck.models.config import VectorstoreConfig

        vs_config = VectorstoreConfig(
            provider="chromadb",
            connection_string="http://localhost:8000",
        )

        result = _convert_vectorstore_to_database_config(vs_config)

        assert result.provider == "chromadb"
        assert result.connection_string == "http://localhost:8000"

    def test_convert_with_options(self) -> None:
        """Test converting with options merged as extra fields."""
        from holodeck.config.loader import _convert_vectorstore_to_database_config
        from holodeck.models.config import VectorstoreConfig

        vs_config = VectorstoreConfig(
            provider="chromadb",
            connection_string="http://localhost:8000",
            options={"index_name": "my_index", "custom_field": "value"},
        )

        result = _convert_vectorstore_to_database_config(vs_config)

        # Options should be merged as extra fields
        result_dict = result.model_dump()
        assert result_dict.get("index_name") == "my_index"
        assert result_dict.get("custom_field") == "value"


class TestDeepMerge:
    """Tests for _deep_merge static method."""

    def test_deep_merge_simple(self) -> None:
        """Test deep merge with simple values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        ConfigLoader._deep_merge(base, override)

        assert base == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self) -> None:
        """Test deep merge with nested dicts."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 20, "z": 30}}

        ConfigLoader._deep_merge(base, override)

        assert base == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}

    def test_deep_merge_override_non_dict_with_dict(self) -> None:
        """Test that non-dict value is replaced with dict."""
        base = {"a": "string"}
        override = {"a": {"nested": "value"}}

        ConfigLoader._deep_merge(base, override)

        assert base == {"a": {"nested": "value"}}


class TestResolveFilePath:
    """Tests for resolve_file_path method."""

    def test_resolve_absolute_path(self, temp_dir: Path) -> None:
        """Test resolving absolute path returns as-is."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        loader = ConfigLoader()
        result = loader.resolve_file_path(str(test_file), "/other/dir")

        assert result == str(test_file)

    def test_resolve_relative_path(self, temp_dir: Path) -> None:
        """Test resolving relative path against base directory."""
        test_file = temp_dir / "subdir" / "test.txt"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("content")

        loader = ConfigLoader()
        result = loader.resolve_file_path("subdir/test.txt", str(temp_dir))

        assert result == str(test_file.resolve())

    def test_resolve_file_not_found(self, temp_dir: Path) -> None:
        """Test resolve_file_path raises FileNotFoundError for missing file."""
        import pytest

        from holodeck.lib.errors import FileNotFoundError

        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.resolve_file_path("nonexistent.txt", str(temp_dir))

        assert "not found" in str(exc_info.value).lower()


class TestLoadInstructions:
    """Tests for load_instructions method."""

    def test_load_inline_instructions(self, temp_dir: Path) -> None:
        """Test loading inline instructions."""
        from holodeck.models.agent import Agent, Instructions

        agent = Agent(
            name="test",
            model={"provider": "openai", "name": "gpt-4o"},
            instructions=Instructions(inline="You are helpful."),
        )

        loader = ConfigLoader()
        result = loader.load_instructions(str(temp_dir / "agent.yaml"), agent)

        assert result == "You are helpful."

    def test_load_file_instructions(self, temp_dir: Path) -> None:
        """Test loading instructions from file."""
        from holodeck.models.agent import Agent, Instructions

        instructions_file = temp_dir / "instructions.txt"
        instructions_file.write_text("Instructions from file.")

        agent = Agent(
            name="test",
            model={"provider": "openai", "name": "gpt-4o"},
            instructions=Instructions(file="instructions.txt"),
        )

        loader = ConfigLoader()
        result = loader.load_instructions(str(temp_dir / "agent.yaml"), agent)

        assert result == "Instructions from file."

    def test_load_no_instructions_returns_none(self, temp_dir: Path) -> None:
        """Test that missing instructions returns None."""
        from unittest.mock import MagicMock

        from holodeck.models.agent import Agent, Instructions

        # Create agent with mocked instructions that has neither inline nor file
        agent = MagicMock(spec=Agent)
        agent.instructions = MagicMock(spec=Instructions)
        agent.instructions.inline = None
        agent.instructions.file = None

        loader = ConfigLoader()
        result = loader.load_instructions(str(temp_dir / "agent.yaml"), agent)

        assert result is None


class TestResolveVectorstoreDatabaseConfig:
    """Tests for resolve_vectorstore_database_config method."""

    def test_resolve_from_project_config(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test resolving vectorstore config from project config."""
        # Create project config with vectorstores
        project_config = {
            "vectorstores": {
                "postgres_store": {
                    "provider": "postgres",
                    "connection_string": "postgresql://localhost:5432/db",
                }
            }
        }
        (temp_dir / "config.yml").write_text(yaml.dump(project_config))

        # Create agent yaml
        agent_yaml = temp_dir / "agent.yaml"
        agent_yaml.write_text("name: test")

        loader = ConfigLoader()
        result = loader.resolve_vectorstore_database_config(str(agent_yaml))

        assert result is not None
        assert result["provider"] == "postgres"
        assert result["connection_string"] == "postgresql://localhost:5432/db"

    def test_resolve_by_name(self, temp_dir: Path, monkeypatch: Any) -> None:
        """Test resolving specific vectorstore by name."""
        project_config = {
            "vectorstores": {
                "chromadb_store": {
                    "provider": "chromadb",
                    "connection_string": "http://localhost:8000",
                },
                "postgres_store": {
                    "provider": "postgres",
                    "connection_string": "postgresql://localhost:5432/db",
                },
            }
        }
        (temp_dir / "config.yml").write_text(yaml.dump(project_config))

        agent_yaml = temp_dir / "agent.yaml"
        agent_yaml.write_text("name: test")

        loader = ConfigLoader()
        result = loader.resolve_vectorstore_database_config(
            str(agent_yaml), vectorstore_name="postgres_store"
        )

        assert result is not None
        assert result["provider"] == "postgres"

    def test_resolve_unknown_name_returns_none(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test resolving unknown vectorstore name returns None."""
        project_config = {
            "vectorstores": {
                "postgres_store": {
                    "provider": "postgres",
                    "connection_string": "postgresql://localhost:5432/db",
                },
            }
        }
        (temp_dir / "config.yml").write_text(yaml.dump(project_config))

        agent_yaml = temp_dir / "agent.yaml"
        agent_yaml.write_text("name: test")

        loader = ConfigLoader()
        result = loader.resolve_vectorstore_database_config(
            str(agent_yaml), vectorstore_name="nonexistent"
        )

        assert result is None

    def test_resolve_no_vectorstores_returns_none(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test resolving when no vectorstores configured returns None."""
        monkeypatch.setenv("HOME", str(temp_dir))
        (temp_dir / ".holodeck").mkdir()

        agent_yaml = temp_dir / "agent.yaml"
        agent_yaml.write_text("name: test")

        loader = ConfigLoader()
        result = loader.resolve_vectorstore_database_config(str(agent_yaml))

        assert result is None

    def test_resolve_from_user_config_fallback(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test fallback to user config when no project config."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # Create user config with vectorstores
        user_config = {
            "vectorstores": {
                "user_chromadb": {
                    "provider": "chromadb",
                    "connection_string": "http://localhost:8000",
                }
            }
        }
        (holodeck_dir / "config.yml").write_text(yaml.dump(user_config))

        # Create project without vectorstores
        project_dir = temp_dir / "project"
        project_dir.mkdir()
        agent_yaml = project_dir / "agent.yaml"
        agent_yaml.write_text("name: test")

        loader = ConfigLoader()
        result = loader.resolve_vectorstore_database_config(str(agent_yaml))

        assert result is not None
        assert result["connection_string"] == "http://localhost:8000"


class TestMergeConfigsWithVectorstores:
    """Tests for merge_configs with vectorstore tool resolution."""

    def test_merge_configs_resolves_vectorstore_references(self) -> None:
        """Test that merge_configs resolves vectorstore database references."""
        from holodeck.models.config import GlobalConfig

        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "tools": [
                {
                    "name": "kb",
                    "type": "vectorstore",
                    "database": "postgres_store",
                    "source": "data/",
                }
            ],
        }

        global_config = GlobalConfig(
            vectorstores={
                "postgres_store": {
                    "provider": "postgres",
                    "connection_string": "postgresql://localhost:5432/db",
                }
            }
        )

        loader = ConfigLoader()
        result = loader.merge_configs(agent_config, global_config)

        # Database should be resolved
        assert isinstance(result["tools"][0]["database"], dict)
        assert result["tools"][0]["database"]["provider"] == "postgres"


class TestResolveVectorstoreReferencesNonDictTool:
    """Test _resolve_vectorstore_references with non-dict tools."""

    def test_skips_non_dict_tool(self) -> None:
        """Test that non-dict tools are skipped without error."""
        from holodeck.models.config import VectorstoreConfig

        # Include a non-dict item (shouldn't happen but defensive)
        tools: list[Any] = [
            "not_a_dict",
            {
                "name": "kb",
                "type": "vectorstore",
                "database": "postgres_store",
                "source": "data/",
            },
        ]

        vectorstores = {
            "postgres_store": VectorstoreConfig(
                provider="postgres",
                connection_string="postgresql://localhost:5432/db",
            )
        }

        loader = ConfigLoader()
        # Should not raise, just skip non-dict
        loader._resolve_vectorstore_references(tools, vectorstores)

        # Second tool should still be resolved
        assert tools[1]["database"]["provider"] == "postgres"


class TestLoadConfigFileEnvSubstitutionEmpty:
    """Test _load_config_file when env substitution results in empty."""

    def test_config_file_empty_after_env_substitution(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test config file that becomes empty dict after substitution."""
        # Test case where env var substitution results in an empty string value
        # which after YAML parse gives an empty/falsy result
        monkeypatch.setenv("MY_VAR", "null")

        config_file = temp_dir / "config.yml"
        # This parses to None/null after substitution
        config_file.write_text("${MY_VAR}")

        loader = ConfigLoader()
        result = loader.load_project_config(str(temp_dir))

        # The file content after substitution parses to None, should return None
        assert result is None


class TestEnvValueParsing:
    """Tests for _parse_env_value and _get_env_value functions."""

    def test_parse_env_value_timeout_int(self) -> None:
        """Test parsing timeout value as int."""
        from holodeck.config.loader import _parse_env_value

        result = _parse_env_value("file_timeout", "42")
        assert result == 42
        assert isinstance(result, int)

    def test_parse_env_value_bool_true(self) -> None:
        """Test parsing boolean true variants."""
        from holodeck.config.loader import _parse_env_value

        for value in ["true", "1", "yes", "on", "TRUE", "Yes", "ON"]:
            result = _parse_env_value("cache_enabled", value)
            assert result is True

    def test_parse_env_value_bool_false(self) -> None:
        """Test parsing boolean false variants."""
        from holodeck.config.loader import _parse_env_value

        for value in ["false", "0", "no", "off", "FALSE"]:
            result = _parse_env_value("verbose", value)
            assert result is False

    def test_parse_env_value_string(self) -> None:
        """Test parsing string value."""
        from holodeck.config.loader import _parse_env_value

        result = _parse_env_value("cache_dir", "/custom/path")
        assert result == "/custom/path"

    def test_get_env_value_not_found(self) -> None:
        """Test _get_env_value returns None for missing env var."""
        from holodeck.config.loader import _get_env_value

        result = _get_env_value("file_timeout", {})
        assert result is None

    def test_get_env_value_invalid_int(self) -> None:
        """Test _get_env_value returns None for invalid int."""
        from holodeck.config.loader import _get_env_value

        result = _get_env_value(
            "file_timeout", {"HOLODECK_FILE_TIMEOUT": "not_a_number"}
        )
        assert result is None


# --- MCP Server Config Merge Tests (T035a-c) ---


def _create_mcp_server(name: str, description: str) -> MCPTool:
    """Helper to create a sample MCP server for testing."""
    return MCPTool(
        name=name,
        description=description,
        type="mcp",
        transport=TransportType.STDIO,
        command=CommandType.NPX,
        args=["-y", f"@modelcontextprotocol/server-{name}@1.0.0"],
        registry_name=f"io.github.modelcontextprotocol/server-{name}",
    )


class TestMergeMcpServers:
    """Tests for _merge_mcp_servers() method."""

    def test_merge_mcp_servers_empty_tools(self) -> None:
        """Test merging when agent has no tools."""
        loader = ConfigLoader()
        tools: list[dict[str, Any]] = []
        global_servers = [
            _create_mcp_server("filesystem", "Read and explore files"),
            _create_mcp_server("github", "Interact with GitHub"),
        ]

        loader._merge_mcp_servers(tools, global_servers)

        assert len(tools) == 2
        assert tools[0]["name"] == "filesystem"
        assert tools[1]["name"] == "github"

    def test_merge_mcp_servers_no_overlap(self) -> None:
        """Test merging when agent has different tools (no name conflict)."""
        loader = ConfigLoader()
        tools: list[dict[str, Any]] = [
            {"name": "vectorstore_tool", "type": "vectorstore"},
            {"name": "custom_function", "type": "function"},
        ]
        global_servers = [_create_mcp_server("filesystem", "Read files")]

        loader._merge_mcp_servers(tools, global_servers)

        assert len(tools) == 3
        assert tools[0]["name"] == "vectorstore_tool"
        assert tools[1]["name"] == "custom_function"
        assert tools[2]["name"] == "filesystem"

    def test_merge_mcp_servers_with_overlap(self) -> None:
        """Test that agent tools override global with same name."""
        loader = ConfigLoader()
        # Agent has a tool named "filesystem" (different config)
        tools: list[dict[str, Any]] = [
            {
                "name": "filesystem",
                "type": "mcp",
                "description": "Agent-level filesystem (takes precedence)",
                "transport": "stdio",
                "command": "uvx",  # Different from global
                "args": ["custom-filesystem"],
            }
        ]
        global_servers = [_create_mcp_server("filesystem", "Global filesystem")]

        loader._merge_mcp_servers(tools, global_servers)

        # Should NOT add global filesystem (agent takes precedence)
        assert len(tools) == 1
        assert tools[0]["name"] == "filesystem"
        assert tools[0]["command"] == "uvx"  # Agent's config preserved
        assert tools[0]["description"] == "Agent-level filesystem (takes precedence)"

    def test_merge_mcp_servers_partial_overlap(self) -> None:
        """Test partial overlap - some names conflict, some don't."""
        loader = ConfigLoader()
        # Agent has "filesystem" but not "github"
        tools: list[dict[str, Any]] = [
            {
                "name": "filesystem",
                "type": "mcp",
                "description": "Agent filesystem",
                "transport": "stdio",
                "command": "uvx",
                "args": ["agent-fs"],
            }
        ]
        global_servers = [
            _create_mcp_server("filesystem", "Global filesystem"),
            _create_mcp_server("github", "Global GitHub"),
        ]

        loader._merge_mcp_servers(tools, global_servers)

        # Should have agent's filesystem + global's github
        assert len(tools) == 2
        assert tools[0]["name"] == "filesystem"
        assert tools[0]["command"] == "uvx"  # Agent's config
        assert tools[1]["name"] == "github"
        assert tools[1]["command"] == "npx"  # Global's config

    def test_merge_mcp_servers_non_mcp_overlap(self) -> None:
        """Test that non-MCP tools with same name also block global MCP."""
        loader = ConfigLoader()
        # Agent has a vectorstore tool named "filesystem"
        tools: list[dict[str, Any]] = [
            {"name": "filesystem", "type": "vectorstore", "database": None}
        ]
        global_servers = [_create_mcp_server("filesystem", "Global filesystem")]

        loader._merge_mcp_servers(tools, global_servers)

        # Should NOT add global MCP (name conflict with non-MCP tool)
        assert len(tools) == 1
        assert tools[0]["type"] == "vectorstore"


class TestMergeConfigsWithMcpServers:
    """Tests for merge_configs() with MCP server merging."""

    def test_merge_configs_with_mcp_servers(self) -> None:
        """Test full merge_configs() integration with MCP servers."""
        loader = ConfigLoader()
        agent_config: dict[str, Any] = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are helpful."},
            "tools": [],
        }
        global_config = GlobalConfig(
            providers=None,
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=[_create_mcp_server("filesystem", "Read files")],
        )

        result = loader.merge_configs(agent_config, global_config)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "filesystem"

    def test_merge_configs_no_global_mcp_servers(self) -> None:
        """Test merge_configs() when global config has no MCP servers."""
        loader = ConfigLoader()
        agent_config: dict[str, Any] = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are helpful."},
            "tools": [{"name": "my_tool", "type": "function"}],
        }
        global_config = GlobalConfig(
            providers=None,
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=None,
        )

        result = loader.merge_configs(agent_config, global_config)

        # Agent tools unchanged
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "my_tool"

    def test_merge_configs_agent_precedence(self) -> None:
        """Test that agent config takes precedence over global."""
        loader = ConfigLoader()
        agent_config: dict[str, Any] = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are helpful."},
            "tools": [
                {
                    "name": "filesystem",
                    "type": "mcp",
                    "description": "Agent's custom filesystem",
                    "transport": "stdio",
                    "command": "docker",
                    "args": ["run", "my-fs-image"],
                }
            ],
        }
        global_config = GlobalConfig(
            providers=None,
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=[_create_mcp_server("filesystem", "Global filesystem")],
        )

        result = loader.merge_configs(agent_config, global_config)

        # Agent's filesystem preserved, global's skipped
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "filesystem"
        assert result["tools"][0]["command"] == "docker"
        assert result["tools"][0]["description"] == "Agent's custom filesystem"

    def test_merge_configs_no_tools_section(self) -> None:
        """Test merge_configs() when agent has no tools section."""
        loader = ConfigLoader()
        agent_config: dict[str, Any] = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are helpful."},
            # No tools section
        }
        global_config = GlobalConfig(
            providers=None,
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=[_create_mcp_server("filesystem", "Read files")],
        )

        result = loader.merge_configs(agent_config, global_config)

        # Tools section created with global MCP servers
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "filesystem"

    def test_merge_configs_empty_global_mcp_servers(self) -> None:
        """Test merge_configs() with empty mcp_servers list."""
        loader = ConfigLoader()
        agent_config: dict[str, Any] = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are helpful."},
            "tools": [{"name": "my_tool", "type": "function"}],
        }
        global_config = GlobalConfig(
            providers=None,
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=[],  # Empty list
        )

        result = loader.merge_configs(agent_config, global_config)

        # Agent tools unchanged
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "my_tool"

    def test_merge_configs_no_global_config(self) -> None:
        """Test merge_configs() with no global config."""
        loader = ConfigLoader()
        agent_config: dict[str, Any] = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "You are helpful."},
            "tools": [{"name": "my_tool", "type": "function"}],
        }

        result = loader.merge_configs(agent_config, None)

        # Agent config returned as-is
        assert result == agent_config
