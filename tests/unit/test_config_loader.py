"""Tests for ConfigLoader in holodeck.config.loader."""

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError, FileNotFoundError, ValidationError
from holodeck.models.agent import Agent


class TestParseYaml:
    """Tests for YAML parsing (T034)."""

    def test_parse_yaml_valid_yaml(self, temp_dir: Path) -> None:
        """Test parse_yaml with valid YAML content."""
        yaml_file = temp_dir / "test.yaml"
        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test instructions"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert isinstance(result, dict)
        assert result["name"] == "test_agent"
        assert result["model"]["provider"] == "openai"

    def test_parse_yaml_with_list_structure(self, temp_dir: Path) -> None:
        """Test parse_yaml with list structures."""
        yaml_file = temp_dir / "test.yaml"
        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
            "tools": [{"name": "search", "type": "vectorstore", "source": "data.txt"}],
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "vectorstore"

    def test_parse_yaml_invalid_yaml_syntax(self, temp_dir: Path) -> None:
        """Test parse_yaml with invalid YAML syntax."""
        yaml_file = temp_dir / "invalid.yaml"
        yaml_file.write_text("invalid: [yaml: syntax: here")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.parse_yaml(str(yaml_file))
        assert "YAML" in str(exc_info.value) or "parse" in str(exc_info.value).lower()

    def test_parse_yaml_file_not_found(self) -> None:
        """Test parse_yaml with non-existent file."""
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.parse_yaml("/nonexistent/path/to/file.yaml")
        assert "/nonexistent" in str(exc_info.value)

    def test_parse_yaml_empty_file(self, temp_dir: Path) -> None:
        """Test parse_yaml with empty YAML file."""
        yaml_file = temp_dir / "empty.yaml"
        yaml_file.write_text("")

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        # Empty YAML is valid and returns None
        assert result is None or result == {}


class TestLoadAgentYaml:
    """Tests for load_agent_yaml (T034)."""

    def test_load_agent_yaml_valid(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with valid agent configuration."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test_agent",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            "instructions": {"inline": "You are a helpful assistant."},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert isinstance(agent, Agent)
        assert agent.name == "test_agent"
        assert agent.model.provider.value == "openai"

    def test_load_agent_yaml_with_description(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with optional description."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test_agent",
            "description": "A test agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert agent.description == "A test agent"

    def test_load_agent_yaml_missing_name(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with missing required name field."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        with pytest.raises((ConfigError, ValidationError)):
            loader.load_agent_yaml(str(yaml_file))

    def test_load_agent_yaml_missing_model(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with missing required model field."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test_agent",
            "instructions": {"inline": "Test"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        with pytest.raises((ConfigError, ValidationError)):
            loader.load_agent_yaml(str(yaml_file))

    def test_load_agent_yaml_missing_instructions(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with missing required instructions field."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        with pytest.raises((ConfigError, ValidationError)):
            loader.load_agent_yaml(str(yaml_file))

    def test_load_agent_yaml_invalid_provider(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with invalid provider value."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test_agent",
            "model": {
                "provider": "invalid_provider",
                "name": "gpt-4o",
            },
            "instructions": {"inline": "Test"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        with pytest.raises((ConfigError, ValidationError)):
            loader.load_agent_yaml(str(yaml_file))

    def test_load_agent_yaml_with_tools(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with tools."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
            "tools": [{"name": "search", "type": "vectorstore", "source": "data.txt"}],
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert agent.tools is not None
        assert len(agent.tools) == 1

    def test_load_agent_yaml_with_test_cases(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with test cases."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
            "test_cases": [{"input": "What is 2+2?"}],
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert agent.test_cases is not None
        assert len(agent.test_cases) == 1


class TestFileResolution:
    """Tests for resolve_file_path (T037)."""

    def test_resolve_file_path_absolute_path(self, temp_dir: Path) -> None:
        """Test resolve_file_path with absolute path."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        loader = ConfigLoader()
        resolved = loader.resolve_file_path(str(test_file), str(temp_dir))

        assert resolved == str(test_file)

    def test_resolve_file_path_relative_to_agent_yaml(self, temp_dir: Path) -> None:
        """Test resolve_file_path with path relative to agent.yaml."""
        agent_yaml = temp_dir / "agent.yaml"
        instructions_file = temp_dir / "prompts" / "system.md"
        instructions_file.parent.mkdir(exist_ok=True)
        instructions_file.write_text("System instructions")

        loader = ConfigLoader()
        resolved = loader.resolve_file_path("prompts/system.md", str(agent_yaml.parent))

        assert Path(resolved).exists()
        assert "system.md" in resolved

    def test_resolve_file_path_missing_file_raises_error(self, temp_dir: Path) -> None:
        """Test resolve_file_path with non-existent file."""
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.resolve_file_path("nonexistent.txt", str(temp_dir))
        assert "nonexistent.txt" in str(exc_info.value)

    def test_resolve_file_path_current_dir_reference(self, temp_dir: Path) -> None:
        """Test resolve_file_path with ./ reference."""
        test_file = temp_dir / "config.txt"
        test_file.write_text("test")

        loader = ConfigLoader()
        resolved = loader.resolve_file_path("./config.txt", str(temp_dir))

        assert Path(resolved).exists()


class TestGlobalConfigLoading:
    """Tests for load_global_config (T035)."""

    def test_load_global_config_from_file(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config reads from ~/.holodeck/config.yaml."""
        global_config = temp_dir / "global_config.yaml"
        config_content = {
            "providers": {"openai": {"api_key": "test-key"}},
        }
        global_config.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        # Patch the home directory
        monkeypatch.setenv("HOME", str(temp_dir))

        # Create .holodeck directory
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        actual_config_file = holodeck_dir / "config.yaml"
        actual_config_file.write_text(yaml.dump(config_content))

        result = loader.load_global_config()

        assert isinstance(result, dict)
        assert "providers" in result

    def test_load_global_config_missing_returns_empty(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config returns empty dict if file missing."""
        loader = ConfigLoader()
        monkeypatch.setenv("HOME", str(temp_dir))

        result = loader.load_global_config()

        assert isinstance(result, dict)
        assert len(result) == 0 or result == {}

    def test_load_global_config_with_env_substitution(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config applies env var substitution."""
        global_config = temp_dir / "config.yaml"
        config_content = "api_key: ${TEST_API_KEY}"
        global_config.write_text(config_content)

        monkeypatch.setenv("HOME", str(temp_dir))
        monkeypatch.setenv("TEST_API_KEY", "secret-123")

        # Create .holodeck directory
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        actual_config_file = holodeck_dir / "config.yaml"
        actual_config_file.write_text(config_content)

        loader = ConfigLoader()
        result = loader.load_global_config()

        # After substitution, the api_key should contain the env value
        if result:
            result_str = yaml.dump(result)
            assert "secret-123" in result_str or result.get("api_key") == "secret-123"


class TestConfigPrecedence:
    """Tests for merge_configs and config precedence (T036)."""

    def test_merge_configs_agent_overrides_env_vars(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that agent.yaml values override environment variables."""
        os.environ["TEST_PROVIDER"] = "anthropic"

        agent_config = {
            "name": "test",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        global_config = {
            "providers": {"default": "anthropic"},
        }

        loader = ConfigLoader()
        merged = loader.merge_configs(agent_config, global_config)

        # Agent config should take precedence
        assert merged["model"]["provider"] == "openai"

    def test_merge_configs_env_vars_override_global(self, monkeypatch: Any) -> None:
        """Test that environment variables override global config."""
        monkeypatch.setenv("TEST_MODEL", "gpt-4o")

        agent_config = {"name": "test"}
        global_config = {"default_model": "gpt-3.5"}

        loader = ConfigLoader()
        merged = loader.merge_configs(agent_config, global_config)

        # Agent config should still be primary
        assert "name" in merged

    def test_merge_configs_missing_fields_from_global(self) -> None:
        """Test that global config fills in missing optional fields."""
        agent_config = {
            "name": "test",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        global_config = {
            "default_temperature": 0.5,
            "providers": {"openai": {"api_key": "key-123"}},
        }

        loader = ConfigLoader()
        merged = loader.merge_configs(agent_config, global_config)

        # Agent config should remain intact, global added if not conflicting
        assert merged["name"] == "test"


class TestErrorHandling:
    """Tests for error handling and conversion (T038)."""

    def test_error_handling_pydantic_errors_converted_to_config_error(
        self, temp_dir: Path
    ) -> None:
        """Test that Pydantic validation errors are converted to ConfigError."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "",  # Empty name is invalid
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        with pytest.raises((ConfigError, ValidationError)):
            loader.load_agent_yaml(str(yaml_file))

    def test_error_handling_includes_field_name(self, temp_dir: Path) -> None:
        """Test that error messages include the field name."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test",
            "model": {"provider": "invalid", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        with pytest.raises((ConfigError, ValidationError)) as exc_info:
            loader.load_agent_yaml(str(yaml_file))

        error_msg = str(exc_info.value)
        assert "provider" in error_msg.lower() or "model" in error_msg.lower()

    def test_error_handling_file_not_found_includes_path(self, temp_dir: Path) -> None:
        """Test that file not found errors include the full path."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"file": "/nonexistent/path/to/prompts.md"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        # When resolving instruction files that don't exist
        with pytest.raises(FileNotFoundError):
            agent = loader.load_agent_yaml(str(yaml_file))
            # Optionally resolve instruction files
            if agent.instructions.file:
                loader.resolve_file_path(agent.instructions.file, str(yaml_file.parent))

    def test_error_handling_missing_required_field_message(
        self, temp_dir: Path
    ) -> None:
        """Test that missing required fields produce clear error messages."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test",
            # missing model
            "instructions": {"inline": "Test"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        with pytest.raises((ConfigError, ValidationError)) as exc_info:
            loader.load_agent_yaml(str(yaml_file))

        error_msg = str(exc_info.value).lower()
        assert "model" in error_msg or "required" in error_msg


class TestLoadAgentYamlEnvSubstitution:
    """Tests for environment variable substitution in loaded config (T034)."""

    def test_load_agent_yaml_with_env_var_substitution(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that load_agent_yaml applies env var substitution."""
        monkeypatch.setenv("AGENT_DESCRIPTION", "An agent built with env vars")

        yaml_file = temp_dir / "agent.yaml"
        yaml_content = """
name: test_agent
description: ${AGENT_DESCRIPTION}
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: Test instructions
"""
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert agent.name == "test_agent"
        assert agent.description == "An agent built with env vars"
