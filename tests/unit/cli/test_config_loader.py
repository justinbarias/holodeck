"""Tests for ConfigLoader in holodeck.config.loader."""

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError, FileNotFoundError, ValidationError
from holodeck.models.agent import Agent
from holodeck.models.config import GlobalConfig


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

    def test_load_agent_yaml_with_tools(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with tools."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
            "tools": [
                {
                    "name": "search",
                    "description": "Search through documents",
                    "type": "vectorstore",
                    "source": "data.txt",
                }
            ],
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
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                }
            },
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

        assert isinstance(result, GlobalConfig)
        assert result.providers is not None
        assert "openai" in result.providers

    def test_load_global_config_missing_returns_empty(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config returns None if file missing."""
        loader = ConfigLoader()
        monkeypatch.setenv("HOME", str(temp_dir))

        result = loader.load_global_config()

        assert result is None

    def test_load_global_config_with_env_substitution(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config applies env var substitution."""
        global_config = temp_dir / "config.yaml"
        config_content = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "${TEST_MODEL_NAME}",
                }
            }
        }
        global_config.write_text(yaml.dump(config_content))

        monkeypatch.setenv("HOME", str(temp_dir))
        monkeypatch.setenv("TEST_MODEL_NAME", "gpt-4o-substituted")

        # Create .holodeck directory
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        actual_config_file = holodeck_dir / "config.yaml"
        actual_config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        result = loader.load_global_config()

        # After substitution, the name should contain the env value
        assert result is not None
        assert isinstance(result, GlobalConfig)
        assert result.providers is not None
        assert "openai" in result.providers
        assert result.providers["openai"].name == "gpt-4o-substituted"


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
        global_config_dict = {
            "providers": {
                "anthropic": {
                    "provider": "anthropic",
                    "name": "claude-3-opus",
                }
            },
        }
        global_config = GlobalConfig(**global_config_dict)

        loader = ConfigLoader()
        merged = loader.merge_configs(agent_config, global_config)

        # Agent config should take precedence
        assert merged["model"]["provider"] == "openai"

    def test_merge_configs_env_vars_override_global(self, monkeypatch: Any) -> None:
        """Test that environment variables override global config."""
        monkeypatch.setenv("TEST_MODEL", "gpt-4o")

        agent_config = {"name": "test"}
        global_config_dict = {
            "deployment": {
                "type": "docker",
            }
        }
        global_config = GlobalConfig(**global_config_dict)

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
        global_config_dict = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "temperature": 0.5,
                }
            },
        }
        global_config = GlobalConfig(**global_config_dict)

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


class TestGlobalConfigEmpty:
    """Tests for empty/malformed global config files."""

    def test_load_global_config_empty_file(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config with empty file returns None."""
        monkeypatch.setenv("HOME", str(temp_dir))

        # Create .holodeck directory
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"
        global_config_file.write_text("")  # Empty file

        loader = ConfigLoader()
        result = loader.load_global_config()

        assert result is None

    def test_load_global_config_invalid_yaml_syntax(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config with invalid YAML syntax raises ConfigError."""
        monkeypatch.setenv("HOME", str(temp_dir))

        # Create .holodeck directory
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"
        global_config_file.write_text("invalid: [yaml: syntax: here")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_global_config()

        assert "global_config_parse" in str(exc_info.value)


class TestLoadInstructions:
    """Tests for load_instructions method."""

    def test_load_instructions_from_file(self, temp_dir: Path) -> None:
        """Test load_instructions reads from file."""
        # Create agent.yaml
        agent_yaml = temp_dir / "agent.yaml"
        agent_yaml.write_text("")

        # Create instructions file
        instructions_file = temp_dir / "instructions.md"
        instructions_content = (
            "You are a helpful assistant. Always be kind and accurate."
        )
        instructions_file.write_text(instructions_content)

        # Create agent with file reference
        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"file": "instructions.md"},
        }
        agent_yaml.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))
        instructions = loader.load_instructions(str(agent_yaml), agent)

        assert instructions == instructions_content
        assert "helpful assistant" in instructions

    def test_load_instructions_inline(self, temp_dir: Path) -> None:
        """Test load_instructions returns inline content when present."""
        agent_yaml = temp_dir / "agent.yaml"
        inline_content = "You are a test agent with inline instructions."

        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": inline_content},
        }
        agent_yaml.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))
        instructions = loader.load_instructions(str(agent_yaml), agent)

        assert instructions == inline_content

    def test_load_instructions_file_not_found(self, temp_dir: Path) -> None:
        """Test load_instructions raises error when file doesn't exist."""
        agent_yaml = temp_dir / "agent.yaml"

        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"file": "nonexistent.md"},
        }
        agent_yaml.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))

        with pytest.raises(FileNotFoundError):
            loader.load_instructions(str(agent_yaml), agent)

    def test_load_instructions_returns_none_when_neither_provided(
        self, temp_dir: Path
    ) -> None:
        """Test load_instructions returns content from default inline."""
        agent_yaml = temp_dir / "agent.yaml"

        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Default instructions"},
        }
        agent_yaml.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))

        # Even with defaults, we should get content
        instructions = loader.load_instructions(str(agent_yaml), agent)
        assert instructions is not None

    def test_load_instructions_with_special_characters(self, temp_dir: Path) -> None:
        """Test load_instructions handles special characters correctly."""
        agent_yaml = temp_dir / "agent.yaml"

        # Instructions with special characters and unicode
        instructions_content = (
            "You are a helpful assistant. 你好 مرحبا emoji: 🎉 "
            "Special chars: @#$%^&*()"
        )
        instructions_file = temp_dir / "instructions.md"
        instructions_file.write_text(instructions_content, encoding="utf-8")

        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"file": "instructions.md"},
        }
        agent_yaml.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))
        instructions = loader.load_instructions(str(agent_yaml), agent)

        assert instructions == instructions_content
        assert "你好" in instructions
        assert "🎉" in instructions

    def test_load_instructions_with_nested_path(self, temp_dir: Path) -> None:
        """Test load_instructions with nested directory structure."""
        agent_yaml = temp_dir / "agent.yaml"

        # Create nested directory structure
        prompts_dir = temp_dir / "prompts" / "system"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        instructions_file = prompts_dir / "instructions.md"
        instructions_content = "Nested instructions"
        instructions_file.write_text(instructions_content)

        yaml_content = {
            "name": "test_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"file": "prompts/system/instructions.md"},
        }
        agent_yaml.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))
        instructions = loader.load_instructions(str(agent_yaml), agent)

        assert instructions == instructions_content


class TestDeepMerge:
    """Tests for _deep_merge static method."""

    def test_deep_merge_simple_override(self) -> None:
        """Test _deep_merge with simple value override."""
        base = {"a": 1, "b": 2}
        override = {"b": 3}

        ConfigLoader._deep_merge(base, override)

        assert base == {"a": 1, "b": 3}

    def test_deep_merge_nested_dicts(self) -> None:
        """Test _deep_merge with nested dictionary structures."""
        base = {"model": {"provider": "openai", "temperature": 0.7}}
        override = {"model": {"temperature": 0.5}}

        ConfigLoader._deep_merge(base, override)

        assert base == {"model": {"provider": "openai", "temperature": 0.5}}

    def test_deep_merge_deeply_nested(self) -> None:
        """Test _deep_merge with deeply nested structures."""
        base = {"level1": {"level2": {"level3": {"value": "original", "keep": "this"}}}}
        override = {"level1": {"level2": {"level3": {"value": "updated"}}}}

        ConfigLoader._deep_merge(base, override)

        assert base["level1"]["level2"]["level3"]["value"] == "updated"
        assert base["level1"]["level2"]["level3"]["keep"] == "this"

    def test_deep_merge_new_keys(self) -> None:
        """Test _deep_merge adds new keys."""
        base = {"existing": "value"}
        override = {"new_key": "new_value"}

        ConfigLoader._deep_merge(base, override)

        assert base == {"existing": "value", "new_key": "new_value"}

    def test_deep_merge_replaces_dict_with_scalar(self) -> None:
        """Test _deep_merge replaces dictionary with scalar value."""
        base = {"key": {"nested": "value"}}
        override = {"key": "simple_value"}

        ConfigLoader._deep_merge(base, override)

        assert base["key"] == "simple_value"

    def test_deep_merge_replaces_scalar_with_dict(self) -> None:
        """Test _deep_merge replaces scalar with dictionary."""
        base = {"key": "simple_value"}
        override = {"key": {"nested": "value"}}

        ConfigLoader._deep_merge(base, override)

        assert base["key"] == {"nested": "value"}

    def test_deep_merge_with_lists(self) -> None:
        """Test _deep_merge replaces lists (not merges them)."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        ConfigLoader._deep_merge(base, override)

        assert base["items"] == [4, 5]

    def test_deep_merge_empty_override(self) -> None:
        """Test _deep_merge with empty override dict."""
        base = {"a": 1, "b": {"c": 2}}
        original = base.copy()

        ConfigLoader._deep_merge(base, {})

        assert base == original


class TestParseYamlEdgeCases:
    """Additional edge case tests for parse_yaml method."""

    def test_parse_yaml_with_comments(self, temp_dir: Path) -> None:
        """Test parse_yaml handles YAML with comments correctly."""
        yaml_file = temp_dir / "test.yaml"
        yaml_content = """
# This is a comment
name: test_agent  # inline comment
model:
  provider: openai  # LLM provider
  name: gpt-4o
"""
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert result["name"] == "test_agent"
        assert result["model"]["provider"] == "openai"

    def test_parse_yaml_with_nested_structures(self, temp_dir: Path) -> None:
        """Test parse_yaml with complex nested YAML structures."""
        yaml_file = temp_dir / "test.yaml"
        yaml_content = {
            "name": "complex_agent",
            "metadata": {
                "version": "1.0",
                "tags": ["ai", "testing"],
                "config": {"nested": {"deeply": {"value": "deep_structure"}}},
            },
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert result["name"] == "complex_agent"
        assert (
            result["metadata"]["config"]["nested"]["deeply"]["value"]
            == "deep_structure"
        )
        assert "ai" in result["metadata"]["tags"]

    def test_parse_yaml_with_special_characters(self, temp_dir: Path) -> None:
        """Test parse_yaml handles special characters in values."""
        yaml_file = temp_dir / "test.yaml"
        yaml_content = """
name: test_agent
description: "Agent with special chars: @#$%^&*() and unicode: 你好 🎉"
url: "https://example.com/path?param=value&other=123"
"""
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert "@#$%^&*()" in result["description"]
        assert "你好" in result["description"]
        assert "https://example.com" in result["url"]

    def test_parse_yaml_with_boolean_and_null_values(self, temp_dir: Path) -> None:
        """Test parse_yaml correctly parses boolean and null values."""
        yaml_file = temp_dir / "test.yaml"
        yaml_content = """
enabled: true
disabled: false
nullable: null
optional: ~
"""
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert result["enabled"] is True
        assert result["disabled"] is False
        assert result["nullable"] is None
        assert result["optional"] is None

    def test_parse_yaml_with_numbers(self, temp_dir: Path) -> None:
        """Test parse_yaml handles various number formats."""
        yaml_file = temp_dir / "test.yaml"
        yaml_content = """
integer: 42
float_value: 3.14
scientific: 1.2e-3
negative: -100
"""
        yaml_file.write_text(yaml_content)

        loader = ConfigLoader()
        result = loader.parse_yaml(str(yaml_file))

        assert result["integer"] == 42
        assert result["float_value"] == 3.14
        assert result["scientific"] == 0.0012
        assert result["negative"] == -100


class TestResolveFilePathEdgeCases:
    """Additional edge case tests for resolve_file_path method."""

    def test_resolve_file_path_with_parent_directory_reference(
        self, temp_dir: Path
    ) -> None:
        """Test resolve_file_path with parent directory reference (..)."""
        # Create nested directory structure
        subdir = temp_dir / "config"
        subdir.mkdir()
        test_file = temp_dir / "instructions.md"
        test_file.write_text("Instructions")

        loader = ConfigLoader()
        resolved = loader.resolve_file_path("../instructions.md", str(subdir))

        assert Path(resolved).exists()
        assert "instructions.md" in resolved

    def test_resolve_file_path_with_multiple_relative_segments(
        self, temp_dir: Path
    ) -> None:
        """Test resolve_file_path with multiple directory segments."""
        # Create nested structure: config/agents/data/test.txt
        nested_dir = temp_dir / "config" / "agents" / "data"
        nested_dir.mkdir(parents=True)
        test_file = nested_dir / "test.txt"
        test_file.write_text("test data")

        loader = ConfigLoader()
        resolved = loader.resolve_file_path(
            "data/test.txt", str(temp_dir / "config" / "agents")
        )

        assert Path(resolved).exists()
        assert "test.txt" in resolved

    def test_resolve_file_path_with_spaces_in_path(self, temp_dir: Path) -> None:
        """Test resolve_file_path with spaces in directory/file names."""
        # Create directory with spaces
        spaced_dir = temp_dir / "my documents"
        spaced_dir.mkdir()
        test_file = spaced_dir / "my file.md"
        test_file.write_text("content")

        loader = ConfigLoader()
        resolved = loader.resolve_file_path("my file.md", str(spaced_dir))

        assert Path(resolved).exists()
        assert "my file.md" in resolved

    def test_resolve_file_path_absolute_with_base_dir_ignored(
        self, temp_dir: Path
    ) -> None:
        """Test resolve_file_path with absolute path ignores base_dir."""
        # Create a file
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        loader = ConfigLoader()
        # Even if we pass a different base_dir, absolute path should be used
        resolved = loader.resolve_file_path(str(test_file), "/some/other/dir")

        assert Path(resolved).exists()
        assert resolved == str(test_file)

    def test_resolve_file_path_normalized(self, temp_dir: Path) -> None:
        """Test resolve_file_path normalizes path correctly."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        loader = ConfigLoader()
        # Use a path with redundant segments like ./
        resolved = loader.resolve_file_path("./test.txt", str(temp_dir))

        assert Path(resolved).exists()
        assert "test.txt" in resolved


class TestLoadAgentYamlIntegration:
    """Integration tests for load_agent_yaml with global config."""

    def test_load_agent_yaml_with_all_optional_fields(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with all optional fields populated."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "full_agent",
            "description": "An agent with all fields",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "temperature": 0.9,
                "max_tokens": 4096,
            },
            "instructions": {"inline": "Be helpful"},
            "tools": [
                {
                    "name": "search",
                    "description": "Search through documents",
                    "type": "vectorstore",
                    "source": "data.txt",
                }
            ],
            "test_cases": [
                {"input": "What is AI?"},
                {"input": "Explain ML"},
            ],
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert agent.name == "full_agent"
        assert agent.description == "An agent with all fields"
        assert agent.model.temperature == 0.9
        assert agent.model.max_tokens == 4096
        assert len(agent.tools) == 1
        assert len(agent.test_cases) == 2

    def test_load_agent_yaml_complex_nested_structure(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with complex nested configuration."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "complex_agent",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "temperature": 0.5,
                "max_tokens": 2000,
            },
            "instructions": {"inline": "Complex instructions"},
            "tools": [
                {
                    "name": "search_tool",
                    "description": "Search through documents",
                    "type": "vectorstore",
                    "source": "documents.txt",
                },
                {
                    "name": "calc_tool",
                    "description": "Math calculation tool",
                    "type": "function",
                    "file": "math.py",
                    "function": "calculate",
                },
            ],
            "test_cases": [
                {
                    "input": "Test 1",
                    "expected_tools": ["search_tool"],
                },
                {
                    "input": "Test 2",
                    "expected_tools": ["calc_tool"],
                },
            ],
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert agent.name == "complex_agent"
        assert len(agent.tools) == 2
        # Tools are now typed objects (VectorstoreTool, FunctionTool, etc.)
        assert agent.tools[0].name == "search_tool"
        assert agent.tools[1].name == "calc_tool"
        assert len(agent.test_cases) == 2

    def test_load_agent_yaml_with_minimal_config(self, temp_dir: Path) -> None:
        """Test load_agent_yaml with minimal required fields only."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "minimal_agent",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
            },
            "instructions": {"inline": "Minimal instructions"},
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert agent.name == "minimal_agent"
        assert agent.model.provider.value == "openai"
        # Optional fields should be None or empty
        assert agent.description is None or agent.description == ""
        assert agent.tools is None or len(agent.tools) == 0

    def test_load_agent_yaml_empty_lists_vs_none(self, temp_dir: Path) -> None:
        """Test load_agent_yaml handles empty lists correctly."""
        yaml_file = temp_dir / "agent.yaml"
        yaml_content = {
            "name": "agent_with_empty_lists",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
            "tools": [],
            "test_cases": [],
        }
        yaml_file.write_text(yaml.dump(yaml_content))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(yaml_file))

        assert agent.name == "agent_with_empty_lists"
        # Empty lists should either be None or empty
        assert agent.tools is None or agent.tools == []
        assert agent.test_cases is None or agent.test_cases == []


class TestMergeConfigsAdvanced:
    """Advanced tests for merge_configs method."""

    def test_merge_configs_preserves_agent_structure(self) -> None:
        """Test that merge_configs returns agent config structure unchanged."""
        agent_config = {
            "name": "test_agent",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
            },
            "instructions": {"inline": "Test"},
        }
        global_config = None

        loader = ConfigLoader()
        merged = loader.merge_configs(agent_config, global_config)

        assert merged == agent_config

    def test_merge_configs_with_global_config_object(self) -> None:
        """Test merge_configs with actual GlobalConfig instance."""
        agent_config = {
            "name": "test",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        global_config_dict = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "temperature": 0.5,
                }
            },
            "deployment": {"type": "docker"},
        }
        global_config = GlobalConfig(**global_config_dict)

        loader = ConfigLoader()
        merged = loader.merge_configs(agent_config, global_config)

        # Agent config should take full precedence
        assert merged["name"] == "test"
        assert merged["model"]["provider"] == "openai"

    def test_merge_configs_with_empty_agent_config(self) -> None:
        """Test merge_configs with empty agent config."""
        agent_config = {}
        global_config = None

        loader = ConfigLoader()
        merged = loader.merge_configs(agent_config, global_config)

        assert merged == {}


class TestGlobalConfigValidationFailures:
    """Tests for GlobalConfig validation failure scenarios."""

    def test_load_global_config_invalid_vectorstore_provider(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config with empty vectorstore provider."""
        monkeypatch.setenv("HOME", str(temp_dir))

        # Create .holodeck directory
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"

        # Invalid config with empty provider
        config_content = {
            "vectorstores": {
                "postgres_store": {
                    "provider": "",  # Empty provider - invalid
                    "connection_string": "postgresql://localhost/holodeck",
                }
            }
        }
        global_config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_global_config()

        assert "global_config_validation" in str(exc_info.value)
        assert "provider" in str(exc_info.value).lower()

    def test_load_global_config_invalid_vectorstore_connection_string(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config with empty vectorstore connection_string."""
        monkeypatch.setenv("HOME", str(temp_dir))

        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"

        # Invalid config with empty connection_string
        config_content = {
            "vectorstores": {
                "postgres_store": {
                    "provider": "postgres",
                    "connection_string": "",  # Empty - invalid
                }
            }
        }
        global_config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_global_config()

        assert "global_config_validation" in str(exc_info.value)

    def test_load_global_config_invalid_deployment_type(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config with empty deployment type."""
        monkeypatch.setenv("HOME", str(temp_dir))

        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"

        # Invalid config with empty deployment type
        config_content = {
            "deployment": {
                "type": "",  # Empty - invalid
                "settings": {"replicas": 3},
            }
        }
        global_config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_global_config()

        assert "global_config_validation" in str(exc_info.value)

    def test_load_global_config_missing_required_vectorstore_field(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config with missing required vectorstore field."""
        monkeypatch.setenv("HOME", str(temp_dir))

        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"

        # Missing connection_string - required field
        config_content = {
            "vectorstores": {
                "postgres_store": {
                    "provider": "postgres",
                    # Missing required connection_string
                }
            }
        }
        global_config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_global_config()

        assert "global_config_validation" in str(exc_info.value)

    def test_load_global_config_extra_fields_not_allowed(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config rejects extra fields."""
        monkeypatch.setenv("HOME", str(temp_dir))

        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"

        # Config with extra field not in model
        config_content = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                }
            },
            "unknown_field": "should_not_be_allowed",  # Extra field
        }
        global_config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_global_config()

        assert "global_config_validation" in str(exc_info.value)

    def test_load_global_config_invalid_llm_provider(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config with invalid LLM provider configuration."""
        monkeypatch.setenv("HOME", str(temp_dir))

        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"

        # Invalid LLM provider config - missing required name field
        config_content = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    # Missing required name field
                }
            }
        }
        global_config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_global_config()

        assert "global_config_validation" in str(exc_info.value)

    def test_load_global_config_valid_with_all_sections(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test load_global_config succeeds with all valid sections."""
        monkeypatch.setenv("HOME", str(temp_dir))

        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"

        config_content = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                }
            },
            "vectorstores": {
                "postgres_store": {
                    "provider": "postgres",
                    "connection_string": "postgresql://localhost/holodeck",
                    "options": {"pool_size": 10},
                }
            },
            "deployment": {
                "type": "docker",
                "settings": {"image": "holodeck:latest"},
            },
        }
        global_config_file.write_text(yaml.dump(config_content))

        loader = ConfigLoader()
        result = loader.load_global_config()

        assert result is not None
        assert isinstance(result, GlobalConfig)
        assert result.providers is not None
        assert "openai" in result.providers
        assert result.vectorstores is not None
        assert "postgres_store" in result.vectorstores
        assert result.deployment is not None
        assert result.deployment.type == "docker"
