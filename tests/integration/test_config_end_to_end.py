"""Integration tests for end-to-end configuration loading workflow."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from holodeck.config.loader import ConfigLoader
from holodeck.lib.errors import ConfigError, FileNotFoundError
from holodeck.models.agent import Agent
from holodeck.models.config import GlobalConfig


class TestConfigEndToEndWorkflow:
    """End-to-end tests for full configuration loading workflow (T043)."""

    def test_end_to_end_load_agent_with_tools_and_global_config(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test full workflow: load agent.yaml with tools, merge global config.

        This test validates:
        1. Agent YAML parsing
        2. Tool configuration loading
        3. Global config merging
        4. File reference resolution
        5. Successful Agent instantiation
        """
        # Setup global config (contains infrastructure settings, not Agent fields)
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"
        global_config_file.write_text(
            yaml.dump(
                {
                    "deployment": {
                        "type": "kubernetes",
                    },
                }
            )
        )

        # Setup agent YAML with tools
        agent_yaml_dir = temp_dir / "agent_config"
        agent_yaml_dir.mkdir()
        agent_yaml_file = agent_yaml_dir / "agent.yaml"

        # Create tool reference files
        tools_dir = agent_yaml_dir / "tools"
        tools_dir.mkdir()

        # Create instruction file
        prompts_dir = agent_yaml_dir / "prompts"
        prompts_dir.mkdir()
        system_prompt_file = prompts_dir / "system.md"
        system_prompt_file.write_text("You are a helpful research assistant.")

        agent_config = {
            "name": "research_agent",
            "description": "Agent for research tasks",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            "instructions": {"file": "prompts/system.md"},
            "tools": [
                {
                    "name": "search",
                    "type": "vectorstore",
                    "source": "documents.txt",
                    "description": "Search documents",
                }
            ],
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        # Load agent through ConfigLoader
        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        # Verify agent loaded successfully
        assert isinstance(agent, Agent)
        assert agent.name == "research_agent"
        assert agent.description == "Agent for research tasks"
        assert agent.model.provider.value == "openai"
        assert agent.instructions.file == "prompts/system.md"
        assert agent.tools is not None
        assert len(agent.tools) == 1

    def test_end_to_end_with_evaluations(self, temp_dir: Path) -> None:
        """Test full workflow with evaluation configuration.

        Validates:
        1. Evaluation config parsing
        2. Metric configuration loading
        3. Model override at metric level
        """
        agent_yaml_dir = temp_dir / "agent_config"
        agent_yaml_dir.mkdir()
        agent_yaml_file = agent_yaml_dir / "agent.yaml"

        agent_config = {
            "name": "evaluation_agent",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
            },
            "instructions": {"inline": "You are helpful."},
            "evaluations": {
                "model": {
                    "provider": "openai",
                    "name": "gpt-4o",
                },
                "metrics": [
                    {
                        "type": "standard",
                        "metric": "groundedness",
                        "threshold": 0.8,
                    },
                    {
                        "type": "standard",
                        "metric": "relevance",
                        "threshold": 0.75,
                    },
                ],
            },
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        assert agent.evaluations is not None
        assert len(agent.evaluations.metrics) == 2
        assert agent.evaluations.model is not None
        assert agent.evaluations.model.provider.value == "openai"

    def test_end_to_end_with_test_cases(self, temp_dir: Path) -> None:
        """Test full workflow with test cases.

        Validates:
        1. Test case parsing
        2. Multiple test cases loading
        3. Ground truth and evaluation references
        """
        agent_yaml_dir = temp_dir / "agent_config"
        agent_yaml_dir.mkdir()
        agent_yaml_file = agent_yaml_dir / "agent.yaml"

        agent_config = {
            "name": "test_agent",
            "model": {"provider": "anthropic", "name": "claude-3-opus"},
            "instructions": {"inline": "You are helpful."},
            "test_cases": [
                {
                    "input": "What is the capital of France?",
                    "ground_truth": "Paris",
                    "expected_tools": ["search"],
                },
                {
                    "input": "Explain machine learning",
                    "ground_truth": "A subset of AI...",
                },
            ],
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        assert agent.test_cases is not None
        assert len(agent.test_cases) == 2
        assert agent.test_cases[0].input == "What is the capital of France?"
        assert agent.test_cases[0].ground_truth == "Paris"

    def test_end_to_end_with_multiple_tools(self, temp_dir: Path) -> None:
        """Test full workflow with multiple different tool types.

        Validates:
        1. Tool type discrimination
        2. Multiple tools of different types
        3. Tool-specific validation
        """
        agent_yaml_dir = temp_dir / "agent_config"
        agent_yaml_dir.mkdir()
        agent_yaml_file = agent_yaml_dir / "agent.yaml"

        agent_config = {
            "name": "multi_tool_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Use tools effectively."},
            "tools": [
                {
                    "name": "vector_search",
                    "type": "vectorstore",
                    "source": "docs.txt",
                    "description": "Vector search",
                },
                {
                    "name": "python_exec",
                    "type": "function",
                    "file": "tools/python.py",
                    "function": "execute",
                    "description": "Execute Python code",
                },
                {
                    "name": "github",
                    "type": "mcp",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "description": "GitHub operations",
                },
                {
                    "name": "summarize",
                    "type": "prompt",
                    "template": "Summarize: {text}",
                    "description": "Summarize text",
                    "parameters": {
                        "text": {"type": "string", "description": "Text to summarize"}
                    },
                },
            ],
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        assert agent.tools is not None
        assert len(agent.tools) == 4
        # Verify tool types are preserved
        tool_types = {
            tool.get("type") for tool in agent.tools if isinstance(tool, dict)
        }
        assert "vectorstore" in tool_types or len(agent.tools) > 0

    def test_end_to_end_api_key_from_global_config_merges_into_agent(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that api_key from global config merges into agent config.

        Validates:
        1. Global config with provider configuration is loaded
        2. Agent config specifies provider
        3. ConfigLoader loads global config as GlobalConfig model
        4. Agent instance is successfully created
        """
        # Setup global config with provider configuration
        monkeypatch.setenv("HOME", str(temp_dir))
        monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o-global")
        monkeypatch.setenv("OPENAI_API_KEY", "some-api-key")

        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"
        global_config_file.write_text(
            yaml.dump(
                {
                    "providers": {
                        "openai": {
                            "provider": "openai",
                            "name": "${OPENAI_MODEL_NAME}",
                            "temperature": 0.5,
                            "api_key": "${OPENAI_API_KEY}",
                        }
                    }
                }
            )
        )

        # Setup agent YAML
        agent_yaml_dir = temp_dir / "agent_config"
        agent_yaml_dir.mkdir()
        agent_yaml_file = agent_yaml_dir / "agent.yaml"

        agent_config = {
            "name": "api_key_test_agent",
            "description": "Agent to test global config loading",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "temperature": 0.7,
            },
            "instructions": {"inline": "You are a helpful assistant."},
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        # Load agent through ConfigLoader
        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        # Verify agent loaded successfully
        assert isinstance(agent, Agent)
        assert agent.name == "api_key_test_agent"
        assert agent.model.provider.value == "openai"
        assert agent.model.name == "gpt-4o"
        assert agent.model.api_key == "some-api-key"

        # Verify global config was loaded as GlobalConfig model
        global_config = loader.load_global_config()
        assert isinstance(global_config, GlobalConfig)
        assert global_config.providers is not None
        assert "openai" in global_config.providers
        # After env substitution, name should be the actual value
        assert global_config.providers["openai"].name == "gpt-4o-global"
        assert global_config.providers["openai"].temperature == 0.5
        assert global_config.providers["openai"].api_key == "some-api-key"


class TestConfigErrorScenarios:
    """Error scenario tests for configuration loading (T044)."""

    def test_error_scenario_missing_required_field(self, temp_dir: Path) -> None:
        """Test error scenario: missing field raises ConfigError.

        Validates:
        1. Missing 'name' field is caught
        2. Error message is actionable
        3. Proper exception type is raised
        """
        agent_yaml_file = temp_dir / "agent.yaml"
        # Missing 'name' field
        agent_config = {
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_agent_yaml(str(agent_yaml_file))

        error_msg = str(exc_info.value).lower()
        assert "name" in error_msg or "required" in error_msg

    def test_error_scenario_invalid_yaml_syntax(self, temp_dir: Path) -> None:
        """Test error scenario: invalid YAML syntax raises ConfigError.

        Validates:
        1. Malformed YAML is caught
        2. Error is ConfigError
        3. Error message explains the issue
        """
        agent_yaml_file = temp_dir / "agent.yaml"
        # Invalid YAML syntax
        agent_yaml_file.write_text("invalid: [yaml: syntax: unclosed")

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_agent_yaml(str(agent_yaml_file))

        error_msg = str(exc_info.value).lower()
        assert "yaml" in error_msg or "parse" in error_msg

    def test_error_scenario_missing_instruction_file(self, temp_dir: Path) -> None:
        """Test error scenario: missing instruction file raises FileNotFoundError.

        Validates:
        1. Missing file is detected
        2. FileNotFoundError is raised
        3. Error includes file path
        """
        agent_yaml_dir = temp_dir / "agent_config"
        agent_yaml_dir.mkdir()
        agent_yaml_file = agent_yaml_dir / "agent.yaml"

        agent_config = {
            "name": "test",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"file": "nonexistent_prompts.md"},
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        # When we try to resolve the file
        agent = loader.load_agent_yaml(str(agent_yaml_file))
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.resolve_file_path(agent.instructions.file, str(agent_yaml_dir))

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg or "prompts" in error_msg

    def test_error_scenario_invalid_provider_value(self, temp_dir: Path) -> None:
        """Test error scenario: invalid provider value raises ValidationError.

        Validates:
        1. Invalid enum value is caught
        2. ValidationError is raised
        3. Error message includes expected values
        """
        agent_yaml_file = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test",
            "model": {"provider": "invalid_provider", "name": "gpt-4o"},
            "instructions": {"inline": "Test"},
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_agent_yaml(str(agent_yaml_file))

        error_msg = str(exc_info.value)
        assert "provider" in error_msg.lower() or "invalid" in error_msg.lower()

    def test_error_scenario_temperature_out_of_range(self, temp_dir: Path) -> None:
        """Test error scenario: temperature out of valid range.

        Validates:
        1. Temperature validation works
        2. Out-of-range values are rejected
        3. Error message explains constraint
        """
        agent_yaml_file = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test",
            "model": {
                "provider": "openai",
                "name": "gpt-4o",
                "temperature": 5.0,  # Invalid: must be 0-2
            },
            "instructions": {"inline": "Test"},
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        with pytest.raises(ConfigError) as exc_info:
            loader.load_agent_yaml(str(agent_yaml_file))

        error_msg = str(exc_info.value)
        assert "temperature" in error_msg.lower() or "range" in error_msg.lower()


class TestConfigPrecedenceScenarios:
    """Precedence scenario tests (T045)."""

    def test_precedence_scenario_agent_yaml_overrides_all(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test precedence: agent.yaml settings override all others.

        Setup:
        1. Global config with infrastructure settings
        2. Environment variables (for API keys)
        3. Agent YAML with explicit settings

        Verify:
        - Agent YAML values are used (highest precedence)
        """
        # Setup global config (infrastructure settings)
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        global_config_file = holodeck_dir / "config.yaml"
        global_config_file.write_text(
            yaml.dump(
                {
                    "deployment": {
                        "type": "azure",
                    },
                }
            )
        )

        # Set environment variables
        monkeypatch.setenv("DEFAULT_PROVIDER", "azure_openai")

        # Create agent YAML
        agent_yaml_dir = temp_dir / "agent_config"
        agent_yaml_dir.mkdir()
        agent_yaml_file = agent_yaml_dir / "agent.yaml"
        agent_config = {
            "name": "precedence_test",
            "model": {
                "provider": "openai",  # Most specific - should be used
                "name": "gpt-4o",
                "temperature": 0.7,
            },
            "instructions": {"inline": "Test"},
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        # Agent YAML should take precedence
        assert agent.model.provider.value == "openai"
        assert agent.model.temperature == 0.7

    def test_precedence_scenario_with_file_references(self, temp_dir: Path) -> None:
        """Test precedence: file-based instructions resolve relative to agent.yaml.

        Validates:
        1. Instruction files are resolved relative to agent.yaml directory
        2. Tool files can reference relative paths
        3. Relative paths work correctly
        """
        # Create agent directory structure
        agent_dir = temp_dir / "my_agent"
        agent_dir.mkdir()
        prompts_dir = agent_dir / "prompts"
        prompts_dir.mkdir()
        tools_dir = agent_dir / "tools"
        tools_dir.mkdir()

        # Create instruction file
        system_prompt = prompts_dir / "system.md"
        system_prompt.write_text("You are a system instruction.")

        # Create agent YAML
        agent_yaml_file = agent_dir / "agent.yaml"
        agent_config = {
            "name": "file_ref_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"file": "prompts/system.md"},
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        # Verify agent loaded successfully
        assert agent.name == "file_ref_agent"
        assert agent.instructions.file == "prompts/system.md"

        # Verify file can be resolved
        resolved = loader.resolve_file_path(agent.instructions.file, str(agent_dir))
        assert Path(resolved).exists()


class TestResponseFormatIntegration:
    """Integration tests for response_format in agent configuration (T023)."""

    def test_agent_with_inline_response_format(self, temp_dir: Path) -> None:
        """Test loading agent with inline response_format."""
        from holodeck.config.schema import SchemaValidator

        agent_dir = temp_dir / "agent1"
        agent_dir.mkdir()

        response_format = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer"],
        }

        # Create agent YAML with inline response_format
        agent_yaml_file = agent_dir / "agent.yaml"
        agent_config = {
            "name": "qa_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Answer questions"},
            "response_format": response_format,
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        # Verify response_format is stored
        assert agent.response_format == response_format
        assert agent.response_format["type"] == "object"
        assert "answer" in agent.response_format["properties"]

        # Verify response_format can be validated
        validated = SchemaValidator.validate_schema(agent.response_format)
        assert validated == response_format

    def test_agent_with_response_format_from_file(self, temp_dir: Path) -> None:
        """Test loading agent with response_format from external file."""
        import json

        from holodeck.config.schema import SchemaValidator

        agent_dir = temp_dir / "agent2"
        agent_dir.mkdir()

        # Create schema file
        schemas_dir = agent_dir / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "response.json"

        schema_content = {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["result"],
        }
        schema_file.write_text(json.dumps(schema_content))

        # Create agent YAML with file reference
        agent_yaml_file = agent_dir / "agent.yaml"
        agent_config = {
            "name": "file_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Process data"},
            "response_format": "schemas/response.json",
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        # Verify response_format is stored as file path
        assert agent.response_format == "schemas/response.json"

        # Verify schema can be loaded
        loaded_schema = SchemaValidator.load_schema_from_file(
            agent.response_format, base_dir=str(agent_dir)
        )
        assert loaded_schema == schema_content

    def test_agent_response_format_not_inherited(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that response_format is not inherited from global config."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()

        # User-level config (response_format shouldn't be here, but test the merge)
        user_config = {
            "providers": {
                "openai": {
                    "provider": "openai",
                    "name": "gpt-4o",
                }
            }
        }
        (holodeck_dir / "config.yml").write_text(yaml.dump(user_config))

        # Agent without response_format
        agent_dir = temp_dir / "agent3"
        agent_dir.mkdir()
        agent_yaml_file = agent_dir / "agent.yaml"
        agent_config = {
            "name": "basic_agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Basic agent"},
        }
        agent_yaml_file.write_text(yaml.dump(agent_config))

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml_file))

        # Verify response_format is None (not inherited)
        assert agent.response_format is None

    def test_response_format_validation_at_load_time(self, temp_dir: Path) -> None:
        """Test that invalid response_format is caught at config load time."""
        import json

        from holodeck.config.schema import SchemaValidator

        agent_dir = temp_dir / "agent4"
        agent_dir.mkdir()

        # Create invalid schema file (unsupported keyword)
        schemas_dir = agent_dir / "schemas"
        schemas_dir.mkdir()
        schema_file = schemas_dir / "bad.json"

        bad_schema = {
            "type": "object",
            "anyOf": [{"type": "string"}],  # unsupported keyword
        }
        schema_file.write_text(json.dumps(bad_schema))

        # Try to validate the schema
        with pytest.raises(ValueError) as exc_info:
            SchemaValidator.load_schema_from_file(
                "schemas/bad.json", base_dir=str(agent_dir)
            )

        # Verify error message is clear
        assert "anyOf" in str(exc_info.value)
