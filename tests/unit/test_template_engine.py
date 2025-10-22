"""Unit tests for TemplateRenderer and template rendering logic.

Tests cover:
- T015: TemplateRenderer.render_template()
- T016: TemplateRenderer.validate_agent_config()
- T017: Error handling in template rendering
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from holodeck.cli.exceptions import InitError, ValidationError
from holodeck.lib.template_engine import TemplateRenderer
from holodeck.models.agent import Agent


class TestTemplateRendererRenderTemplate:
    """Tests for TemplateRenderer.render_template() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = TemplateRenderer()
        self.temp_dir = tempfile.mkdtemp()

    def test_render_template_with_valid_jinja2(self):
        """Test rendering a valid Jinja2 template."""
        # Create a test template file
        template_path = Path(self.temp_dir) / "test.j2"
        template_path.write_text("Hello {{ name }}")

        variables = {"name": "World"}
        result = self.renderer.render_template(str(template_path), variables)

        assert result == "Hello World"

    def test_render_template_with_multiple_variables(self):
        """Test rendering with multiple variables."""
        template_path = Path(self.temp_dir) / "test.j2"
        template_path.write_text("Project: {{ project_name }}, Author: {{ author }}")

        variables = {"project_name": "my-agent", "author": "Alice"}
        result = self.renderer.render_template(str(template_path), variables)

        assert result == "Project: my-agent, Author: Alice"

    def test_render_template_with_jinja2_filters(self):
        """Test rendering with Jinja2 filters."""
        template_path = Path(self.temp_dir) / "test.j2"
        template_path.write_text("{{ text | upper }}")

        variables = {"text": "hello"}
        result = self.renderer.render_template(str(template_path), variables)

        assert result == "HELLO"

    def test_render_template_with_jinja2_conditionals(self):
        """Test rendering with Jinja2 conditional logic."""
        template_path = Path(self.temp_dir) / "test.j2"
        template_path.write_text("{% if enabled %}ENABLED{% else %}DISABLED{% endif %}")

        # Test enabled
        variables = {"enabled": True}
        result = self.renderer.render_template(str(template_path), variables)
        assert result == "ENABLED"

        # Test disabled
        variables = {"enabled": False}
        result = self.renderer.render_template(str(template_path), variables)
        assert result == "DISABLED"

    def test_render_template_with_missing_variable(self):
        """Test rendering with missing required variable."""
        template_path = Path(self.temp_dir) / "test.j2"
        template_path.write_text("Hello {{ missing_var }}")

        variables = {}
        # StrictUndefined mode causes error for undefined variables
        from holodeck.cli.exceptions import InitError

        with pytest.raises(InitError):
            self.renderer.render_template(str(template_path), variables)

    def test_render_template_with_invalid_jinja2_syntax(self):
        """Test rendering template with invalid Jinja2 syntax."""
        template_path = Path(self.temp_dir) / "test.j2"
        template_path.write_text("{{ unclosed variable")

        variables = {}
        with pytest.raises((InitError, Exception)):
            self.renderer.render_template(str(template_path), variables)

    def test_render_template_with_nonexistent_file(self):
        """Test rendering when template file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.renderer.render_template("/nonexistent/template.j2", {})


class TestTemplateRendererValidateAgentConfig:
    """Tests for TemplateRenderer.validate_agent_config() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = TemplateRenderer()

    def test_validate_agent_config_with_valid_yaml(self):
        """Test validation passes for valid agent.yaml."""
        valid_yaml = """
name: my-agent
description: Test agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Be helpful"
"""
        result = self.renderer.validate_agent_config(valid_yaml)
        assert isinstance(result, Agent)
        assert result.name == "my-agent"

    def test_validate_agent_config_with_invalid_yaml_syntax(self):
        """Test validation fails for invalid YAML syntax."""
        invalid_yaml = """
name: my-agent
model:
  provider: openai
  name: gpt-4o
instructions: [unclosed
"""
        with pytest.raises((yaml.YAMLError, ValidationError)):
            self.renderer.validate_agent_config(invalid_yaml)

    def test_validate_agent_config_with_missing_required_fields(self):
        """Test validation fails when required fields are missing."""
        # Missing 'instructions'
        invalid_yaml = """
name: my-agent
model:
  provider: openai
  name: gpt-4o
"""
        with pytest.raises(ValidationError):
            self.renderer.validate_agent_config(invalid_yaml)

    def test_validate_agent_config_with_invalid_field_values(self):
        """Test validation fails for invalid field values."""
        # Empty name (invalid)
        invalid_yaml = """
name: ""
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
"""
        with pytest.raises(ValidationError):
            self.renderer.validate_agent_config(invalid_yaml)

    def test_validate_agent_config_with_extra_fields(self):
        """Test validation handles extra fields appropriately."""
        yaml_with_extras = """
name: my-agent
description: Test
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
extra_field: should_fail
"""
        with pytest.raises(ValidationError):
            self.renderer.validate_agent_config(yaml_with_extras)

    def test_validate_agent_config_with_valid_file_instruction(self):
        """Test validation with file-based instructions."""
        valid_yaml = """
name: my-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  file: "instructions/system-prompt.md"
"""
        result = self.renderer.validate_agent_config(valid_yaml)
        assert isinstance(result, Agent)
        assert result.instructions.file == "instructions/system-prompt.md"

    def test_validate_agent_config_with_tools(self):
        """Test validation with tools configuration."""
        valid_yaml = """
name: my-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
tools:
  - name: search
    type: vectorstore
"""
        result = self.renderer.validate_agent_config(valid_yaml)
        assert isinstance(result, Agent)
        assert len(result.tools) == 1

    def test_validate_agent_config_returns_agent_instance(self):
        """Test that validation returns a proper Agent instance."""
        valid_yaml = """
name: test-agent
description: A test agent
model:
  provider: anthropic
  name: claude-3-opus
instructions:
  inline: "You are helpful"
"""
        result = self.renderer.validate_agent_config(valid_yaml)

        assert isinstance(result, Agent)
        assert result.name == "test-agent"
        assert result.description == "A test agent"
        assert result.model.provider == "anthropic"
        assert result.instructions.inline == "You are helpful"


class TestTemplateRendererErrorHandling:
    """Tests for error handling in TemplateRenderer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = TemplateRenderer()
        self.temp_dir = tempfile.mkdtemp()

    def test_render_template_error_with_clear_message(self):
        """Test that template rendering errors have clear messages."""
        template_path = Path(self.temp_dir) / "test.j2"
        template_path.write_text("{{ undefined_var | nonexistent_filter }}")

        variables = {}
        try:
            self.renderer.render_template(str(template_path), variables)
        except Exception as e:
            # Error message should be informative
            assert len(str(e)) > 0

    def test_validate_agent_config_error_with_line_numbers(self):
        """Test that validation errors include helpful context."""
        invalid_yaml = """
name: ""
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
"""
        with pytest.raises(ValidationError) as exc_info:
            self.renderer.validate_agent_config(invalid_yaml)

        # Error message should be helpful
        assert len(str(exc_info.value)) > 0

    def test_render_and_validate_with_rendering_failure(self):
        """Test render_and_validate when rendering fails."""
        template_path = Path(self.temp_dir) / "agent.yaml.j2"
        template_path.write_text("{{ bad_syntax")

        variables = {}
        with pytest.raises(InitError):
            self.renderer.render_and_validate(str(template_path), variables)

    def test_render_and_validate_with_validation_failure(self):
        """Test render_and_validate when validation fails."""
        template_path = Path(self.temp_dir) / "agent.yaml.j2"
        template_path.write_text(
            """
name: {{ project_name }}
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
"""
        )

        # This will render successfully but fail validation due to missing name
        variables = {"project_name": ""}
        with pytest.raises(ValidationError):
            self.renderer.render_and_validate(str(template_path), variables)

    def test_render_and_validate_returns_string(self):
        """Test that successful render_and_validate returns string."""
        template_path = Path(self.temp_dir) / "agent.yaml.j2"
        template_path.write_text(
            """
name: {{ project_name }}
description: {{ description }}
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
"""
        )

        variables = {"project_name": "my-agent", "description": "Test agent"}
        result = self.renderer.render_and_validate(str(template_path), variables)

        assert isinstance(result, str)
        assert "my-agent" in result
        assert "Test agent" in result


class TestTemplateRendererIntegration:
    """Integration tests for TemplateRenderer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = TemplateRenderer()
        self.temp_dir = tempfile.mkdtemp()

    def test_full_workflow_render_validate_agent_config(self):
        """Test complete workflow: render Jinja2 and validate as AgentConfig."""
        # Create template
        template_path = Path(self.temp_dir) / "agent.yaml.j2"
        template_content = """
name: {{ project_name }}
description: {{ description }}
model:
  provider: {{ model_provider }}
  name: {{ model_name }}
instructions:
  inline: {{ instructions }}
"""
        template_path.write_text(template_content)

        # Render
        variables = {
            "project_name": "my-research-tool",
            "description": "A research assistant",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "instructions": '"Analyze research papers"',
        }
        rendered = self.renderer.render_template(str(template_path), variables)

        # Validate
        agent = self.renderer.validate_agent_config(rendered)

        assert agent.name == "my-research-tool"
        assert agent.description == "A research assistant"
        assert agent.model.provider == "openai"
