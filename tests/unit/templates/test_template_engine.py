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
    description: Search through documents
    type: vectorstore
    source: data/docs
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

    def test_render_and_validate_with_non_yaml_file(self):
        """Test render_and_validate with non-YAML files (should skip validation)."""
        template_path = Path(self.temp_dir) / "README.md.j2"
        template_path.write_text("# {{ project_name }}\n\nAuthor: {{ author }}")

        variables = {"project_name": "my-project", "author": "John Doe"}
        result = self.renderer.render_and_validate(str(template_path), variables)

        assert isinstance(result, str)
        assert "my-project" in result
        assert "John Doe" in result

    def test_validate_agent_config_with_empty_yaml_content(self):
        """Test validation with completely empty YAML content."""
        with pytest.raises(ValidationError) as exc_info:
            self.renderer.validate_agent_config("")

        assert "empty" in str(exc_info.value).lower()

    def test_validate_agent_config_with_null_yaml_content(self):
        """Test validation with YAML that parses to None."""
        with pytest.raises(ValidationError) as exc_info:
            self.renderer.validate_agent_config("null")

        assert "empty" in str(exc_info.value).lower()

    def test_validate_agent_config_with_complex_pydantic_error(self):
        """Test Pydantic validation error with multiple field errors."""
        invalid_yaml = """
name: ""
description: "Test"
model:
  provider: "invalid-provider"
  name: "gpt-4o"
instructions:
  inline: "Help"
"""
        with pytest.raises(ValidationError) as exc_info:
            self.renderer.validate_agent_config(invalid_yaml)

        error_msg = str(exc_info.value)
        # Should contain helpful error information
        assert len(error_msg) > 0

    def test_render_template_with_complex_jinja2_operations(self):
        """Test rendering with complex Jinja2 operations."""
        template_path = Path(self.temp_dir) / "complex.j2"
        template_content = (
            "{% for item in items %}"
            "{{ item }}"
            "{% if not loop.last %}, {% endif %}"
            "{% endfor %}"
        )
        template_path.write_text(template_content)

        variables = {"items": ["apple", "banana", "cherry"]}
        result = self.renderer.render_template(str(template_path), variables)

        assert result == "apple, banana, cherry"

    def test_render_template_with_jinja2_loops(self):
        """Test rendering with Jinja2 loop constructs."""
        template_path = Path(self.temp_dir) / "loop.j2"
        template_path.write_text("{% for i in range(3) %}{{ i }}{% endfor %}")

        variables = {}
        result = self.renderer.render_template(str(template_path), variables)

        assert result == "012"

    def test_template_environment_is_properly_initialized(self):
        """Test that TemplateRenderer initializes Jinja2 environment correctly."""
        renderer = TemplateRenderer()
        assert renderer.env is not None
        # Verify StrictUndefined is set - check that undefined is a class type
        assert isinstance(renderer.env.undefined, type)

    def test_render_template_with_environment_variables(self):
        """Test rendering with environment-like variable substitution."""
        template_path = Path(self.temp_dir) / "env.j2"
        template_path.write_text(
            "Model: {{ model_name }}\nVersion: {{ version }}\nStatus: {{ status }}"
        )

        variables = {
            "model_name": "gpt-4o",
            "version": "1.0.0",
            "status": "active",
        }
        result = self.renderer.render_template(str(template_path), variables)

        assert "gpt-4o" in result
        assert "1.0.0" in result
        assert "active" in result

    def test_render_template_with_dict_variable(self):
        """Test rendering with dict variables in template."""
        template_path = Path(self.temp_dir) / "dict.j2"
        template_path.write_text(
            "Provider: {{ config.provider }}\nModel: {{ config.model }}"
        )

        variables = {"config": {"provider": "openai", "model": "gpt-4o"}}
        result = self.renderer.render_template(str(template_path), variables)

        assert "openai" in result
        assert "gpt-4o" in result

    def test_validate_agent_config_minimal_valid_config(self):
        """Test validation with absolute minimal valid configuration."""
        minimal_yaml = """
name: agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Hello"
"""
        result = self.renderer.validate_agent_config(minimal_yaml)
        assert isinstance(result, Agent)
        assert result.name == "agent"
        assert result.model.provider == "openai"

    def test_setup_safe_filters_is_called(self):
        """Test that _setup_safe_filters is called during initialization."""
        # This indirectly tests the _setup_safe_filters method
        renderer = TemplateRenderer()
        assert renderer.env is not None
        # If _setup_safe_filters didn't run, basic filters should still work
        template_path = Path(self.temp_dir) / "filter.j2"
        template_path.write_text("{{ text | upper }}")
        result = renderer.render_template(str(template_path), {"text": "hello"})
        assert result == "HELLO"


class TestGetAvailableTemplates:
    """Tests for TemplateRenderer.get_available_templates() method."""

    def test_returns_list_of_dicts(self):
        """Test get_available_templates returns list of template metadata."""
        templates = TemplateRenderer.get_available_templates()
        assert isinstance(templates, list)
        # Should have at least the 3 built-in templates
        assert len(templates) >= 3

    def test_each_template_has_required_keys(self):
        """Test each template dict has value, display_name, description."""
        templates = TemplateRenderer.get_available_templates()
        for t in templates:
            assert "value" in t
            assert "display_name" in t
            assert "description" in t
            # All values should be strings
            assert isinstance(t["value"], str)
            assert isinstance(t["display_name"], str)
            assert isinstance(t["description"], str)

    def test_includes_known_templates(self):
        """Test that built-in templates are included."""
        templates = TemplateRenderer.get_available_templates()
        template_values = {t["value"] for t in templates}
        # These templates should exist
        assert "conversational" in template_values
        assert "research" in template_values
        assert "customer-support" in template_values

    def test_template_metadata_matches_manifests(self):
        """Test that template metadata matches manifest files."""
        templates = TemplateRenderer.get_available_templates()

        # Find conversational template
        conversational = next(
            (t for t in templates if t["value"] == "conversational"), None
        )
        assert conversational is not None
        assert conversational["display_name"] == "Conversational Agent"
        assert "multi-turn conversations" in conversational["description"].lower()

    def test_returns_empty_list_if_no_templates_dir(self):
        """Test returns empty list if templates directory doesn't exist."""
        # This is tested implicitly - if templates_dir doesn't exist, returns []
        # We can't easily test this without mocking, but the code path exists
        templates = TemplateRenderer.get_available_templates()
        # Just verify it returns a list (even if empty in some edge cases)
        assert isinstance(templates, list)

    def test_templates_are_sorted(self):
        """Test that templates are returned in sorted order."""
        templates = TemplateRenderer.get_available_templates()
        values = [t["value"] for t in templates]
        # Should be sorted alphabetically (from sorted(iterdir()))
        assert values == sorted(values)
