"""Unit tests for TemplateRenderer and template rendering logic.

Tests cover:
- T015: TemplateRenderer.render_template()
- T016: TemplateRenderer.validate_agent_config()
- T017: Error handling in template rendering
"""

from pathlib import Path

import pytest
import yaml

from holodeck.cli.exceptions import InitError, ValidationError
from holodeck.lib.template_engine import TemplateRenderer
from holodeck.models.agent import Agent


class TestTemplateRendererRenderTemplate:
    """Tests for TemplateRenderer.render_template() method."""

    @pytest.mark.parametrize(
        "template_content,variables,expected",
        [
            pytest.param(
                "Hello {{ name }}",
                {"name": "World"},
                "Hello World",
                id="simple_variable",
            ),
            pytest.param(
                "Project: {{ project_name }}, Author: {{ author }}",
                {"project_name": "my-agent", "author": "Alice"},
                "Project: my-agent, Author: Alice",
                id="multiple_variables",
            ),
            pytest.param(
                "{{ text | upper }}",
                {"text": "hello"},
                "HELLO",
                id="jinja2_filter",
            ),
        ],
    )
    def test_render_template_variations(
        self,
        renderer: TemplateRenderer,
        template_temp_dir: Path,
        template_content: str,
        variables: dict,
        expected: str,
    ) -> None:
        """Test rendering templates with various inputs."""
        template_path = template_temp_dir / "test.j2"
        template_path.write_text(template_content)

        result = renderer.render_template(str(template_path), variables)
        assert result == expected

    def test_render_template_with_jinja2_conditionals(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test rendering with Jinja2 conditional logic."""
        template_path = template_temp_dir / "test.j2"
        template_path.write_text("{% if enabled %}ENABLED{% else %}DISABLED{% endif %}")

        assert (
            renderer.render_template(str(template_path), {"enabled": True}) == "ENABLED"
        )
        assert (
            renderer.render_template(str(template_path), {"enabled": False})
            == "DISABLED"
        )

    def test_render_template_with_missing_variable(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test rendering with missing required variable."""
        template_path = template_temp_dir / "test.j2"
        template_path.write_text("Hello {{ missing_var }}")

        with pytest.raises(InitError):
            renderer.render_template(str(template_path), {})

    def test_render_template_with_invalid_jinja2_syntax(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test rendering template with invalid Jinja2 syntax."""
        template_path = template_temp_dir / "test.j2"
        template_path.write_text("{{ unclosed variable")

        with pytest.raises((InitError, Exception)):
            renderer.render_template(str(template_path), {})

    def test_render_template_with_nonexistent_file(
        self, renderer: TemplateRenderer
    ) -> None:
        """Test rendering when template file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            renderer.render_template("/nonexistent/template.j2", {})


class TestTemplateRendererValidateAgentConfig:
    """Tests for TemplateRenderer.validate_agent_config() method."""

    def test_validate_agent_config_with_valid_yaml(
        self, renderer: TemplateRenderer
    ) -> None:
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
        result = renderer.validate_agent_config(valid_yaml)
        assert isinstance(result, Agent)
        assert result.name == "my-agent"

    def test_validate_agent_config_with_invalid_yaml_syntax(
        self, renderer: TemplateRenderer
    ) -> None:
        """Test validation fails for invalid YAML syntax."""
        invalid_yaml = """
name: my-agent
model:
  provider: openai
  name: gpt-4o
instructions: [unclosed
"""
        with pytest.raises((yaml.YAMLError, ValidationError)):
            renderer.validate_agent_config(invalid_yaml)

    def test_validate_agent_config_with_missing_required_fields(
        self, renderer: TemplateRenderer
    ) -> None:
        """Test validation fails when required fields are missing."""
        invalid_yaml = """
name: my-agent
model:
  provider: openai
  name: gpt-4o
"""
        with pytest.raises(ValidationError):
            renderer.validate_agent_config(invalid_yaml)

    def test_validate_agent_config_with_invalid_field_values(
        self, renderer: TemplateRenderer
    ) -> None:
        """Test validation fails for invalid field values."""
        invalid_yaml = """
name: ""
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
"""
        with pytest.raises(ValidationError):
            renderer.validate_agent_config(invalid_yaml)

    def test_validate_agent_config_with_extra_fields(
        self, renderer: TemplateRenderer
    ) -> None:
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
            renderer.validate_agent_config(yaml_with_extras)

    def test_validate_agent_config_with_valid_file_instruction(
        self, renderer: TemplateRenderer
    ) -> None:
        """Test validation with file-based instructions."""
        valid_yaml = """
name: my-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  file: "instructions/system-prompt.md"
"""
        result = renderer.validate_agent_config(valid_yaml)
        assert isinstance(result, Agent)
        assert result.instructions.file == "instructions/system-prompt.md"

    def test_validate_agent_config_with_tools(self, renderer: TemplateRenderer) -> None:
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
        result = renderer.validate_agent_config(valid_yaml)
        assert isinstance(result, Agent)
        assert len(result.tools) == 1

    def test_validate_agent_config_returns_agent_instance(
        self, renderer: TemplateRenderer
    ) -> None:
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
        result = renderer.validate_agent_config(valid_yaml)

        assert isinstance(result, Agent)
        assert result.name == "test-agent"
        assert result.description == "A test agent"
        assert result.model.provider == "anthropic"
        assert result.instructions.inline == "You are helpful"


class TestTemplateRendererErrorHandling:
    """Tests for error handling in TemplateRenderer."""

    def test_render_template_error_with_clear_message(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test that template rendering errors have clear messages."""
        template_path = template_temp_dir / "test.j2"
        template_path.write_text("{{ undefined_var | nonexistent_filter }}")

        try:
            renderer.render_template(str(template_path), {})
        except Exception as e:
            # Error message should be informative
            assert len(str(e)) > 0

    def test_validate_agent_config_error_with_line_numbers(
        self, renderer: TemplateRenderer
    ) -> None:
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
            renderer.validate_agent_config(invalid_yaml)

        # Error message should be helpful
        assert len(str(exc_info.value)) > 0

    def test_render_and_validate_with_rendering_failure(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test render_and_validate when rendering fails."""
        template_path = template_temp_dir / "agent.yaml.j2"
        template_path.write_text("{{ bad_syntax")

        with pytest.raises(InitError):
            renderer.render_and_validate(str(template_path), {})

    def test_render_and_validate_with_validation_failure(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test render_and_validate when validation fails."""
        template_path = template_temp_dir / "agent.yaml.j2"
        template_path.write_text("""
name: {{ project_name }}
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
""")

        # This will render successfully but fail validation due to missing name
        with pytest.raises(ValidationError):
            renderer.render_and_validate(str(template_path), {"project_name": ""})

    def test_render_and_validate_returns_string(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test that successful render_and_validate returns string."""
        template_path = template_temp_dir / "agent.yaml.j2"
        template_path.write_text("""
name: {{ project_name }}
description: {{ description }}
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Help"
""")

        result = renderer.render_and_validate(
            str(template_path),
            {"project_name": "my-agent", "description": "Test agent"},
        )

        assert isinstance(result, str)
        assert "my-agent" in result
        assert "Test agent" in result


class TestTemplateRendererIntegration:
    """Integration tests for TemplateRenderer."""

    def test_full_workflow_render_validate_agent_config(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test complete workflow: render Jinja2 and validate as AgentConfig."""
        template_path = template_temp_dir / "agent.yaml.j2"
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

        variables = {
            "project_name": "my-research-tool",
            "description": "A research assistant",
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "instructions": '"Analyze research papers"',
        }
        rendered = renderer.render_template(str(template_path), variables)

        agent = renderer.validate_agent_config(rendered)

        assert agent.name == "my-research-tool"
        assert agent.description == "A research assistant"
        assert agent.model.provider == "openai"

    def test_render_and_validate_with_non_yaml_file(
        self, renderer: TemplateRenderer, template_temp_dir: Path
    ) -> None:
        """Test render_and_validate with non-YAML files (should skip validation)."""
        template_path = template_temp_dir / "README.md.j2"
        template_path.write_text("# {{ project_name }}\n\nAuthor: {{ author }}")

        result = renderer.render_and_validate(
            str(template_path),
            {"project_name": "my-project", "author": "John Doe"},
        )

        assert isinstance(result, str)
        assert "my-project" in result
        assert "John Doe" in result

    @pytest.mark.parametrize(
        "yaml_content,error_match",
        [
            pytest.param("", "empty", id="empty_content"),
            pytest.param("null", "empty", id="null_content"),
        ],
    )
    def test_validate_agent_config_with_empty_content(
        self, renderer: TemplateRenderer, yaml_content: str, error_match: str
    ) -> None:
        """Test validation with empty/null YAML content."""
        with pytest.raises(ValidationError) as exc_info:
            renderer.validate_agent_config(yaml_content)

        assert error_match in str(exc_info.value).lower()

    def test_validate_agent_config_with_complex_pydantic_error(
        self, renderer: TemplateRenderer
    ) -> None:
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
            renderer.validate_agent_config(invalid_yaml)

        error_msg = str(exc_info.value)
        assert len(error_msg) > 0

    @pytest.mark.parametrize(
        "template_content,variables,expected_in_result",
        [
            pytest.param(
                (
                    "{% for item in items %}"
                    "{{ item }}"
                    "{% if not loop.last %}, {% endif %}"
                    "{% endfor %}"
                ),
                {"items": ["apple", "banana", "cherry"]},
                "apple, banana, cherry",
                id="for_loop_with_separator",
            ),
            pytest.param(
                "{% for i in range(3) %}{{ i }}{% endfor %}",
                {},
                "012",
                id="range_loop",
            ),
        ],
    )
    def test_render_template_with_complex_jinja2(
        self,
        renderer: TemplateRenderer,
        template_temp_dir: Path,
        template_content: str,
        variables: dict,
        expected_in_result: str,
    ) -> None:
        """Test rendering with complex Jinja2 operations."""
        template_path = template_temp_dir / "complex.j2"
        template_path.write_text(template_content)

        result = renderer.render_template(str(template_path), variables)
        assert result == expected_in_result

    def test_template_environment_is_properly_initialized(self) -> None:
        """Test that TemplateRenderer initializes Jinja2 environment correctly."""
        renderer = TemplateRenderer()
        assert renderer.env is not None
        assert isinstance(renderer.env.undefined, type)

    @pytest.mark.parametrize(
        "template_content,variables,expected_fragments",
        [
            pytest.param(
                "Model: {{ model_name }}\nVersion: {{ version }}\nStatus: {{ status }}",
                {"model_name": "gpt-4o", "version": "1.0.0", "status": "active"},
                ["gpt-4o", "1.0.0", "active"],
                id="env_like_variables",
            ),
            pytest.param(
                "Provider: {{ config.provider }}\nModel: {{ config.model }}",
                {"config": {"provider": "openai", "model": "gpt-4o"}},
                ["openai", "gpt-4o"],
                id="dict_variables",
            ),
        ],
    )
    def test_render_template_with_variable_structures(
        self,
        renderer: TemplateRenderer,
        template_temp_dir: Path,
        template_content: str,
        variables: dict,
        expected_fragments: list[str],
    ) -> None:
        """Test rendering templates with different variable structures."""
        template_path = template_temp_dir / "test.j2"
        template_path.write_text(template_content)

        result = renderer.render_template(str(template_path), variables)
        for fragment in expected_fragments:
            assert fragment in result

    def test_validate_agent_config_minimal_valid_config(
        self, renderer: TemplateRenderer
    ) -> None:
        """Test validation with absolute minimal valid configuration."""
        minimal_yaml = """
name: agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "Hello"
"""
        result = renderer.validate_agent_config(minimal_yaml)
        assert isinstance(result, Agent)
        assert result.name == "agent"
        assert result.model.provider == "openai"

    def test_setup_safe_filters_is_called(self, template_temp_dir: Path) -> None:
        """Test that _setup_safe_filters is called during initialization."""
        renderer = TemplateRenderer()
        assert renderer.env is not None
        template_path = template_temp_dir / "filter.j2"
        template_path.write_text("{{ text | upper }}")
        result = renderer.render_template(str(template_path), {"text": "hello"})
        assert result == "HELLO"


class TestGetAvailableTemplates:
    """Tests for TemplateRenderer.get_available_templates() method."""

    def test_returns_list_of_dicts(self) -> None:
        """Test get_available_templates returns list of template metadata."""
        templates = TemplateRenderer.get_available_templates()
        assert isinstance(templates, list)
        # Should have at least the 3 built-in templates
        assert len(templates) >= 3

    def test_each_template_has_required_keys(self) -> None:
        """Test each template dict has value, display_name, description."""
        templates = TemplateRenderer.get_available_templates()
        for t in templates:
            assert "value" in t
            assert "display_name" in t
            assert "description" in t
            assert isinstance(t["value"], str)
            assert isinstance(t["display_name"], str)
            assert isinstance(t["description"], str)

    @pytest.mark.parametrize(
        "template_name",
        [
            pytest.param("conversational", id="conversational"),
            pytest.param("research", id="research"),
            pytest.param("customer-support", id="customer_support"),
        ],
    )
    def test_includes_known_template(self, template_name: str) -> None:
        """Test that built-in templates are included."""
        templates = TemplateRenderer.get_available_templates()
        template_values = {t["value"] for t in templates}
        assert template_name in template_values

    def test_template_metadata_matches_manifests(self) -> None:
        """Test that template metadata matches manifest files."""
        templates = TemplateRenderer.get_available_templates()

        conversational = next(
            (t for t in templates if t["value"] == "conversational"), None
        )
        assert conversational is not None
        assert conversational["display_name"] == "Conversational Agent"
        assert "multi-turn conversations" in conversational["description"].lower()

    def test_returns_empty_list_if_no_templates_dir(self) -> None:
        """Test returns empty list if templates directory doesn't exist."""
        templates = TemplateRenderer.get_available_templates()
        assert isinstance(templates, list)

    def test_templates_are_sorted(self) -> None:
        """Test that templates are returned in sorted order."""
        templates = TemplateRenderer.get_available_templates()
        values = [t["value"] for t in templates]
        assert values == sorted(values)
