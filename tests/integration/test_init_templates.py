"""Integration tests for template selection (Phase 4: US2).

Tests for the holodeck init command with:
- Research template selection
- Customer-support template selection
- Invalid template error handling
- All templates producing valid agent.yaml
- Template-specific instructions

Refactored to use template_project_module fixture and parameterization
to eliminate redundant subprocess calls.
"""

from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
class TestInitTemplateSelection:
    """Test template selection functionality (T040, T041).

    Consolidates research and customer-support template tests to use
    parameterized template_project_module fixture instead of creating
    projects multiple times.
    """

    def test_template_creates_project(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify all templates create valid project structure.

        Test case T040, T041: Template creation for all templates
        - Research template
        - Customer-support template
        - Conversational template (default)

        Tests all templates via parameterization with single subprocess
        call per template.
        """
        project_dir, template, result = template_project_module

        # Verify project directory was created
        assert project_dir.exists(), f"{template}: Project directory not created"
        assert project_dir.is_dir(), f"{template}: Project path is not a directory"

    def test_template_creates_valid_agent_yaml(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify all templates create valid agent.yaml.

        Test case T040, T041: Verify agent.yaml is valid for all templates
        """
        project_dir, template, result = template_project_module

        # Verify agent.yaml exists
        agent_yaml = project_dir / "agent.yaml"
        assert agent_yaml.exists(), f"{template}: agent.yaml not created"

        # Verify agent.yaml is valid YAML
        config = yaml.safe_load(agent_yaml.read_text())
        assert config is not None, f"{template}: agent.yaml is not valid YAML"
        assert "name" in config, f"{template}: agent.yaml missing 'name' field"
        assert "model" in config, f"{template}: agent.yaml missing 'model' field"


@pytest.mark.integration
class TestInvalidTemplateHandling:
    """Test invalid template error handling (T042)."""

    def test_invalid_template_shows_error(self, init_project) -> None:
        """Verify invalid template selection shows helpful error message.

        Test case T042: Invalid template error handling
        """
        project_name = "test-invalid"

        # Run holodeck init with invalid template
        project_dir, result = init_project(
            project_name, template="invalid-template-xyz"
        )

        # Command should fail
        assert result.returncode != 0, "Command should fail for invalid template"

        # Error message should mention available templates
        error_output = result.stderr.lower()
        assert (
            "template" in error_output or "available" in error_output
        ), f"Error message should mention templates: {result.stderr}"

    def test_invalid_template_no_project_created(self, init_project, temp_dir) -> None:
        """Verify no project is created when template is invalid.

        Test case T042: No partial projects on template error
        """
        project_name = "test-invalid-2"

        project_dir, result = init_project(
            project_name, template="nonexistent-template"
        )

        # Command should fail
        assert result.returncode != 0, "Command should fail for invalid template"

        # Project directory should not exist
        assert not project_dir.exists(), (
            f"Project directory should not be created for invalid "
            f"template: {project_dir}"
        )


@pytest.mark.integration
class TestAllTemplatesProduceValidYAML:
    """Test all templates produce valid agent.yaml (T043)."""

    def test_template_produces_valid_agent_yaml(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify all templates produce parseable agent.yaml with required fields.

        Test case T043: All templates generate valid agent.yaml
        - Conversational template
        - Research template
        - Customer-support template

        Tests all templates via parameterization.
        """
        project_dir, template, result = template_project_module

        agent_yaml = project_dir / "agent.yaml"
        assert agent_yaml.exists(), f"{template}: agent.yaml not created"

        # Parse and validate YAML structure
        config = yaml.safe_load(agent_yaml.read_text())
        assert config is not None, f"{template}: agent.yaml is not valid YAML"

        # Verify required fields
        required_fields = ["name", "model", "instructions", "tools"]
        for field in required_fields:
            assert (
                field in config
            ), f"{template}: agent.yaml missing required field '{field}'"

        # Verify model configuration
        assert isinstance(
            config["model"], dict
        ), f"{template}: model field is not a dictionary"
        assert (
            "provider" in config["model"]
        ), f"{template}: model missing 'provider' field"


@pytest.mark.integration
class TestTemplateSpecificInstructions:
    """Test template-specific instructions (T044)."""

    def test_template_has_specific_instructions(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify each template includes template-specific instructions.

        Test case T044: Template-specific instructions
        - Research: mentions analysis, papers, citations
        - Customer-support: mentions tickets, customers, issues
        - Conversational: mentions chat, dialogue, conversation

        Tests all templates via parameterization.
        """
        project_dir, template, result = template_project_module

        # Read system prompt
        system_prompt = project_dir / "instructions" / "system-prompt.md"
        assert system_prompt.exists(), f"{template}: system-prompt.md not found"

        content = system_prompt.read_text().lower()

        # Define template-specific keywords
        template_keywords = {
            "conversational": ["conversation", "chat", "dialogue", "friendly"],
            "research": ["research", "analysis", "paper", "academic", "citation"],
            "customer-support": [
                "support",
                "customer",
                "help",
                "ticket",
                "issue",
                "resolve",
            ],
        }

        expected_keywords = template_keywords.get(template, [])
        matching_keywords = [kw for kw in expected_keywords if kw in content]

        assert len(matching_keywords) > 0, (
            f"{template}: Instructions don't contain template-specific keywords. "
            f"Expected one of {expected_keywords}, found none"
        )

    def test_template_has_different_instructions(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify templates have meaningfully different instructions.

        Test case T044: Template instructions are not generic
        """
        project_dir, template, result = template_project_module

        # Read system prompt
        system_prompt = project_dir / "instructions" / "system-prompt.md"
        content = system_prompt.read_text()

        # Instructions should not be empty
        assert (
            len(content.strip()) > 100
        ), f"{template}: Instructions are too short (likely generic)"

        # Should contain template name or related terms
        content_lower = content.lower()

        # Each template should mention its domain
        if template == "conversational":
            domain_terms = ["conversation", "chat", "dialogue"]
        elif template == "research":
            domain_terms = ["research", "paper", "academic"]
        elif template == "customer-support":
            domain_terms = ["support", "customer", "ticket", "issue"]
        else:
            domain_terms = []

        has_domain_term = any(term in content_lower for term in domain_terms)
        assert has_domain_term, (
            f"{template}: Instructions don't mention domain-specific terms. "
            f"Expected one of {domain_terms}"
        )
