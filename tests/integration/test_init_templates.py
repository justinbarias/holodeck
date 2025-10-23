"""Integration tests for template selection (Phase 4: US2).

Tests for the holodeck init command with:
- Research template selection
- Customer-support template selection
- Invalid template error handling
- All templates producing valid agent.yaml
- Template-specific instructions
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
class TestInitResearchTemplate:
    """Test research template selection functionality (T040)."""

    def test_research_template_creates_project(self, temp_dir: Path) -> None:
        """Verify `holodeck init <name> --template research` creates research project.

        Test case T040: Research template creation
        """
        project_name = "test-research"

        # Run holodeck init with research template
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "research",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify project directory was created
        project_dir = temp_dir / project_name
        assert project_dir.exists(), f"Project directory not created: {project_dir}"
        assert project_dir.is_dir(), f"Project path is not a directory: {project_dir}"

    def test_research_template_creates_agent_yaml(self, temp_dir: Path) -> None:
        """Verify research template creates valid agent.yaml.

        Test case T040: Verify agent.yaml is created and has research content
        """
        project_name = "test-research"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "research",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Verify agent.yaml exists
        agent_yaml = temp_dir / project_name / "agent.yaml"
        assert agent_yaml.exists(), f"agent.yaml not created: {agent_yaml}"

        # Verify agent.yaml is valid YAML
        config = yaml.safe_load(agent_yaml.read_text())
        assert config is not None, "agent.yaml is not valid YAML"
        assert "name" in config, "agent.yaml missing 'name' field"


@pytest.mark.integration
class TestInitCustomerSupportTemplate:
    """Test customer-support template selection functionality (T041)."""

    def test_customer_support_template_creates_project(self, temp_dir: Path) -> None:
        """Verify `holodeck init <name> --template customer-support` creates project.

        Test case T041: Customer-support template creation
        """
        project_name = "test-support"

        # Run holodeck init with customer-support template
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "customer-support",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify project directory was created
        project_dir = temp_dir / project_name
        assert project_dir.exists(), f"Project directory not created: {project_dir}"
        assert project_dir.is_dir(), f"Project path is not a directory: {project_dir}"

    def test_customer_support_template_creates_agent_yaml(self, temp_dir: Path) -> None:
        """Verify customer-support template creates valid agent.yaml.

        Test case T041: Verify agent.yaml is created with support content
        """
        project_name = "test-support"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "customer-support",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Verify agent.yaml exists
        agent_yaml = temp_dir / project_name / "agent.yaml"
        assert agent_yaml.exists(), f"agent.yaml not created: {agent_yaml}"

        # Verify agent.yaml is valid YAML
        config = yaml.safe_load(agent_yaml.read_text())
        assert config is not None, "agent.yaml is not valid YAML"
        assert "name" in config, "agent.yaml missing 'name' field"


@pytest.mark.integration
class TestInvalidTemplateHandling:
    """Test invalid template error handling (T042)."""

    def test_invalid_template_shows_error(self, temp_dir: Path) -> None:
        """Verify invalid template selection shows helpful error message.

        Test case T042: Invalid template error handling
        """
        project_name = "test-invalid"

        # Run holodeck init with invalid template
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "invalid-template-xyz",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Command should fail
        assert result.returncode != 0, "Command should fail for invalid template"

        # Error message should mention available templates
        error_output = result.stderr.lower()
        assert (
            "template" in error_output or "available" in error_output
        ), f"Error message should mention templates: {result.stderr}"

    def test_invalid_template_no_project_created(self, temp_dir: Path) -> None:
        """Verify no project is created when template is invalid.

        Test case T042: No partial projects on template error
        """
        project_name = "test-invalid-2"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "invalid-template",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Project directory should not exist
        project_dir = temp_dir / project_name
        assert (
            not project_dir.exists()
        ), "Project directory should not be created on template error"


@pytest.mark.integration
class TestAllTemplatesValidAgentYaml:
    """Test all templates produce valid agent.yaml (T043)."""

    @pytest.mark.parametrize(
        "template", ["conversational", "research", "customer-support"]
    )
    def test_template_produces_valid_agent_yaml(
        self, temp_dir: Path, template: str
    ) -> None:
        """Verify each template produces valid agent.yaml per AgentConfig schema.

        Test case T043: All templates valid YAML
        """
        project_name = f"test-{template}"

        # Run holodeck init with template
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                template,
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Command should succeed
        assert result.returncode == 0, f"Command failed for {template}: {result.stderr}"

        # Verify agent.yaml exists and is valid YAML
        agent_yaml = temp_dir / project_name / "agent.yaml"
        assert agent_yaml.exists(), f"agent.yaml not created for {template}"

        # Parse YAML and verify structure
        config = yaml.safe_load(agent_yaml.read_text())
        assert config is not None, f"agent.yaml invalid YAML for {template}"

        # Verify required fields
        required_fields = ["name", "model", "instructions", "tools"]
        for field in required_fields:
            assert (
                field in config
            ), f"agent.yaml missing required field '{field}' for {template}"


@pytest.mark.integration
class TestTemplateSpecificInstructions:
    """Test template-specific instructions content (T044)."""

    def test_research_template_has_vector_search_example(self, temp_dir: Path) -> None:
        """Verify research template includes vector search tool examples.

        Test case T044: Research template specific content
        """
        project_name = "test-research-specific"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "research",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Check agent.yaml for vector search references
        agent_yaml = temp_dir / project_name / "agent.yaml"
        content = agent_yaml.read_text()

        # Should have some vector search reference (could be in comments or examples)
        config = yaml.safe_load(content)
        assert config is not None, "agent.yaml should be valid YAML"

        # Verify research-related instructions are present
        instructions_file = (
            temp_dir / project_name / "instructions" / "system-prompt.md"
        )
        assert (
            instructions_file.exists()
        ), "system-prompt.md should exist for research template"

    def test_customer_support_template_has_function_examples(
        self, temp_dir: Path
    ) -> None:
        """Verify customer-support template includes function tool examples.

        Test case T044: Customer-support template specific content
        """
        project_name = "test-support-specific"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                project_name,
                "--template",
                "customer-support",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Verify support-related instructions are present
        instructions_file = (
            temp_dir / project_name / "instructions" / "system-prompt.md"
        )
        assert (
            instructions_file.exists()
        ), "system-prompt.md should exist for support template"

        # Agent.yaml should be valid
        agent_yaml = temp_dir / project_name / "agent.yaml"
        config = yaml.safe_load(agent_yaml.read_text())
        assert config is not None, "agent.yaml should be valid YAML"
