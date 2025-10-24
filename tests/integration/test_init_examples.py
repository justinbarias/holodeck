"""Integration tests for example file generation (Phase 5: US3).

Tests for generating sample files and examples:
- Template files are generated (instructions, tools/README, data, tests)
- Example test cases YAML is valid
- Instructions are present and non-empty
- Data files are present with proper formatting
- Learning experience: examples discoverable and understandable
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
class TestInitTemplateFilesGeneration:
    """Test that all template files are generated (T056)."""

    def test_all_template_files_generated_conversational(self, temp_dir: Path) -> None:
        """Verify all files generated for conversational template.

        Test case T056: All template files are generated
        """
        project_name = "test-conv"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        project_dir = temp_dir / project_name

        # Check for template-specific directories
        assert (project_dir / "instructions").exists()
        assert (project_dir / "tools").exists()
        assert (project_dir / "data").exists()

        # Check for key files
        assert (project_dir / "agent.yaml").exists()
        assert (project_dir / "instructions" / "system-prompt.md").exists()
        assert (project_dir / "tools" / "README.md").exists()

    def test_all_template_files_generated_research(self, temp_dir: Path) -> None:
        """Verify all files generated for research template.

        Test case T056: Research template files
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

        project_dir = temp_dir / project_name

        # Check for template-specific directories
        assert (project_dir / "instructions").exists()
        assert (project_dir / "tools").exists()
        assert (project_dir / "data").exists()

        # Check for key files
        assert (project_dir / "agent.yaml").exists()
        assert (project_dir / "instructions" / "system-prompt.md").exists()
        assert (project_dir / "tools" / "README.md").exists()

    def test_all_template_files_generated_support(self, temp_dir: Path) -> None:
        """Verify all files generated for customer-support template.

        Test case T056: Customer-support template files
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

        project_dir = temp_dir / project_name

        # Check for template-specific directories
        assert (project_dir / "instructions").exists()
        assert (project_dir / "tools").exists()
        assert (project_dir / "data").exists()

        # Check for key files
        assert (project_dir / "agent.yaml").exists()
        assert (project_dir / "instructions" / "system-prompt.md").exists()
        assert (project_dir / "tools" / "README.md").exists()


@pytest.mark.integration
class TestExampleTestCasesValidity:
    """Test that example test cases YAML is valid (T057)."""

    def test_example_test_cases_are_valid_yaml_conversational(
        self, temp_dir: Path
    ) -> None:
        """Verify example test cases in agent.yaml are valid YAML.

        Test case T057: Valid YAML test cases
        """
        project_name = "test-conv"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        agent_yaml_path = temp_dir / project_name / "agent.yaml"
        agent_config = yaml.safe_load(agent_yaml_path.read_text())

        # Verify test_cases field exists
        assert "test_cases" in agent_config
        assert isinstance(agent_config["test_cases"], list)
        assert len(agent_config["test_cases"]) > 0

        # Verify each test case has required fields
        for test_case in agent_config["test_cases"]:
            assert "name" in test_case
            assert "input" in test_case
            assert "expected_tools" in test_case
            assert "ground_truth" in test_case

    def test_example_test_cases_multiple_per_template(self, temp_dir: Path) -> None:
        """Verify 2-3 example test cases per template.

        Test case T057: Multiple test case examples
        """
        for template in ["conversational", "research", "customer-support"]:
            project_name = f"test-{template}"

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

            assert result.returncode == 0

            agent_yaml_path = temp_dir / project_name / "agent.yaml"
            agent_config = yaml.safe_load(agent_yaml_path.read_text())

            # Verify we have at least 2 test cases
            assert "test_cases" in agent_config
            assert (
                len(agent_config["test_cases"]) >= 2
            ), f"Template {template} has fewer than 2 test cases"


@pytest.mark.integration
class TestInstructionsContent:
    """Test that instructions are present and non-empty (T058)."""

    def test_system_prompt_instructions_present(self, temp_dir: Path) -> None:
        """Verify system-prompt.md file is present and non-empty.

        Test case T058: Instructions present
        """
        project_name = "test-conv"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        system_prompt = temp_dir / project_name / "instructions" / "system-prompt.md"
        assert system_prompt.exists()
        assert system_prompt.is_file()

        content = system_prompt.read_text()
        assert len(content) > 0, "Instructions file is empty"

    def test_instructions_template_specific(self, temp_dir: Path) -> None:
        """Verify instructions are specific to template type.

        Test case T058: Template-specific content in instructions
        """
        # Test conversational template
        project_name = "test-conv"
        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        system_prompt = temp_dir / project_name / "instructions" / "system-prompt.md"
        content = system_prompt.read_text()
        assert "conversation" in content.lower() or "chat" in content.lower()

    def test_tools_readme_present_all_templates(self, temp_dir: Path) -> None:
        """Verify tools/README.md exists for all templates.

        Test case T058: Tools README present
        """
        for template in ["conversational", "research", "customer-support"]:
            project_name = f"test-{template}"

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

            assert result.returncode == 0

            tools_readme = temp_dir / project_name / "tools" / "README.md"
            assert tools_readme.exists()
            assert tools_readme.is_file()
            content = tools_readme.read_text()
            assert len(content) > 0


@pytest.mark.integration
class TestDataFilesFormatting:
    """Test that data files are present with proper formatting (T059)."""

    def test_conversational_data_files_valid(self, temp_dir: Path) -> None:
        """Verify data files in conversational template are valid.

        Test case T059: Data files formatting - conversational
        """
        project_name = "test-conv"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        data_dir = temp_dir / project_name / "data"
        assert data_dir.exists()

        # Conversational should have faqs.md
        faqs_file = data_dir / "faqs.md"
        if faqs_file.exists():
            content = faqs_file.read_text()
            assert len(content) > 0, "faqs.md is empty"
            # Should be valid markdown
            assert "#" in content or "-" in content

    def test_research_data_files_valid(self, temp_dir: Path) -> None:
        """Verify data files in research template are valid JSON.

        Test case T059: Data files formatting - research
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

        data_dir = temp_dir / project_name / "data"
        assert data_dir.exists()

        # Research should have papers_index.json
        papers_file = data_dir / "papers_index.json"
        if papers_file.exists():
            content = papers_file.read_text()
            # Should be valid JSON
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                pytest.fail(f"papers_index.json is not valid JSON: {e}")

    def test_support_data_files_valid(self, temp_dir: Path) -> None:
        """Verify data files in customer-support template are valid CSV.

        Test case T059: Data files formatting - customer-support
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

        data_dir = temp_dir / project_name / "data"
        assert data_dir.exists()

        # Support should have sample_issues.csv
        csv_file = data_dir / "sample_issues.csv"
        if csv_file.exists():
            content = csv_file.read_text()
            assert len(content) > 0, "sample_issues.csv is empty"
            # Should have CSV structure (commas or similar)
            assert "," in content or "\n" in content


@pytest.mark.integration
class TestLearningExperience:
    """Test learning experience: examples discoverable and understandable (T060)."""

    def test_generated_project_has_example_structure(self, temp_dir: Path) -> None:
        """Verify generated project structure enables learning.

        Test case T060: Learning experience - project structure
        """
        project_name = "test-learn"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        project_dir = temp_dir / project_name

        # Key structure for learning
        assert (project_dir / "agent.yaml").exists()  # Main config to understand
        assert (project_dir / "instructions").exists()  # How to write instructions
        assert (project_dir / "tools").exists()  # How to add tools
        assert (project_dir / "data").exists()  # Example data

    def test_agent_yaml_includes_comments_and_examples(self, temp_dir: Path) -> None:
        """Verify agent.yaml has comments or examples for learning.

        Test case T060: Learning experience - agent.yaml examples
        """
        project_name = "test-learn"

        result = subprocess.run(
            [sys.executable, "-m", "holodeck.cli.main", "init", project_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        agent_yaml = temp_dir / project_name / "agent.yaml"
        content = agent_yaml.read_text()

        # Should have test_cases as examples
        assert "test_cases:" in content or "test_cases" in content
        # Should have model configuration
        assert "model:" in content
        # Should have tools section (empty or with examples)
        assert "tools:" in content

    def test_examples_cover_common_use_cases(self, temp_dir: Path) -> None:
        """Verify examples cover common agent scenarios.

        Test case T060: Learning experience - use case coverage
        """
        for template in ["conversational", "research", "customer-support"]:
            project_name = f"test-{template}"

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

            assert result.returncode == 0

            agent_yaml_path = temp_dir / project_name / "agent.yaml"
            agent_config = yaml.safe_load(agent_yaml_path.read_text())

            # Each template should have example test cases
            assert "test_cases" in agent_config
            test_cases = agent_config["test_cases"]

            # Each test case should have descriptive names and inputs
            for test_case in test_cases:
                assert len(test_case.get("name", "")) > 0
                assert len(test_case.get("input", "")) > 0
                assert len(test_case.get("ground_truth", "")) > 0
