"""Integration tests for example file generation (Phase 5: US3).

Tests for generating sample files and examples:
- Template files are generated (instructions, tools/README, data, tests)
- Example test cases YAML is valid
- Instructions are present and non-empty
- Data files are present with proper formatting
- Learning experience: examples discoverable and understandable

Refactored to use parameterized fixtures and eliminate redundant subprocess calls.
"""

import json
from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
class TestInitTemplateFilesGeneration:
    """Test that all template files are generated (T056).

    Uses template_project_module fixture to test all templates with single
    subprocess call per template instead of multiple calls.
    """

    def test_all_template_files_generated(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify all files generated for all templates.

        Test case T056: All template files are generated.
        Tests conversational, research, and customer-support templates
        via parameterization.
        """
        project_dir, template, result = template_project_module

        # Check for template-specific directories
        assert (
            project_dir / "instructions"
        ).exists(), f"{template}: instructions dir missing"
        assert (project_dir / "tools").exists(), f"{template}: tools dir missing"
        assert (project_dir / "data").exists(), f"{template}: data dir missing"

        # Check for key files
        assert (project_dir / "agent.yaml").exists(), f"{template}: agent.yaml missing"
        assert (
            project_dir / "instructions" / "system-prompt.md"
        ).exists(), f"{template}: system-prompt.md missing"
        assert (
            project_dir / "tools" / "README.md"
        ).exists(), f"{template}: tools/README.md missing"


@pytest.mark.integration
class TestExampleTestCasesValidity:
    """Test that example test cases YAML is valid (T057)."""

    def test_example_test_cases_are_valid_yaml(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify example test cases in agent.yaml are valid YAML.

        Test case T057: Valid YAML test cases with required fields.
        Tests all templates via parameterization.
        """
        project_dir, template, result = template_project_module

        agent_yaml_path = project_dir / "agent.yaml"
        agent_config = yaml.safe_load(agent_yaml_path.read_text())

        # Verify test_cases field exists
        assert "test_cases" in agent_config, f"{template}: test_cases missing"
        assert isinstance(
            agent_config["test_cases"], list
        ), f"{template}: test_cases not a list"
        assert len(agent_config["test_cases"]) > 0, f"{template}: test_cases is empty"

        # Verify each test case has required fields
        for i, test_case in enumerate(agent_config["test_cases"]):
            assert "name" in test_case, f"{template}: test_case[{i}] missing 'name'"
            assert "input" in test_case, f"{template}: test_case[{i}] missing 'input'"
            assert (
                "expected_tools" in test_case
            ), f"{template}: test_case[{i}] missing 'expected_tools'"
            assert (
                "ground_truth" in test_case
            ), f"{template}: test_case[{i}] missing 'ground_truth'"

    def test_example_test_cases_multiple_per_template(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify 2-3 example test cases per template.

        Test case T057: Multiple test case examples (at least 2 per template).
        """
        project_dir, template, result = template_project_module

        agent_yaml_path = project_dir / "agent.yaml"
        agent_config = yaml.safe_load(agent_yaml_path.read_text())

        # Verify we have at least 2 test cases
        assert "test_cases" in agent_config, f"{template}: test_cases missing"
        num_cases = len(agent_config["test_cases"])
        assert (
            num_cases >= 2
        ), f"Template {template} has {num_cases} test cases (expected >= 2)"


@pytest.mark.integration
class TestInstructionsContent:
    """Test that instructions are present and non-empty (T058)."""

    def test_system_prompt_instructions_present(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify system-prompt.md file is present and non-empty.

        Test case T058: Instructions present for all templates.
        """
        project_dir, template, result = template_project_module

        system_prompt = project_dir / "instructions" / "system-prompt.md"
        assert system_prompt.exists(), f"{template}: system-prompt.md missing"
        assert system_prompt.is_file(), f"{template}: system-prompt.md not a file"

        content = system_prompt.read_text()
        assert len(content) > 0, f"{template}: Instructions file is empty"

    def test_instructions_template_specific(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify instructions are specific to template type.

        Test case T058: Template-specific content in instructions.
        """
        project_dir, template, result = template_project_module

        system_prompt = project_dir / "instructions" / "system-prompt.md"
        content = system_prompt.read_text().lower()

        # Template-specific checks
        template_keywords = {
            "conversational": ["conversation", "chat", "dialogue"],
            "research": ["research", "analysis", "paper", "academic"],
            "customer-support": ["support", "customer", "help", "ticket", "issue"],
        }

        expected_keywords = template_keywords.get(template, [])
        has_keyword = any(keyword in content for keyword in expected_keywords)

        assert has_keyword, (
            f"{template}: Instructions don't contain template-specific keywords. "
            f"Expected one of {expected_keywords}"
        )

    def test_tools_readme_present_all_templates(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify tools/README.md exists for all templates.

        Test case T058: Tools README present with content.
        """
        project_dir, template, result = template_project_module

        tools_readme = project_dir / "tools" / "README.md"
        assert tools_readme.exists(), f"{template}: tools/README.md missing"
        assert tools_readme.is_file(), f"{template}: tools/README.md not a file"

        content = tools_readme.read_text()
        assert len(content) > 0, f"{template}: tools/README.md is empty"


@pytest.mark.integration
class TestDataFilesFormatting:
    """Test that data files are present with proper formatting (T059)."""

    def test_data_directory_exists(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify data directory exists for all templates.

        Test case T059: Data directory present.
        """
        project_dir, template, result = template_project_module

        data_dir = project_dir / "data"
        assert data_dir.exists(), f"{template}: data directory missing"
        assert data_dir.is_dir(), f"{template}: data is not a directory"

    def test_template_specific_data_files_valid(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify template-specific data files are valid.

        Test case T059: Data files formatting for all templates.
        Tests conversational (markdown), research (JSON), and customer-support (CSV).
        """
        project_dir, template, result = template_project_module

        data_dir = project_dir / "data"

        if template == "conversational":
            # Conversational should have faqs.md
            faqs_file = data_dir / "faqs.md"
            if faqs_file.exists():
                content = faqs_file.read_text()
                assert len(content) > 0, f"{template}: faqs.md is empty"
                # Should be valid markdown (has headers or lists)
                assert (
                    "#" in content or "-" in content
                ), f"{template}: faqs.md doesn't appear to be markdown"

        elif template == "research":
            # Research should have papers_index.json
            papers_file = data_dir / "papers_index.json"
            if papers_file.exists():
                content = papers_file.read_text()
                # Should be valid JSON
                try:
                    data = json.loads(content)
                    assert isinstance(data, dict | list), (
                        f"{template}: papers_index.json doesn't contain "
                        "valid JSON structure"
                    )
                except json.JSONDecodeError as e:
                    pytest.fail(f"{template}: papers_index.json is not valid JSON: {e}")

        elif template == "customer-support":
            # Support should have sample_issues.csv
            csv_file = data_dir / "sample_issues.csv"
            if csv_file.exists():
                content = csv_file.read_text()
                assert len(content) > 0, f"{template}: sample_issues.csv is empty"
                # Should have CSV structure (commas and newlines)
                assert (
                    "," in content or "\n" in content
                ), f"{template}: sample_issues.csv doesn't appear to be CSV format"


@pytest.mark.integration
class TestLearningExperience:
    """Test learning experience: examples discoverable and understandable (T060)."""

    def test_generated_project_has_example_structure(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify generated project structure enables learning.

        Test case T060: Learning experience - project structure.
        """
        project_dir, template, result = template_project_module

        # Key structure for learning
        assert (
            project_dir / "agent.yaml"
        ).exists(), f"{template}: agent.yaml missing (main config)"
        assert (
            project_dir / "instructions"
        ).exists(), f"{template}: instructions dir missing"
        assert (project_dir / "tools").exists(), f"{template}: tools dir missing"
        assert (project_dir / "data").exists(), f"{template}: data dir missing"

    def test_agent_yaml_includes_comments_and_examples(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify agent.yaml has comments or examples for learning.

        Test case T060: Learning experience - agent.yaml examples.
        """
        project_dir, template, result = template_project_module

        agent_yaml = project_dir / "agent.yaml"
        content = agent_yaml.read_text()

        # Should have test_cases as examples
        assert (
            "test_cases:" in content or "test_cases" in content
        ), f"{template}: agent.yaml missing test_cases examples"
        # Should have model configuration
        assert "model:" in content, f"{template}: agent.yaml missing model config"
        # Should have tools section (empty or with examples)
        assert "tools:" in content, f"{template}: agent.yaml missing tools section"

    def test_examples_cover_common_use_cases(
        self, template_project_module: tuple[Path, str, object]
    ) -> None:
        """Verify examples cover common agent scenarios.

        Test case T060: Learning experience - use case coverage.
        """
        project_dir, template, result = template_project_module

        agent_yaml_path = project_dir / "agent.yaml"
        agent_config = yaml.safe_load(agent_yaml_path.read_text())

        # Each template should have example test cases
        assert "test_cases" in agent_config, f"{template}: test_cases missing"
        test_cases = agent_config["test_cases"]

        # Each test case should have descriptive names and inputs
        for i, test_case in enumerate(test_cases):
            name = test_case.get("name", "")
            input_text = test_case.get("input", "")
            ground_truth = test_case.get("ground_truth", "")

            assert len(name) > 0, f"{template}: test_case[{i}] has empty name"
            assert len(input_text) > 0, f"{template}: test_case[{i}] has empty input"
            assert (
                len(ground_truth) > 0
            ), f"{template}: test_case[{i}] has empty ground_truth"
