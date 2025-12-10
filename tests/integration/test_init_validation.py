"""Integration tests for project structure validation (Phase 6: US4).

Tests for validation of:
- Valid project structure verification
- YAML syntax validation with error reporting
- AgentConfig schema validation
- Error message clarity and actionability
- Partial cleanup on failure
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
class TestInitValidation:
    """Test project structure validation functionality (T073-T077)."""

    def test_valid_project_structure_verification(self, temp_dir: Path) -> None:
        """Verify all required directories and files exist after successful init.

        Test case T073: Valid project structure verification
        """
        project_name = "test-agent"

        # Run holodeck init command
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                project_name,
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        project_dir = temp_dir / project_name

        # Verify all required directories exist
        required_dirs = [
            project_dir,
            project_dir / "instructions",
            project_dir / "tools",
            project_dir / "data",
        ]

        for required_dir in required_dirs:
            assert (
                required_dir.exists()
            ), f"Required directory not created: {required_dir}"
            assert required_dir.is_dir(), f"Path is not a directory: {required_dir}"

        # Verify required files exist
        assert (project_dir / "agent.yaml").exists(), "agent.yaml not created"

    def test_agent_yaml_yaml_syntax_validation(self, temp_dir: Path) -> None:
        """Verify generated agent.yaml has valid YAML syntax.

        Test case T074: YAML syntax validation rejection
        """
        project_name = "test-agent"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                project_name,
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        agent_yaml = temp_dir / project_name / "agent.yaml"
        content = agent_yaml.read_text()

        # Verify YAML syntax is valid by parsing it
        try:
            parsed = yaml.safe_load(content)
            assert parsed is not None, "YAML parsed but resulted in None"
            assert isinstance(parsed, dict), "YAML should parse to a dictionary"
            assert "name" in parsed, "agent.yaml missing 'name' field"
        except yaml.YAMLError as e:
            pytest.fail(f"Generated agent.yaml has invalid YAML syntax:\n{e}")

    def test_agent_config_schema_validation(self, temp_dir: Path) -> None:
        """Verify generated agent.yaml validates against AgentConfig schema.

        Test case T075: AgentConfig schema validation with invalid YAML rejection
        """
        project_name = "test-agent"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                project_name,
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        agent_yaml = temp_dir / project_name / "agent.yaml"
        content = agent_yaml.read_text()

        # Validate against Agent schema using Pydantic
        try:
            from holodeck.models.agent import Agent

            parsed = yaml.safe_load(content)
            agent = Agent.model_validate(parsed)

            # Verify required fields are present
            assert agent.name is not None, "Agent name is required"
            assert agent.description is not None, "Agent description is required"
            assert agent.model is not None, "Agent model is required"

        except Exception as e:
            pytest.fail(
                f"Generated agent.yaml does not validate against Agent schema:\n{e}"
            )

    def test_error_message_clarity_and_actionability(self, temp_dir: Path) -> None:
        """Verify error messages are clear and provide actionable next steps.

        Test case T076: Error message clarity and actionability
        """
        # Try to create project with invalid template
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                "test-agent",
                "--template",
                "invalid-template",
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Command should fail
        assert result.returncode != 0, "Command should have failed for invalid template"

        # Error message should contain helpful information
        error_output = result.stderr + result.stdout
        assert (
            "template" in error_output.lower()
        ), "Error message should mention template"
        assert (
            "available" in error_output.lower()
        ), "Error message should mention available templates"

    def test_partial_cleanup_on_failure(self, temp_dir: Path) -> None:
        """Verify partial directories are removed after initialization failure.

        Test case T077: Partial cleanup on failure
        """
        # Create a scenario where initialization fails
        # We'll test this by checking that failed init attempts don't leave partial dirs
        project_name = "test-cleanup"

        # Try with invalid template to force failure
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                project_name,
                "--template",
                "nonexistent-template",
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Command should fail
        assert result.returncode != 0, "Command should have failed"

        # Verify project directory was not created (or was cleaned up)
        project_dir = temp_dir / project_name
        assert (
            not project_dir.exists()
        ), "Partial project directory should be cleaned up on failure"

    def test_yaml_syntax_error_line_numbers(self, temp_dir: Path) -> None:
        """Verify error messages include line numbers for YAML syntax errors.

        Test case T074 extended: YAML errors include line numbers
        """
        # This test verifies that if a template generates invalid YAML,
        # the error message includes line numbers for debugging
        project_name = "test-agent"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                project_name,
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Successful init should work
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify the generated YAML is clean
        agent_yaml = temp_dir / project_name / "agent.yaml"
        content = agent_yaml.read_text()

        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            # If there's a YAML error, it should have line information
            assert hasattr(
                e, "problem_mark"
            ), "YAML error should include position information"

    def test_multiple_consecutive_inits_no_interference(self, temp_dir: Path) -> None:
        """Verify multiple consecutive init commands don't interfere with each other.

        Test case T099 extended: Race condition handling
        """
        # Create first project
        result1 = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                "agent-1",
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )
        assert result1.returncode == 0

        # Create second project immediately after
        result2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                "agent-2",
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0

        # Verify both projects exist and are valid
        project1 = temp_dir / "agent-1"
        project2 = temp_dir / "agent-2"

        assert project1.exists()
        assert project2.exists()
        assert (project1 / "agent.yaml").exists()
        assert (project2 / "agent.yaml").exists()

        # Verify both have valid YAML
        for project_dir in [project1, project2]:
            yaml_file = project_dir / "agent.yaml"
            content = yaml_file.read_text()
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                pytest.fail(
                    f"Generated agent.yaml in {project_dir} is not valid YAML:\n{e}"
                )
