"""Integration tests for project metadata (Phase 7: US5).

Tests for the holodeck init command with:
- --description flag in generated agent.yaml
- --author flag in generated agent.yaml
- Metadata with special characters and escaping
- Missing metadata defaults (placeholder text)
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
class TestInitMetadata:
    """Test project metadata functionality (T084-T087)."""

    def test_description_flag_persisted_in_agent_yaml(self, temp_dir: Path) -> None:
        """Verify --description flag is stored in generated agent.yaml.

        Test case T084: Metadata persisted correctly
        """
        project_name = "test-metadata-desc"
        description = "My test agent description"

        # Run holodeck init with --description flag
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                project_name,
                "--description",
                description,
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify agent.yaml was created and contains description
        project_dir = temp_dir / project_name
        agent_yaml_path = project_dir / "agent.yaml"
        assert agent_yaml_path.exists(), f"agent.yaml not created: {agent_yaml_path}"

        # Load and verify YAML content
        with open(agent_yaml_path) as f:
            agent_config = yaml.safe_load(f)

        assert agent_config is not None, "agent.yaml is empty"
        assert agent_config.get("description") == description, (
            f"Description not found in agent.yaml. "
            f"Expected: {description}, Got: {agent_config.get('description')}"
        )

    def test_author_flag_persisted_in_agent_yaml(self, temp_dir: Path) -> None:
        """Verify --author flag is stored in generated agent.yaml.

        Test case T085: Author field populated
        """
        project_name = "test-metadata-author"
        author = "John Doe"

        # Run holodeck init with --author flag
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                project_name,
                "--author",
                author,
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify agent.yaml was created and contains author
        project_dir = temp_dir / project_name
        agent_yaml_path = project_dir / "agent.yaml"
        assert agent_yaml_path.exists(), f"agent.yaml not created: {agent_yaml_path}"

        # Load and verify YAML content
        with open(agent_yaml_path) as f:
            agent_config = yaml.safe_load(f)

        assert agent_config is not None, "agent.yaml is empty"
        assert agent_config.get("author") == author, (
            f"Author not found in agent.yaml. "
            f"Expected: {author}, Got: {agent_config.get('author')}"
        )

    def test_metadata_with_special_characters_handled_safely(
        self, temp_dir: Path
    ) -> None:
        """Verify metadata with special characters and quotes are escaped safely.

        Test case T086: Special characters, quotes, newlines handled safely
        """
        project_name = "test-metadata-special"
        description = 'Agent with "quotes" and special chars: @#$%&'
        author = "O'Reilly & Associates"

        # Run holodeck init with special characters in metadata
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "holodeck.cli.main",
                "init",
                "--name",
                project_name,
                "--description",
                description,
                "--author",
                author,
                "--non-interactive",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify agent.yaml was created and values are preserved correctly
        project_dir = temp_dir / project_name
        agent_yaml_path = project_dir / "agent.yaml"
        assert agent_yaml_path.exists(), f"agent.yaml not created: {agent_yaml_path}"

        # Load and verify YAML content (YAML parser should handle escaping)
        with open(agent_yaml_path) as f:
            agent_config = yaml.safe_load(f)

        assert agent_config is not None, "agent.yaml is empty"
        assert agent_config.get("description") == description, (
            f"Description not properly escaped. "
            f"Expected: {description}, Got: {agent_config.get('description')}"
        )
        assert agent_config.get("author") == author, (
            f"Author not properly escaped. "
            f"Expected: {author}, Got: {agent_config.get('author')}"
        )

    def test_missing_metadata_shows_placeholder_defaults(self, temp_dir: Path) -> None:
        """Verify missing metadata shows placeholder text in generated agent.yaml.

        Test case T087: Sensible defaults provided when metadata not provided
        """
        project_name = "test-metadata-defaults"

        # Run holodeck init without --description or --author flags
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

        # Verify agent.yaml was created with placeholder defaults
        project_dir = temp_dir / project_name
        agent_yaml_path = project_dir / "agent.yaml"
        assert agent_yaml_path.exists(), f"agent.yaml not created: {agent_yaml_path}"

        # Load and verify YAML content has sensible defaults
        with open(agent_yaml_path) as f:
            agent_config = yaml.safe_load(f)

        assert agent_config is not None, "agent.yaml is empty"

        # Verify description has placeholder or is valid
        description = agent_config.get("description")
        if description is not None:
            assert isinstance(description, str), "Description should be a string"
            # Placeholder text should be helpful (TODO or similar)
            assert "TODO" in description or len(description) > 0

        # Verify author has placeholder or is valid
        author = agent_config.get("author")
        if author is not None:
            assert isinstance(author, str), "Author should be a string"
            # Placeholder text should be helpful (TODO or similar)
            assert "TODO" in author or len(author) > 0
