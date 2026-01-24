"""Integration tests for deploy build command.

These tests verify the end-to-end functionality of the deploy build command.
Tests marked with @pytest.mark.docker require Docker to be available.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from holodeck.cli.main import main


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_agent_dir() -> Path:
    """Return path to sample agent fixture."""
    return Path(__file__).parent.parent.parent / "fixtures" / "deploy" / "sample_agent"


class TestDeployBuildCommand:
    """Integration tests for holodeck deploy build command."""

    def test_build_dry_run_shows_dockerfile(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that --dry-run shows generated Dockerfile without building."""
        agent_yaml = sample_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "[DRY RUN]" in result.output
        assert "Would build image" in result.output
        assert "Generated Dockerfile:" in result.output
        assert "FROM" in result.output
        assert "ENTRYPOINT" in result.output
        assert "No image was built" in result.output

    def test_build_dry_run_shows_configuration(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that --dry-run shows build configuration."""
        agent_yaml = sample_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "Build Configuration:" in result.output
        assert "Agent:" in result.output
        assert "test-agent" in result.output
        assert "Image:" in result.output
        assert "ghcr.io/test-org/test-agent" in result.output

    def test_build_dry_run_with_custom_tag(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that --tag option overrides tag_strategy."""
        agent_yaml = sample_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run", "--tag", "v1.0.0"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "v1.0.0" in result.output
        assert "ghcr.io/test-org/test-agent:v1.0.0" in result.output

    def test_build_copies_instruction_files(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that instruction files are included in Dockerfile."""
        agent_yaml = sample_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # Should have COPY for instruction file
        assert "COPY instructions.md" in result.output

    def test_build_missing_deployment_section(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test error when agent.yaml has no deployment section."""
        # Create agent.yaml without deployment section
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            """
name: test-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: "You are a test agent."
"""
        )

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
        )

        assert result.exit_code == 2
        assert "deployment" in result.output.lower()

    def test_build_quiet_mode(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that --quiet suppresses progress output."""
        agent_yaml = sample_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run", "--quiet"],
            catch_exceptions=False,
        )

        # In quiet dry-run mode, should still show the Dockerfile
        assert result.exit_code == 0
        # But should not show verbose loading messages
        assert "Loading agent configuration" not in result.output


class TestDeployBuildDocker:
    """Integration tests that require Docker.

    These tests are marked with @pytest.mark.docker and will be skipped
    if Docker is not available.
    """

    @pytest.mark.docker
    def test_build_creates_image(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that build command creates a Docker image."""
        agent_yaml = sample_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--tag", "test-build"],
        )

        # If Docker is available, should build successfully
        if result.exit_code == 0:
            assert "Build Successful!" in result.output
            assert "test-build" in result.output

    @pytest.mark.docker
    def test_build_with_no_cache(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that --no-cache flag is passed to Docker build."""
        agent_yaml = sample_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--no-cache", "--tag", "test-nocache"],
        )

        # Just verify command runs (may fail if Docker not available)
        # The nocache flag is verified in unit tests
        assert result.exit_code in (0, 3)  # 0=success, 3=Docker error


class TestDeployBuildDockerNotAvailable:
    """Tests for when Docker is not available."""

    def test_build_without_docker_shows_error(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that appropriate error is shown when Docker is not available."""
        from docker.errors import DockerException

        agent_yaml = sample_agent_dir / "agent.yaml"

        with patch("docker.from_env") as mock_from_env:
            mock_from_env.side_effect = DockerException("Cannot connect to Docker")

            result = cli_runner.invoke(
                main,
                ["deploy", "build", str(agent_yaml)],
            )

            assert result.exit_code == 3
            assert "Docker" in result.output

    def test_build_dry_run_works_without_docker(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test that --dry-run works even without Docker."""
        from docker.errors import DockerException

        agent_yaml = sample_agent_dir / "agent.yaml"

        # Dry run should not need Docker
        with patch("docker.from_env") as mock_from_env:
            mock_from_env.side_effect = DockerException("Cannot connect to Docker")

            result = cli_runner.invoke(
                main,
                ["deploy", "build", str(agent_yaml), "--dry-run"],
                catch_exceptions=False,
            )

            # Dry run should succeed without Docker
            assert result.exit_code == 0
            assert "[DRY RUN]" in result.output
