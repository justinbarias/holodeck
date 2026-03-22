"""Integration tests for deploying Claude-backed agents.

These tests verify that the deploy build command correctly detects
Anthropic provider and includes Node.js in the generated Dockerfile.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from holodeck.cli.main import main


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def claude_agent_dir() -> Path:
    """Return path to Claude agent fixture."""
    return Path(__file__).parent.parent.parent / "fixtures" / "claude_agent"


@pytest.fixture
def sample_agent_dir() -> Path:
    """Return path to OpenAI sample agent fixture."""
    return Path(__file__).parent.parent.parent / "fixtures" / "deploy" / "sample_agent"


class TestDeployBuildClaudeAgent:
    """Integration tests for Claude agent deploy build command."""

    def test_build_claude_agent_dry_run_shows_nodejs(
        self, cli_runner: CliRunner, claude_agent_dir: Path
    ) -> None:
        """Test dry-run build of Claude agent includes Node.js in Dockerfile."""
        agent_yaml = claude_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "nodesource" in result.output.lower()
        assert "nodejs" in result.output.lower()

    def test_build_openai_agent_dry_run_no_nodejs(
        self, cli_runner: CliRunner, sample_agent_dir: Path
    ) -> None:
        """Test dry-run build of OpenAI agent does not include Node.js."""
        agent_yaml = sample_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "nodesource" not in result.output.lower()

    def test_build_claude_agent_dry_run_shows_standard_elements(
        self, cli_runner: CliRunner, claude_agent_dir: Path
    ) -> None:
        """Test dry-run build of Claude agent still has standard Dockerfile elements."""
        agent_yaml = claude_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "FROM" in result.output
        assert "ENTRYPOINT" in result.output
        assert "HEALTHCHECK" in result.output


class TestAcceptanceScenarios:
    """Acceptance tests for Claude deploy scenarios."""

    def test_anthropic_deploy_build_includes_nodejs(
        self, cli_runner: CliRunner, claude_agent_dir: Path
    ) -> None:
        """AC-1: Anthropic provider triggers Node.js installation in Dockerfile."""
        agent_yaml = claude_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # Both Python base image and Node.js install present
        assert "FROM" in result.output
        assert "nodesource.com/setup_22.x" in result.output

    def test_dry_run_shows_claude_dockerfile_additions(
        self, cli_runner: CliRunner, claude_agent_dir: Path
    ) -> None:
        """AC-5: Dry-run output shows Claude-specific Dockerfile additions."""
        agent_yaml = claude_agent_dir / "agent.yaml"

        result = cli_runner.invoke(
            main,
            ["deploy", "build", str(agent_yaml), "--dry-run"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        output = result.output
        # Node.js setup
        assert "nodesource.com/setup_22.x" in output
        assert "--no-install-recommends" in output
        assert "rm -rf /var/lib/apt/lists/*" in output
        # Security: non-root user
        assert "USER holodeck" in output
        # Health check present
        assert "HEALTHCHECK" in output
