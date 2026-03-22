"""Unit tests for Dockerfile generation.

TDD tests - these should fail until T009 implements the generator.
"""

import re
from datetime import datetime
from pathlib import Path

import pytest


class TestDockerfileTemplate:
    """Tests for the Dockerfile template constant."""

    def test_template_exists(self) -> None:
        """Test that the Dockerfile template constant exists."""
        from holodeck.deploy.dockerfile import HOLODECK_DOCKERFILE_TEMPLATE

        assert HOLODECK_DOCKERFILE_TEMPLATE is not None
        assert isinstance(HOLODECK_DOCKERFILE_TEMPLATE, str)
        assert len(HOLODECK_DOCKERFILE_TEMPLATE) > 0


class TestGenerateDockerfile:
    """Tests for the generate_dockerfile function."""

    def test_minimal_dockerfile_generation(self) -> None:
        """Test minimal Dockerfile generation."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        # Should be a non-empty string
        assert isinstance(dockerfile, str)
        assert len(dockerfile) > 0

        # Should contain FROM instruction
        assert "FROM" in dockerfile

    def test_dockerfile_oci_labels(self) -> None:
        """Test OCI labels are included in generated Dockerfile."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="my-agent",
            port=8080,
            protocol="rest",
        )

        # Check for required OCI labels
        assert 'org.opencontainers.image.title="my-agent"' in dockerfile
        assert "org.opencontainers.image.version" in dockerfile
        assert "org.opencontainers.image.created" in dockerfile
        assert "org.opencontainers.image.source" in dockerfile
        assert 'com.holodeck.managed="true"' in dockerfile

    def test_dockerfile_instruction_files_copy(self) -> None:
        """Test COPY statements for instruction files."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            instruction_files=["instructions.md", "prompts/system.md"],
        )

        # Should have COPY for each instruction file
        assert "COPY instructions.md" in dockerfile
        assert "COPY prompts/system.md" in dockerfile

    def test_dockerfile_data_directories_copy(self) -> None:
        """Test COPY statements for data directories."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            data_directories=["data/", "knowledge/"],
        )

        # Should have COPY for each data directory
        assert "COPY data/" in dockerfile
        assert "COPY knowledge/" in dockerfile

    def test_dockerfile_protocol_configuration_rest(self) -> None:
        """Test protocol configuration for REST."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        assert (
            'HOLODECK_PROTOCOL="rest"' in dockerfile
            or "HOLODECK_PROTOCOL=rest" in dockerfile
        )

    def test_dockerfile_protocol_configuration_ag_ui(self) -> None:
        """Test protocol configuration for AG-UI."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="ag-ui",
        )

        assert (
            'HOLODECK_PROTOCOL="ag-ui"' in dockerfile
            or "HOLODECK_PROTOCOL=ag-ui" in dockerfile
        )

    def test_dockerfile_protocol_configuration_both(self) -> None:
        """Test protocol configuration for both protocols."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="both",
        )

        assert (
            'HOLODECK_PROTOCOL="both"' in dockerfile
            or "HOLODECK_PROTOCOL=both" in dockerfile
        )

    def test_dockerfile_custom_port(self) -> None:
        """Test custom port configuration."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=3000,
            protocol="rest",
        )

        # Should expose custom port
        assert "EXPOSE 3000" in dockerfile
        # Should set port in ENV
        assert (
            'HOLODECK_PORT="3000"' in dockerfile or "HOLODECK_PORT=3000" in dockerfile
        )

    def test_dockerfile_custom_base_image(self) -> None:
        """Test custom base image configuration."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            base_image="ghcr.io/holodeck-ai/base:v1.0.0",
        )

        assert "FROM ghcr.io/holodeck-ai/base:v1.0.0" in dockerfile

    def test_dockerfile_default_base_image(self) -> None:
        """Test default base image when not specified."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        # Should use default holodeck base image
        assert "FROM ghcr.io/justinbarias/holodeck-base:latest" in dockerfile

    def test_dockerfile_healthcheck_present(self) -> None:
        """Test HEALTHCHECK instruction is present."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        assert "HEALTHCHECK" in dockerfile
        # Should check /health endpoint
        assert "/health" in dockerfile

    def test_dockerfile_entrypoint_present(self) -> None:
        """Test ENTRYPOINT instruction is present."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        assert "ENTRYPOINT" in dockerfile
        # Should reference the entrypoint script
        assert "entrypoint.sh" in dockerfile or "holodeck" in dockerfile

    def test_dockerfile_agent_yaml_copy(self) -> None:
        """Test that agent.yaml is copied."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        assert "COPY agent.yaml" in dockerfile or "COPY ./agent.yaml" in dockerfile

    def test_dockerfile_environment_variables(self) -> None:
        """Test environment variables are set."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            environment={"DEBUG": "true", "LOG_LEVEL": "info"},
        )

        # ENV instructions should be present
        assert "ENV" in dockerfile
        assert "DEBUG" in dockerfile
        assert "LOG_LEVEL" in dockerfile

    def test_dockerfile_version_label(self) -> None:
        """Test version label with custom version."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            version="1.2.3",
        )

        assert 'org.opencontainers.image.version="1.2.3"' in dockerfile

    def test_dockerfile_created_timestamp_format(self) -> None:
        """Test created timestamp is in ISO 8601 format."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        # Find the created label value
        match = re.search(r'org\.opencontainers\.image\.created="([^"]+)"', dockerfile)
        assert match is not None, "Created label not found"

        timestamp = match.group(1)
        # Should be valid ISO 8601 format
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp}")

    def test_dockerfile_source_label(self) -> None:
        """Test source label with custom source URL."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            source_url="https://github.com/my-org/my-agent",
        )

        assert (
            'org.opencontainers.image.source="https://github.com/my-org/my-agent"'
            in dockerfile
        )

    def test_dockerfile_workdir_set(self) -> None:
        """Test WORKDIR is set to /app."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        assert "WORKDIR /app" in dockerfile

    def test_dockerfile_user_set(self) -> None:
        """Test non-root USER is set."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        # Should have USER instruction with non-root user
        assert "USER" in dockerfile
        # Common non-root users
        assert (
            "holodeck" in dockerfile or "appuser" in dockerfile or "1000" in dockerfile
        )

    def test_dockerfile_no_instruction_files(self) -> None:
        """Test Dockerfile generation without instruction files."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            instruction_files=[],
        )

        # Should still generate valid Dockerfile
        assert "FROM" in dockerfile
        assert "ENTRYPOINT" in dockerfile

    def test_dockerfile_no_data_directories(self) -> None:
        """Test Dockerfile generation without data directories."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            data_directories=[],
        )

        # Should still generate valid Dockerfile
        assert "FROM" in dockerfile
        assert "ENTRYPOINT" in dockerfile

    def test_dockerfile_entrypoint_copy(self) -> None:
        """Test that entrypoint.sh is copied."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        # Should copy entrypoint script
        assert "COPY" in dockerfile
        assert "entrypoint.sh" in dockerfile


class TestDockerfileValidation:
    """Tests for Dockerfile validation utilities."""

    def test_generated_dockerfile_is_valid_syntax(self) -> None:
        """Test that generated Dockerfile has valid syntax structure."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
        )

        # Check for required Dockerfile instructions in correct order
        lines = dockerfile.strip().split("\n")
        instructions = [
            line.split()[0]
            for line in lines
            if line.strip() and not line.startswith("#")
        ]

        # FROM must be first instruction
        assert instructions[0] == "FROM"

        # ENTRYPOINT or CMD must be present
        assert "ENTRYPOINT" in instructions or "CMD" in instructions

    def test_no_secrets_in_dockerfile(self) -> None:
        """Test that no secrets are embedded in Dockerfile."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test-agent",
            port=8080,
            protocol="rest",
            environment={"API_KEY": "${API_KEY}", "SECRET": "${SECRET}"},
        )

        # Should use variable references, not actual values
        assert "${API_KEY}" in dockerfile or "API_KEY" not in dockerfile
        # Should never contain actual secret values
        assert "sk-" not in dockerfile
        assert "password" not in dockerfile.lower() or "${" in dockerfile


class TestClaudeAgentFixture:
    """Tests for the Claude agent test fixture."""

    def test_claude_agent_fixture_loads(self) -> None:
        """Verify Claude agent fixture loads with anthropic provider."""
        from holodeck.config.loader import ConfigLoader
        from holodeck.models.llm import ProviderEnum

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(
            str(
                Path(__file__).parent.parent.parent
                / "fixtures"
                / "claude_agent"
                / "agent.yaml"
            )
        )
        assert agent.model.provider == ProviderEnum.ANTHROPIC
        assert agent.claude.max_concurrent_sessions == 5


class TestDockerfileNodejs:
    """Tests for Node.js conditional block in Dockerfile generation."""

    def test_generate_dockerfile_without_nodejs(self) -> None:
        """Test Dockerfile without Node.js has no nodejs references."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test",
            port=8080,
            protocol="rest",
            needs_nodejs=False,
        )

        assert "nodejs" not in dockerfile.lower()
        assert "nodesource" not in dockerfile.lower()

    def test_generate_dockerfile_with_nodejs(self) -> None:
        """Test Dockerfile with Node.js includes nodesource setup."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test",
            port=8080,
            protocol="rest",
            needs_nodejs=True,
        )

        assert "nodesource.com/setup_22.x" in dockerfile
        assert "apt-get install -y --no-install-recommends nodejs" in dockerfile

    def test_generate_dockerfile_nodejs_cleanup(self) -> None:
        """Test Node.js install includes apt cache cleanup."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test",
            port=8080,
            protocol="rest",
            needs_nodejs=True,
        )

        assert "rm -rf /var/lib/apt/lists/*" in dockerfile

    def test_generate_dockerfile_nodejs_before_user_switch(self) -> None:
        """Test Node.js install appears before USER holodeck switch."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test",
            port=8080,
            protocol="rest",
            needs_nodejs=True,
        )

        lines = dockerfile.split("\n")
        nodejs_line = next(
            i for i, line in enumerate(lines) if "nodesource" in line.lower()
        )
        user_line = next(
            i for i, line in enumerate(lines) if line.strip() == "USER holodeck"
        )
        assert nodejs_line < user_line

    def test_generate_dockerfile_default_needs_nodejs_false(self) -> None:
        """Test that needs_nodejs defaults to False (no Node.js block)."""
        from holodeck.deploy.dockerfile import generate_dockerfile

        dockerfile = generate_dockerfile(
            agent_name="test",
            port=8080,
            protocol="rest",
        )

        assert "nodejs" not in dockerfile.lower()
        assert "nodesource" not in dockerfile.lower()


class TestProviderDetection:
    """Tests for provider-based needs_nodejs detection in deploy command."""

    @pytest.fixture
    def claude_agent_path(self) -> Path:
        """Return path to Claude agent fixture."""
        return (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "claude_agent"
            / "agent.yaml"
        )

    @pytest.fixture
    def openai_agent_path(self) -> Path:
        """Return path to OpenAI agent fixture."""
        return (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "deploy"
            / "sample_agent"
            / "agent.yaml"
        )

    def test_generate_dockerfile_content_detects_anthropic_provider(
        self, claude_agent_path: Path
    ) -> None:
        """Test that Anthropic provider triggers Node.js in Dockerfile."""
        from holodeck.cli.commands.deploy import _generate_dockerfile_content
        from holodeck.config.loader import ConfigLoader

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(claude_agent_path))
        content = _generate_dockerfile_content(agent, agent.deployment, "test")

        assert "nodesource" in content.lower()
        assert "nodejs" in content.lower()

    def test_generate_dockerfile_content_skips_nodejs_for_openai(
        self, openai_agent_path: Path
    ) -> None:
        """Test that OpenAI provider does not include Node.js."""
        from holodeck.cli.commands.deploy import _generate_dockerfile_content
        from holodeck.config.loader import ConfigLoader

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(openai_agent_path))
        content = _generate_dockerfile_content(agent, agent.deployment, "test")

        assert "nodesource" not in content.lower()
        assert "nodejs" not in content.lower()

    def test_generate_dockerfile_content_skips_nodejs_for_ollama(
        self, tmp_path: Path
    ) -> None:
        """Test that Ollama provider does not include Node.js."""
        from holodeck.cli.commands.deploy import _generate_dockerfile_content
        from holodeck.config.loader import ConfigLoader

        # Create minimal ollama agent config
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: ollama-test\n"
            "model:\n"
            "  provider: ollama\n"
            "  name: llama3\n"
            "instructions:\n"
            '  inline: "Test"\n'
            "deployment:\n"
            "  registry:\n"
            "    url: ghcr.io\n"
            "    repository: test/ollama\n"
            "    tag_strategy: latest\n"
            "  target:\n"
            "    provider: aws\n"
            "    aws:\n"
            "      region: us-east-1\n"
            "      cpu: 1\n"
            "      memory: 2048\n"
        )
        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(agent_yaml))
        content = _generate_dockerfile_content(agent, agent.deployment, "test")

        assert "nodesource" not in content.lower()
        assert "nodejs" not in content.lower()

    def test_dry_run_shows_nodejs_for_claude_agent(
        self, claude_agent_path: Path
    ) -> None:
        """Test dry-run output includes all Claude-specific Dockerfile additions."""
        from holodeck.cli.commands.deploy import _generate_dockerfile_content
        from holodeck.config.loader import ConfigLoader

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(claude_agent_path))
        content = _generate_dockerfile_content(agent, agent.deployment, "test")

        # Node.js installation
        assert "nodesource.com/setup_22.x" in content
        assert "--no-install-recommends" in content
        assert "rm -rf /var/lib/apt/lists/*" in content
        # Standard Dockerfile elements still present
        assert "USER holodeck" in content
        assert "HEALTHCHECK" in content
        assert "ENTRYPOINT" in content

    def test_dry_run_skips_nodejs_for_non_claude_agent(
        self, openai_agent_path: Path
    ) -> None:
        """Test dry-run output for non-Claude agent has no Node.js additions."""
        from holodeck.cli.commands.deploy import _generate_dockerfile_content
        from holodeck.config.loader import ConfigLoader

        loader = ConfigLoader()
        agent = loader.load_agent_yaml(str(openai_agent_path))
        content = _generate_dockerfile_content(agent, agent.deployment, "test")

        assert "nodesource" not in content
        assert "--no-install-recommends" not in content
        # Standard elements still present
        assert "USER holodeck" in content
        assert "HEALTHCHECK" in content
