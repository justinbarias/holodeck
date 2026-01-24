"""Unit tests for Dockerfile generation.

TDD tests - these should fail until T009 implements the generator.
"""

import re
from datetime import datetime

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
        assert "FROM holodeck-ai/base" in dockerfile or "FROM python:" in dockerfile

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
