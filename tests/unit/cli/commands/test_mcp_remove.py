"""Unit tests for the 'holodeck mcp remove' command (T032c-e).

Tests cover basic functionality, options handling, and error handling.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from holodeck.cli.commands.mcp import remove
from holodeck.lib.errors import ConfigError, ServerNotFoundError
from holodeck.lib.errors import FileNotFoundError as HoloDeckFileNotFoundError
from holodeck.models.tool import CommandType, MCPTool, TransportType

# --- Fixtures ---


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_mcp_tool() -> MCPTool:
    """Create a sample MCPTool for testing."""
    return MCPTool(
        name="filesystem",
        description="Read and explore files",
        type="mcp",
        transport=TransportType.STDIO,
        command=CommandType.NPX,
        args=["-y", "@modelcontextprotocol/server-filesystem@1.0.0"],
        registry_name="io.github.modelcontextprotocol/server-filesystem",
    )


@pytest.fixture
def mock_remove_from_agent() -> Any:
    """Mock remove_mcp_server_from_agent for testing."""
    with patch("holodeck.cli.commands.mcp.remove_mcp_server_from_agent") as mock:
        yield mock


@pytest.fixture
def mock_remove_from_global() -> Any:
    """Mock remove_mcp_server_from_global for testing."""
    with patch("holodeck.cli.commands.mcp.remove_mcp_server_from_global") as mock:
        yield mock


# --- Test Classes ---


@pytest.mark.unit
class TestRemoveBasicFunctionality:
    """Tests for basic remove command functionality (T032c)."""

    def test_remove_from_agent_success(
        self,
        cli_runner: CliRunner,
        mock_remove_from_agent: MagicMock,
    ) -> None:
        """Test successful removal from agent config."""
        result = cli_runner.invoke(remove, ["filesystem"])

        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "filesystem" in result.output
        assert "agent.yaml" in result.output
        mock_remove_from_agent.assert_called_once()

    def test_remove_success_message_format(
        self,
        cli_runner: CliRunner,
        mock_remove_from_agent: MagicMock,
    ) -> None:
        """Test that success message follows expected format."""
        result = cli_runner.invoke(remove, ["my-server"])

        assert result.exit_code == 0
        assert "Removed 'my-server' from agent.yaml" in result.output

    def test_remove_from_global_success(
        self,
        cli_runner: CliRunner,
        mock_remove_from_global: MagicMock,
    ) -> None:
        """Test successful removal from global config."""
        result = cli_runner.invoke(remove, ["filesystem", "-g"])

        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "filesystem" in result.output
        assert "~/.holodeck/config.yaml" in result.output
        mock_remove_from_global.assert_called_once_with("filesystem")


@pytest.mark.unit
class TestRemoveOptions:
    """Tests for remove command options (T032d)."""

    def test_agent_option(
        self,
        cli_runner: CliRunner,
        mock_remove_from_agent: MagicMock,
    ) -> None:
        """Test --agent option for custom agent file."""
        result = cli_runner.invoke(
            remove,
            ["filesystem", "--agent", "custom-agent.yaml"],
        )

        assert result.exit_code == 0
        call_args = mock_remove_from_agent.call_args
        assert str(call_args[0][0]) == "custom-agent.yaml"
        assert call_args[0][1] == "filesystem"

    def test_global_short_option(
        self,
        cli_runner: CliRunner,
        mock_remove_from_global: MagicMock,
    ) -> None:
        """Test -g short option for global config."""
        result = cli_runner.invoke(remove, ["filesystem", "-g"])

        assert result.exit_code == 0
        assert "~/.holodeck/config.yaml" in result.output
        mock_remove_from_global.assert_called_once_with("filesystem")

    def test_global_long_option(
        self,
        cli_runner: CliRunner,
        mock_remove_from_global: MagicMock,
    ) -> None:
        """Test --global long option."""
        result = cli_runner.invoke(remove, ["filesystem", "--global"])

        assert result.exit_code == 0
        mock_remove_from_global.assert_called_once_with("filesystem")

    def test_default_agent_file(
        self,
        cli_runner: CliRunner,
        mock_remove_from_agent: MagicMock,
    ) -> None:
        """Test that default agent file is agent.yaml."""
        result = cli_runner.invoke(remove, ["filesystem"])

        assert result.exit_code == 0
        call_args = mock_remove_from_agent.call_args
        assert str(call_args[0][0]) == "agent.yaml"


@pytest.mark.unit
class TestRemoveErrorHandling:
    """Tests for error handling in remove command (T032e)."""

    def test_server_not_found_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error when server not found in config."""
        with patch("holodeck.cli.commands.mcp.remove_mcp_server_from_agent") as mock:
            mock.side_effect = ServerNotFoundError("nonexistent", "agent.yaml")

            result = cli_runner.invoke(remove, ["nonexistent"])

            assert result.exit_code == 1
            assert "Error" in result.output
            assert "nonexistent" in result.output

    def test_file_not_found_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error when agent.yaml doesn't exist."""
        with patch("holodeck.cli.commands.mcp.remove_mcp_server_from_agent") as mock:
            mock.side_effect = HoloDeckFileNotFoundError(
                "agent.yaml",
                "Agent file not found: agent.yaml",
            )

            result = cli_runner.invoke(remove, ["filesystem"])

            assert result.exit_code == 1
            assert "Error" in result.output

    def test_config_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error when config write fails."""
        with patch("holodeck.cli.commands.mcp.remove_mcp_server_from_agent") as mock:
            mock.side_effect = ConfigError(
                "agent_config_write",
                "Failed to write agent configuration",
            )

            result = cli_runner.invoke(remove, ["filesystem"])

            assert result.exit_code == 1
            assert "Error" in result.output

    def test_missing_server_argument(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error when SERVER argument is missing."""
        result = cli_runner.invoke(remove, [])

        assert result.exit_code != 0
        # Click should complain about missing argument
        assert "SERVER" in result.output or "Missing argument" in result.output

    def test_global_server_not_found_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error when server not found in global config."""
        with patch("holodeck.cli.commands.mcp.remove_mcp_server_from_global") as mock:
            mock.side_effect = ServerNotFoundError(
                "nonexistent", "global configuration"
            )

            result = cli_runner.invoke(remove, ["nonexistent", "-g"])

            assert result.exit_code == 1
            assert "Error" in result.output
            assert "nonexistent" in result.output


@pytest.mark.unit
class TestRemoveIntegration:
    """Integration tests for remove command with real file operations."""

    def test_remove_from_agent_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
    ) -> None:
        """Test actual removal from agent.yaml file."""
        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(
                remove, ["filesystem", "--agent", str(agent_path)]
            )

        assert result.exit_code == 0
        updated_config = yaml.safe_load(agent_path.read_text())
        assert len(updated_config["tools"]) == 0

    def test_remove_from_global_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
        monkeypatch: Any,
    ) -> None:
        """Test actual removal from global config file."""
        monkeypatch.setenv("HOME", str(tmp_path))
        holodeck_dir = tmp_path / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        config = {
            "mcp_servers": [sample_mcp_tool.model_dump(mode="json")],
        }
        config_path.write_text(yaml.dump(config))

        result = cli_runner.invoke(remove, ["filesystem", "-g"])

        assert result.exit_code == 0
        updated_config = yaml.safe_load(config_path.read_text())
        assert len(updated_config.get("mcp_servers", [])) == 0
