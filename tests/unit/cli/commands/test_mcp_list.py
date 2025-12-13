"""Unit tests for MCP list command (Phase 5 - T027a-f).

Tests for the `holodeck mcp list` command including:
- Default agent.yaml listing
- Global config listing (-g)
- Combined listing (--all)
- JSON output
- Empty state handling
- Table formatting with VERSION column
"""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from holodeck.cli.commands.mcp import _extract_version_from_args, list_cmd
from holodeck.models.tool import CommandType, MCPTool, TransportType


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_mcp_tool() -> MCPTool:
    """Create a sample MCPTool for testing."""
    return MCPTool(
        name="filesystem",
        description="Read and explore files on filesystem",
        type="mcp",
        transport=TransportType.STDIO,
        command=CommandType.NPX,
        args=["-y", "@modelcontextprotocol/server-filesystem@1.0.0"],
        registry_name="io.github.modelcontextprotocol/server-filesystem",
    )


@pytest.fixture
def sample_global_mcp_tool() -> MCPTool:
    """Create a sample MCPTool for global config testing."""
    return MCPTool(
        name="github",
        description="Interact with GitHub repositories",
        type="mcp",
        transport=TransportType.STDIO,
        command=CommandType.NPX,
        args=["-y", "@modelcontextprotocol/server-github@2.0.0"],
        registry_name="io.github.modelcontextprotocol/server-github",
    )


@pytest.mark.unit
class TestExtractVersionFromArgs:
    """Tests for _extract_version_from_args helper function."""

    def test_extracts_version_from_npm_package(self) -> None:
        """Test version extraction from npm package specifier."""
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "@modelcontextprotocol/server-filesystem@1.0.0"],
        )
        assert _extract_version_from_args(tool) == "1.0.0"

    def test_extracts_semver_with_patch(self) -> None:
        """Test version extraction with patch number."""
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "package@2.3.4"],
        )
        assert _extract_version_from_args(tool) == "2.3.4"

    def test_extracts_prerelease_version(self) -> None:
        """Test version extraction with prerelease tag."""
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "package@1.0.0-beta"],
        )
        assert _extract_version_from_args(tool) == "1.0.0-beta"

    def test_returns_dash_for_no_version(self) -> None:
        """Test that '-' is returned when no version found."""
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "package-without-version"],
        )
        assert _extract_version_from_args(tool) == "-"

    def test_returns_dash_for_no_args(self) -> None:
        """Test that '-' is returned when args is None."""
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            transport=TransportType.SSE,  # SSE doesn't require command
            url="http://localhost:8080",
            args=None,
        )
        assert _extract_version_from_args(tool) == "-"

    def test_returns_dash_for_empty_args(self) -> None:
        """Test that '-' is returned when args is empty."""
        tool = MCPTool(
            name="test",
            description="Test",
            type="mcp",
            transport=TransportType.SSE,  # SSE doesn't require command
            url="http://localhost:8080",
            args=[],
        )
        assert _extract_version_from_args(tool) == "-"


@pytest.mark.unit
class TestListCommandDefaultBehavior:
    """Tests for default list command behavior (agent.yaml)."""

    def test_list_shows_agent_servers(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
    ) -> None:
        """Test that list shows servers from agent.yaml."""
        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        with cli_runner.isolated_filesystem(temp_dir=tmp_path):
            result = cli_runner.invoke(list_cmd, ["--agent", str(agent_path)])

        assert result.exit_code == 0
        assert "filesystem" in result.output
        assert "1.0.0" in result.output
        assert "stdio" in result.output

    def test_list_custom_agent_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
    ) -> None:
        """Test that --agent option specifies custom agent file."""
        custom_agent = tmp_path / "custom-agent.yaml"
        agent_config = {
            "name": "custom-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        custom_agent.write_text(yaml.dump(agent_config))

        result = cli_runner.invoke(list_cmd, ["--agent", str(custom_agent)])

        assert result.exit_code == 0
        assert "filesystem" in result.output

    def test_list_shows_empty_message(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that empty state message is shown when no servers."""
        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = cli_runner.invoke(list_cmd, ["--agent", str(agent_path)])

        assert result.exit_code == 0
        assert "No MCP servers configured" in result.output
        assert "holodeck mcp search" in result.output


@pytest.mark.unit
class TestListCommandGlobalFlag:
    """Tests for list command with -g/--global flag."""

    def test_list_global_shows_global_servers(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_global_mcp_tool: MCPTool,
        monkeypatch: Any,
    ) -> None:
        """Test that -g shows servers from global config."""
        monkeypatch.setenv("HOME", str(tmp_path))
        holodeck_dir = tmp_path / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        config = {
            "mcp_servers": [sample_global_mcp_tool.model_dump(mode="json")],
        }
        config_path.write_text(yaml.dump(config))

        result = cli_runner.invoke(list_cmd, ["-g"])

        assert result.exit_code == 0
        assert "github" in result.output
        assert "2.0.0" in result.output

    def test_list_global_empty_state(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        """Test empty state for global config with no servers."""
        monkeypatch.setenv("HOME", str(tmp_path))

        result = cli_runner.invoke(list_cmd, ["-g"])

        assert result.exit_code == 0
        assert "No MCP servers configured" in result.output


@pytest.mark.unit
class TestListCommandAllFlag:
    """Tests for list command with --all flag."""

    def test_list_all_shows_both_sources(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
        sample_global_mcp_tool: MCPTool,
        monkeypatch: Any,
    ) -> None:
        """Test that --all shows servers from both agent and global."""
        # Setup global config
        monkeypatch.setenv("HOME", str(tmp_path))
        holodeck_dir = tmp_path / ".holodeck"
        holodeck_dir.mkdir()
        global_config = holodeck_dir / "config.yaml"
        global_config.write_text(
            yaml.dump({"mcp_servers": [sample_global_mcp_tool.model_dump(mode="json")]})
        )

        # Setup agent config
        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = cli_runner.invoke(list_cmd, ["--all", "--agent", str(agent_path)])

        assert result.exit_code == 0
        assert "filesystem" in result.output
        assert "github" in result.output
        assert "agent" in result.output
        assert "global" in result.output
        assert "SOURCE" in result.output

    def test_list_all_continues_without_agent(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_global_mcp_tool: MCPTool,
        monkeypatch: Any,
    ) -> None:
        """Test that --all continues with global if agent doesn't exist."""
        monkeypatch.setenv("HOME", str(tmp_path))
        holodeck_dir = tmp_path / ".holodeck"
        holodeck_dir.mkdir()
        global_config = holodeck_dir / "config.yaml"
        global_config.write_text(
            yaml.dump({"mcp_servers": [sample_global_mcp_tool.model_dump(mode="json")]})
        )

        # No agent.yaml in temp dir
        result = cli_runner.invoke(list_cmd, ["--all"])

        assert result.exit_code == 0
        assert "github" in result.output


@pytest.mark.unit
class TestListCommandJsonOutput:
    """Tests for list command with --json flag."""

    def test_list_json_output(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
    ) -> None:
        """Test that --json outputs valid JSON."""
        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = cli_runner.invoke(list_cmd, ["--json", "--agent", str(agent_path)])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "servers" in output
        assert "total_count" in output
        assert len(output["servers"]) == 1
        assert output["servers"][0]["name"] == "filesystem"
        assert output["servers"][0]["version"] == "1.0.0"

    def test_list_json_includes_source_with_all(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
        sample_global_mcp_tool: MCPTool,
        monkeypatch: Any,
    ) -> None:
        """Test that --json with --all includes source field."""
        monkeypatch.setenv("HOME", str(tmp_path))
        holodeck_dir = tmp_path / ".holodeck"
        holodeck_dir.mkdir()
        global_config = holodeck_dir / "config.yaml"
        global_config.write_text(
            yaml.dump({"mcp_servers": [sample_global_mcp_tool.model_dump(mode="json")]})
        )

        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = cli_runner.invoke(
            list_cmd, ["--json", "--all", "--agent", str(agent_path)]
        )

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert len(output["servers"]) == 2

        # Check sources are included
        sources = {s["source"] for s in output["servers"]}
        assert "agent" in sources
        assert "global" in sources

    def test_list_json_empty_state(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that --json outputs empty list for no servers."""
        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = cli_runner.invoke(list_cmd, ["--json", "--agent", str(agent_path)])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["servers"] == []
        assert output["total_count"] == 0


@pytest.mark.unit
class TestListCommandTableFormat:
    """Tests for table output formatting."""

    def test_table_has_version_column(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
    ) -> None:
        """Test that table output includes VERSION column."""
        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = cli_runner.invoke(list_cmd, ["--agent", str(agent_path)])

        assert result.exit_code == 0
        assert "VERSION" in result.output
        assert "NAME" in result.output
        assert "TRANSPORT" in result.output
        assert "DESCRIPTION" in result.output

    def test_table_source_column_only_with_all(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_mcp_tool: MCPTool,
    ) -> None:
        """Test that SOURCE column only appears with --all flag."""
        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        # Without --all, no SOURCE column
        result = cli_runner.invoke(list_cmd, ["--agent", str(agent_path)])
        assert "SOURCE" not in result.output

    def test_table_truncates_long_description(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that long descriptions are truncated."""
        long_desc = "A" * 100
        tool = MCPTool(
            name="test",
            description=long_desc,
            type="mcp",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "test@1.0.0"],
        )

        agent_path = tmp_path / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = cli_runner.invoke(list_cmd, ["--agent", str(agent_path)])

        assert result.exit_code == 0
        # Should have truncation (ellipsis)
        assert "..." in result.output


@pytest.mark.unit
class TestListCommandErrorHandling:
    """Tests for error handling in list command."""

    def test_list_error_on_missing_agent_file(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test that error is shown for missing agent file."""
        result = cli_runner.invoke(
            list_cmd, ["--agent", str(tmp_path / "nonexistent.yaml")]
        )

        assert result.exit_code == 1
        assert "Error" in result.output
