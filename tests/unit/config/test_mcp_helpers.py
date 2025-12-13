"""Unit tests for MCP server helper functions (T021a-c).

Tests for add_mcp_server_to_agent(), add_mcp_server_to_global(),
save_global_config(), and duplicate detection logic.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from holodeck.config.loader import (
    _check_mcp_duplicate,
    add_mcp_server_to_agent,
    add_mcp_server_to_global,
    get_mcp_servers_from_agent,
    get_mcp_servers_from_global,
    remove_mcp_server_from_agent,
    remove_mcp_server_from_global,
    save_global_config,
)
from holodeck.lib.errors import DuplicateServerError, ServerNotFoundError
from holodeck.lib.errors import FileNotFoundError as HoloDeckFileNotFoundError
from holodeck.models.config import GlobalConfig
from holodeck.models.tool import CommandType, MCPTool, TransportType


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
def sample_agent_yaml() -> dict[str, Any]:
    """Create a sample agent configuration."""
    return {
        "name": "test-agent",
        "model": {"provider": "openai", "name": "gpt-4o"},
        "instructions": {"inline": "You are a helpful assistant."},
        "tools": [],
    }


class TestSaveGlobalConfig:
    """Tests for save_global_config() function."""

    def test_creates_directory_if_not_exists(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that save_global_config creates ~/.holodeck/ if needed."""
        monkeypatch.setenv("HOME", str(temp_dir))
        config_path = temp_dir / ".holodeck" / "config.yaml"

        config = GlobalConfig(
            providers=None,
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=None,
        )
        result_path = save_global_config(config, config_path)

        assert result_path == config_path
        assert config_path.exists()
        assert config_path.parent.exists()

    def test_saves_valid_yaml(self, temp_dir: Path) -> None:
        """Test that saved config is valid YAML."""
        config_path = temp_dir / "config.yaml"
        config = GlobalConfig(
            providers=None,
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=[
                MCPTool(
                    name="test",
                    description="Test server",
                    type="mcp",
                    transport=TransportType.STDIO,
                    command=CommandType.NPX,
                    args=["-y", "test-package"],
                )
            ],
        )

        save_global_config(config, config_path)

        # Verify YAML is valid and parseable
        content = yaml.safe_load(config_path.read_text())
        assert "mcp_servers" in content
        assert len(content["mcp_servers"]) == 1
        assert content["mcp_servers"][0]["name"] == "test"

    def test_preserves_existing_fields(self, temp_dir: Path) -> None:
        """Test that existing config fields are preserved."""
        config_path = temp_dir / "config.yaml"
        config = GlobalConfig(
            providers={"openai": {"provider": "openai", "name": "gpt-4o"}},
            vectorstores=None,
            execution=None,
            deployment=None,
            mcp_servers=[
                MCPTool(
                    name="test",
                    description="Test server",
                    type="mcp",
                    transport=TransportType.STDIO,
                    command=CommandType.NPX,
                    args=["-y", "test-package"],
                )
            ],
        )

        save_global_config(config, config_path)

        content = yaml.safe_load(config_path.read_text())
        assert "providers" in content
        assert "mcp_servers" in content


class TestCheckMcpDuplicate:
    """Tests for _check_mcp_duplicate() helper function."""

    def test_detects_exact_registry_duplicate(self, sample_mcp_tool: MCPTool) -> None:
        """Test that exact registry_name match is detected as duplicate."""
        existing_tools = [
            {
                "name": "filesystem",
                "type": "mcp",
                "registry_name": "io.github.modelcontextprotocol/server-filesystem",
            }
        ]

        with pytest.raises(DuplicateServerError) as exc_info:
            _check_mcp_duplicate(existing_tools, sample_mcp_tool)

        assert "already configured" in str(exc_info.value)

    def test_detects_name_conflict(self, sample_mcp_tool: MCPTool) -> None:
        """Test that same name with different registry is detected."""
        existing_tools = [
            {
                "name": "filesystem",
                "type": "mcp",
                "registry_name": "io.github.other/filesystem",
            }
        ]

        with pytest.raises(DuplicateServerError) as exc_info:
            _check_mcp_duplicate(existing_tools, sample_mcp_tool)

        assert "already exists" in str(exc_info.value)
        assert "--name" in str(exc_info.value)

    def test_allows_different_names(self, sample_mcp_tool: MCPTool) -> None:
        """Test that different names are allowed."""
        existing_tools = [
            {
                "name": "github",
                "type": "mcp",
                "registry_name": "io.github.other/github",
            }
        ]

        # Should not raise
        _check_mcp_duplicate(existing_tools, sample_mcp_tool)

    def test_ignores_non_mcp_tools(self, sample_mcp_tool: MCPTool) -> None:
        """Test that non-MCP tools are ignored in duplicate check."""
        existing_tools = [
            {
                "name": "filesystem",
                "type": "vectorstore",
            }
        ]

        # Should not raise even though name matches
        _check_mcp_duplicate(existing_tools, sample_mcp_tool)


class TestAddMcpServerToAgent:
    """Tests for add_mcp_server_to_agent() function."""

    def test_creates_tools_list_if_missing(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that tools list is created if missing from agent config."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test instructions"},
        }
        agent_path.write_text(yaml.dump(agent_config))

        add_mcp_server_to_agent(agent_path, sample_mcp_tool)

        updated_config = yaml.safe_load(agent_path.read_text())
        assert "tools" in updated_config
        assert len(updated_config["tools"]) == 1
        assert updated_config["tools"][0]["name"] == "filesystem"

    def test_appends_to_existing_tools(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that new tool is appended to existing tools list."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai", "name": "gpt-4o"},
            "instructions": {"inline": "Test instructions"},
            "tools": [
                {"name": "existing", "type": "function", "function": "test_func"}
            ],
        }
        agent_path.write_text(yaml.dump(agent_config))

        add_mcp_server_to_agent(agent_path, sample_mcp_tool)

        updated_config = yaml.safe_load(agent_path.read_text())
        assert len(updated_config["tools"]) == 2
        assert updated_config["tools"][1]["name"] == "filesystem"

    def test_raises_on_missing_file(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that FileNotFoundError is raised for missing agent.yaml."""
        from holodeck.lib.errors import FileNotFoundError

        agent_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            add_mcp_server_to_agent(agent_path, sample_mcp_tool)

        assert "agent.yaml" in str(exc_info.value) or "No agent" in str(exc_info.value)

    def test_raises_on_duplicate(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that DuplicateServerError is raised for duplicate servers."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [
                {
                    "name": "filesystem",
                    "type": "mcp",
                    "registry_name": "io.github.modelcontextprotocol/server-filesystem",
                }
            ],
        }
        agent_path.write_text(yaml.dump(agent_config))

        with pytest.raises(DuplicateServerError):
            add_mcp_server_to_agent(agent_path, sample_mcp_tool)


class TestAddMcpServerToGlobal:
    """Tests for add_mcp_server_to_global() function."""

    def test_creates_config_if_not_exists(
        self, temp_dir: Path, monkeypatch: Any, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that global config is created if it doesn't exist."""
        monkeypatch.setenv("HOME", str(temp_dir))
        config_path = temp_dir / ".holodeck" / "config.yaml"

        result_path = add_mcp_server_to_global(sample_mcp_tool, config_path)

        assert result_path == config_path
        assert config_path.exists()

        content = yaml.safe_load(config_path.read_text())
        assert "mcp_servers" in content
        assert len(content["mcp_servers"]) == 1

    def test_appends_to_existing_servers(
        self, temp_dir: Path, monkeypatch: Any, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that new server is appended to existing mcp_servers."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        # Create initial config with one server
        initial_config = {
            "mcp_servers": [
                {
                    "name": "github",
                    "description": "GitHub API",
                    "type": "mcp",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "github-server"],
                    "registry_name": "io.github.other/github",
                }
            ]
        }
        config_path.write_text(yaml.dump(initial_config))

        add_mcp_server_to_global(sample_mcp_tool, config_path)

        content = yaml.safe_load(config_path.read_text())
        assert len(content["mcp_servers"]) == 2

    def test_raises_on_duplicate(
        self, temp_dir: Path, monkeypatch: Any, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that DuplicateServerError is raised for duplicate servers."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        # Create initial config with the same server
        initial_config = {
            "mcp_servers": [
                {
                    "name": "filesystem",
                    "description": "Filesystem",
                    "type": "mcp",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "fs-server"],
                    "registry_name": "io.github.modelcontextprotocol/server-filesystem",
                }
            ]
        }
        config_path.write_text(yaml.dump(initial_config))

        with pytest.raises(DuplicateServerError):
            add_mcp_server_to_global(sample_mcp_tool, config_path)


class TestGetMcpServersFromAgent:
    """Tests for get_mcp_servers_from_agent() function."""

    def test_returns_mcp_tools_from_agent(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that MCP tools are returned from agent config."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = get_mcp_servers_from_agent(agent_path)

        assert len(result) == 1
        assert result[0].name == "filesystem"
        assert (
            result[0].registry_name
            == "io.github.modelcontextprotocol/server-filesystem"
        )

    def test_filters_non_mcp_tools(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that non-MCP tools are filtered out."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [
                {"name": "vectorstore_tool", "type": "vectorstore"},
                sample_mcp_tool.model_dump(mode="json"),
                {"name": "function_tool", "type": "function", "function": "test_func"},
            ],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = get_mcp_servers_from_agent(agent_path)

        assert len(result) == 1
        assert result[0].name == "filesystem"

    def test_returns_empty_for_no_tools(self, temp_dir: Path) -> None:
        """Test that empty list is returned when no tools configured."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = get_mcp_servers_from_agent(agent_path)

        assert result == []

    def test_returns_empty_for_empty_tools_list(self, temp_dir: Path) -> None:
        """Test that empty list is returned for empty tools list."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [],
        }
        agent_path.write_text(yaml.dump(agent_config))

        result = get_mcp_servers_from_agent(agent_path)

        assert result == []

    def test_raises_on_missing_file(self, temp_dir: Path) -> None:
        """Test that FileNotFoundError is raised for missing agent file."""
        from holodeck.lib.errors import FileNotFoundError

        agent_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            get_mcp_servers_from_agent(agent_path)


class TestGetMcpServersFromGlobal:
    """Tests for get_mcp_servers_from_global() function."""

    def test_returns_mcp_tools_from_global(
        self, temp_dir: Path, monkeypatch: Any, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that MCP tools are returned from global config."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        config = {
            "mcp_servers": [sample_mcp_tool.model_dump(mode="json")],
        }
        config_path.write_text(yaml.dump(config))

        result = get_mcp_servers_from_global()

        assert len(result) == 1
        assert result[0].name == "filesystem"

    def test_returns_empty_when_no_global_config(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that empty list is returned when no global config exists."""
        monkeypatch.setenv("HOME", str(temp_dir))

        result = get_mcp_servers_from_global()

        assert result == []

    def test_returns_empty_when_no_mcp_servers(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that empty list is returned when mcp_servers is null."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        config = {"providers": {"openai": {"provider": "openai", "name": "gpt-4o"}}}
        config_path.write_text(yaml.dump(config))

        result = get_mcp_servers_from_global()

        assert result == []

    def test_returns_multiple_servers(self, temp_dir: Path, monkeypatch: Any) -> None:
        """Test that multiple servers are returned."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        config = {
            "mcp_servers": [
                {
                    "name": "filesystem",
                    "description": "Filesystem server",
                    "type": "mcp",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@mcp/server-filesystem@1.0.0"],
                },
                {
                    "name": "github",
                    "description": "GitHub server",
                    "type": "mcp",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@mcp/server-github@2.0.0"],
                },
            ],
        }
        config_path.write_text(yaml.dump(config))

        result = get_mcp_servers_from_global()

        assert len(result) == 2
        assert result[0].name == "filesystem"
        assert result[1].name == "github"


@pytest.mark.unit
class TestRemoveMcpServerFromAgent:
    """Tests for remove_mcp_server_from_agent() function (T032a)."""

    def test_removes_server_by_name(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test successful removal of MCP server by name."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [
                sample_mcp_tool.model_dump(mode="json"),
            ],
        }
        agent_path.write_text(yaml.dump(agent_config))

        remove_mcp_server_from_agent(agent_path, "filesystem")

        updated_config = yaml.safe_load(agent_path.read_text())
        assert len(updated_config["tools"]) == 0

    def test_preserves_other_tools(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that other tools are preserved after removal."""
        agent_path = temp_dir / "agent.yaml"
        other_tool = MCPTool(
            name="github",
            description="GitHub server",
            type="mcp",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "@mcp/server-github@1.0.0"],
        )
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [
                sample_mcp_tool.model_dump(mode="json"),
                other_tool.model_dump(mode="json"),
                {"name": "vectorstore", "type": "vectorstore", "provider": "local"},
            ],
        }
        agent_path.write_text(yaml.dump(agent_config))

        remove_mcp_server_from_agent(agent_path, "filesystem")

        updated_config = yaml.safe_load(agent_path.read_text())
        assert len(updated_config["tools"]) == 2
        assert updated_config["tools"][0]["name"] == "github"
        assert updated_config["tools"][1]["name"] == "vectorstore"

    def test_raises_on_missing_file(self, temp_dir: Path) -> None:
        """Test HoloDeckFileNotFoundError for missing agent.yaml."""
        agent_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(HoloDeckFileNotFoundError):
            remove_mcp_server_from_agent(agent_path, "filesystem")

    def test_raises_on_server_not_found(
        self, temp_dir: Path, sample_mcp_tool: MCPTool
    ) -> None:
        """Test ServerNotFoundError when server doesn't exist."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [sample_mcp_tool.model_dump(mode="json")],
        }
        agent_path.write_text(yaml.dump(agent_config))

        with pytest.raises(ServerNotFoundError) as exc_info:
            remove_mcp_server_from_agent(agent_path, "nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_raises_on_empty_tools_list(self, temp_dir: Path) -> None:
        """Test ServerNotFoundError when tools list is empty."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
            "tools": [],
        }
        agent_path.write_text(yaml.dump(agent_config))

        with pytest.raises(ServerNotFoundError):
            remove_mcp_server_from_agent(agent_path, "filesystem")

    def test_raises_on_no_tools_key(self, temp_dir: Path) -> None:
        """Test ServerNotFoundError when no tools key exists."""
        agent_path = temp_dir / "agent.yaml"
        agent_config = {
            "name": "test-agent",
            "model": {"provider": "openai"},
            "instructions": {"inline": "Test"},
        }
        agent_path.write_text(yaml.dump(agent_config))

        with pytest.raises(ServerNotFoundError):
            remove_mcp_server_from_agent(agent_path, "filesystem")


@pytest.mark.unit
class TestRemoveMcpServerFromGlobal:
    """Tests for remove_mcp_server_from_global() function (T032b)."""

    def test_removes_server_by_name(
        self, temp_dir: Path, monkeypatch: Any, sample_mcp_tool: MCPTool
    ) -> None:
        """Test successful removal of MCP server from global config."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        config = {
            "mcp_servers": [sample_mcp_tool.model_dump(mode="json")],
        }
        config_path.write_text(yaml.dump(config))

        result_path = remove_mcp_server_from_global("filesystem")

        assert result_path == config_path
        updated_config = yaml.safe_load(config_path.read_text())
        assert len(updated_config.get("mcp_servers", [])) == 0

    def test_preserves_other_servers(
        self, temp_dir: Path, monkeypatch: Any, sample_mcp_tool: MCPTool
    ) -> None:
        """Test that other servers are preserved after removal."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        other_tool = MCPTool(
            name="github",
            description="GitHub server",
            type="mcp",
            transport=TransportType.STDIO,
            command=CommandType.NPX,
            args=["-y", "@mcp/server-github@1.0.0"],
        )
        config = {
            "mcp_servers": [
                sample_mcp_tool.model_dump(mode="json"),
                other_tool.model_dump(mode="json"),
            ],
        }
        config_path.write_text(yaml.dump(config))

        remove_mcp_server_from_global("filesystem")

        updated_config = yaml.safe_load(config_path.read_text())
        assert len(updated_config["mcp_servers"]) == 1
        assert updated_config["mcp_servers"][0]["name"] == "github"

    def test_raises_when_no_global_config(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test ServerNotFoundError when no global config exists."""
        monkeypatch.setenv("HOME", str(temp_dir))
        # Don't create the .holodeck directory

        with pytest.raises(ServerNotFoundError) as exc_info:
            remove_mcp_server_from_global("filesystem")

        assert "global configuration" in str(exc_info.value)

    def test_raises_on_server_not_found(
        self, temp_dir: Path, monkeypatch: Any, sample_mcp_tool: MCPTool
    ) -> None:
        """Test ServerNotFoundError when server doesn't exist."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        config = {
            "mcp_servers": [sample_mcp_tool.model_dump(mode="json")],
        }
        config_path.write_text(yaml.dump(config))

        with pytest.raises(ServerNotFoundError) as exc_info:
            remove_mcp_server_from_global("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_raises_when_mcp_servers_empty(
        self, temp_dir: Path, monkeypatch: Any
    ) -> None:
        """Test ServerNotFoundError when mcp_servers list is empty."""
        monkeypatch.setenv("HOME", str(temp_dir))
        holodeck_dir = temp_dir / ".holodeck"
        holodeck_dir.mkdir()
        config_path = holodeck_dir / "config.yaml"

        config = {"mcp_servers": []}
        config_path.write_text(yaml.dump(config))

        with pytest.raises(ServerNotFoundError):
            remove_mcp_server_from_global("filesystem")

    def test_custom_global_path(self, temp_dir: Path, sample_mcp_tool: MCPTool) -> None:
        """Test removal with custom global path."""
        custom_path = temp_dir / "custom" / "config.yaml"
        custom_path.parent.mkdir(parents=True)

        config = {
            "mcp_servers": [sample_mcp_tool.model_dump(mode="json")],
        }
        custom_path.write_text(yaml.dump(config))

        result_path = remove_mcp_server_from_global("filesystem", custom_path)

        assert result_path == custom_path
        updated_config = yaml.safe_load(custom_path.read_text())
        assert len(updated_config.get("mcp_servers", [])) == 0
