"""Unit tests for the 'holodeck mcp add' command (T021d-g).

Tests cover basic functionality, options handling, duplicate detection,
error handling, and environment variable display.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.mcp import add
from holodeck.lib.errors import (
    DuplicateServerError,
    RegistryAPIError,
    RegistryConnectionError,
    ServerNotFoundError,
)
from holodeck.models.registry import (
    EnvVarConfig,
    RegistryServer,
    RegistryServerPackage,
    TransportConfig,
)
from holodeck.models.tool import CommandType, MCPTool, TransportType

# --- Fixtures ---


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_registry_server() -> RegistryServer:
    """Create a sample RegistryServer for testing."""
    return RegistryServer(
        name="io.github.modelcontextprotocol/server-filesystem",
        description="Read and explore files on filesystem",
        version="1.0.0",
        packages=[
            RegistryServerPackage(
                registry_type="npm",
                identifier="@modelcontextprotocol/server-filesystem",
                version="1.0.0",
                transport=TransportConfig(type="stdio"),
                environment_variables=[
                    EnvVarConfig(
                        name="FS_ROOT",
                        description="Root directory for filesystem access",
                        required=True,
                    )
                ],
            )
        ],
    )


@pytest.fixture
def sample_mcp_tool() -> MCPTool:
    """Create a sample MCPTool for testing."""
    return MCPTool(
        name="server_filesystem",
        description="Read and explore files on filesystem",
        type="mcp",
        transport=TransportType.STDIO,
        command=CommandType.NPX,
        args=["-y", "@modelcontextprotocol/server-filesystem@1.0.0"],
        registry_name="io.github.modelcontextprotocol/server-filesystem",
    )


@pytest.fixture
def mock_registry_client(sample_registry_server: RegistryServer):
    """Mock MCPRegistryClient for testing with context manager support."""
    with patch("holodeck.cli.commands.mcp.MCPRegistryClient") as mock_class:
        mock_instance = MagicMock()
        mock_instance.get_server.return_value = sample_registry_server
        # Configure as context manager
        mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock_class.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_class, mock_instance


@pytest.fixture
def mock_add_to_agent():
    """Mock add_mcp_server_to_agent for testing."""
    with patch("holodeck.cli.commands.mcp.add_mcp_server_to_agent") as mock:
        yield mock


@pytest.fixture
def mock_add_to_global():
    """Mock add_mcp_server_to_global for testing."""
    with patch("holodeck.cli.commands.mcp.add_mcp_server_to_global") as mock:
        yield mock


# --- Test Classes ---


@pytest.mark.unit
class TestAddBasicFunctionality:
    """Tests for basic add command functionality (T021d)."""

    def test_add_server_success(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        mock_add_to_agent: MagicMock,
    ) -> None:
        """Test successful add of an MCP server."""
        _, mock_instance = mock_registry_client

        result = cli_runner.invoke(
            add,
            ["io.github.modelcontextprotocol/server-filesystem"],
        )

        assert result.exit_code == 0
        assert "Added" in result.output
        mock_instance.get_server.assert_called_once_with(
            "io.github.modelcontextprotocol/server-filesystem", "latest"
        )
        mock_add_to_agent.assert_called_once()

    def test_add_displays_env_vars(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        mock_add_to_agent: MagicMock,
    ) -> None:
        """Test that environment variables are displayed after add."""
        result = cli_runner.invoke(
            add,
            ["io.github.modelcontextprotocol/server-filesystem"],
        )

        assert result.exit_code == 0
        assert "Required environment variables:" in result.output
        assert "FS_ROOT" in result.output

    def test_add_success_message_includes_target(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        mock_add_to_agent: MagicMock,
    ) -> None:
        """Test that success message includes target file."""
        result = cli_runner.invoke(
            add,
            ["io.github.modelcontextprotocol/server-filesystem"],
        )

        assert result.exit_code == 0
        assert "agent.yaml" in result.output


@pytest.mark.unit
class TestAddOptions:
    """Tests for add command options (T021e)."""

    def test_agent_option(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        mock_add_to_agent: MagicMock,
    ) -> None:
        """Test --agent option for custom agent file."""
        result = cli_runner.invoke(
            add,
            [
                "io.github.modelcontextprotocol/server-filesystem",
                "--agent",
                "custom-agent.yaml",
            ],
        )

        assert result.exit_code == 0
        # Verify the path passed to add_mcp_server_to_agent
        call_args = mock_add_to_agent.call_args
        assert str(call_args[0][0]) == "custom-agent.yaml"

    def test_global_option(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        mock_add_to_global: MagicMock,
    ) -> None:
        """Test -g/--global option for global config."""
        result = cli_runner.invoke(
            add,
            [
                "io.github.modelcontextprotocol/server-filesystem",
                "-g",
            ],
        )

        assert result.exit_code == 0
        assert "~/.holodeck/config.yaml" in result.output
        mock_add_to_global.assert_called_once()

    def test_version_option(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        mock_add_to_agent: MagicMock,
    ) -> None:
        """Test --version option for specific version."""
        _, mock_instance = mock_registry_client

        result = cli_runner.invoke(
            add,
            [
                "io.github.modelcontextprotocol/server-filesystem",
                "--version",
                "1.2.0",
            ],
        )

        assert result.exit_code == 0
        mock_instance.get_server.assert_called_once_with(
            "io.github.modelcontextprotocol/server-filesystem", "1.2.0"
        )

    def test_name_option(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        mock_add_to_agent: MagicMock,
    ) -> None:
        """Test --name option for custom server name."""
        result = cli_runner.invoke(
            add,
            [
                "io.github.modelcontextprotocol/server-filesystem",
                "--name",
                "my_fs",
            ],
        )

        assert result.exit_code == 0
        # Verify custom name was applied
        call_args = mock_add_to_agent.call_args
        mcp_tool = call_args[0][1]
        assert mcp_tool.name == "my_fs"


@pytest.mark.unit
class TestAddDuplicateDetection:
    """Tests for duplicate detection in add command (T021f)."""

    def test_duplicate_server_error(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
    ) -> None:
        """Test error handling for duplicate server."""
        with patch("holodeck.cli.commands.mcp.add_mcp_server_to_agent") as mock_add:
            mock_add.side_effect = DuplicateServerError(
                "filesystem",
                "io.github.modelcontextprotocol/server-filesystem",
                "io.github.modelcontextprotocol/server-filesystem",
            )

            result = cli_runner.invoke(
                add,
                ["io.github.modelcontextprotocol/server-filesystem"],
            )

            assert result.exit_code == 1
            assert "already configured" in result.output


@pytest.mark.unit
class TestAddErrorHandling:
    """Tests for error handling in add command (T021g)."""

    def test_server_not_found_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error handling when server not found in registry."""
        with patch("holodeck.cli.commands.mcp.MCPRegistryClient") as mock_class:
            mock_instance = MagicMock()
            mock_instance.get_server.side_effect = ServerNotFoundError(
                "nonexistent/server"
            )
            mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_class.return_value.__exit__ = MagicMock(return_value=False)

            result = cli_runner.invoke(
                add,
                ["nonexistent/server"],
            )

            assert result.exit_code == 1
            assert "not found" in result.output

    def test_registry_connection_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error handling for registry connection issues."""
        with patch("holodeck.cli.commands.mcp.MCPRegistryClient") as mock_class:
            mock_instance = MagicMock()
            mock_instance.get_server.side_effect = RegistryConnectionError(
                "https://registry.modelcontextprotocol.io"
            )
            mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_class.return_value.__exit__ = MagicMock(return_value=False)

            result = cli_runner.invoke(
                add,
                ["io.github.example/server"],
            )

            assert result.exit_code == 1
            assert "unavailable" in result.output

    def test_registry_api_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error handling for registry API errors."""
        with patch("holodeck.cli.commands.mcp.MCPRegistryClient") as mock_class:
            mock_instance = MagicMock()
            mock_instance.get_server.side_effect = RegistryAPIError(
                "https://registry.modelcontextprotocol.io/v0.1/servers/test",
                500,
                "Internal Server Error",
            )
            mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_class.return_value.__exit__ = MagicMock(return_value=False)

            result = cli_runner.invoke(
                add,
                ["io.github.example/server"],
            )

            assert result.exit_code == 1
            assert "error" in result.output.lower()

    def test_no_stdio_transport_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test error when server doesn't support stdio transport."""
        with patch("holodeck.cli.commands.mcp.MCPRegistryClient") as mock_class:
            mock_instance = MagicMock()
            # Server with only SSE transport
            mock_instance.get_server.return_value = RegistryServer(
                name="io.github.example/sse-server",
                description="SSE only server",
                version="1.0.0",
                packages=[
                    RegistryServerPackage(
                        registry_type="npm",
                        identifier="@example/sse-server",
                        transport=TransportConfig(
                            type="sse", url="https://example.com"
                        ),
                    )
                ],
            )
            mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_class.return_value.__exit__ = MagicMock(return_value=False)

            result = cli_runner.invoke(
                add,
                ["io.github.example/sse-server"],
            )

            assert result.exit_code == 1
            assert "stdio" in result.output.lower()


@pytest.mark.unit
class TestFindStdioPackage:
    """Tests for find_stdio_package helper function."""

    def test_finds_stdio_package(self) -> None:
        """Test finding a stdio package."""
        from holodeck.services.mcp_registry import find_stdio_package

        server = RegistryServer(
            name="test/server",
            description="Test",
            version="1.0.0",
            packages=[
                RegistryServerPackage(
                    registry_type="npm",
                    identifier="test-package",
                    transport=TransportConfig(type="stdio"),
                )
            ],
        )

        result = find_stdio_package(server)
        assert result is not None
        assert result.transport.type == "stdio"

    def test_returns_none_for_no_stdio(self) -> None:
        """Test returning None when no stdio package available."""
        from holodeck.services.mcp_registry import find_stdio_package

        server = RegistryServer(
            name="test/server",
            description="Test",
            version="1.0.0",
            packages=[
                RegistryServerPackage(
                    registry_type="npm",
                    identifier="test-package",
                    transport=TransportConfig(type="sse", url="https://example.com"),
                )
            ],
        )

        result = find_stdio_package(server)
        assert result is None

    def test_prioritizes_supported_registry_types(self) -> None:
        """Test that supported registry types are prioritized."""
        from holodeck.services.mcp_registry import find_stdio_package

        server = RegistryServer(
            name="test/server",
            description="Test",
            version="1.0.0",
            packages=[
                # Unsupported type first
                RegistryServerPackage(
                    registry_type="nuget",
                    identifier="nuget-package",
                    transport=TransportConfig(type="stdio"),
                ),
                # Supported type second
                RegistryServerPackage(
                    registry_type="npm",
                    identifier="npm-package",
                    transport=TransportConfig(type="stdio"),
                ),
            ],
        )

        result = find_stdio_package(server)
        assert result is not None
        assert result.registry_type == "npm"
