"""Unit tests for the 'holodeck mcp search' command.

Tests cover argument parsing, options, output formatting, error handling,
and pagination support for the MCP registry search functionality.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from holodeck.cli.commands.mcp import search
from holodeck.lib.errors import RegistryAPIError, RegistryConnectionError
from holodeck.models.registry import (
    RegistryServer,
    RegistryServerPackage,
    SearchResult,
    TransportConfig,
)

# --- Fixtures ---


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_server() -> RegistryServer:
    """Create a sample RegistryServer for testing."""
    return RegistryServer(
        name="io.github.example/server-test",
        description="A test server for unit testing",
        version="1.0.0",
        packages=[
            RegistryServerPackage(
                registry_type="npm",
                identifier="@example/server-test",
                transport=TransportConfig(type="stdio"),
            )
        ],
    )


@pytest.fixture
def sample_search_result(sample_server: RegistryServer) -> SearchResult:
    """Create a sample SearchResult for testing."""
    return SearchResult(
        servers=[sample_server],
        total_count=1,
        next_cursor=None,
    )


@pytest.fixture
def mock_registry_client():
    """Mock MCPRegistryClient for testing with context manager support."""
    with patch("holodeck.cli.commands.mcp.MCPRegistryClient") as mock_class:
        mock_instance = MagicMock()
        # Configure as context manager
        mock_class.return_value.__enter__ = MagicMock(return_value=mock_instance)
        mock_class.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_class, mock_instance


# --- Test Classes ---


@pytest.mark.unit
class TestSearchArgumentParsing:
    """Tests for argument parsing of the search command (T013a)."""

    def test_search_without_query(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that search works without a query argument."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        mock_instance.search.assert_called_once_with(query=None, limit=25)

    def test_search_with_query(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that search passes query to the client."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, ["filesystem"])

        assert result.exit_code == 0
        mock_instance.search.assert_called_once_with(query="filesystem", limit=25)


@pytest.mark.unit
class TestSearchOptions:
    """Tests for --limit and --json options (T013b)."""

    def test_limit_option_default(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that default limit is 25."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        mock_instance.search.assert_called_once_with(query=None, limit=25)

    def test_limit_option_custom(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that custom limit is passed to client."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, ["--limit", "50"])

        assert result.exit_code == 0
        mock_instance.search.assert_called_once_with(query=None, limit=50)

    def test_limit_option_validation_too_low(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that limit below 1 is rejected."""
        result = cli_runner.invoke(search, ["--limit", "0"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "0" in result.output

    def test_limit_option_validation_too_high(
        self,
        cli_runner: CliRunner,
    ) -> None:
        """Test that limit above 100 is rejected."""
        result = cli_runner.invoke(search, ["--limit", "101"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "101" in result.output

    def test_json_flag_outputs_json(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that --json flag produces valid JSON output."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, ["--json"])

        assert result.exit_code == 0
        # Verify output is valid JSON
        output = json.loads(result.output)
        assert "servers" in output
        assert "total_count" in output
        assert "has_more" in output


@pytest.mark.unit
class TestTableFormatter:
    """Tests for table output formatting (T013c)."""

    def test_table_output_format(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that table output has header row and data."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        # Check for header
        assert "NAME" in result.output
        assert "DESCRIPTION" in result.output
        assert "TRANSPORT" in result.output
        # Check for data
        assert "io.github.example/server-test" in result.output

    def test_table_truncates_long_names(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
    ) -> None:
        """Test that long names are truncated with ellipsis."""
        _, mock_instance = mock_registry_client
        long_name_server = RegistryServer(
            name="io.github.verylongorganizationname/very-long-server-name-exceeds",
            description="A very long description that should be truncated in table",
            packages=[
                RegistryServerPackage(
                    registry_type="npm",
                    identifier="@example/long",
                    transport=TransportConfig(type="stdio"),
                )
            ],
        )
        mock_instance.search.return_value = SearchResult(
            servers=[long_name_server],
            total_count=1,
            next_cursor=None,
        )

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        # Long text should be truncated with ellipsis
        assert "..." in result.output

    def test_table_shows_transports(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
    ) -> None:
        """Test that transport column shows transport types."""
        _, mock_instance = mock_registry_client
        multi_transport_server = RegistryServer(
            name="io.github.example/multi-transport",
            description="Server with multiple transports",
            packages=[
                RegistryServerPackage(
                    registry_type="npm",
                    identifier="@example/multi",
                    transport=TransportConfig(type="stdio"),
                ),
                RegistryServerPackage(
                    registry_type="npm",
                    identifier="@example/multi-sse",
                    transport=TransportConfig(type="sse"),
                ),
            ],
        )
        mock_instance.search.return_value = SearchResult(
            servers=[multi_transport_server],
            total_count=1,
            next_cursor=None,
        )

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        # Both transports should be shown
        assert "sse" in result.output
        assert "stdio" in result.output


@pytest.mark.unit
class TestJSONFormatter:
    """Tests for JSON output formatting (T013d)."""

    def test_json_output_structure(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that JSON output has correct structure."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, ["--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert isinstance(output["servers"], list)
        assert isinstance(output["total_count"], int)
        assert isinstance(output["has_more"], bool)

    def test_json_server_fields(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that each server in JSON has required fields."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, ["--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        server = output["servers"][0]
        assert "name" in server
        assert "description" in server
        assert "transports" in server
        assert isinstance(server["transports"], list)

    def test_json_valid_format(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that output is valid, parseable JSON."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result

        result = cli_runner.invoke(search, ["--json"])

        assert result.exit_code == 0
        # Should not raise JSONDecodeError
        parsed = json.loads(result.output)
        assert parsed is not None


@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling (T013e)."""

    def test_network_timeout_error(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
    ) -> None:
        """Test that network timeout shows appropriate error message."""
        _, mock_instance = mock_registry_client
        mock_instance.search.side_effect = RegistryConnectionError(
            url="https://registry.modelcontextprotocol.io",
            original_error=TimeoutError("Connection timed out"),
        )

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 1
        assert "Registry unavailable" in result.output

    def test_api_error_handling(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
    ) -> None:
        """Test that API errors show appropriate error message."""
        _, mock_instance = mock_registry_client
        mock_instance.search.side_effect = RegistryAPIError(
            url="https://registry.modelcontextprotocol.io",
            status_code=500,
            detail="Internal server error",
        )

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 1
        assert "Registry service error" in result.output

    def test_empty_results_message(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
    ) -> None:
        """Test that empty results show appropriate message."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = SearchResult(
            servers=[],
            total_count=0,
            next_cursor=None,
        )

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        assert "No servers found" in result.output


@pytest.mark.unit
class TestPagination:
    """Tests for pagination support (T013f)."""

    def test_pagination_hint_shown(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_server: RegistryServer,
    ) -> None:
        """Test that pagination hint is shown when more results available."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = SearchResult(
            servers=[sample_server],
            total_count=100,
            next_cursor="abc123",
        )

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        assert "100 total results" in result.output
        assert "More available" in result.output

    def test_pagination_hint_hidden(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_search_result: SearchResult,
    ) -> None:
        """Test that pagination hint is hidden when no more results."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = sample_search_result  # next_cursor=None

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        assert "More available" not in result.output

    def test_pagination_hint_not_in_json(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_server: RegistryServer,
    ) -> None:
        """Test that pagination hint is not echoed in JSON output mode."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = SearchResult(
            servers=[sample_server],
            total_count=100,
            next_cursor="abc123",
        )

        result = cli_runner.invoke(search, ["--json"])

        assert result.exit_code == 0
        # The output should be valid JSON (pagination hint would break it)
        output = json.loads(result.output)
        # has_more in JSON indicates pagination
        assert output["has_more"] is True

    def test_total_count_in_json(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
        sample_server: RegistryServer,
    ) -> None:
        """Test that total_count is included in JSON output."""
        _, mock_instance = mock_registry_client
        mock_instance.search.return_value = SearchResult(
            servers=[sample_server],
            total_count=42,
            next_cursor=None,
        )

        result = cli_runner.invoke(search, ["--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["total_count"] == 42


@pytest.mark.unit
class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_server_with_no_packages_shows_default_transport(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
    ) -> None:
        """Test that servers with no packages show default 'stdio' transport."""
        _, mock_instance = mock_registry_client
        server_no_packages = RegistryServer(
            name="io.github.example/no-packages",
            description="Server without packages",
            packages=[],
        )
        mock_instance.search.return_value = SearchResult(
            servers=[server_no_packages],
            total_count=1,
            next_cursor=None,
        )

        result = cli_runner.invoke(search, [])

        assert result.exit_code == 0
        assert "stdio" in result.output

    def test_json_server_with_no_packages_shows_default_transport(
        self,
        cli_runner: CliRunner,
        mock_registry_client: tuple,
    ) -> None:
        """Test that JSON output for servers with no packages shows 'stdio'."""
        _, mock_instance = mock_registry_client
        server_no_packages = RegistryServer(
            name="io.github.example/no-packages",
            description="Server without packages",
            packages=[],
        )
        mock_instance.search.return_value = SearchResult(
            servers=[server_no_packages],
            total_count=1,
            next_cursor=None,
        )

        result = cli_runner.invoke(search, ["--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["servers"][0]["transports"] == ["stdio"]
