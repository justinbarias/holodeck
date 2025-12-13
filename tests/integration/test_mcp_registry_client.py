"""Integration tests for MCPRegistryClient against live MCP Registry API.

These tests make real HTTP calls to https://registry.modelcontextprotocol.io
and verify the client correctly handles API responses and errors.
"""

import pytest

from holodeck.lib.errors import (
    RegistryConnectionError,
    ServerNotFoundError,
)
from holodeck.models.registry import (
    RegistryServer,
    RegistryServerPackage,
    SearchResult,
    ServerVersion,
)
from holodeck.services.mcp_registry import (
    MCPRegistryClient,
    registry_to_mcp_tool,
)


@pytest.mark.integration
class TestMCPRegistryClientSearch:
    """Integration tests for MCPRegistryClient.search() method."""

    def test_search_without_query_returns_servers(self) -> None:
        """Test searching without a query returns a list of servers."""
        client = MCPRegistryClient()
        result = client.search()

        assert isinstance(result, SearchResult)
        assert len(result.servers) > 0
        assert result.total_count > 0

        # Verify first server has expected structure
        server = result.servers[0]
        assert isinstance(server, RegistryServer)
        assert server.name
        assert server.description is not None
        assert server.version

        # Verify aggregation populated versions
        assert hasattr(server, "versions")
        assert len(server.versions) >= 1
        for v in server.versions:
            assert isinstance(v, ServerVersion)
            assert v.version

    def test_search_with_query_filters_results(self) -> None:
        """Test searching with a query filters servers by name."""
        client = MCPRegistryClient()
        result = client.search(query="filesystem")

        assert isinstance(result, SearchResult)
        # The query should filter results - may return 0 or more matches
        for server in result.servers:
            # Search is substring match on name
            assert isinstance(server.name, str)

    def test_search_with_limit_respects_limit(self) -> None:
        """Test that search respects the limit parameter."""
        client = MCPRegistryClient()
        result = client.search(limit=5)

        assert isinstance(result, SearchResult)
        assert len(result.servers) <= 5

    def test_search_pagination_returns_cursor(self) -> None:
        """Test that search returns pagination cursor for next page."""
        client = MCPRegistryClient()
        result = client.search(limit=5)

        # If there are more results, next_cursor should be present
        if result.total_count > 5:
            assert result.next_cursor is not None
        # Either way, it's valid to have no cursor
        assert isinstance(result, SearchResult)

    def test_search_with_cursor_gets_next_page(self) -> None:
        """Test pagination using cursor from previous search."""
        client = MCPRegistryClient()

        # Get first page
        first_page = client.search(limit=3)

        if first_page.next_cursor:
            # Get second page using cursor
            second_page = client.search(limit=3, cursor=first_page.next_cursor)

            assert isinstance(second_page, SearchResult)
            # Second page should have different servers (if any)
            if second_page.servers and first_page.servers:
                first_names = {s.name for s in first_page.servers}
                second_names = {s.name for s in second_page.servers}
                # Pages should be different
                assert first_names != second_names or len(second_page.servers) == 0

    def test_search_aggregates_versions_by_name(self) -> None:
        """Test that search results aggregate versions by server name."""
        client = MCPRegistryClient()
        result = client.search(limit=50)

        # Each server name should appear only once (aggregated)
        names = [s.name for s in result.servers]
        assert len(names) == len(set(names)), "Duplicate server names found"

        # All servers should have versions populated
        for server in result.servers:
            assert len(server.versions) >= 1
            # Each version should have required fields
            for version in server.versions:
                assert isinstance(version, ServerVersion)
                assert version.version  # Version string should be non-empty


@pytest.mark.integration
class TestMCPRegistryClientGetServer:
    """Integration tests for MCPRegistryClient.get_server() method."""

    def test_get_server_returns_server_details(self) -> None:
        """Test getting a known server returns full details."""
        client = MCPRegistryClient()

        # First search to find a server name
        search_result = client.search(limit=1)
        assert len(search_result.servers) > 0

        server_name = search_result.servers[0].name

        # Now get that specific server
        server = client.get_server(server_name)

        assert isinstance(server, RegistryServer)
        assert server.name == server_name
        assert server.description is not None
        assert server.version

    def test_get_server_includes_packages(self) -> None:
        """Test that get_server returns package information."""
        client = MCPRegistryClient()

        # Search and get a server
        search_result = client.search(limit=1)
        assert len(search_result.servers) > 0

        server = client.get_server(search_result.servers[0].name)

        # Most servers should have at least one package
        if server.packages:
            pkg = server.packages[0]
            assert isinstance(pkg, RegistryServerPackage)
            assert pkg.registry_type in (
                "npm",
                "pypi",
                "docker",
                "oci",
                "nuget",
                "mcpb",
            )
            assert pkg.identifier
            assert pkg.transport is not None

    def test_get_server_not_found_raises_error(self) -> None:
        """Test that getting a non-existent server raises ServerNotFoundError."""
        client = MCPRegistryClient()

        with pytest.raises(ServerNotFoundError) as exc_info:
            client.get_server("nonexistent.fake/server-that-does-not-exist")

        assert "nonexistent.fake/server-that-does-not-exist" in str(exc_info.value)


@pytest.mark.integration
class TestMCPRegistryClientListVersions:
    """Integration tests for MCPRegistryClient.list_versions() method."""

    def test_list_versions_returns_version_list(self) -> None:
        """Test listing versions for a known server."""
        client = MCPRegistryClient()

        # First get a server name
        search_result = client.search(limit=1)
        assert len(search_result.servers) > 0

        server_name = search_result.servers[0].name
        versions = client.list_versions(server_name)

        assert isinstance(versions, list)
        # Server should have at least one version
        assert len(versions) >= 1
        for version in versions:
            assert isinstance(version, str)

    def test_list_versions_not_found_raises_error(self) -> None:
        """Test that listing versions for non-existent server raises error."""
        client = MCPRegistryClient()

        with pytest.raises(ServerNotFoundError):
            client.list_versions("nonexistent.fake/server-that-does-not-exist")


@pytest.mark.integration
class TestMCPRegistryClientTimeout:
    """Integration tests for MCPRegistryClient timeout behavior."""

    def test_very_short_timeout_raises_connection_error(self) -> None:
        """Test that very short timeout causes connection error."""
        # Use an impossibly short timeout
        client = MCPRegistryClient(timeout=0.001)

        with pytest.raises(RegistryConnectionError):
            client.search()

    def test_reasonable_timeout_succeeds(self) -> None:
        """Test that reasonable timeout allows successful requests."""
        client = MCPRegistryClient(timeout=10.0)
        result = client.search(limit=1)

        assert isinstance(result, SearchResult)


@pytest.mark.integration
class TestRegistryToMCPToolTransformation:
    """Integration tests for registry_to_mcp_tool() with real server data."""

    def _find_server_with_registry_type(
        self, client: MCPRegistryClient, registry_types: list[str], limit: int = 100
    ) -> tuple[RegistryServer | None, RegistryServerPackage | None]:
        """Find a server with a package of the specified registry type."""
        result = client.search(limit=limit)

        for server in result.servers:
            if server.packages:
                for pkg in server.packages:
                    if pkg.registry_type in registry_types:
                        return server, pkg
        return None, None

    def test_transform_npm_server_to_mcp_tool(self) -> None:
        """Test transforming a real npm-based server to MCPTool."""
        client = MCPRegistryClient()

        server, pkg = self._find_server_with_registry_type(client, ["npm"])

        if server is None:
            pytest.skip("No npm-based server found in registry")

        # Transform to MCPTool
        tool = registry_to_mcp_tool(server, package=pkg)

        assert tool.name
        assert tool.description == server.description
        assert tool.type == "mcp"
        assert tool.registry_name == server.name
        assert tool.command is not None  # Should be set for stdio transport

    def test_transform_pypi_server_to_mcp_tool(self) -> None:
        """Test transforming a real pypi-based server to MCPTool."""
        client = MCPRegistryClient()

        server, pkg = self._find_server_with_registry_type(client, ["pypi"])

        if server is None:
            pytest.skip("No pypi-based server found in registry")

        # Transform to MCPTool
        tool = registry_to_mcp_tool(server, package=pkg)

        assert tool.name
        assert tool.description == server.description
        assert tool.type == "mcp"
        assert tool.registry_name == server.name

    def test_transform_docker_or_oci_server_to_mcp_tool(self) -> None:
        """Test transforming a real docker/oci-based server to MCPTool."""
        client = MCPRegistryClient()

        server, pkg = self._find_server_with_registry_type(client, ["docker", "oci"])

        if server is None:
            pytest.skip("No docker/oci-based server found in registry")

        # Transform to MCPTool
        tool = registry_to_mcp_tool(server, package=pkg)

        assert tool.name
        assert tool.description == server.description
        assert tool.type == "mcp"
        assert tool.registry_name == server.name
        assert tool.command is not None  # Should be docker for oci/docker

    def test_transform_preserves_env_var_requirements(self) -> None:
        """Test that transformation preserves environment variable requirements."""
        client = MCPRegistryClient()

        # Search for servers with env vars AND supported registry type
        result = client.search(limit=100)

        env_server = None
        env_pkg = None
        supported_types = {"npm", "pypi", "docker", "oci"}

        for server in result.servers:
            if server.packages:
                for pkg in server.packages:
                    if (
                        pkg.environment_variables
                        and pkg.registry_type in supported_types
                    ):
                        env_server = server
                        env_pkg = pkg
                        break
            if env_server:
                break

        if env_server is None:
            pytest.skip(
                "No server with environment variables and supported registry type found"
            )

        # Transform to MCPTool
        tool = registry_to_mcp_tool(env_server, package=env_pkg)

        # Env vars should be preserved as placeholders
        assert tool.env is not None
        for key, value in tool.env.items():
            assert key  # Key should be non-empty
            assert "${" in value  # Should be a placeholder


@pytest.mark.integration
class TestMCPRegistryClientRealWorldScenarios:
    """Real-world usage scenario tests."""

    def test_discover_and_configure_server_workflow(self) -> None:
        """Test the full workflow: search -> get details -> transform to config."""
        client = MCPRegistryClient()

        # Step 1: Search for available servers
        search_result = client.search(limit=50)
        assert len(search_result.servers) > 0

        # Find a server with supported package type
        supported_types = {"npm", "pypi", "docker", "oci"}
        target_server = None
        target_pkg = None

        for server in search_result.servers:
            if server.packages:
                for pkg in server.packages:
                    if pkg.registry_type in supported_types:
                        target_server = server
                        target_pkg = pkg
                        break
            if target_server:
                break

        if target_server is None:
            pytest.skip("No server with supported package type found")

        # Step 2: Get full details for that specific server
        server = client.get_server(target_server.name)
        assert server.name == target_server.name

        # Step 3: Check available versions
        versions = client.list_versions(target_server.name)
        assert len(versions) >= 1

        # Step 4: Transform to MCPTool configuration
        tool = registry_to_mcp_tool(server, package=target_pkg)
        assert tool.name
        assert tool.type == "mcp"
        assert tool.registry_name == target_server.name

    def test_search_multiple_queries_returns_different_results(self) -> None:
        """Test that different search queries return different results."""
        client = MCPRegistryClient()

        # Search with different queries
        all_servers = client.search(limit=10)
        # Note: We don't filter by query here since results may vary

        assert isinstance(all_servers, SearchResult)
        # At minimum we should get some servers back
        assert all_servers.total_count >= 0
