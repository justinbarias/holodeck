"""MCP Registry API client.

This module provides the MCPRegistryClient for interacting with the
official MCP Registry at https://registry.modelcontextprotocol.io.
"""

import logging
from urllib.parse import quote

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout

from holodeck.lib.errors import (
    RegistryAPIError,
    RegistryConnectionError,
    ServerNotFoundError,
)
from holodeck.lib.validation import sanitize_tool_name
from holodeck.models.registry import (
    RegistryServer,
    RegistryServerMeta,
    RegistryServerPackage,
    SearchResult,
    ServerVersion,
)
from holodeck.models.tool import CommandType, MCPTool, TransportType

logger = logging.getLogger(__name__)

# Supported package registry types for stdio transport
# Maps to command types: npm -> npx, pypi -> uvx, docker/oci -> docker
SUPPORTED_REGISTRY_TYPES: frozenset[str] = frozenset({"npm", "pypi", "docker", "oci"})


class MCPRegistryClient:
    """Client for MCP Registry API.

    Provides methods to search, retrieve, and list MCP servers from
    the official registry.

    Example:
        >>> client = MCPRegistryClient()
        >>> result = client.search(query="filesystem")
        >>> for server in result.servers:
        ...     print(f"{server.name}: {server.description}")
    """

    DEFAULT_BASE_URL = "https://registry.modelcontextprotocol.io"
    DEFAULT_TIMEOUT = 5.0  # seconds - fail fast

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize client with base URL and timeout.

        Args:
            base_url: Registry API base URL
            timeout: Request timeout in seconds (default: 5.0)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def __enter__(self) -> "MCPRegistryClient":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and close session."""
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP session.

        Should be called when done with the client to release resources.
        Alternatively, use the client as a context manager.

        Example:
            >>> with MCPRegistryClient() as client:
            ...     result = client.search("filesystem")
            # Session automatically closed
        """
        if hasattr(self, "_session") and self._session is not None:
            self._session.close()

    def search(
        self,
        query: str | None = None,
        limit: int = 25,
        cursor: str | None = None,
    ) -> SearchResult:
        """Search for MCP servers.

        Args:
            query: Optional search term (substring match on name)
            limit: Maximum results per page (default 25)
            cursor: Pagination cursor for next page

        Returns:
            SearchResult with servers and pagination info

        Raises:
            RegistryConnectionError: Network/timeout issues
            RegistryAPIError: API returned error status
        """
        logger.debug("Searching MCP registry: query=%s, limit=%d", query, limit)
        url = f"{self.base_url}/v0.1/servers"
        params: dict[str, str | int] = {"limit": limit}

        if query:
            params["search"] = query
        if cursor:
            params["cursor"] = cursor

        response = self._request("GET", url, params=params)
        data = response.json()

        # Parse response structure
        servers: list[RegistryServer] = []
        for item in data.get("servers", []):
            server_data = item.get("server", item)
            meta_data = item.get("_meta")
            servers.append(self._parse_server(server_data, meta_data))

        # Aggregate servers by name (each version comes as separate entry)
        servers = self._aggregate_by_name(servers)

        metadata = data.get("metadata", {})
        result = SearchResult(
            servers=servers,
            next_cursor=metadata.get("nextCursor"),
            total_count=metadata.get("count", len(servers)),
        )
        logger.debug("Search returned %d unique servers", len(result.servers))
        return result

    def get_server(
        self,
        name: str,
        version: str = "latest",
    ) -> RegistryServer:
        """Get specific server by name and version.

        Args:
            name: Server name (reverse-DNS format)
            version: Version string or "latest"

        Returns:
            RegistryServer with full details

        Raises:
            ServerNotFoundError: Server doesn't exist
            RegistryConnectionError: Network/timeout issues
        """
        logger.debug("Fetching server: name=%s, version=%s", name, version)
        # URL-encode the server name (contains '/' in reverse-DNS format)
        encoded_name = quote(name, safe="")
        url = f"{self.base_url}/v0.1/servers/{encoded_name}/versions/{version}"

        try:
            response = self._request("GET", url)
        except RegistryAPIError as e:
            if e.status_code == 404:
                logger.debug("Server not found: %s", name)
                raise ServerNotFoundError(name) from e
            raise

        data = response.json()
        server_data = data.get("server", data)
        server = self._parse_server(server_data, data.get("_meta"))
        logger.debug("Retrieved server: %s v%s", server.name, server.version)
        return server

    def list_versions(self, name: str) -> list[str]:
        """List available versions for a server.

        Args:
            name: Server name (reverse-DNS format)

        Returns:
            List of version strings (newest first)

        Raises:
            ServerNotFoundError: Server doesn't exist
            RegistryConnectionError: Network/timeout issues
        """
        logger.debug("Listing versions for server: %s", name)
        # URL-encode the server name (contains '/' in reverse-DNS format)
        encoded_name = quote(name, safe="")
        url = f"{self.base_url}/v0.1/servers/{encoded_name}/versions"

        try:
            response = self._request("GET", url)
        except RegistryAPIError as e:
            if e.status_code == 404:
                logger.debug("Server not found: %s", name)
                raise ServerNotFoundError(name) from e
            raise

        data = response.json()
        versions: list[str] = []
        # API returns servers array with version info embedded in each server object
        for item in data.get("servers", []):
            server_data = item.get("server", item)
            version_str = server_data.get("version")
            if version_str:
                versions.append(version_str)
        logger.debug("Found %d versions for %s", len(versions), name)
        return versions

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, str | int] | None = None,
    ) -> requests.Response:
        """Execute HTTP request with error handling.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters

        Returns:
            Response object

        Raises:
            RegistryConnectionError: Connection/timeout issues
            RegistryAPIError: Non-2xx status code
        """
        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
            )
        except Timeout as e:
            raise RegistryConnectionError(
                self.base_url,
                original_error=e,
            ) from e
        except RequestsConnectionError as e:
            raise RegistryConnectionError(
                self.base_url,
                original_error=e,
            ) from e

        if not response.ok:
            detail = None
            try:
                error_body = response.json()
                detail = error_body.get("message") or error_body.get("error")
            except (ValueError, KeyError):
                # JSON parsing failed or no error message field
                detail = response.text[:200] if response.text else None
            raise RegistryAPIError(url, response.status_code, detail)

        return response

    def _parse_server(
        self,
        server_data: dict[str, object],
        meta_data: dict[str, object] | None = None,
    ) -> RegistryServer:
        """Parse server data from API response to RegistryServer model.

        Uses Pydantic's model_validate for type-safe parsing with camelCase
        alias support.

        Args:
            server_data: Server data dictionary from API response
            meta_data: Optional metadata dictionary

        Returns:
            RegistryServer instance

        Raises:
            ValidationError: If required fields are missing or invalid
        """
        from pydantic import ValidationError

        # Parse metadata from the nested registry format
        meta: RegistryServerMeta | None = None
        if meta_data:
            registry_meta = meta_data.get("io.modelcontextprotocol.registry/official")
            if isinstance(registry_meta, dict):
                try:
                    meta = RegistryServerMeta.model_validate(registry_meta)
                except ValidationError:
                    logger.debug("Failed to parse server metadata, using defaults")
                    meta = None

        # Build the server data with parsed metadata
        parsed_data = dict(server_data)
        if meta:
            parsed_data["meta"] = meta

        try:
            return RegistryServer.model_validate(parsed_data)
        except ValidationError as e:
            # Log validation errors for debugging
            logger.warning(
                "Failed to validate server data for '%s': %s",
                server_data.get("name", "unknown"),
                e,
            )
            raise

    def _aggregate_by_name(self, servers: list[RegistryServer]) -> list[RegistryServer]:
        """Aggregate servers by name, collecting versions into a list.

        The MCP registry returns each version of a server as a separate entry.
        This method groups them by name and creates a single entry per server
        with all versions in the `versions` field.

        Args:
            servers: List of servers (potentially multiple entries per server name)

        Returns:
            List of servers with one entry per unique name, versions aggregated
        """
        from collections import defaultdict

        # Group servers by name
        grouped: dict[str, list[RegistryServer]] = defaultdict(list)
        for server in servers:
            grouped[server.name].append(server)

        # Create aggregated list
        aggregated: list[RegistryServer] = []
        for server_entries in grouped.values():
            # Use first entry as base (typically the latest version)
            base = server_entries[0]

            # Build versions list from all entries
            versions = [
                ServerVersion(
                    version=s.version,
                    packages=s.packages,
                    meta=s.meta,
                )
                for s in server_entries
            ]

            # Create aggregated server with versions
            aggregated.append(base.model_copy(update={"versions": versions}))

        logger.debug(
            "Aggregated %d server entries into %d unique servers",
            len(servers),
            len(aggregated),
        )
        return aggregated


def find_stdio_package(server: RegistryServer) -> RegistryServerPackage | None:
    """Find a package that supports stdio transport.

    Prioritizes supported registry types (npm, pypi, docker, oci) when
    multiple stdio packages are available.

    Args:
        server: Registry server with packages

    Returns:
        First stdio-compatible package with supported registry type,
        or any stdio package as fallback, or None if none found
    """
    # First pass: find stdio package with supported registry type
    for pkg in server.packages:
        if (
            pkg.transport.type == "stdio"
            and pkg.registry_type in SUPPORTED_REGISTRY_TYPES
        ):
            return pkg

    # Fallback: any stdio package (will fail later with unsupported type error)
    for pkg in server.packages:
        if pkg.transport.type == "stdio":
            return pkg

    return None


def registry_to_mcp_tool(
    server: RegistryServer,
    package: RegistryServerPackage | None = None,
    transport_override: str | None = None,
) -> MCPTool:
    """Convert registry server to MCPTool configuration.

    Transforms an MCP server from the registry into a tool configuration
    that can be added to agent.yaml or global config.

    Args:
        server: RegistryServer from registry API
        package: Specific package to use (defaults to first package)
        transport_override: Optional transport type override

    Returns:
        MCPTool configuration ready for YAML serialization

    Raises:
        ValueError: If server has no packages
    """
    if not server.packages:
        raise ValueError(f"Server '{server.name}' has no packages configured")

    # Use specified package or default to first
    pkg = package or server.packages[0]

    # Map transport type (with override support)
    transport_map: dict[str, TransportType] = {
        "stdio": TransportType.STDIO,
        "sse": TransportType.SSE,
        "streamable-http": TransportType.HTTP,
    }

    if transport_override:
        transport = transport_map.get(transport_override, TransportType.STDIO)
    else:
        transport = transport_map.get(pkg.transport.type, TransportType.STDIO)

    # Map registry type to command
    # OCI is treated like docker (container images)
    command_map: dict[str, CommandType] = {
        "npm": CommandType.NPX,
        "pypi": CommandType.UVX,
        "docker": CommandType.DOCKER,
        "oci": CommandType.DOCKER,
    }
    command = command_map.get(pkg.registry_type)

    # Validate that we have a supported registry type for stdio transport
    if transport == TransportType.STDIO and command is None:
        raise ValueError(
            f"Unsupported registry type '{pkg.registry_type}' for stdio transport. "
            f"Supported types: {', '.join(sorted(SUPPORTED_REGISTRY_TYPES))}"
        )

    # Build args based on registry type
    args: list[str] = []
    if pkg.registry_type == "npm":
        version_suffix = f"@{pkg.version}" if pkg.version else "@latest"
        args = ["-y", f"{pkg.identifier}{version_suffix}"]
    elif pkg.registry_type == "pypi":
        args = [f"{pkg.identifier}=={pkg.version}"] if pkg.version else [pkg.identifier]
    elif pkg.registry_type in ("docker", "oci"):
        # OCI identifiers may already include the tag
        if ":" in pkg.identifier:
            args = ["run", "-i", pkg.identifier]
        else:
            version_tag = pkg.version or "latest"
            args = ["run", "-i", f"{pkg.identifier}:{version_tag}"]

    # Extract env vars as placeholders
    env: dict[str, str] | None = None
    if pkg.environment_variables:
        env = {ev.name: f"${{{ev.name}}}" for ev in pkg.environment_variables}

    # Validate server name before extracting short name
    if not server.name:
        raise ValueError("Server name is required")

    # Extract short name from reverse-DNS format (e.g., "io.github.user/server-name")
    raw_name = server.name.split("/")[-1] if "/" in server.name else server.name

    # Sanitize to valid tool name (alphanumeric and underscores only)
    short_name = sanitize_tool_name(raw_name)

    return MCPTool(
        name=short_name,
        description=server.description,
        type="mcp",
        transport=transport,
        command=command if transport == TransportType.STDIO else None,
        args=args if transport == TransportType.STDIO else None,
        env=env,
        env_file=None,
        encoding=None,
        url=pkg.transport.url if transport != TransportType.STDIO else None,
        headers=None,
        timeout=None,
        sse_read_timeout=None,
        terminate_on_close=None,
        config=None,
        load_tools=True,
        load_prompts=True,
        request_timeout=60,
        is_retrieval=False,
        registry_name=server.name,  # Store full name for duplicate detection
    )
