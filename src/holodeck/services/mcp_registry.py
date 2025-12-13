"""MCP Registry API client.

This module provides the MCPRegistryClient for interacting with the
official MCP Registry at https://registry.modelcontextprotocol.io.
"""

import contextlib
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
from holodeck.models.registry import (
    EnvVarConfig,
    RegistryServer,
    RegistryServerMeta,
    RegistryServerPackage,
    RepositoryInfo,
    SearchResult,
    TransportConfig,
)
from holodeck.models.tool import CommandType, MCPTool, TransportType

logger = logging.getLogger(__name__)


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

        metadata = data.get("metadata", {})
        return SearchResult(
            servers=servers,
            next_cursor=metadata.get("nextCursor"),
            total_count=metadata.get("count", len(servers)),
        )

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
        # URL-encode the server name (contains '/' in reverse-DNS format)
        encoded_name = quote(name, safe="")
        url = f"{self.base_url}/v0.1/servers/{encoded_name}/versions/{version}"

        try:
            response = self._request("GET", url)
        except RegistryAPIError as e:
            if e.status_code == 404:
                raise ServerNotFoundError(name) from e
            raise

        data = response.json()
        server_data = data.get("server", data)
        return self._parse_server(server_data, data.get("_meta"))

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
        # URL-encode the server name (contains '/' in reverse-DNS format)
        encoded_name = quote(name, safe="")
        url = f"{self.base_url}/v0.1/servers/{encoded_name}/versions"

        try:
            response = self._request("GET", url)
        except RegistryAPIError as e:
            if e.status_code == 404:
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
            with contextlib.suppress(Exception):
                detail = response.json().get("message")
            raise RegistryAPIError(url, response.status_code, detail)

        return response

    def _parse_server(
        self,
        server_data: dict[str, object],
        meta_data: dict[str, object] | None = None,
    ) -> RegistryServer:
        """Parse server data from API response to RegistryServer model.

        Args:
            server_data: Server data dictionary
            meta_data: Optional metadata dictionary

        Returns:
            RegistryServer instance
        """
        from typing import Any, cast

        # Cast to Any for dynamic access since API response structure is known
        data = cast(dict[str, Any], server_data)

        # Parse packages with camelCase to snake_case conversion
        packages: list[RegistryServerPackage] = []
        raw_packages = data.get("packages", [])
        if isinstance(raw_packages, list):
            for pkg in raw_packages:
                if not isinstance(pkg, dict):
                    continue
                transport_data = pkg.get("transport", {})
                if not isinstance(transport_data, dict):
                    transport_data = {}

                env_vars: list[EnvVarConfig] = []
                raw_env_vars = pkg.get("environmentVariables", [])
                if isinstance(raw_env_vars, list):
                    for ev in raw_env_vars:
                        if isinstance(ev, dict):
                            desc = ev.get("description")
                            env_vars.append(
                                EnvVarConfig(
                                    name=str(ev.get("name", "")),
                                    description=str(desc) if desc else None,
                                    required=bool(ev.get("required", True)),
                                )
                            )

                registry_type = str(pkg.get("registryType", "npm"))
                transport_type = str(transport_data.get("type", "stdio"))
                transport_url = transport_data.get("url")
                pkg_version = pkg.get("version")

                packages.append(
                    RegistryServerPackage(
                        registry_type=registry_type,  # type: ignore[arg-type]
                        identifier=str(pkg.get("identifier", "")),
                        version=str(pkg_version) if pkg_version else None,
                        transport=TransportConfig(
                            type=transport_type,  # type: ignore[arg-type]
                            url=str(transport_url) if transport_url else None,
                        ),
                        environment_variables=env_vars,
                    )
                )

        # Parse repository
        repo: RepositoryInfo | None = None
        repo_data = data.get("repository")
        if isinstance(repo_data, dict):
            source = repo_data.get("source")
            repo = RepositoryInfo(
                url=str(repo_data.get("url", "")),
                source=str(source) if source else None,
            )

        # Parse metadata
        meta: RegistryServerMeta | None = None
        if meta_data:
            meta_dict = cast(dict[str, Any], meta_data)
            registry_meta = meta_dict.get(
                "io.modelcontextprotocol.registry/official", {}
            )
            if isinstance(registry_meta, dict):
                status = registry_meta.get("status", "active")
                published = registry_meta.get("publishedAt")
                updated = registry_meta.get("updatedAt")
                meta = RegistryServerMeta(
                    status=str(status) if status else "active",  # type: ignore[arg-type]
                    published_at=published,
                    updated_at=updated,
                    is_latest=bool(registry_meta.get("isLatest", False)),
                )

        title = data.get("title")
        website = data.get("websiteUrl")

        return RegistryServer(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            title=str(title) if title else None,
            version=str(data.get("version", "")),
            repository=repo,
            website_url=str(website) if website else None,
            packages=packages,
            meta=meta,
        )


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
        supported_types = ", ".join(command_map.keys())
        raise ValueError(
            f"Unsupported registry type '{pkg.registry_type}' for stdio transport. "
            f"Supported types: {supported_types}"
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

    # Extract short name for display
    short_name = server.name.split("/")[-1] if "/" in server.name else server.name

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
