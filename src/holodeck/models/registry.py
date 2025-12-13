"""MCP Registry API data models.

This module defines Pydantic models that map to the MCP Registry API
response structure at https://registry.modelcontextprotocol.io.

These models are used by the MCPRegistryClient to parse API responses
and provide type-safe access to server metadata.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EnvVarConfig(BaseModel):
    """Environment variable requirement from MCP registry.

    Describes an environment variable that an MCP server requires
    for configuration (e.g., API keys, credentials).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Environment variable name")
    description: str | None = Field(None, description="Description of the variable")
    required: bool = Field(True, description="Whether the variable is required")


class TransportConfig(BaseModel):
    """Transport configuration for MCP server communication.

    Defines how to connect to an MCP server, supporting stdio (local process),
    SSE (server-sent events), or streamable HTTP transports.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["stdio", "sse", "streamable-http"] = Field(
        ..., description="Transport protocol type"
    )
    url: str | None = Field(
        None, description="Server URL for HTTP-based transports (sse, streamable-http)"
    )


class RegistryServerPackage(BaseModel):
    """Package distribution information from MCP registry.

    Describes how to install and run an MCP server, including the package
    manager, package identifier, and required environment variables.
    """

    model_config = ConfigDict(extra="forbid")

    registry_type: Literal["npm", "pypi", "docker", "oci", "nuget", "mcpb"] = Field(
        ..., description="Package registry type (npm, pypi, docker, etc.)"
    )
    identifier: str = Field(
        ..., description="Package identifier (e.g., '@modelcontextprotocol/server-fs')"
    )
    version: str | None = Field(None, description="Package version")
    transport: TransportConfig = Field(..., description="Transport configuration")
    environment_variables: list[EnvVarConfig] = Field(
        default_factory=list, description="Required environment variables"
    )


class RepositoryInfo(BaseModel):
    """Source repository information for an MCP server.

    Links to the source code repository where the MCP server is maintained.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="Repository URL")
    source: str | None = Field(
        None, description="Repository host (github, gitlab, etc.)"
    )


class RegistryServerMeta(BaseModel):
    """Registry metadata for an MCP server.

    Contains administrative information about the server's status and
    publication history in the registry.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["active", "deprecated", "deleted"] = Field(
        default="active", description="Server status in registry"
    )
    published_at: datetime | None = Field(
        None, description="When the server was first published"
    )
    updated_at: datetime | None = Field(
        None, description="When the server was last updated"
    )
    is_latest: bool = Field(False, description="Whether this is the latest version")


class RegistryServer(BaseModel):
    """Complete MCP server representation from registry.

    The main model representing an MCP server as returned by the registry API.
    Contains all metadata needed to discover, display, and install a server.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Server name in reverse-DNS format (e.g., 'io.github.user/server')",
    )
    description: str = Field(..., description="Human-readable server description")
    title: str | None = Field(None, description="Display title for the server")
    version: str = Field(..., description="Server version")
    repository: RepositoryInfo | None = Field(
        None, description="Source repository information"
    )
    website_url: str | None = Field(None, description="Project website URL")
    packages: list[RegistryServerPackage] = Field(
        default_factory=list, description="Available package distributions"
    )
    meta: RegistryServerMeta | None = Field(
        None, description="Registry metadata (status, timestamps)"
    )


class SearchResult(BaseModel):
    """Search result from MCP registry with pagination.

    Returned by the registry search endpoint with a list of matching servers
    and pagination information for fetching additional results.
    """

    model_config = ConfigDict(extra="forbid")

    servers: list[RegistryServer] = Field(
        default_factory=list, description="List of matching servers"
    )
    next_cursor: str | None = Field(
        None, description="Pagination cursor for next page of results"
    )
    total_count: int = Field(0, description="Total number of matching servers")
